import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from typing import List, Dict, Optional, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class LLMWrapper(nn.Module):
    """
    LLM封装器类
    封装HuggingFace Transformers模型，添加关系分类头，支持LoRA微调
    """
    
    def __init__(self, 
                 model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
                 num_relations: int = 7,
                 output_mode: str = "relation_head",
                 use_lora: bool = True,
                 lora_r: int = 16,
                 lora_alpha: int = 32,
                 lora_dropout: float = 0.05,
                 use_quantization: bool = False,
                 load_in_4bit: bool = False,
                 load_in_8bit: bool = False,
                 bnb_4bit_compute_dtype: str = "float16",
                 bnb_4bit_quant_type: str = "nf4",
                 max_length: int = 512,
                 device: str = "cuda",
                 target_modules: Optional[List[str]] = None,
                 enable_gradient_checkpointing: bool = True,
                 train_backbone: bool = False,
                 log_inputs: bool = False,
                 generation_max_new_tokens: int = 16,
                 generation_do_sample: bool = False,
                 generation_temperature: float = 0.1,
                 generation_top_p: float = 0.9,
                 generation_top_k: int = 50
                 ):
        """
        初始化LLM封装器
        
        Args:
            model_name (str): 模型名称或路径
            num_relations (int): 关系类型数量
            use_lora (bool): 是否使用LoRA
            lora_r (int): LoRA秩
            lora_alpha (int): LoRA缩放因子
            lora_dropout (float): LoRA dropout率
            load_in_8bit (bool): 是否以8bit加载模型
            max_length (int): 最大序列长度
            device (str): 设备
            target_modules (List[str], optional): LoRA 针对的模块
        """
        super().__init__()
        
        self.model_name = model_name
        self.num_relations = num_relations
        self.output_mode = str(output_mode or "relation_head")
        self.use_lora = use_lora
        self.max_length = max_length
        self.target_modules = target_modules or ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        # 设备选择：在分布式训练中，使用传入的 device
        self.device = torch.device(device)
        self.use_fallback = False
        self.hidden_dim = 256  # 默认隐藏维度
        self.train_backbone = bool(train_backbone)
        self.log_inputs = bool(log_inputs)
        self.generation_max_new_tokens = int(generation_max_new_tokens)
        self.generation_do_sample = bool(generation_do_sample)
        self.generation_temperature = float(generation_temperature)
        self.generation_top_p = float(generation_top_p)
        self.generation_top_k = int(generation_top_k)
        self.relation_labels_en = [
            "Activation",
            "Inhibition",
            "Binding",
            "Catalysis",
            "Expression Regulation",
            "Physical Interaction",
            "Genetic Interaction",
        ]
        self.backbone_trainable = False
        self.is_kbit_model = False

        # 加载模型和分词器
        resolved_model_name = model_name
        try:
            repo_root = Path(__file__).resolve().parents[1]
            raw = str(model_name).rstrip("/")
            if raw:
                if raw.startswith("models/") or raw.startswith("./models/") or raw.startswith("../models/"):
                    candidate = (repo_root / raw).resolve()
                    if candidate.exists():
                        resolved_model_name = str(candidate)
                else:
                    p = Path(raw)
                    if not p.exists() and "/models/llm/" in raw:
                        suffix = raw.split("/models/llm/", 1)[1].strip("/")
                        candidate = (repo_root / "models" / "llm" / suffix).resolve()
                        if candidate.exists():
                            resolved_model_name = str(candidate)
        except Exception:
            pass

        self.model_name = resolved_model_name
        logger.info(f"正在加载模型: {resolved_model_name}")
        try:
            if self.device.type != "cuda":
                raise RuntimeError("当前环境为 CPU，仅做联调验证时使用本地回退分类器；如需真实加载 Llama 3B，请使用 CUDA 环境。")
            bnb_config = None
            is_kbit_model = False
            if bool(use_quantization) and (bool(load_in_4bit) or bool(load_in_8bit)):
                try:
                    compute_dtype = torch.float16
                    if str(bnb_4bit_compute_dtype).lower() in ("bfloat16", "bf16"):
                        compute_dtype = torch.bfloat16
                    if bool(load_in_4bit):
                        bnb_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_quant_type=str(bnb_4bit_quant_type),
                            bnb_4bit_compute_dtype=compute_dtype,
                        )
                        is_kbit_model = True
                        logger.info("启用bitsandbytes 4bit量化加载")
                    else:
                        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
                        is_kbit_model = True
                        logger.info("启用bitsandbytes 8bit量化加载")
                except Exception as e:
                    logger.warning(f"bitsandbytes量化不可用，将使用非量化加载: {e}")

            self.tokenizer = AutoTokenizer.from_pretrained(resolved_model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 在 DDP 环境下，不能使用 device_map="auto"
            # 我们根据当前的进程 rank 手动指定 device
            self.model = AutoModelForCausalLM.from_pretrained(
                resolved_model_name,
                quantization_config=bnb_config,
                torch_dtype=torch.float16,
                device_map={"": self.device.index} if self.device.type == "cuda" else None
            )
            self.is_kbit_model = bool(is_kbit_model)
            
            # 获取模型隐藏层维度
            self.hidden_dim = self.model.config.hidden_size
            logger.info(f"LLM模型 {resolved_model_name} 成功加载到设备 {self.device}")
            logger.info(f"模型参数: hidden_size={self.hidden_dim}, vocab_size={self.model.config.vocab_size}")
        except Exception as e:
            logger.warning(f"LLM加载失败，启用本地回退分类器: {e}")
            self.tokenizer = None
            self.model = None
            self.use_fallback = True
            # 简单字符嵌入编码器
            self.char_embed = nn.Embedding(256, self.hidden_dim).to(self.device)
        
        # 配置LoRA（仅当成功加载模型）
        if use_lora and not self.use_fallback:
            logger.info(f"配置LoRA，r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
            
            # 准备模型进行kbit训练
            if bool(self.is_kbit_model):
                self.model = prepare_model_for_kbit_training(self.model)
            
            # LoRA配置
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=self.target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM"
            )
            
            # 创建LoRA模型
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()

        if self.model is not None:
            try:
                self.backbone_trainable = any(
                    bool(getattr(p, "requires_grad", False)) for p in self.model.parameters()
                )
            except Exception:
                self.backbone_trainable = False

            if bool(enable_gradient_checkpointing) and self.backbone_trainable:
                try:
                    self.model.gradient_checkpointing_enable()
                except Exception:
                    pass
            else:
                try:
                    self.model.gradient_checkpointing_disable()
                except Exception:
                    pass
        
        # 添加关系分类头
        logger.info(f"添加关系分类头，输入维度: {self.hidden_dim}, 输出维度: {num_relations}")
        self.relation_classifier = nn.Linear(self.hidden_dim, num_relations).to(self.device, dtype=torch.float32)
        
        logger.info(f"LLMWrapper初始化完成，模型: {resolved_model_name}, 设备: {self.device}")
    
    def get_cls_embedding(self, outputs: Dict) -> torch.Tensor:
        """
        获取CLS嵌入（对于不同模型有不同的实现）
        
        Args:
            outputs (Dict): 模型输出
            
        Returns:
            torch.Tensor: CLS嵌入
        """
        # 对于因果LM，优先使用hidden_states获取最后一个token的嵌入
        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            last_hidden_state = outputs.last_hidden_state  # [batch_size, seq_len, hidden_dim]
            cls_embedding = last_hidden_state[:, -1, :]
        elif hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            last_hidden = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_dim]
            cls_embedding = last_hidden[:, -1, :]
        else:
            # 回退到logits进行表示（近似），取最后一个token的logits并线性映射到hidden_dim
            logits = outputs.logits  # [batch_size, seq_len, vocab_size]
            last_logits = logits[:, -1, :]  # [batch_size, vocab_size]
            # 使用一个简单的线性层将vocab空间映射到hidden_dim（不训练，仅用于占位）
            with torch.no_grad():
                proj = torch.randn(last_logits.size(-1), self.hidden_dim, device=last_logits.device) * 0.01
                cls_embedding = last_logits @ proj
        
        return cls_embedding
    
    def forward(self, 
               texts: List[str],
               labels: Optional[torch.Tensor] = None
               ) -> Dict:
        """
        前向传播
        
        Args:
            texts (List[str]): 输入文本列表
            labels (Optional[torch.Tensor]): 标签张量 [batch_size]
            
        Returns:
            Dict: 包含logits、loss（如果提供标签）的字典
        """
        if self.use_fallback:
            # 基于字符的简单编码：求平均嵌入
            batch_embeddings = []
            for text in texts:
                # 将文本转换为字符ID
                char_ids = [ord(c) % 256 for c in text[:self.max_length]]
                if len(char_ids) == 0:
                    char_ids = [0]
                char_tensor = torch.tensor(char_ids, dtype=torch.long, device=self.device)
                emb = self.char_embed(char_tensor)  # [seq_len, hidden_dim]
                cls_emb = emb.mean(dim=0)  # [hidden_dim]
                batch_embeddings.append(cls_emb)
            cls_embedding = torch.stack(batch_embeddings, dim=0)
        else:
            # 分词
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
            )
            
            # 将输入移到设备上
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 模型前向传播
            backbone_requires_grad = bool(self.training and self.backbone_trainable)
            with (torch.enable_grad() if backbone_requires_grad else torch.no_grad()):
                if self.log_inputs:
                    for i, text in enumerate(texts):
                        logger.info("="*30 + f" LLM Input {i+1} " + "="*30)
                        logger.info(f"Text:\n{text}")
                        logger.info("="*75)
                base_model = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
                decoder = getattr(base_model, "model", None)
                if decoder is not None:
                    outputs = decoder(**inputs, output_hidden_states=False, use_cache=False)
                else:
                    outputs = self.model(**inputs, output_hidden_states=True)
            
            # 获取CLS嵌入
            cls_embedding = self.get_cls_embedding(outputs)
        
        # 关系分类
        if cls_embedding.dtype != torch.float32:
            cls_embedding = cls_embedding.float()
        relation_logits = self.relation_classifier(cls_embedding)  # [batch_size, num_relations]
        
        # 计算损失
        loss = None
        if labels is not None:
            labels = labels.to(self.device).float()  # 多标签需要 float 类型的 target
            # 使用 BCEWithLogitsLoss 处理多标签分类
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(relation_logits.float(), labels)
        
        return {
            'logits': relation_logits,
            'loss': loss,
            'cls_embedding': cls_embedding
        }
    
    def predict(self, 
                texts: List[str],
                return_type: str = 'probabilities' # 'probabilities', 'logits'
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        预测关系类型
        
        Args:
            texts (List[str]): 输入文本列表
            return_type (str): 第二个返回值类型：'probabilities' 或 'logits'
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (预测类别, 概率或Logits)
        """
        if str(self.output_mode).lower() == "text" and not self.use_fallback and self.model is not None and self.tokenizer is not None:
            pred, pseudo_logits = self._predict_via_generation(texts)
            if return_type == "probabilities":
                return pred, torch.sigmoid(pseudo_logits)
            return pred, pseudo_logits

        self.eval()
        with torch.no_grad():
            outputs = self.forward(texts)
            logits = outputs['logits'].float()
        probs = torch.sigmoid(logits)

        threshold = float(getattr(self, "prediction_threshold", 0.5))
        max_labels = getattr(self, "max_labels", None)
        force_topk = getattr(self, "force_topk", None)
        if max_labels is not None:
            try:
                max_labels = int(max_labels)
            except Exception:
                max_labels = None
        if force_topk is not None:
            try:
                force_topk = int(force_topk)
            except Exception:
                force_topk = None

        batch_size, num_rel = probs.size(0), probs.size(1)
        predictions = torch.zeros_like(probs)
        for i in range(batch_size):
            row = probs[i]
            if force_topk is not None and force_topk > 0:
                k = min(int(force_topk), int(num_rel))
                top_idx = torch.topk(row, k=k, largest=True).indices
                predictions[i, top_idx] = 1.0
                continue

            selected = (row >= threshold).nonzero(as_tuple=False).flatten()
            if selected.numel() == 0:
                predictions[i, torch.argmax(row).item()] = 1.0
                continue
            if max_labels is not None and max_labels > 0 and selected.numel() > max_labels:
                top_idx = torch.topk(row, k=int(max_labels), largest=True).indices
                predictions[i, top_idx] = 1.0
                continue
            predictions[i, selected] = 1.0

        if return_type == 'probabilities':
            return predictions, probs
        return predictions, logits
    
    def get_trainable_parameters(self) -> int:
        """
        获取可训练参数数量
        
        Returns:
            int: 可训练参数数量
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def save_model(self, save_path: str) -> None:
        """
        保存模型
        
        Args:
            save_path (str): 保存路径
        """
        logger.info(f"保存模型到 {save_path}")
        
        # 保存模型和分词器
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        # 保存关系分类头
        classifier_path = f"{save_path}/relation_classifier.pt"
        torch.save(self.relation_classifier.state_dict(), classifier_path)
        
        logger.info(f"模型保存完成，关系分类头保存到 {classifier_path}")
    
    @classmethod
    def from_pretrained(cls, 
                       model_path: str,
                       num_relations: int = 7,
                       load_in_8bit: bool = True,
                       device: str = "cuda"
                       ) -> "LLMWrapper":
        """
        从预训练模型加载
        
        Args:
            model_path (str): 模型路径
            num_relations (int): 关系类型数量
            load_in_8bit (bool): 是否以8bit加载
            device (str): 设备
            
        Returns:
            LLMWrapper: LLM封装器实例
        """
        # 创建实例
        wrapper = cls(
            model_name=model_path,
            num_relations=num_relations,
            use_lora=False,  # 先不使用LoRA
            load_in_8bit=load_in_8bit,
            device=device
        )
        
        # 加载关系分类头
        classifier_path = f"{model_path}/relation_classifier.pt"
        if torch.cuda.is_available():
            state_dict = torch.load(classifier_path)
        else:
            state_dict = torch.load(classifier_path, map_location=torch.device('cpu'))
        
        wrapper.relation_classifier.load_state_dict(state_dict)
        
        logger.info(f"从 {model_path} 加载模型完成")
        return wrapper
    
    def generate_text(self, 
                     texts: List[str],
                     max_new_tokens: int = 100,
                     temperature: float = 0.7,
                     top_p: float = 0.9,
                     top_k: int = 50,
                     do_sample: bool = True
                     ) -> List[str]:
        """
        生成文本
        
        Args:
            texts (List[str]): 输入文本列表
            max_new_tokens (int): 生成的最大新token数
            temperature (float): 温度参数
            top_p (float): Top-p采样参数
            
        Returns:
            List[str]: 生成的文本列表
        """
        self.eval()
        
        # 分词
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 生成文本
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=int(top_k),
                do_sample=bool(do_sample),
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # 解码
        generated_texts = self.tokenizer.batch_decode(
            outputs, 
            skip_special_tokens=True
        )
        
        return generated_texts

    def compute_text_loss(self, prompts: List[str], target_texts: List[str]) -> Optional[torch.Tensor]:
        if self.use_fallback or self.model is None or self.tokenizer is None:
            return None
        if len(prompts) != len(target_texts):
            raise ValueError("prompts and target_texts must have the same length")

        self.model.train()
        self.relation_classifier.train()

        joint_texts = [p + "\n" + t for p, t in zip(prompts, target_texts)]

        prompt_inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        joint_inputs = self.tokenizer(
            joint_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        prompt_inputs = {k: v.to(self.device) for k, v in prompt_inputs.items()}
        joint_inputs = {k: v.to(self.device) for k, v in joint_inputs.items()}

        labels = joint_inputs["input_ids"].clone()
        attn = joint_inputs.get("attention_mask")
        if attn is not None:
            labels[attn == 0] = -100

        prompt_lens = prompt_inputs["attention_mask"].sum(dim=1).tolist()
        for i, l in enumerate(prompt_lens):
            l = int(l)
            if l > 0 and l < labels.size(1):
                labels[i, :l] = -100

        out = self.model(**joint_inputs, labels=labels)
        return out.loss

    def _predict_via_generation(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        generated = self.generate_text(
            texts,
            max_new_tokens=self.generation_max_new_tokens,
            temperature=self.generation_temperature,
            top_p=self.generation_top_p,
            top_k=self.generation_top_k,
            do_sample=self.generation_do_sample
        )
        num_rel = int(self.num_relations)
        preds = torch.zeros((len(texts), num_rel), dtype=torch.float32, device=self.device)
        for i, out in enumerate(generated):
            out_l = str(out).lower()
            found_any = False
            for j, name in enumerate(self.relation_labels_en[:num_rel]):
                if str(name).lower() in out_l:
                    preds[i, j] = 1.0
                    found_any = True
            if not found_any:
                fallback_idx = 5 if num_rel > 5 else 0
                preds[i, fallback_idx] = 1.0

        pseudo_logits = torch.full_like(preds, -5.0)
        pseudo_logits[preds > 0] = 5.0
        return preds, pseudo_logits
    
    def get_embeddings(self, texts: List[str]) -> torch.Tensor:
        """
        获取文本嵌入
        
        Args:
            texts (List[str]): 输入文本列表
            
        Returns:
            torch.Tensor: 文本嵌入 [batch_size, hidden_dim]
        """
        self.eval()
        
        with torch.no_grad():
            outputs = self.forward(texts)
            embeddings = outputs['cls_embedding']
        
        return embeddings
