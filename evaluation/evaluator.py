import torch
import torch.nn.functional as F
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import json
import os
import time
from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              roc_auc_score, confusion_matrix, classification_report)

from graph.builder import PPIGraphBuilder
from models.cot_generator import ExploratoryCOTGenerator
from models.component_builder import ComponentBuilder
from models.esm_encoder import ESMEncoder

from llm.prompt_designer import PromptDesigner
from llm.wrapper import LLMWrapper

from utils.metrics import PPIMetrics
from utils.protein_mapper import ProteinSimilarityMapper, compute_non_isolated_ids

logger = logging.getLogger(__name__)


class PPIEvaluator:
    """
    PPI评估器
    用于评估模型在PPI预测任务上的性能
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化评估器
        
        Args:
            config (Dict[str, Any]): 配置字典
        """
        self.config = config
        self.device = self._get_device()
        
        # 初始化组件
        self.graph_builder = None
        self.cot_generator = None
        self.llm_wrapper = None
        self.prompt_designer = None
        self.metrics_calculator = None
        
        # 评估配置
        self.eval_batch_size = config.get('evaluation', {}).get('batch_size', 32)
        self.eval_mode = config.get('evaluation', {}).get('mode', 'standard')  # standard, novel_protein, zero_shot_relation
        
        # 初始化指标
        self.metrics = {
            'accuracy': [],
            'micro_f1': [],
            'macro_f1': [],
            'weighted_f1': [],
            'auc': [],
            'auprc': [],
            'loss': []
        }

        # LLAPA 基准结果 (SHS27k 数据集)
        self.llapa_benchmarks = {
            'shs27k': {
                'accuracy': 0.887,
                'micro_f1': 0.885,
                'macro_f1': 0.882,
                'auc': 0.942
            }
        }
        
        # 关系类型映射
        self.relation_type_mapping = {
            0: "激活作用",
            1: "结合作用",
            2: "催化作用",
            3: "表达作用",
            4: "抑制作用",
            5: "翻译后修饰",
            6: "反应作用",
        }
        
        logger.info(f"PPIEvaluator初始化完成，使用设备：{self.device}，评估模式：{self.eval_mode}")
    
    def _get_device(self) -> str:
        """
        获取设备
        
        Returns:
            str: 设备字符串
        """
        device_cfg = (self.config.get('device', {}) or {}).get('device_type', 'cuda')
        device_cfg = str(device_cfg).lower()

        if device_cfg == 'cpu' or device_cfg.startswith('cpu'):
            return "cpu"

        if not torch.cuda.is_available():
            return "cpu"

        device_ids = (self.config.get('device', {}) or {}).get('device_ids')
        if isinstance(device_ids, (list, tuple)) and len(device_ids) > 0:
            try:
                return f"cuda:{int(device_ids[0])}"
            except Exception:
                return "cuda:0"

        return "cuda:0"
    
    def build_components(self):
        """
        构建所有组件
        """
        logger.info("开始构建评估组件...")
        
        # 1. 构建图构建器
        self._build_graph_builder()
        
        # 2. 构建模型组件
        self._build_model_components()
        
        # 3. 构建LLM组件
        self._build_llm_components()
        
        # 4. 构建指标计算器
        self._build_metrics_calculator()
        
        logger.info("所有评估组件构建完成")
    
    def _build_graph_builder(self):
        """
        构建图构建器
        """
        self.graph_builder = ComponentBuilder.build_graph_builder(self.config)
    
    def _build_model_components(self):
        """
        构建模型组件
        """
        esm_model_name = self.config.get('dataset', {}).get('preprocessing', {}).get('feature_extraction', {}).get('esm_model', 'facebook/esm2_t6_8M_UR50D')
        self.esm_encoder = ESMEncoder(model_name=esm_model_name, device=self.device)
        self.cot_generator = ComponentBuilder.build_cot_generator(self.config, self.device)
    
    def _build_llm_components(self):
        """
        构建LLM组件
        """
        device_str = str(self.device) if hasattr(self, 'device') else None
        self.prompt_designer, self.llm_wrapper = ComponentBuilder.build_llm_components(self.config, device=device_str)
        
        # 加载蛋白质信息（用于提示生成）
        data_dir = self.config.get('paths', {}).get('data_dir', 'data')
        info_file = self.config.get('dataset', {}).get('protein_info_file', 'protein_info.csv')
        # 补充：通常这些文件都在 data/processed 下
        info_path = os.path.join(data_dir, 'processed', info_file)
        
        if not os.path.exists(info_path):
            # 备选路径：直接在 data_dir 下
            info_path = os.path.join(data_dir, info_file)
        
        if os.path.exists(info_path):
            logger.info(f"正在从 {info_path} 加载蛋白质补充信息...")
            self.prompt_designer.load_protein_info(info_path)
        else:
            logger.warning(f"未找到蛋白质信息文件: {info_path}")
    
    def _build_metrics_calculator(self):
        """
        构建指标计算器
        """
        self.metrics_calculator = PPIMetrics()
        logger.info("指标计算器已构建完成")
    
    def load_model(self, checkpoint_path: str):
        """
        加载模型
        
        Args:
            checkpoint_path (str): 模型检查点路径
        """
        if not os.path.exists(checkpoint_path):
            logger.error(f"模型检查点不存在：{checkpoint_path}")
            raise FileNotFoundError(f"模型检查点不存在：{checkpoint_path}")
        
        # 构建组件
        self.build_components()
        
        # 加载检查点
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 加载COT生成器状态
        if isinstance(checkpoint, dict) and 'cot_generator_state_dict' in checkpoint:
            self.cot_generator.load_state_dict(checkpoint['cot_generator_state_dict'], strict=False)
            logger.info("COT生成器状态已加载")
        elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.cot_generator.load_state_dict(checkpoint['model_state_dict'], strict=False)
            logger.info("COT生成器状态已从model_state_dict加载")
        elif isinstance(checkpoint, dict):
            try:
                self.cot_generator.load_state_dict(checkpoint, strict=False)
                logger.info("COT生成器状态已从原始state_dict加载")
            except Exception:
                logger.warning("检查点中没有可识别的COT生成器状态")
        else:
            logger.warning("检查点格式不支持，无法加载COT生成器状态")
        
        # 加载LLM状态
        if self.llm_wrapper and self.llm_wrapper.model:
            llm_state = checkpoint.get('llm_state_dict')
            if llm_state:
                self.llm_wrapper.model.load_state_dict(llm_state, strict=False)
                logger.info("LLM状态已加载")
            llm_peft_state = checkpoint.get('llm_peft_state_dict')
            if llm_peft_state is None:
                llm_peft_state = (checkpoint.get('logs') or {}).get('llm_peft_state_dict')
            if llm_peft_state:
                self.llm_wrapper.model.load_state_dict(llm_peft_state, strict=False)
                logger.info("LLM LoRA 适配器状态已加载")

        llm_cls_state = checkpoint.get('llm_classifier_state_dict')
        if llm_cls_state is None:
            llm_cls_state = (checkpoint.get('logs') or {}).get('llm_classifier_state_dict')
        if self.llm_wrapper and llm_cls_state:
            self.llm_wrapper.relation_classifier.load_state_dict(llm_cls_state)
            logger.info("LLM分类头状态已加载")
        
        logger.info(f"模型已从检查点加载：{checkpoint_path}")
    
    def evaluate(self, data: Any, evaluation_mode: str = "standard") -> Dict[str, float]:
        """
        评估模型
        
        Args:
            data (Any): 评估数据 (graph, dataset) tuple or dataset
            evaluation_mode (str): 评估模式 ('standard', 'new_protein', 'zero_shot')
            
        Returns:
            Dict[str, float]: 评估结果
        """
        logger.info(f"开始{evaluation_mode}模式评估...")
        
        # 确保组件已构建
        if not self.cot_generator or not self.llm_wrapper or not self.prompt_designer:
            self.build_components()

        # Unpack data if it is a tuple (graph, dataset)
        if isinstance(data, tuple):
            self.graph, dataset = data
            self.graph = self.graph.to(self.device)
        else:
            dataset = data
            # 如果没有全局图，从数据中构建
            if self.graph_builder and (not hasattr(self, 'graph') or self.graph is None):
                logger.info("构建训练图用于评估...")
                train_file = self.config.get('dataset', {}).get('train_file')
                self.graph, _ = self.graph_builder.build_graph_and_load_data(split='train', data_file=train_file)
                self.graph = self.graph.to(self.device)

        if not hasattr(self, "_train_protein_sequences"):
            self._train_protein_sequences = {}
        if not hasattr(self, "_non_isolated_protein_ids"):
            self._non_isolated_protein_ids = set()

        if self.graph is not None and not self._non_isolated_protein_ids:
            self._non_isolated_protein_ids = compute_non_isolated_ids(self.graph)

        if not self._train_protein_sequences and self.graph_builder:
            train_file = self.config.get('dataset', {}).get('train_file')
            try:
                _, train_dataset = self.graph_builder.build_graph_and_load_data(split='train', data_file=train_file)
                if hasattr(train_dataset, "get_protein_sequences"):
                    self._train_protein_sequences = train_dataset.get_protein_sequences()
            except Exception:
                self._train_protein_sequences = {}

        if not hasattr(self, "protein_similarity_mapper") or self.protein_similarity_mapper is None:
            sim_cfg = (self.config.get("preprocessing", {}) or {}).get("similarity_mapping", {}) or self.config.get("similarity_mapping", {}) or {}
            sim_enabled = bool(sim_cfg.get("enabled", True))
            sim_batch_size = int(sim_cfg.get("batch_size", 16))
            sim_method = str(sim_cfg.get("method", "esm"))
            sim_allow_fallback_to_esm = bool(sim_cfg.get("allow_fallback_to_esm", True))
            sim_blastp_evalue = float(sim_cfg.get("blastp_evalue", 1e-5))
            sim_blastp_num_threads = int(sim_cfg.get("blastp_num_threads", 4))
            sim_cache_dir = sim_cfg.get("cache_dir")
            if not sim_cache_dir:
                sim_cache_dir = os.path.join(self.config.get("paths", {}).get("data_dir", "."), "cache", "protein_similarity_mapper")
            self.protein_similarity_mapper = ProteinSimilarityMapper(
                esm_encoder=self.esm_encoder,
                cache_dir=sim_cache_dir,
                enabled=sim_enabled,
                batch_size=sim_batch_size,
                method=sim_method,
                allow_fallback_to_esm=sim_allow_fallback_to_esm,
                blastp_evalue=sim_blastp_evalue,
                blastp_num_threads=sim_blastp_num_threads,
            )
            self.protein_similarity_mapper.fit(
                train_protein_sequences=self._train_protein_sequences,
                non_isolated_ids=self._non_isolated_protein_ids,
            )

        self._ensure_graph_node_features()
        
        # 设置模型为评估模式
        self.cot_generator.eval()
        if self.llm_wrapper and self.llm_wrapper.model:
            self.llm_wrapper.model.eval()
        
        # 根据评估模式选择评估方法
        if evaluation_mode == "standard":
            return self._evaluate_standard(dataset)
        elif evaluation_mode == "new_protein":
            return self._evaluate_new_protein(dataset)
        elif evaluation_mode == "zero_shot":
            return self._evaluate_zero_shot(dataset)
        else:
            raise ValueError(f"不支持的评估模式：{evaluation_mode}")

    def _ensure_graph_node_features(self) -> None:
        if not hasattr(self, "graph") or self.graph is None:
            return
        if float(self.graph.x.abs().sum().detach().cpu().item()) > 0:
            return
        if not hasattr(self, "esm_encoder") or self.esm_encoder is None:
            return
        seqs = getattr(self, "_train_protein_sequences", {}) or {}
        protein_ids = getattr(self.graph, "protein_ids", None) or []
        if not protein_ids:
            return
        data_dir = (self.config.get("paths", {}) or {}).get("data_dir", "data")
        cache_dir = os.path.join(data_dir, "cache")
        os.makedirs(cache_dir, exist_ok=True)
        safe_esm = str(getattr(self.esm_encoder, "model_name", "esm")).replace("/", "_").replace("\\", "_")
        cache_path = os.path.join(cache_dir, f"graph_x_{safe_esm}_{len(protein_ids)}.pt")
        if os.path.exists(cache_path):
            x = torch.load(cache_path, map_location="cpu")
            self.graph.x.copy_(x.to(self.graph.x.device))
            return
        indices = []
        sequences = []
        for pid in protein_ids:
            seq = seqs.get(pid)
            if not seq:
                continue
            idx = self.graph.protein_id_to_idx.get(pid)
            if idx is None:
                continue
            indices.append(int(idx))
            sequences.append(seq)
        if not sequences:
            return
        embeddings = self.esm_encoder.get_batch_embeddings(sequences, batch_size=16)
        x_cpu = torch.zeros((len(protein_ids), int(self.graph.x.size(-1))), dtype=torch.float32)
        for idx, emb in zip(indices, embeddings):
            if emb is None:
                continue
            x_cpu[idx] = emb.detach().to("cpu").float()
        tmp_path = f"{cache_path}.tmp.{os.getpid()}"
        torch.save(x_cpu, tmp_path)
        os.replace(tmp_path, cache_path)
        self.graph.x.copy_(x_cpu.to(self.graph.x.device))
    
    def _evaluate_standard(self, data: Any) -> Dict[str, float]:
        """
        标准测试集评估
        
        Args:
            data (Any): 评估数据
            
        Returns:
            Dict[str, float]: 评估结果
        """
        # 创建数据加载器
        dataloader = self._create_dataloader(data, self.config.get('evaluation', {}).get('batch_size', 32))
        
        # 初始化评估结果
        all_predictions = []
        all_labels = []
        all_probabilities = []
        total_loss = 0
        total_steps = 0
        
        start_time = time.time()

        max_batches = self.config.get('evaluation', {}).get('max_batches')
        max_batches = int(max_batches) if max_batches is not None else None
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                logger.info(f"处理批次 {batch_idx+1}/{len(dataloader)}")
                
                # 移动数据到设备 (only tensors)
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.to(self.device)
                
                # 使用全局图
                graph_data = self.graph
                
                # 获取节点索引
                source_ids = batch['protein_a_id']
                target_ids = batch['protein_b_id']

                source_seqs = batch.get("protein_a", [])
                target_seqs = batch.get("protein_b", [])

                mapped_source_ids = source_ids
                mapped_target_ids = target_ids
                if hasattr(self, "protein_similarity_mapper") and self.protein_similarity_mapper:
                    mapped_source_ids = self.protein_similarity_mapper.map_batch(
                        protein_ids=source_ids,
                        protein_sequences=source_seqs,
                        non_isolated_ids=getattr(self, "_non_isolated_protein_ids", set()),
                    ).mapped_ids
                    mapped_target_ids = self.protein_similarity_mapper.map_batch(
                        protein_ids=target_ids,
                        protein_sequences=target_seqs,
                        non_isolated_ids=getattr(self, "_non_isolated_protein_ids", set()),
                    ).mapped_ids

                source_nodes = [self.graph.protein_id_to_idx[pid] for pid in mapped_source_ids]
                target_nodes = [self.graph.protein_id_to_idx[pid] for pid in mapped_target_ids]
                
                # 构造目标信息
                mapped_target_seqs = [
                    (getattr(self, "_train_protein_sequences", {}) or {}).get(mapped_target_ids[i], target_seqs[i])
                    for i in range(len(target_nodes))
                ]
                target_esms = self.esm_encoder.get_batch_embeddings(mapped_target_seqs, batch_size=min(16, len(mapped_target_seqs)))
                target_infos = [{'esm_features': target_esms[i], 'protein_id': target_nodes[i]} for i in range(len(target_nodes))]
                
                # 获取生成的链数量配置
                num_chains = self.config.get('model', {}).get('num_chains', 3)
                
                # 生成COT路径
                paths_info_batch = self.cot_generator.batch_generate_chains(
                    start_protein_ids=source_nodes,
                    target_protein_infos=target_infos,
                    graph_data=graph_data,
                    training=False,
                    device=self.device,
                    num_chains=num_chains
                )
                
                protein_info = {}
                for pid, idx in self.graph.protein_id_to_idx.items():
                    if self.prompt_designer and pid in self.prompt_designer.protein_id_to_info:
                        protein_info[idx] = self.prompt_designer.protein_id_to_info[pid]
                    else:
                        protein_info[idx] = {'name': pid, 'function': '暂无功能描述'}

                for i, idx in enumerate(source_nodes):
                    seq_a = batch['protein_a'][i]
                    protein_info[idx] = self.graph_builder.get_protein_info_by_sequence(seq_a, self.prompt_designer)
                for i, idx in enumerate(target_nodes):
                    seq_b = batch['protein_b'][i]
                    protein_info[idx] = self.graph_builder.get_protein_info_by_sequence(seq_b, self.prompt_designer)
                
                prompts = []
                for i in range(len(source_nodes)):
                    # paths_info_batch[i] 是一个 List[Dict]，包含多条路径
                    paths_info = paths_info_batch[i]
                    
                    multi_paths = [p['path'] for p in paths_info]
                    multi_relations = [p.get('relations', []) for p in paths_info]
                    
                    prompt_data = {
                        'source_protein': source_ids[i],
                        'target_protein': target_ids[i],
                        'path': multi_paths,
                        'relations': multi_relations,
                        'protein_info': protein_info
                    }
                    prompt = self.prompt_designer.generate_prompt(
                        template_type='exploratory_reasoning',
                        **prompt_data
                    )
                    prompts.append(prompt)

                
                # LLM预测
                pred_indices, logits = self.llm_wrapper.predict(prompts, return_type='logits')
                
                # 确保预测结果不是None
                if pred_indices is None:
                    logger.warning(f"批次 {batch_idx} LLM预测返回None，跳过")
                    continue
                
                # 计算损失 (多标签使用 BCEWithLogitsLoss)
                if batch['label'].dim() > 1 and batch['label'].size(1) > 1:
                    if not torch.isfinite(logits).all():
                        logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
                    loss = F.binary_cross_entropy_with_logits(logits.float(), batch['label'].float())
                else:
                    loss = F.cross_entropy(logits, batch['label'].long().view(-1))
                total_loss += loss.item()
                total_steps += 1
                
                # 记录结果
                all_predictions.extend(pred_indices.cpu().numpy())
                all_labels.extend(batch['label'].cpu().numpy())
                # 多标签使用 sigmoid 而非 softmax
                probs = torch.sigmoid(logits)
                all_probabilities.extend(probs.cpu().numpy())
        
        end_time = time.time()
        
        # 计算评估指标
        results = self._calculate_metrics(
            np.array(all_predictions),
            np.array(all_labels),
            np.array(all_probabilities)
        )
        
        # 添加损失
        results['loss'] = total_loss / total_steps
        results['evaluation_time'] = end_time - start_time
        
        # 与 LLAPA 进行对比并保存对比结果
        llapa_comparison = {}
        dataset_name = self.config.get('dataset', {}).get('name', 'shs27k').lower()
        if dataset_name in self.llapa_benchmarks:
            logger.info(f"\n" + "="*50)
            logger.info(f"性能对比结果 (当前模型 vs LLAPA):")
            benchmark = self.llapa_benchmarks[dataset_name]
            for metric, value in results.items():
                if metric in benchmark:
                    diff = (value - benchmark[metric]) / benchmark[metric] * 100
                    comparison_str = f"{value:.4f} (LLAPA: {benchmark[metric]:.4f}, 差异: {diff:+.2f}%)"
                    logger.info(f"  {metric}: {comparison_str}")
                    llapa_comparison[metric] = {
                        'current': value,
                        'llapa': benchmark[metric],
                        'diff_percent': diff
                    }
            logger.info("="*50 + "\n")
        
        # 保存详细结果 (传入对比数据)
        self._save_evaluation_results(all_predictions, all_labels, all_probabilities, "standard", llapa_comparison)
        
        logger.info(f"标准评估完成，耗时：{end_time - start_time:.2f}秒")
        
        return results
    
    def _evaluate_new_protein(self, data: Any) -> Dict[str, float]:
        """
        新蛋白质评估（留出蛋白质）
        
        Args:
            data (Any): 评估数据
            
        Returns:
            Dict[str, float]: 评估结果
        """
        # 这里需要实现新蛋白质评估逻辑
        # 新蛋白质评估：测试集中包含训练集中没有的蛋白质
        logger.info("新蛋白质评估（留出蛋白质）")
        
        # 创建数据加载器
        dataloader = self._create_dataloader(data, self.config.get('evaluation', {}).get('batch_size', 32))
        
        # 初始化评估结果
        all_predictions = []
        all_labels = []
        all_probabilities = []
        total_loss = 0
        total_steps = 0
        
        start_time = time.time()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                logger.info(f"处理批次 {batch_idx+1}/{len(dataloader)}")
                
                # 移动数据到设备
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.to(self.device)
                
                # 使用全局图
                graph_data = self.graph
                
                # 获取节点索引
                source_ids = batch['protein_a_id']
                target_ids = batch['protein_b_id']

                source_seqs = batch.get("protein_a", [])
                target_seqs = batch.get("protein_b", [])

                mapped_source_ids = source_ids
                mapped_target_ids = target_ids
                if hasattr(self, "protein_similarity_mapper") and self.protein_similarity_mapper:
                    mapped_source_ids = self.protein_similarity_mapper.map_batch(
                        protein_ids=source_ids,
                        protein_sequences=source_seqs,
                        non_isolated_ids=getattr(self, "_non_isolated_protein_ids", set()),
                    ).mapped_ids
                    mapped_target_ids = self.protein_similarity_mapper.map_batch(
                        protein_ids=target_ids,
                        protein_sequences=target_seqs,
                        non_isolated_ids=getattr(self, "_non_isolated_protein_ids", set()),
                    ).mapped_ids

                source_nodes = [self.graph.protein_id_to_idx[pid] for pid in mapped_source_ids]
                target_nodes = [self.graph.protein_id_to_idx[pid] for pid in mapped_target_ids]
                
                # 构造目标信息
                mapped_target_seqs = [
                    (getattr(self, "_train_protein_sequences", {}) or {}).get(mapped_target_ids[i], target_seqs[i])
                    for i in range(len(target_nodes))
                ]
                target_esms = self.esm_encoder.get_batch_embeddings(mapped_target_seqs, batch_size=min(16, len(mapped_target_seqs)))
                target_infos = [{'esm_features': target_esms[i], 'protein_id': target_nodes[i]} for i in range(len(target_nodes))]
                
                # 生成COT路径
                num_chains = self.config.get('model', {}).get('num_chains', 3)
                paths_info_batch = self.cot_generator.batch_generate_chains(
                    start_protein_ids=source_nodes,
                    target_protein_infos=target_infos,
                    graph_data=graph_data,
                    training=False,
                    device=self.device
                    ,num_chains=num_chains
                )
                
                # 准备提示所需的数据
                protein_info = {}
                for pid, idx in self.graph.protein_id_to_idx.items():
                    if self.prompt_designer and pid in self.prompt_designer.protein_id_to_info:
                        protein_info[idx] = self.prompt_designer.protein_id_to_info[pid]
                    else:
                        protein_info[idx] = {'name': pid, 'function': '暂无功能描述'}

                for i, idx in enumerate(source_nodes):
                    seq_a = batch['protein_a'][i]
                    protein_info[idx] = self.graph_builder.get_protein_info_by_sequence(seq_a, self.prompt_designer)
                for i, idx in enumerate(target_nodes):
                    seq_b = batch['protein_b'][i]
                    protein_info[idx] = self.graph_builder.get_protein_info_by_sequence(seq_b, self.prompt_designer)
                
                prompts = []
                for i in range(len(source_nodes)):
                    paths_info = paths_info_batch[i]
                    multi_paths = [p['path'] for p in paths_info]
                    multi_relations = [p.get('relations', []) for p in paths_info]
                    prompt_data = {
                        'source_protein': source_ids[i],
                        'target_protein': target_ids[i],
                        'path': multi_paths,
                        'relations': multi_relations,
                        'protein_info': protein_info
                    }
                    prompt = self.prompt_designer.generate_prompt(
                        template_type='exploratory_reasoning',
                        **prompt_data
                    )
                    prompts.append(prompt)
                
                # LLM预测
                pred_indices, logits = self.llm_wrapper.predict(prompts, return_type='logits')
                
                # 确保预测结果不是None
                if pred_indices is None:
                    logger.warning(f"批次 {batch_idx} LLM预测返回None，跳过")
                    continue
                
                # 计算损失
                loss = F.cross_entropy(logits, batch['label'].long())
                total_loss += loss.item()
                total_steps += 1
                
                # 记录结果
                all_predictions.extend(pred_indices.cpu().numpy())
                all_labels.extend(batch['label'].cpu().numpy())
                all_probabilities.extend(F.softmax(logits, dim=1).cpu().numpy())
        
        end_time = time.time()
        
        # 计算评估指标
        results = self._calculate_metrics(
            np.array(all_predictions),
            np.array(all_labels),
            np.array(all_probabilities)
        )
        
        # 添加损失
        results['loss'] = total_loss / total_steps
        results['evaluation_time'] = end_time - start_time
        
        # 保存详细结果
        self._save_evaluation_results(all_predictions, all_labels, all_probabilities, "new_protein")
        
        logger.info(f"新蛋白质评估完成，耗时：{end_time - start_time:.2f}秒")
        
        return results

    def _evaluate_zero_shot(self, data: Any) -> Dict[str, float]:
        """
        零样本关系评估
        
        Args:
            data (Any): 评估数据
            
        Returns:
            Dict[str, float]: 评估结果
        """
        # 这里需要实现零样本关系评估逻辑
        # 零样本关系评估：测试集中包含训练集中没有的关系类型
        logger.info("零样本关系评估")
        
        # 创建数据加载器
        dataloader = self._create_dataloader(data, self.config.get('evaluation', {}).get('batch_size', 32))
        
        # 初始化评估结果
        all_predictions = []
        all_labels = []
        all_probabilities = []
        total_loss = 0
        total_steps = 0
        
        start_time = time.time()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                logger.info(f"处理批次 {batch_idx+1}/{len(dataloader)}")
                
                # 移动数据到设备
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.to(self.device)
                
                # 使用全局图
                graph_data = self.graph
                
                # 获取节点索引
                source_ids = batch['protein_a_id']
                target_ids = batch['protein_b_id']

                source_seqs = batch.get("protein_a", [])
                target_seqs = batch.get("protein_b", [])

                mapped_source_ids = source_ids
                mapped_target_ids = target_ids
                if hasattr(self, "protein_similarity_mapper") and self.protein_similarity_mapper:
                    mapped_source_ids = self.protein_similarity_mapper.map_batch(
                        protein_ids=source_ids,
                        protein_sequences=source_seqs,
                        non_isolated_ids=getattr(self, "_non_isolated_protein_ids", set()),
                    ).mapped_ids
                    mapped_target_ids = self.protein_similarity_mapper.map_batch(
                        protein_ids=target_ids,
                        protein_sequences=target_seqs,
                        non_isolated_ids=getattr(self, "_non_isolated_protein_ids", set()),
                    ).mapped_ids

                source_nodes = [self.graph.protein_id_to_idx[pid] for pid in mapped_source_ids]
                target_nodes = [self.graph.protein_id_to_idx[pid] for pid in mapped_target_ids]
                
                # 构造目标信息
                mapped_target_seqs = [
                    (getattr(self, "_train_protein_sequences", {}) or {}).get(mapped_target_ids[i], target_seqs[i])
                    for i in range(len(target_nodes))
                ]
                target_esms = self.esm_encoder.get_batch_embeddings(mapped_target_seqs, batch_size=min(16, len(mapped_target_seqs)))
                target_infos = [{'esm_features': target_esms[i], 'protein_id': target_nodes[i]} for i in range(len(target_nodes))]
                
                # 生成COT路径
                num_chains = self.config.get('model', {}).get('num_chains', 3)
                paths_info_batch = self.cot_generator.batch_generate_chains(
                    start_protein_ids=source_nodes,
                    target_protein_infos=target_infos,
                    graph_data=graph_data,
                    training=False,
                    device=self.device
                    ,num_chains=num_chains
                )
                
                # 准备提示所需的数据
                protein_info = {}
                for pid, idx in self.graph.protein_id_to_idx.items():
                    if self.prompt_designer and pid in self.prompt_designer.protein_id_to_info:
                        protein_info[idx] = self.prompt_designer.protein_id_to_info[pid]
                    else:
                        protein_info[idx] = {'name': pid, 'function': '暂无功能描述'}

                for i, idx in enumerate(source_nodes):
                    seq_a = batch['protein_a'][i]
                    protein_info[idx] = self.graph_builder.get_protein_info_by_sequence(seq_a, self.prompt_designer)
                for i, idx in enumerate(target_nodes):
                    seq_b = batch['protein_b'][i]
                    protein_info[idx] = self.graph_builder.get_protein_info_by_sequence(seq_b, self.prompt_designer)
                
                prompts = []
                for i in range(len(source_nodes)):
                    paths_info = paths_info_batch[i]
                    multi_paths = [p['path'] for p in paths_info]
                    multi_relations = [p.get('relations', []) for p in paths_info]
                    prompt_data = {
                        'source_protein': source_ids[i],
                        'target_protein': target_ids[i],
                        'path': multi_paths,
                        'relations': multi_relations,
                        'protein_info': protein_info
                    }
                    prompt = self.prompt_designer.generate_prompt(
                        template_type='exploratory_reasoning',
                        **prompt_data
                    )
                    prompts.append(prompt)
                
                # LLM预测
                pred_indices, logits = self.llm_wrapper.predict(prompts, return_type='logits')
                
                # 确保预测结果不是None
                if pred_indices is None:
                    logger.warning(f"批次 {batch_idx} LLM预测返回None，跳过")
                    continue
                
                # 计算损失
                loss = F.cross_entropy(logits, batch['label'].long())
                total_loss += loss.item()
                total_steps += 1
                
                # 记录结果
                all_predictions.extend(pred_indices.cpu().numpy())
                all_labels.extend(batch['label'].cpu().numpy())
                all_probabilities.extend(F.softmax(logits, dim=1).cpu().numpy())
        
        end_time = time.time()
        
        # 计算评估指标
        results = self._calculate_metrics(
            np.array(all_predictions),
            np.array(all_labels),
            np.array(all_probabilities)
        )
        
        # 添加损失
        results['loss'] = total_loss / total_steps
        results['evaluation_time'] = end_time - start_time
        
        # 保存详细结果
        self._save_evaluation_results(all_predictions, all_labels, all_probabilities, "zero_shot")
        
        logger.info(f"零样本关系评估完成，耗时：{end_time - start_time:.2f}秒")
        
        return results
    
    def evaluate_single_pair(self, 
                             model: torch.nn.Module, 
                             graph_data: Any, 
                             llm_wrapper: Any, 
                             protein_a: str, 
                             protein_b: str,
                             protein_a_id: Optional[str] = None,
                             protein_b_id: Optional[str] = None
                             ) -> Dict[str, Any]:
        """
        评估单个蛋白质对
        
        Args:
            model (torch.nn.Module): GNN模型
            graph_data (Any): 图数据
            llm_wrapper (Any): LLM包装器
            protein_a (str): 蛋白质A的名称或序列
            protein_b (str): 蛋白质B的名称或序列
            protein_a_id (Optional[str]): 蛋白质A在图中的ID
            protein_b_id (Optional[str]): 蛋白质B在图中的ID
            
        Returns:
            Dict[str, Any]: 评估结果
        """
        model.eval()
        if llm_wrapper and llm_wrapper.model:
            llm_wrapper.model.eval()
            
        # 确定节点索引
        if protein_a_id and protein_b_id and hasattr(graph_data, 'protein_id_to_idx'):
            source_node = graph_data.protein_id_to_idx.get(protein_a_id)
            target_node = graph_data.protein_id_to_idx.get(protein_b_id)
        else:
            # 默认使用前两个节点
            source_node = 0
            target_node = 1
            
        if source_node is None or target_node is None:
            logger.error(f"找不到蛋白质索引: {protein_a_id}, {protein_b_id}")
            return {'status': 'error', 'message': 'Protein IDs not found in graph'}

        # 1. 生成COT路径
        # 构造目标信息
        target_info = {'esm_features': graph_data.x[target_node], 'protein_id': target_node}
        
        paths_info = self.cot_generator.batch_generate_chains(
            start_protein_ids=[source_node],
            target_protein_infos=[target_info],
            graph_data=graph_data,
            training=False,
            device=self.device
        )
        if not paths_info or not paths_info[0]:
            return {'status': 'error', 'message': 'No paths generated'}
        path_info = paths_info[0][0]
        path = path_info.get('path', [])
        relations = path_info.get('relations', [])
        
        # 2. 生成提示
        # 构造蛋白质信息字典
        protein_info = {}
        for pid, idx in graph_data.protein_id_to_idx.items():
            if self.prompt_designer and pid in self.prompt_designer.protein_id_to_info:
                protein_info[idx] = self.prompt_designer.protein_id_to_info[pid]
            else:
                protein_info[idx] = {'name': pid, 'function': '暂无功能描述'}
        
        prompt_data = {
            'source_protein': protein_a,
            'target_protein': protein_b,
            'path': path,
            'relations': relations,
            'protein_info': protein_info
        }
        
        prompt = self.prompt_designer.generate_prompt(
            template_type='exploratory_reasoning',
            **prompt_data
        )
        
        # LLM预测
        prediction, probabilities = llm_wrapper.predict([prompt], return_type='probabilities')
        pred_vec = prediction[0]
        prob_vec = probabilities[0]
        if isinstance(pred_vec, torch.Tensor):
            idxs = (pred_vec > 0).nonzero(as_tuple=False).flatten().tolist()
        else:
            idxs = [i for i, v in enumerate(pred_vec) if float(v) > 0.0]
        if not idxs:
            idxs = [int(torch.argmax(prob_vec).item())] if isinstance(prob_vec, torch.Tensor) else [int(np.argmax(prob_vec))]
        
        # 4. 构建结果
        result = {
            'prediction': idxs,
            'prediction_text': [self.relation_type_mapping.get(i, "未知") for i in idxs],
            'probabilities': prob_vec.tolist() if isinstance(prob_vec, torch.Tensor) else list(prob_vec),
            'path': path,
            'path_description': self.prompt_designer.path_to_text(path, protein_info, relations=relations),
            'prompt': prompt
        }
        
        return result

    def _create_dataloader(self, data: Any, batch_size: int) -> DataLoader:
        """
        创建数据加载器
        
        Args:
            data (Any): 数据
            batch_size (int): 批次大小
            
        Returns:
            DataLoader: 数据加载器
        """
        # 这里需要根据实际数据格式实现数据加载器
        # 假设data是一个包含图和节点对的数据集
        return DataLoader(
            data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.config.get('evaluation', {}).get('num_workers', 0),
            pin_memory=self.config.get('evaluation', {}).get('pin_memory', False)
        )
    
    def _calculate_metrics(self, 
                         predictions: np.ndarray, 
                         labels: np.ndarray, 
                         probabilities: np.ndarray
                         ) -> Dict[str, float]:
        """
        计算评估指标
        
        Args:
            predictions (np.ndarray): 预测结果 [N]
            labels (np.ndarray): 真实标签 [N]
            probabilities (np.ndarray): 预测概率 [N, C]
            
        Returns:
            Dict[str, float]: 评估指标
        """
        logger.info("开始计算评估指标...")
        
        # 计算基本指标
        accuracy = accuracy_score(labels, predictions)
        micro_f1 = f1_score(labels, predictions, average='micro', zero_division=0)
        macro_f1 = f1_score(labels, predictions, average='macro', zero_division=0)
        weighted_f1 = f1_score(labels, predictions, average='weighted', zero_division=0)
        
        # 计算AUC和AUPRC
        auc = self.metrics_calculator.calculate_roc_auc(labels, probabilities)
        auprc = self.metrics_calculator.calculate_auprc(labels, probabilities)
        
        # 计算每个类别的指标
        class_report = classification_report(
            labels, 
            predictions, 
            target_names=[self.relation_type_mapping.get(i, f"Type_{i}") for i in range(len(self.relation_type_mapping))],
            output_dict=True,
            zero_division=0
        )
        
        # 计算混淆矩阵 (多标签不适用传统混淆矩阵)
        is_multilabel = labels.ndim > 1 and labels.shape[1] > 1
        if not is_multilabel:
            cm = confusion_matrix(labels, predictions)
            cm_list = cm.tolist()
        else:
            cm_list = None
        
        # 构造结果
        results = {
            'accuracy': accuracy,
            'micro_f1': micro_f1,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'auc': auc,
            'auprc': auprc,
            'class_report': class_report,
            'confusion_matrix': cm_list
        }
        
        # 记录指标
        self.metrics['accuracy'].append(accuracy)
        self.metrics['micro_f1'].append(micro_f1)
        self.metrics['macro_f1'].append(macro_f1)
        self.metrics['weighted_f1'].append(weighted_f1)
        self.metrics['auc'].append(auc)
        self.metrics['auprc'].append(auprc)
        
        logger.info(f"评估指标计算完成")
        logger.info(f"准确率: {accuracy:.4f}, Micro-F1: {micro_f1:.4f}, Macro-F1: {macro_f1:.4f}, AUC: {auc:.4f}, AUPRC: {auprc:.4f}")
        
        return results
    
    def _save_evaluation_results(self, 
                               predictions: List[int],
                               labels: List[int],
                               probabilities: List[List[float]],
                               evaluation_mode: str,
                               llapa_comparison: Optional[Dict[str, Any]] = None
                               ):
        """
        保存评估结果
        
        Args:
            predictions (List[int]): 预测结果
            labels (List[int]): 真实标签
            probabilities (List[List[float]]): 预测概率
            evaluation_mode (str): 评估模式
            llapa_comparison (Optional[Dict[str, Any]]): 与 LLAPA 的对比数据
        """
        # 创建结果目录
        default_results_dir = self.config.get('paths', {}).get('eval_detailed_dir', 'artifacts/eval_detailed')
        results_dir = self.config.get('evaluation', {}).get('results_dir', default_results_dir)
        os.makedirs(results_dir, exist_ok=True)
        
        # 生成时间戳
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # 保存详细结果
        results = {
            'timestamp': timestamp,
            'evaluation_mode': evaluation_mode,
            'predictions': np.array(predictions).tolist(),
            'labels': np.array(labels).tolist(),
            'probabilities': np.array(probabilities).tolist(),
            'relation_type_mapping': self.relation_type_mapping
        }
        
        if llapa_comparison:
            results['llapa_comparison'] = llapa_comparison
        
        results_path = os.path.join(results_dir, f"{evaluation_mode}_results_{timestamp}.json")
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        
        logger.info(f"评估详细结果已保存到：{results_path}")
    
    def print_evaluation_results(self, results: Dict[str, float]):
        """
        打印评估结果
        
        Args:
            results (Dict[str, float]): 评估结果
        """
        print("\n" + "="*50)
        print("评估结果")
        print("="*50)
        
        # 打印基本指标
        print(f"准确率: {results['accuracy']:.4f}")
        print(f"精确率: {results['precision']:.4f}")
        print(f"召回率: {results['recall']:.4f}")
        print(f"F1分数: {results['f1']:.4f}")
        print(f"AUC: {results['auc']:.4f}")
        print(f"损失: {results['loss']:.4f}")
        print(f"评估时间: {results['evaluation_time']:.2f}秒")
        
        print("\n" + "="*50)
        print("类别报告")
        print("="*50)
        
        # 打印类别报告
        class_report = results.get('class_report', {})
        for class_name, metrics in class_report.items():
            if class_name in ['accuracy', 'macro avg', 'weighted avg']:
                print(f"\n{class_name}:")
                for metric_name, value in metrics.items():
                    print(f"  {metric_name}: {value:.4f}")
            else:
                print(f"{class_name}:")
                for metric_name, value in metrics.items():
                    print(f"  {metric_name}: {value:.4f}")
        
        print("\n" + "="*50)
        print("混淆矩阵")
        print("="*50)
        
        # 打印混淆矩阵
        cm = np.array(results.get('confusion_matrix', []))
        if cm.shape[0] > 0:
            # 打印行标签
            print("\t" + "\t".join([self.relation_type_mapping[i] for i in range(cm.shape[1])]))
            
            for i in range(cm.shape[0]):
                print(self.relation_type_mapping[i] + "\t" + "\t".join(map(str, cm[i])))
