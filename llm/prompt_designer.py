import torch
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class PromptDesigner:
    """
    设计LLM提示模板的类
    支持多种提示模板类型，将COT路径转换为自然语言描述
    """
    
    def __init__(self):
        """
        初始化提示设计器
        """
        # 定义提示模板
        self.templates = {
            'exploratory_reasoning': self._get_exploratory_template(),
            'similarity_reasoning': self._get_similarity_template(),
            'function_based': self._get_function_based_template(),
            'confidence_reasoning': self._get_confidence_template()
        }
        
        # 关系类型映射
        self.relation_map = {
            0: "Activation",
            1: "Inhibition",
            2: "Binding",
            3: "Catalysis",
            4: "Expression Regulation",
            5: "Physical Interaction",
            6: "Genetic Interaction",
        }
        
        self.protein_id_to_info = {}
        
        logger.info(f"PromptDesigner初始化完成，支持 {len(self.templates)} 种提示模板")
    
    def load_protein_info(self, csv_path: str):
        """
        从CSV文件加载蛋白质信息
        
        Args:
            csv_path (str): CSV文件路径
        """
        import pandas as pd
        import os
        import re
        if not os.path.exists(csv_path):
            logger.warning(f"蛋白质信息文件不存在: {csv_path}")
            return
            
        try:
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                p_id = str(row['protein_id'])
                p_name = str(row['protein_name'])
                p_func = str(row['function'])
                
                # 过滤掉无效信息
                if p_name == "Unknown" or p_name == "Error":
                    p_name = p_id
                if p_func == "Unknown" or p_func == "Error":
                    p_func = "No functional description available"
                
                # 存储原始 ID
                self.protein_id_to_info[p_id] = {
                    'name': p_name,
                    'function': p_func
                }
                
                # 处理 ENSP ID：如果 ID 包含 . 则取点后的部分，或者取 ENSP 开头的部分
                # 例如 9606.ENSP00000260270 -> ENSP00000260270
                if '.' in p_id:
                    short_id = p_id.split('.')[-1]
                    if short_id not in self.protein_id_to_info:
                        self.protein_id_to_info[short_id] = self.protein_id_to_info[p_id]
                
                # 额外处理：如果 ID 包含 ENSP 但不是以 ENSP 开头
                match = re.search(r'ENSP\d+', p_id)
                if match:
                    ensp_id = match.group()
                    if ensp_id not in self.protein_id_to_info:
                        self.protein_id_to_info[ensp_id] = self.protein_id_to_info[p_id]

            logger.info(f"从 {csv_path} 加载了 {len(self.protein_id_to_info)} 条蛋白质信息映射")
        except Exception as e:
            logger.error(f"加载蛋白质信息失败: {e}")
    
    def _get_knowledge_base(self) -> str:
        """
        Get the definition and criteria for Protein-Protein Interaction (PPI) types.
        """
        return """# Interaction Categories
Use the exact label order (index 0~6):
1. Activation: Upstream protein increases downstream activity/function.
2. Inhibition: Upstream protein decreases downstream activity/function.
3. Binding: Non-covalent complex formation between proteins.
4. Catalysis: Enzymatic modification/reaction affecting the partner.
5. Expression Regulation: Changes in mRNA/protein abundance of the partner.
6. Physical Interaction: Direct physical contact/scaffolding interaction.
7. Genetic Interaction: Non-additive phenotype in double perturbation.
"""

    def _get_exploratory_template(self) -> str:
        """
        Get the exploratory reasoning template.
        
        Returns:
            str: Exploratory reasoning template.
        """
        return f"""# Role
You are an expert in protein-protein interaction (PPI) relation prediction.

# Task Context
Given functional information for a source protein and a target protein, plus multiple COT chains (each chain is a sequence of pairwise relations), predict the interaction relation(s) between {{source_protein}} and {{target_protein}} (multi-label).

Important: The ground-truth relation set between the two input proteins is guaranteed to be non-empty, i.e., at least one of the 7 categories applies.

## Protein Information
- Source: **{{source_protein}}** — {{source_function}}
- Target: **{{target_protein}}** — {{target_function}}

## Path Protein Information
{{path_protein_functions}}

## COT Chains
{{path_description}}

{self._get_knowledge_base()}

# Related Knowledge
- Combine protein functions with chain evidence. For example, enzymes/kinases often suggest Catalysis; transcription factors/signaling regulators may suggest Expression Regulation; complex/co-localization evidence may suggest Binding or Physical Interaction.

# Reasoning Workflow (<THINKING>)
<THINKING>
1. Aggregate recurring relation types and key intermediate nodes across chains to form candidate labels.
2. Filter candidates by functional consistency between proteins.
3. If uncertain, output fewer labels but ensure at least one label is chosen.
</THINKING>

# Constraints
- Use only the provided input; do not invent facts.
- Output must use the exact English label names from the 7 categories.
- Do not output the <THINKING> content.
- Never output None. If unsure, output the single most likely label.

# Output Format
One line: a comma-separated list of labels in the exact order shown in Interaction Categories (e.g., `Activation,Binding`)."""

    def _get_similarity_template(self) -> str:
        """
        Get the similarity reasoning template.
        
        Returns:
            str: Similarity reasoning template.
        """
        return f"""# Role
You are an expert in protein-protein interaction (PPI) relation prediction.

# Task Context
Use protein functions, similar-protein interaction hints, and COT chains to predict the interaction relation(s) between {{source_protein}} and {{target_protein}} (multi-label).

Important: The ground-truth relation set between the two input proteins is guaranteed to be non-empty.

{self._get_knowledge_base()}

# Related Knowledge
## Protein Functional Information
{{function_info}}

## Interaction of Similar Proteins
{{similarity_info}}

## COT Chains
{{path_description}}

# Constraints
- Use only the provided input; do not invent facts.
- Output must use the exact English label names from the 7 categories.
- Never output None. If unsure, output the single most likely label.

# Output Format
One line: comma-separated labels in the exact order shown in Interaction Categories."""
    
    def _get_function_based_template(self) -> str:
        """
        Get the function-based reasoning template.
        
        Returns:
            str: Function-based reasoning template.
        """
        return f"""# Role
You are an expert in protein-protein interaction (PPI) relation prediction.

# Task Context
Primarily based on protein functions and COT chains, predict the interaction relation(s) between {{source_protein}} and {{target_protein}} (multi-label).

Important: The ground-truth relation set between the two input proteins is guaranteed to be non-empty.

{self._get_knowledge_base()}

# Related Knowledge
## Protein Information
- Source: **{{source_protein}}** — {{source_function}}
- Target: **{{target_protein}}** — {{target_function}}

## Path Protein Information
{{path_protein_functions}}

## COT Chains
{{path_description}}

# Constraints
- Use only the provided input; do not invent facts.
- Output must use the exact English label names from the 7 categories.
- Never output None. If unsure, output the single most likely label.

# Output Format
One line: comma-separated labels in the exact order shown in Interaction Categories."""
    
    def _get_confidence_template(self) -> str:
        """
        Get the confidence-based reasoning template.
        
        Returns:
            str: Confidence-based reasoning template.
        """
        return f"""# Role
You are an expert in protein-protein interaction (PPI) relation prediction.

# Task Context
Using confidence-annotated COT chains, predict the interaction relation(s) between {{source_protein}} and {{target_protein}} (multi-label).

Important: The ground-truth relation set between the two input proteins is guaranteed to be non-empty.

{self._get_knowledge_base()}

# Related Knowledge
## COT Chains (with confidence)
{{path_with_confidence}}

# Constraints
- Use only the provided input; do not invent facts.
- Output must use the exact English label names from the 7 categories.
- Never output None. If unsure, output the single most likely label.

# Output Format
One line: comma-separated labels in the exact order shown in Interaction Categories."""
    
    def path_to_text(self, 
                    path: List[int],
                    protein_info: Dict[int, Dict[str, str]],
                    relations: Optional[List[int]] = None,
                    confidence_scores: Optional[List[float]] = None
                    ) -> str:
        """
        将一个或多个COT路径转换为自然语言描述
        
        Args:
            path (List[int] or List[List[int]]): 路径列表或单个路径
            protein_info (Dict[int, Dict[str, str]]): 蛋白质信息字典
            relations (Optional[List[int]] or List[List[int]]): 关系列表或单个关系列表
            confidence_scores (Optional[List[float]] or List[List[float]]): 置信度列表或单个置信度列表
            
        Returns:
            str: 自然语言描述的路径
        """
        # 检查是否是多条路径
        if path and isinstance(path[0], list):
            multi_path_desc = []
            for idx, p in enumerate(path):
                rels = relations[idx] if relations and idx < len(relations) else None
                confs = confidence_scores[idx] if confidence_scores and idx < len(confidence_scores) else None
                p_desc = self._single_path_to_text(p, protein_info, rels, confs)
                multi_path_desc.append(f"## Chain {idx+1}\n{p_desc}")
            return "\n\n".join(multi_path_desc)
        else:
            return self._single_path_to_text(path, protein_info, relations, confidence_scores)

    def _single_path_to_text(self, 
                            path: List[int],
                            protein_info: Dict[int, Dict[str, str]],
                            relations: Optional[List[int]] = None,
                            confidence_scores: Optional[List[float]] = None
                            ) -> str:
        """内部方法：将单个路径转换为文本"""
        if len(path) < 2:
            return "Empty path"
        
        path_desc = []
        for i in range(len(path) - 1):
            source_id = path[i]
            target_id = path[i+1]
            
            # 获取蛋白质名称
            source_info = protein_info.get(source_id, {})
            target_info = protein_info.get(target_id, {})
            
            source_name = source_info.get('name', f"Protein {source_id}")
            target_name = target_info.get('name', f"Protein {target_id}")
            
            # 尝试从全局信息中获取名称
            source_id_str = str(source_name) if source_name != f"Protein {source_id}" else str(source_id)
            target_id_str = str(target_name) if target_name != f"Protein {target_id}" else str(target_id)
            
            if source_id_str in self.protein_id_to_info:
                source_name = self.protein_id_to_info[source_id_str]['name']
            if target_id_str in self.protein_id_to_info:
                target_name = self.protein_id_to_info[target_id_str]['name']
            
            # 获取关系类型
            if relations and i < len(relations):
                relation_idx = relations[i]
                if relation_idx == 7:  # 相似性关系
                    relation = "Sequence Similarity"
                else:
                    relation = self.relation_map.get(relation_idx, f"Relation {relation_idx}")
            else:
                relation = "Interaction"
            
            # 构建步骤描述
            step_desc = f"{i+1}. {source_name} —[{relation}]→ {target_name}"
            
            # 添加置信度信息
            if confidence_scores and i < len(confidence_scores):
                step_desc += f" (Confidence: {confidence_scores[i]:.2f})"
            
            path_desc.append(step_desc)
        
        return "\n".join(path_desc)

    
    def generate_prompt(self, 
                       template_type: str,
                       source_protein: str,
                       target_protein: str,
                       path: List[int],
                       protein_info: Dict[int, Dict[str, str]],
                       relations: Optional[List[int]] = None,
                       confidence_scores: Optional[List[float]] = None,
                       additional_info: Optional[Dict] = None
                       ) -> str:
        """
        生成提示文本
        
        Args:
            template_type (str): 提示模板类型
            source_protein (str): 源蛋白质名称
            target_protein (str): 目标蛋白质名称
            path (List[int]): 探索路径
            protein_info (Dict[int, Dict[str, str]]): 蛋白质信息字典
            relations (Optional[List[int]]): 关系类型列表
            confidence_scores (Optional[List[float]]): 置信度分数
            additional_info (Optional[Dict]): 附加信息
            
        Returns:
            str: 生成的提示文本
        """
        # 检查模板类型是否存在
        if template_type not in self.templates:
            logger.warning(f"未知的模板类型: {template_type}，使用默认模板")
            template_type = 'exploratory_reasoning'
        
        template = self.templates[template_type]
        
        # 转换路径为文本
        if confidence_scores:
            path_text = self.path_to_text(path, protein_info, relations, confidence_scores)
        else:
            path_text = self.path_to_text(path, protein_info, relations)
        
        # 获取源和目标蛋白质的信息（如果可用）
        source_id = str(source_protein)
        target_id = str(target_protein)
        
        source_name = source_protein
        target_name = target_protein
        source_function = "No functional description available"
        target_function = "No functional description available"
        
        # 尝试从全局信息中获取更详细的信息
        if source_id in self.protein_id_to_info:
            source_name = self.protein_id_to_info[source_id]['name']
            source_function = self.protein_id_to_info[source_id]['function']
        
        if target_id in self.protein_id_to_info:
            target_name = self.protein_id_to_info[target_id]['name']
            target_function = self.protein_id_to_info[target_id]['function']

        # 构建提示变量
        path_protein_functions = self._path_protein_functions_text(path, protein_info)
        prompt_vars = {
            'source_protein': source_name,
            'target_protein': target_name,
            'source_function': source_function,
            'target_function': target_function,
            'path_protein_functions': path_protein_functions,
            'path_description': path_text
        }
        
        # 添加附加信息
        if additional_info:
            prompt_vars.update(additional_info)
        
        # 生成完整提示
        prompt = template.format(**prompt_vars)
        
        return prompt

    def _path_protein_functions_text(self, path, protein_info: Dict[int, Dict[str, str]]) -> str:
        node_ids: List[int] = []
        if path and isinstance(path, list) and path and isinstance(path[0], list):
            for p in path:
                for nid in p:
                    node_ids.append(int(nid))
        else:
            for nid in (path or []):
                node_ids.append(int(nid))

        seen = set()
        ordered_unique = []
        for nid in node_ids:
            if nid in seen:
                continue
            seen.add(nid)
            ordered_unique.append(nid)

        lines = []
        for nid in ordered_unique:
            info = protein_info.get(nid, {}) or {}
            pid = info.get('name', f"Protein {nid}")
            pid_str = str(pid)
            display_name = pid_str
            func = "No functional description available"
            if pid_str in self.protein_id_to_info:
                display_name = self.protein_id_to_info[pid_str].get('name', pid_str)
                func = self.protein_id_to_info[pid_str].get('function', func)
            lines.append(f"- Function of {pid_str}: {func}")

        return "\n".join(lines) if lines else "- No path protein information available"
    
    def generate_batch_prompts(self, 
                              template_type: str,
                              batch_data: List[Dict]
                              ) -> List[str]:
        """
        批量生成提示文本
        
        Args:
            template_type (str): 提示模板类型
            batch_data (List[Dict]): 批量数据，每个元素包含生成提示所需的信息
            
        Returns:
            List[str]: 生成的提示文本列表
        """
        prompts = []
        
        for data in batch_data:
            prompt = self.generate_prompt(
                template_type=template_type,
                source_protein=data['source_protein'],
                target_protein=data['target_protein'],
                path=data['path'],
                protein_info=data['protein_info'],
                relations=data.get('relations'),
                confidence_scores=data.get('confidence_scores'),
                additional_info=data.get('additional_info')
            )
            prompts.append(prompt)
        
        return prompts
    
    def add_protein_function_info(self, 
                                 protein_id: int,
                                 function_info: str,
                                 protein_info_dict: Dict[int, Dict[str, str]]
                                 ) -> Dict[int, Dict[str, str]]:
        """
        向蛋白质信息字典中添加功能信息
        
        Args:
            protein_id (int): 蛋白质ID
            function_info (str): 蛋白质功能信息
            protein_info_dict (Dict[int, Dict[str, str]]): 蛋白质信息字典
            
        Returns:
            Dict[int, Dict[str, str]]: 更新后的蛋白质信息字典
        """
        if protein_id not in protein_info_dict:
            protein_info_dict[protein_id] = {}
        
        protein_info_dict[protein_id]['function'] = function_info
        
        return protein_info_dict
    
    def relation_to_text(self, relation_idx: int) -> str:
        """
        将关系索引转换为文本描述
        
        Args:
            relation_idx (int): 关系索引
            
        Returns:
            str: 关系文本描述
        """
        return self.relation_map.get(relation_idx, f"未知关系类型 {relation_idx}")
    
    def text_to_relation(self, relation_text: str) -> Optional[int]:
        """
        将关系文本转换为索引
        
        Args:
            relation_text (str): 关系文本描述
            
        Returns:
            Optional[int]: 关系索引，如果未找到则返回None
        """
        # 反向映射
        reverse_map = {v: k for k, v in self.relation_map.items()}
        return reverse_map.get(relation_text)
