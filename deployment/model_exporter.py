import os
import sys
import argparse
import logging
import json
import time
import torch
import numpy as np
from typing import Dict, Any, Optional

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from configs.config_manager import ConfigManager
from models.gnn_ppi import GNN_PPI
from utils.logger import setup_logger

# 设置日志
setup_logger(logging.INFO, os.path.join(project_root, 'logs', 'model_exporter.log'))
logger = logging.getLogger('model_exporter')

class ModelExporter:
    """
    PGPR 模型导出工具，支持 ONNX 格式和其他部署格式
    """
    
    def __init__(self, config_path: str, checkpoint_path: str):
        """
        初始化模型导出工具
        
        Args:
            config_path (str): 配置文件路径
            checkpoint_path (str): 模型检查点路径
        """
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        
        # 加载配置
        self.config_manager = ConfigManager()
        self.config_manager.load_config(config_path)
        self.config = self.config_manager.config
        
        # 设置设备
        self.device = self.config['device']['device_type']
        if self.device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA不可用，将使用CPU")
            self.device = 'cpu'
            self.config['device']['device_type'] = self.device
        
        # 加载模型
        self.model = self._load_model()
        logger.info(f"模型加载完成，使用设备: {self.device}")
    
    def _load_model(self) -> GNN_PPI:
        """
        加载模型
        
        Returns:
            GNN_PPI: 加载的模型
        """
        logger.info(f"开始加载模型检查点: {self.checkpoint_path}")
        
        # 创建模型实例
        model = GNN_PPI(self.config)
        model = model.to(self.device)
        
        # 加载检查点
        if os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            logger.info(f"模型检查点加载成功")
        else:
            logger.error(f"模型检查点不存在: {self.checkpoint_path}")
            raise FileNotFoundError(f"模型检查点不存在: {self.checkpoint_path}")
        
        # 设置模型为评估模式
        model.eval()
        return model
    
    def export_to_onnx(self, output_path: str, dynamic_axes: Optional[Dict] = None, opset_version: int = 13):
        """
        将模型导出为ONNX格式
        
        Args:
            output_path (str): 输出ONNX文件路径
            dynamic_axes (Optional[Dict]): 动态轴配置
            opset_version (int): ONNX opset版本
        """
        logger.info(f"开始将模型导出为ONNX格式，输出路径: {output_path}")
        start_time = time.time()
        
        # 创建输出目录
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # 创建示例输入
        example_input = self._create_example_input()
        
        # 默认动态轴配置
        if dynamic_axes is None:
            dynamic_axes = {
                'node_features': {0: 'num_nodes'},
                'edge_index': {1: 'num_edges'},
                'edge_features': {0: 'num_edges'},
                'target_protein_embedding': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        
        # 导出ONNX
        torch.onnx.export(
            self.model,
            example_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['node_features', 'edge_index', 'edge_features', 'target_protein_embedding'],
            output_names=['output'],
            dynamic_axes=dynamic_axes
        )
        
        logger.info(f"ONNX模型导出完成，耗时: {time.time() - start_time:.2f}秒")
        logger.info(f"ONNX模型大小: {os.path.getsize(output_path) / (1024 * 1024):.2f} MB")
    
    def _create_example_input(self) -> tuple:
        """
        创建示例输入数据
        
        Returns:
            tuple: 示例输入张量
        """
        # 根据模型配置创建示例输入
        node_embedding_dim = self.config['model']['gnn_ppi']['node_representation']['hidden_dim']
        edge_embedding_dim = self.config['model']['gnn_ppi']['graph_attention']['edge_dim']
        
        # 创建示例节点特征
        num_nodes = 100
        node_features = torch.randn(num_nodes, node_embedding_dim).to(self.device)
        
        # 创建示例边索引（无向图）
        num_edges = 200
        edge_index = torch.randint(0, num_nodes, (2, num_edges)).to(self.device)
        
        # 创建示例边特征
        edge_features = torch.randn(num_edges, edge_embedding_dim).to(self.device)
        
        # 创建示例目标蛋白质嵌入
        target_protein_embedding = torch.randn(1, node_embedding_dim).to(self.device)
        
        return (node_features, edge_index, edge_features, target_protein_embedding)
    
    def export_to_torchscript(self, output_path: str, traced: bool = True):
        """
        将模型导出为TorchScript格式
        
        Args:
            output_path (str): 输出TorchScript文件路径
            traced (bool): 是否使用跟踪方式导出
        """
        logger.info(f"开始将模型导出为TorchScript格式，输出路径: {output_path}")
        start_time = time.time()
        
        # 创建输出目录
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        if traced:
            # 使用跟踪方式导出
            example_input = self._create_example_input()
            traced_model = torch.jit.trace(self.model, example_input)
            traced_model.save(output_path)
        else:
            # 使用脚本方式导出
            scripted_model = torch.jit.script(self.model)
            scripted_model.save(output_path)
        
        logger.info(f"TorchScript模型导出完成，耗时: {time.time() - start_time:.2f}秒")
        logger.info(f"TorchScript模型大小: {os.path.getsize(output_path) / (1024 * 1024):.2f} MB")
    
    def quantize_model(self, output_path: str, quantize_type: str = 'dynamic'):
        """
        对模型进行量化
        
        Args:
            output_path (str): 输出量化模型路径
            quantize_type (str): 量化类型 ('dynamic', 'static', 'integer')
        """
        logger.info(f"开始对模型进行量化，类型: {quantize_type}")
        start_time = time.time()
        
        # 创建输出目录
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        if quantize_type == 'dynamic':
            # 动态量化
            quantized_model = torch.quantization.quantize_dynamic(
                self.model,
                {torch.nn.Linear, torch.nn.LSTM},
                dtype=torch.qint8
            )
        
        elif quantize_type == 'static':
            # 静态量化需要校准数据
            logger.info("静态量化需要校准数据，正在生成校准数据...")
            
            # 准备校准数据生成器
            def calibration_data():
                for _ in range(32):  # 使用32个样本进行校准
                    yield self._create_example_input()
            
            # 配置量化
            self.model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            model_prepared = torch.quantization.prepare(self.model)
            
            # 校准模型
            for data in calibration_data():
                model_prepared(*data)
            
            # 转换为量化模型
            quantized_model = torch.quantization.convert(model_prepared)
        
        else:
            raise ValueError(f"不支持的量化类型: {quantize_type}")
        
        # 保存量化模型
        torch.save({
            'model_state_dict': quantized_model.state_dict(),
            'config': self.config,
            'quantization_type': quantize_type
        }, output_path)
        
        logger.info(f"模型量化完成，耗时: {time.time() - start_time:.2f}秒")
        logger.info(f"量化模型大小: {os.path.getsize(output_path) / (1024 * 1024):.2f} MB")
    
    def prune_model(self, output_path: str, amount: float = 0.3, pruning_method: str = 'l1_unstructured'):
        """
        对模型进行剪枝
        
        Args:
            output_path (str): 输出剪枝模型路径
            amount (float): 剪枝比例 (0.0-1.0)
            pruning_method (str): 剪枝方法
        """
        logger.info(f"开始对模型进行剪枝，比例: {amount:.2f}, 方法: {pruning_method}")
        start_time = time.time()
        
        # 创建输出目录
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        import torch.nn.utils.prune as prune
        
        # 对模型中的线性层进行剪枝
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                if pruning_method == 'l1_unstructured':
                    prune.l1_unstructured(module, name='weight', amount=amount)
                elif pruning_method == 'random_unstructured':
                    prune.random_unstructured(module, name='weight', amount=amount)
                elif pruning_method == 'ln_structured':
                    prune.ln_structured(module, name='weight', amount=amount, n=2, dim=0)
                else:
                    logger.warning(f"不支持的剪枝方法: {pruning_method}, 跳过该层")
        
        # 移除剪枝掩码，使剪枝永久化
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                if hasattr(module, 'weight_mask'):
                    prune.remove(module, 'weight')
        
        # 保存剪枝模型
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'pruning_amount': amount,
            'pruning_method': pruning_method
        }, output_path)
        
        logger.info(f"模型剪枝完成，耗时: {time.time() - start_time:.2f}秒")
        logger.info(f"剪枝模型大小: {os.path.getsize(output_path) / (1024 * 1024):.2f} MB")
    
    def create_deployment_package(self, output_dir: str, include_onnx: bool = True, include_torchscript: bool = True, 
                                quantize: bool = True, prune: bool = True):
        """
        创建完整的部署包
        
        Args:
            output_dir (str): 输出目录
            include_onnx (bool): 是否包含ONNX模型
            include_torchscript (bool): 是否包含TorchScript模型
            quantize (bool): 是否包含量化模型
            prune (bool): 是否包含剪枝模型
        """
        logger.info(f"开始创建部署包，输出目录: {output_dir}")
        start_time = time.time()
        
        # 创建部署包目录结构
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'configs'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'docs'), exist_ok=True)
        
        # 导出原始PyTorch模型
        original_model_path = os.path.join(output_dir, 'models', 'model_original.pth')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config
        }, original_model_path)
        
        # 导出ONNX模型
        if include_onnx:
            onnx_path = os.path.join(output_dir, 'models', 'model.onnx')
            self.export_to_onnx(onnx_path)
        
        # 导出TorchScript模型
        if include_torchscript:
            torchscript_path = os.path.join(output_dir, 'models', 'model.pt')
            self.export_to_torchscript(torchscript_path)
        
        # 导出量化模型
        if quantize:
            quantized_path = os.path.join(output_dir, 'models', 'model_quantized.pth')
            self.quantize_model(quantized_path)
        
        # 导出剪枝模型
        if prune:
            pruned_path = os.path.join(output_dir, 'models', 'model_pruned.pth')
            self.prune_model(pruned_path)
        
        # 保存配置文件
        config_path = os.path.join(output_dir, 'configs', 'model_config.yaml')
        self.config_manager.save_config(config_path)
        
        # 生成部署文档
        self._generate_deployment_docs(output_dir)
        
        # 生成README文件
        self._generate_readme(output_dir)
        
        logger.info(f"部署包创建完成，耗时: {time.time() - start_time:.2f}秒")
        logger.info(f"部署包大小: {self._get_directory_size(output_dir) / (1024 * 1024):.2f} MB")
    
    def _generate_deployment_docs(self, output_dir: str):
        """
        生成部署文档
        
        Args:
            output_dir (str): 输出目录
        """
        docs_path = os.path.join(output_dir, 'docs', 'deployment_guide.md')
        
        with open(docs_path, 'w', encoding='utf-8') as f:
            f.write("# PGPR 模型部署指南\n\n")
            f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## 1. 模型文件说明\n\n")
            f.write("### 1.1 PyTorch模型\n")
            f.write("- `model_original.pth`: 原始PyTorch模型\n")
            f.write("- `model_quantized.pth`: 量化后的PyTorch模型\n")
            f.write("- `model_pruned.pth`: 剪枝后的PyTorch模型\n\n")
            f.write("### 1.2 ONNX模型\n")
            f.write("- `model.onnx`: ONNX格式模型，支持跨平台部署\n\n")
            f.write("### 1.3 TorchScript模型\n")
            f.write("- `model.pt`: TorchScript格式模型，用于C++部署\n\n")
            f.write("## 2. 模型加载与推理\n\n")
            f.write("### 2.1 Python推理示例\n")
            f.write("```python\n")
            f.write("import torch\n")
            f.write("from models.gnn_ppi import GNN_PPI\n")
            f.write("from configs.config_manager import ConfigManager\n\n")
            f.write("# 加载配置\n")
            f.write("config_manager = ConfigManager()\n")
            f.write("config_manager.load_config('configs/model_config.yaml')\n")
            f.write("config = config_manager.config\n\n")
            f.write("# 加载模型\n")
            f.write("model = GNN_PPI(config)\n")
            f.write("checkpoint = torch.load('models/model_original.pth')\n")
            f.write("model.load_state_dict(checkpoint['model_state_dict'])\n")
            f.write("model.eval()\n\n")
            f.write("# 执行推理\n")
            f.write("# 请根据实际情况准备输入数据\n")
            f.write("# output = model(node_features, edge_index, edge_features, target_protein_embedding)\n")
            f.write("```\n\n")
            f.write("### 2.2 ONNX推理示例\n")
            f.write("```python\n")
            f.write("import onnxruntime as ort\n")
            f.write("import numpy as np\n\n")
            f.write("# 创建ONNX运行时会话\n")
            f.write("ort_session = ort.InferenceSession('models/model.onnx')\n\n")
            f.write("# 准备输入数据\n")
            f.write("# node_features = np.random.randn(num_nodes, node_embedding_dim).astype(np.float32)\n")
            f.write("# edge_index = np.random.randint(0, num_nodes, (2, num_edges)).astype(np.int64)\n")
            f.write("# edge_features = np.random.randn(num_edges, edge_embedding_dim).astype(np.float32)\n")
            f.write("# target_protein_embedding = np.random.randn(1, node_embedding_dim).astype(np.float32)\n\n")
            f.write("# 执行推理\n")
            f.write("# inputs = {\n")
            f.write("#     'node_features': node_features,\n")
            f.write("#     'edge_index': edge_index,\n")
            f.write("#     'edge_features': edge_features,\n")
            f.write("#     'target_protein_embedding': target_protein_embedding\n")
            f.write("# }\n")
            f.write("# outputs = ort_session.run(None, inputs)\n")
            f.write("```\n\n")
    
    def _generate_readme(self, output_dir: str):
        """
        生成README文件
        
        Args:
            output_dir (str): 输出目录
        """
        readme_path = os.path.join(output_dir, 'README.md')
        
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write("# PGPR 模型部署包\n\n")
            f.write("## 1. 项目介绍\n")
            f.write("PGPR (Policy-Guided Path Reasoning) 是一个用于\n")
            f.write("预测蛋白质-蛋白质相互作用的深度学习模型。\n\n")
            f.write("## 2. 部署包结构\n")
            f.write("```\n")
            f.write("deployment_package/\n")
            f.write("├── models/              # 模型文件\n")
            f.write("│   ├── model_original.pth    # 原始模型\n")
            f.write("│   ├── model_quantized.pth   # 量化模型\n")
            f.write("│   ├── model_pruned.pth      # 剪枝模型\n")
            f.write("│   ├── model.onnx            # ONNX模型\n")
            f.write("│   └── model.pt              # TorchScript模型\n")
            f.write("├── configs/             # 配置文件\n")
            f.write("│   └── model_config.yaml     # 模型配置\n")
            f.write("├── docs/                # 文档\n")
            f.write("│   └── deployment_guide.md   # 部署指南\n")
            f.write("└── README.md            # 说明文件\n")
            f.write("```\n\n")
            f.write("## 3. 使用说明\n")
            f.write("请参考 `docs/deployment_guide.md` 获取详细的部署和推理说明。\n\n")
    
    def _get_directory_size(self, directory: str) -> int:
        """
        获取目录大小
        
        Args:
            directory (str): 目录路径
            
        Returns:
            int: 目录大小（字节）
        """
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)
        return total_size

def export_main():
    """
    模型导出工具主函数
    """
    parser = argparse.ArgumentParser(description="PGPR 模型导出工具")
    parser.add_argument('--config', type=str, required=True, help="配置文件路径")
    parser.add_argument('--checkpoint', type=str, required=True, help="模型检查点路径")
    parser.add_argument('--output', type=str, default='deploy/models', help="输出目录")
    
    # 导出格式选择
    export_group = parser.add_argument_group('导出格式')
    export_group.add_argument('--onnx', action='store_true', help="导出为ONNX格式")
    export_group.add_argument('--torchscript', action='store_true', help="导出为TorchScript格式")
    export_group.add_argument('--package', action='store_true', help="创建完整部署包")
    
    # 模型优化选项
    optimize_group = parser.add_argument_group('模型优化')
    optimize_group.add_argument('--quantize', action='store_true', help="量化模型")
    optimize_group.add_argument('--prune', type=float, help="剪枝模型（比例0.0-1.0）")
    
    args = parser.parse_args()
    
    # 检查配置文件和检查点是否存在
    if not os.path.exists(args.config):
        logger.error(f"配置文件不存在: {args.config}")
        sys.exit(1)
    
    if not os.path.exists(args.checkpoint):
        logger.error(f"模型检查点不存在: {args.checkpoint}")
        sys.exit(1)
    
    # 初始化导出工具
    exporter = ModelExporter(args.config, args.checkpoint)
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 执行导出操作
    if args.onnx:
        onnx_path = os.path.join(args.output, 'model.onnx')
        exporter.export_to_onnx(onnx_path)
    
    if args.torchscript:
        torchscript_path = os.path.join(args.output, 'model.pt')
        exporter.export_to_torchscript(torchscript_path)
    
    if args.quantize:
        quantized_path = os.path.join(args.output, 'model_quantized.pth')
        exporter.quantize_model(quantized_path)
    
    if args.prune:
        if 0.0 <= args.prune <= 1.0:
            pruned_path = os.path.join(args.output, 'model_pruned.pth')
            exporter.prune_model(pruned_path, amount=args.prune)
        else:
            logger.error(f"剪枝比例必须在0.0-1.0之间，当前值: {args.prune}")
            sys.exit(1)
    
    if args.package:
        exporter.create_deployment_package(args.output)
    
    # 如果没有指定任何导出格式，默认导出ONNX和创建部署包
    if not any([args.onnx, args.torchscript, args.quantize, args.prune, args.package]):
        logger.info("没有指定导出格式，默认导出ONNX和创建部署包")
        exporter.export_to_onnx(os.path.join(args.output, 'model.onnx'))
        exporter.create_deployment_package(args.output)
    
    logger.info("模型导出完成")

if __name__ == "__main__":
    export_main()
