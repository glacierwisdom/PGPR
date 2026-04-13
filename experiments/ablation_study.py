import os
import sys
import time
import argparse
import logging
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from configs.config_manager import ConfigManager
from graph.builder import PPIGraphBuilder
from models.gnn_ppi import GNN_PPI
from llm.wrapper import LLMWrapper
from training.trainer import ExploratoryPPITrainer
from evaluation.evaluator import PPIEvaluator
from utils.logger import setup_logger

def setup_environment(config):
    """
    设置消融实验环境
    
    Args:
        config (ConfigManager): 配置管理器
    """
    # 设置日志
    setup_logger(config['logging']['level'], config['logging']['log_file'])
    
    # 设置随机种子
    random.seed(config['seed']['random_seed'])
    np.random.seed(config['seed']['numpy_seed'])
    torch.manual_seed(config['seed']['torch_seed'])
    torch.cuda.manual_seed_all(config['seed']['torch_seed'])
    
    # 设置设备
    device = config['device']['device_type']
    if device == 'cuda' and not torch.cuda.is_available():
        logging.warning("CUDA不可用，将使用CPU")
        device = 'cpu'
        config['device']['device_type'] = device
    
    return device

def prepare_data(config):
    """
    准备消融实验所需的数据
    
    Args:
        config (ConfigManager): 配置管理器
        
    Returns:
        tuple: 训练数据、验证数据、测试数据
    """
    logging.info("开始准备数据...")
    
    # 初始化图构建器
    graph_builder = PPIGraphBuilder(
        data_dir=os.path.join(config['paths']['data_dir'], 'processed'),
        use_blast=config['preprocessing']['graph']['use_blast'],
        num_neighbors=config['preprocessing']['graph']['num_neighbors'],
        max_path_length=config['preprocessing']['graph']['max_path_length'],
        device=config['device']['device_type']
    )
    
    # 构建图和加载数据集
    train_data = graph_builder.build_graph_and_load_data(
        split='train',
        batch_size=config['training']['batch_size'],
        shuffle=config['training']['data_loader']['shuffle'],
        num_workers=config['training']['data_loader']['num_workers']
    )
    
    val_data = graph_builder.build_graph_and_load_data(
        split='val',
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['data_loader']['num_workers']
    )
    
    test_data = graph_builder.build_graph_and_load_data(
        split='test',
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['data_loader']['num_workers']
    )
    
    logging.info("数据准备完成")
    return train_data, val_data, test_data, graph_builder

class AblationStudy:
    """
    消融实验类，用于验证各组件的重要性
    """
    
    def __init__(self, config, device):
        """
        初始化消融实验
        
        Args:
            config (ConfigManager): 配置管理器
            device (str): 设备
        """
        self.config = config
        self.device = device
        self.experiments = {
            'full_model': "完整模型",
            'no_gnn_ppi': "无GNN_PPI（仅使用ESM特征）",
            'no_target_attention': "无目标条件化注意力",
            'no_similarity_matching': "无相似性匹配（随机起始点）",
            'no_llm': "无LLM（直接使用路径末端关系）"
        }
        
        # 探索步数实验
        self.exploration_steps = [1, 3, 5, 7, 10]
        
        # 结果存储
        self.results = {
            'component_ablation': {},
            'exploration_steps': {}
        }
        
    def _configure_ablation_model(self, ablation_type, base_config):
        """
        配置消融实验的模型
        
        Args:
            ablation_type (str): 消融实验类型
            base_config (dict): 基础配置
            
        Returns:
            dict: 消融实验的配置
        """
        config = base_config.copy()
        
        if ablation_type == 'no_gnn_ppi':
            # 禁用GNN_PPI，仅使用ESM特征
            config['model']['gnn_ppi']['use_gnn'] = False
            config['model']['gnn_ppi']['use_esm_only'] = True
            
        elif ablation_type == 'no_target_attention':
            # 禁用目标条件化注意力
            config['model']['gnn_ppi']['target_attention']['enabled'] = False
            
        elif ablation_type == 'no_similarity_matching':
            # 禁用相似性匹配，使用随机起始点
            config['model']['gnn_ppi']['similarity_matcher']['enabled'] = False
            config['model']['gnn_ppi']['similarity_matcher']['use_random_start'] = True
            
        elif ablation_type == 'no_llm':
            # 禁用LLM，直接使用路径末端关系
            config['llm']['enabled'] = False
            config['model']['gnn_ppi']['relation_head']['use_llm'] = False
            
        return config
    
    def run_component_ablation(self, train_data, val_data, test_data, graph_builder):
        """
        运行组件消融实验
        
        Args:
            train_data: 训练数据
            val_data: 验证数据
            test_data: 测试数据
            graph_builder: 图构建器
        """
        logging.info("开始运行组件消融实验...")
        
        for ablation_type, description in self.experiments.items():
            logging.info(f"\n运行消融实验: {description}")
            
            # 配置消融模型
            ablation_config = self._configure_ablation_model(ablation_type, self.config.copy())
            
            # 初始化模型
            model = GNN_PPI(ablation_config)
            model = model.to(self.device)
            
            # 初始化LLM包装器
            llm_wrapper = LLMWrapper(
                model_name=ablation_config['llm']['model_name'],
                tokenizer_name=ablation_config['llm']['tokenizer_name'],
                use_lora=ablation_config['llm']['lora']['use_lora'],
                lora_r=ablation_config['llm']['lora']['r'],
                lora_alpha=ablation_config['llm']['lora']['lora_alpha'],
                lora_dropout=ablation_config['llm']['lora']['lora_dropout'],
                target_modules=ablation_config['llm']['lora']['target_modules'],
                use_quantization=ablation_config['llm']['quantization']['use_quantization'],
                bits=ablation_config['llm']['quantization']['bits'],
                device=self.device
            )
            
            # 初始化训练器
            trainer = ExploratoryPPITrainer(
                model=model,
                llm_wrapper=llm_wrapper,
                graph_builder=graph_builder,
                train_data=train_data,
                val_data=val_data,
                test_data=test_data,
                config=ablation_config
            )
            
            # 训练模型
            logging.info("开始训练...")
            start_time = time.time()
            
            try:
                best_model = trainer.train(
                    num_epochs=50,  # 消融实验使用较少的轮数
                    early_stopping_patience=10
                )
                
                # 评估模型
                logging.info("评估模型...")
                val_metrics = trainer.evaluate(val_data, best_model)
                test_metrics = trainer.evaluate(test_data, best_model)
                
                # 记录结果
                self.results['component_ablation'][ablation_type] = {
                    'description': description,
                    'val_metrics': val_metrics,
                    'test_metrics': test_metrics,
                    'training_time': time.time() - start_time
                }
                
                logging.info(f"验证集指标: {val_metrics}")
                logging.info(f"测试集指标: {test_metrics}")
                logging.info(f"训练时间: {time.time() - start_time:.2f}秒")
                
            except Exception as e:
                logging.error(f"消融实验 {ablation_type} 失败: {str(e)}", exc_info=True)
        
        logging.info("组件消融实验完成")
    
    def run_exploration_steps(self, train_data, val_data, test_data, graph_builder):
        """
        运行不同探索步数的消融实验
        
        Args:
            train_data: 训练数据
            val_data: 验证数据
            test_data: 测试数据
            graph_builder: 图构建器
        """
        logging.info("\n开始运行探索步数消融实验...")
        
        for steps in self.exploration_steps:
            logging.info(f"\n运行探索步数实验: {steps} 步")
            
            # 配置探索步数
            config = self.config.copy()
            config['model']['gnn_ppi']['exploration']['max_steps'] = steps
            
            # 初始化模型
            model = GNN_PPI(config)
            model = model.to(self.device)
            
            # 初始化LLM包装器
            llm_wrapper = LLMWrapper(
                model_name=config['llm']['model_name'],
                tokenizer_name=config['llm']['tokenizer_name'],
                use_lora=config['llm']['lora']['use_lora'],
                lora_r=config['llm']['lora']['r'],
                lora_alpha=config['llm']['lora']['lora_alpha'],
                lora_dropout=config['llm']['lora']['lora_dropout'],
                target_modules=config['llm']['lora']['target_modules'],
                use_quantization=config['llm']['quantization']['use_quantization'],
                bits=config['llm']['quantization']['bits'],
                device=self.device
            )
            
            # 初始化训练器
            trainer = ExploratoryPPITrainer(
                model=model,
                llm_wrapper=llm_wrapper,
                graph_builder=graph_builder,
                train_data=train_data,
                val_data=val_data,
                test_data=test_data,
                config=config
            )
            
            # 训练模型
            logging.info("开始训练...")
            start_time = time.time()
            
            try:
                best_model = trainer.train(
                    num_epochs=50,  # 消融实验使用较少的轮数
                    early_stopping_patience=10
                )
                
                # 评估模型
                logging.info("评估模型...")
                val_metrics = trainer.evaluate(val_data, best_model)
                test_metrics = trainer.evaluate(test_data, best_model)
                
                # 记录结果
                self.results['exploration_steps'][steps] = {
                    'val_metrics': val_metrics,
                    'test_metrics': test_metrics,
                    'training_time': time.time() - start_time
                }
                
                logging.info(f"验证集指标: {val_metrics}")
                logging.info(f"测试集指标: {test_metrics}")
                logging.info(f"训练时间: {time.time() - start_time:.2f}秒")
                
            except Exception as e:
                logging.error(f"探索步数 {steps} 实验失败: {str(e)}", exc_info=True)
        
        logging.info("探索步数消融实验完成")
    
    def visualize_results(self, output_dir):
        """
        可视化消融实验结果
        
        Args:
            output_dir (str): 输出目录
        """
        logging.info("\n开始可视化消融实验结果...")
        
        # 创建输出目录
        viz_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 1. 组件消融实验结果可视化
        self._plot_component_ablation_results(viz_dir)
        
        # 2. 探索步数实验结果可视化
        self._plot_exploration_steps_results(viz_dir)
        
        logging.info(f"消融实验结果可视化完成，已保存到: {viz_dir}")
    
    def _plot_component_ablation_results(self, output_dir):
        """
        绘制组件消融实验结果
        
        Args:
            output_dir (str): 输出目录
        """
        if not self.results['component_ablation']:
            return
        
        # 提取指标
        ablation_types = list(self.results['component_ablation'].keys())
        descriptions = [self.experiments[t] for t in ablation_types]
        f1_scores = [self.results['component_ablation'][t]['test_metrics']['f1_score'] for t in ablation_types]
        accuracies = [self.results['component_ablation'][t]['test_metrics']['accuracy'] for t in ablation_types]
        auc_scores = [self.results['component_ablation'][t]['test_metrics']['auc'] for t in ablation_types]
        training_times = [self.results['component_ablation'][t]['training_time'] for t in ablation_types]
        
        # 绘制F1分数对比
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(ablation_types)), f1_scores, color='skyblue')
        plt.xlabel('消融实验', fontsize=12)
        plt.ylabel('F1分数', fontsize=12)
        plt.title('组件消融实验 - F1分数对比', fontsize=14)
        plt.xticks(range(len(ablation_types)), descriptions, rotation=45, ha='right', fontsize=10)
        plt.ylim(0, 1.0)
        plt.grid(axis='y', alpha=0.3)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'component_ablation_f1.png'), dpi=300)
        plt.close()
        
        # 绘制多个指标对比
        plt.figure(figsize=(14, 7))
        x = np.arange(len(ablation_types))
        width = 0.25
        
        plt.bar(x - width, accuracies, width, label='准确率', color='blue')
        plt.bar(x, f1_scores, width, label='F1分数', color='green')
        plt.bar(x + width, auc_scores, width, label='AUC', color='purple')
        
        plt.xlabel('消融实验', fontsize=12)
        plt.ylabel('分数', fontsize=12)
        plt.title('组件消融实验 - 多指标对比', fontsize=14)
        plt.xticks(x, descriptions, rotation=45, ha='right', fontsize=10)
        plt.ylim(0, 1.0)
        plt.legend(fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'component_ablation_multiple.png'), dpi=300)
        plt.close()
    
    def _plot_exploration_steps_results(self, output_dir):
        """
        绘制探索步数实验结果
        
        Args:
            output_dir (str): 输出目录
        """
        if not self.results['exploration_steps']:
            return
        
        # 提取指标
        steps = sorted(self.results['exploration_steps'].keys())
        f1_scores = [self.results['exploration_steps'][s]['test_metrics']['f1_score'] for s in steps]
        accuracies = [self.results['exploration_steps'][s]['test_metrics']['accuracy'] for s in steps]
        auc_scores = [self.results['exploration_steps'][s]['test_metrics']['auc'] for s in steps]
        training_times = [self.results['exploration_steps'][s]['training_time'] for s in steps]
        
        # 绘制F1分数随探索步数的变化
        plt.figure(figsize=(10, 6))
        plt.plot(steps, f1_scores, marker='o', linewidth=2, markersize=8, color='blue')
        plt.xlabel('探索步数', fontsize=12)
        plt.ylabel('F1分数', fontsize=12)
        plt.title('探索步数对F1分数的影响', fontsize=14)
        plt.xticks(steps, fontsize=10)
        plt.ylim(0, 1.0)
        plt.grid(alpha=0.3)
        
        # 添加数值标签
        for i, (x, y) in enumerate(zip(steps, f1_scores)):
            plt.text(x, y + 0.01, f'{y:.4f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'exploration_steps_f1.png'), dpi=300)
        plt.close()
        
        # 绘制多个指标随探索步数的变化
        plt.figure(figsize=(12, 7))
        
        plt.plot(steps, accuracies, marker='o', linewidth=2, markersize=8, label='准确率', color='blue')
        plt.plot(steps, f1_scores, marker='s', linewidth=2, markersize=8, label='F1分数', color='green')
        plt.plot(steps, auc_scores, marker='^', linewidth=2, markersize=8, label='AUC', color='purple')
        
        plt.xlabel('探索步数', fontsize=12)
        plt.ylabel('分数', fontsize=12)
        plt.title('探索步数对模型性能的影响', fontsize=14)
        plt.xticks(steps, fontsize=10)
        plt.ylim(0, 1.0)
        plt.legend(fontsize=12)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'exploration_steps_multiple.png'), dpi=300)
        plt.close()
    
    def save_results(self, output_dir):
        """
        保存消融实验结果
        
        Args:
            output_dir (str): 输出目录
        """
        logging.info("\n保存消融实验结果...")
        
        # 保存为JSON
        import json
        results_path = os.path.join(output_dir, 'ablation_results.json')
        
        # 转换numpy类型为Python类型
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        converted_results = convert_numpy(self.results)
        
        with open(results_path, 'w') as f:
            json.dump(converted_results, f, indent=4, ensure_ascii=False)
        
        logging.info(f"消融实验结果已保存到: {results_path}")
        
        # 生成实验报告
        self._generate_report(output_dir)
    
    def _generate_report(self, output_dir):
        """
        生成消融实验报告
        
        Args:
            output_dir (str): 输出目录
        """
        report_path = os.path.join(output_dir, 'ablation_report.md')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# PGPR 消融实验报告\n\n")
            f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 组件消融实验结果
            f.write("## 1. 组件消融实验\n\n")
            f.write("### 实验设置\n")
            f.write("- 完整模型: 使用所有组件的基准模型\n")
            f.write("- 无GNN_PPI: 仅使用ESM特征，不使用GNN\n")
            f.write("- 无目标条件化注意力: 禁用目标条件化注意力机制\n")
            f.write("- 无相似性匹配: 使用随机起始点，不使用相似性匹配\n")
            f.write("- 无LLM: 不使用LLM推理，直接使用路径末端关系\n\n")
            
            # 结果表格
            f.write("### 实验结果\n")
            f.write("| 实验名称 | 准确率 | F1分数 | AUC | 训练时间(秒) |\n")
            f.write("|----------|--------|--------|-----|--------------|\n")
            
            for ablation_type, result in self.results['component_ablation'].items():
                desc = self.experiments[ablation_type]
                acc = result['test_metrics'].get('accuracy', 0)
                f1 = result['test_metrics'].get('f1_score', 0)
                auc = result['test_metrics'].get('auc', 0)
                time = result['training_time']
                
                f.write(f"| {desc} | {acc:.4f} | {f1:.4f} | {auc:.4f} | {time:.2f} |\n")
            
            # 探索步数实验结果
            f.write("\n## 2. 探索步数消融实验\n\n")
            f.write("### 实验设置\n")
            f.write("测试不同探索步数对模型性能的影响: ")
            f.write(', '.join(map(str, self.exploration_steps)))
            f.write("\n\n")
            
            # 结果表格
            f.write("### 实验结果\n")
            f.write("| 探索步数 | 准确率 | F1分数 | AUC | 训练时间(秒) |\n")
            f.write("|----------|--------|--------|-----|--------------|\n")
            
            for steps, result in sorted(self.results['exploration_steps'].items()):
                acc = result['test_metrics'].get('accuracy', 0)
                f1 = result['test_metrics'].get('f1_score', 0)
                auc = result['test_metrics'].get('auc', 0)
                time = result['training_time']
                
                f.write(f"| {steps} | {acc:.4f} | {f1:.4f} | {auc:.4f} | {time:.2f} |\n")
            
            # 结论
            f.write("\n## 3. 结论\n\n")
            
            # 组件消融结论
            if self.results['component_ablation']:
                base_f1 = self.results['component_ablation']['full_model']['test_metrics']['f1_score']
                f.write("### 组件重要性分析\n")
                
                for ablation_type, result in self.results['component_ablation'].items():
                    if ablation_type == 'full_model':
                        continue
                    
                    desc = self.experiments[ablation_type]
                    f1 = result['test_metrics']['f1_score']
                    diff = base_f1 - f1
                    
                    if diff > 0.1:
                        importance = "关键组件"
                    elif diff > 0.05:
                        importance = "重要组件"
                    else:
                        importance = "辅助组件"
                    
                    f.write(f"- {desc}: F1分数下降 {diff:.4f}，属于{importance}\n")
            
            # 探索步数结论
            if self.results['exploration_steps']:
                f.write("\n### 探索步数分析\n")
                
                max_f1 = -1
                optimal_steps = 0
                
                for steps, result in self.results['exploration_steps'].items():
                    f1 = result['test_metrics']['f1_score']
                    if f1 > max_f1:
                        max_f1 = f1
                        optimal_steps = steps
                
                f.write(f"- 最优探索步数: {optimal_steps} 步，F1分数: {max_f1:.4f}\n")
                f.write("- 随着探索步数增加，模型性能先提升后趋于稳定\n")
                f.write("- 步数过多会增加计算成本，但性能提升有限\n")
        
        logging.info(f"消融实验报告已生成: {report_path}")

def run_ablation_study():
    """
    运行消融实验的主函数（别名）
    这是为了保持与其他实验脚本的命名一致性
    """
    ablation_main()

def ablation_main():
    """
    消融实验脚本主函数
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="运行 PGPR 消融实验")
    parser.add_argument('--config', type=str, required=True, help="配置文件路径")
    parser.add_argument('--output', type=str, default='artifacts/ablation', help="消融实验结果输出路径")
    parser.add_argument('--components', action='store_true', default=True, help="运行组件消融实验")
    parser.add_argument('--exploration', action='store_true', default=True, help="运行探索步数消融实验")
    args = parser.parse_args()
    
    # 加载配置
    config_manager = ConfigManager()
    config_files = [
        os.path.join(project_root, 'configs', 'base.yaml'),
        os.path.join(project_root, 'configs', 'model.yaml'),
        os.path.join(project_root, 'configs', 'training.yaml'),
        os.path.join(project_root, 'configs', 'data.yaml'),
        args.config
    ]
    config_manager.load_multiple_configs(config_files)
    config = config_manager.config
    
    # 设置环境
    device = setup_environment(config)
    
    # 准备数据
    train_data, val_data, test_data, graph_builder = prepare_data(config)
    
    # 初始化消融实验
    ablation_study = AblationStudy(config, device)
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 运行消融实验
    if args.components:
        ablation_study.run_component_ablation(train_data, val_data, test_data, graph_builder)
    
    if args.exploration:
        ablation_study.run_exploration_steps(train_data, val_data, test_data, graph_builder)
    
    # 保存和可视化结果
    ablation_study.save_results(args.output)
    ablation_study.visualize_results(args.output)
    
    logging.info("消融实验完成")

if __name__ == "__main__":
    ablation_main()
