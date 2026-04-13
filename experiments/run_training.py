import os
import sys

# 设置 Hugging Face 镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import time
import argparse
import logging
import random
import numpy as np
import torch

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from configs.config_manager import ConfigManager
from graph.builder import PPIGraphBuilder
from training.trainer import ExploratoryPPITrainer
from utils.logger import setup_logger
import utils

def setup_seed(seed):
    """
    设置随机种子以确保实验可复现
    
    Args:
        seed (int): 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多GPU设置
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def prepare_data(config):
    """
    准备训练、验证和测试数据
    
    Args:
        config (ConfigManager): 配置管理器
        
    Returns:
        tuple: 训练数据、验证数据、测试数据, 图构建器
    """
    logging.info("开始准备数据...")
    
    # 统一维度设置
    esm_dim_value = config.get('gnn_ppi', {}).get('node_representation', {}).get('feature_dim', 320)
    edge_dim_value = config.get('model', {}).get('num_edge_features', 64)
    
    # 初始化图构建器（确保与模型维度一致）
    # 优先使用配置中的 processed_data_dir，如果没有则使用 data_dir/processed
    data_dir = config['paths'].get('processed_data_dir')
    if not data_dir:
        data_dir = os.path.join(config['paths']['data_dir'], 'processed')
        
    graph_builder = PPIGraphBuilder(
        esm_dim=esm_dim_value,
        edge_dim=edge_dim_value,
        device=config['device']['device_type'],
        data_dir=data_dir,
        use_blast=config['preprocessing']['graph']['use_blast'],
        num_neighbors=config['preprocessing']['graph']['num_neighbors'],
        max_path_length=config['preprocessing']['graph']['max_path_length']
    )
    
    # 获取配置中的文件名
    train_file = config['dataset'].get('train_file')
    val_file = config['dataset'].get('val_file')
    test_file = config['dataset'].get('test_file')
    
    # 构建图和加载数据集
    train_data = graph_builder.build_graph_and_load_data(
        split='train',
        data_file=train_file,
        batch_size=config['training']['batch_size'],
        shuffle=config['training']['data_loader']['shuffle'],
        num_workers=config['training']['data_loader']['num_workers']
    )
    
    val_data = graph_builder.build_graph_and_load_data(
        split='val',
        data_file=val_file,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['data_loader']['num_workers']
    )
    
    test_data = graph_builder.build_graph_and_load_data(
        split='test',
        data_file=test_file,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['data_loader']['num_workers']
    )
    
    logging.info("数据准备完成")
    return train_data, val_data, test_data, graph_builder

def apply_dataset_defaults(config: dict) -> dict:
    dataset_cfg = config.get('dataset', {}) or {}
    dataset_name = str(dataset_cfg.get('name', '')).lower()
    if dataset_name in ("shs27k", "shs148k"):
        processed_dir = f"data/{dataset_name}_llapa/processed"
        dataset_cfg.setdefault('train_file', f"{processed_dir}/random_train.tsv")
        dataset_cfg.setdefault('val_file', f"{processed_dir}/random_val.tsv")
        dataset_cfg.setdefault('test_file', f"{processed_dir}/random_test.tsv")
        dataset_cfg.setdefault('protein_info_file', 'protein_info.csv')
        config['dataset'] = dataset_cfg
    return config

def train_main(args_list):
    """
    训练脚本主函数
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="训练 PGPR 模型")
    parser.add_argument('--config', type=str, required=True, help="配置文件路径")
    parser.add_argument('--model-config', type=str, default=None, help="模型配置文件路径")
    parser.add_argument('--data-config', type=str, default=None, help="数据配置文件路径")
    parser.add_argument('--checkpoint', type=str, default=None, help="检查点路径（用于恢复训练或微调）")
    parser.add_argument('--mode', type=str, choices=['train', 'finetune', 'continue'], default='train', help="训练模式")
    parser.add_argument('--batch_size', type=int, default=None, help="批处理大小")
    parser.add_argument('--epochs', type=int, default=None, help="训练轮数")
    parser.add_argument('--lr', type=float, default=None, help="学习率")
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default=None, help="设备")
    parser.add_argument('--fine-tune', action='store_true', help="启用微调模式")
    parser.add_argument('--gpu', type=int, nargs='+', default=None, help="GPU设备ID列表")
    parser.add_argument('--verbose', action='store_true', help="显示详细日志")
    parser.add_argument('--config-override', type=str, nargs='*', default=[], help="覆盖配置文件中的参数")
    args = parser.parse_args(args_list)
    
    # 加载配置
    config_manager = ConfigManager()
    config_files = [
        os.path.join(project_root, 'configs', 'base.yaml'),
        os.path.join(project_root, 'configs', 'model.yaml'),
        os.path.join(project_root, 'configs', 'training.yaml'),
        os.path.join(project_root, 'configs', 'data.yaml'),
        args.config
    ]
    
    # 如果提供了其他配置文件，添加到列表中
    if args.model_config:
        config_files.append(args.model_config)
    if args.data_config:
        config_files.append(args.data_config)
    
    config_manager.load_multiple_configs(config_files)
    resume_checkpoint = args.checkpoint
    args.checkpoint = None
    config_manager.parse_cli_args(args)
    args.checkpoint = resume_checkpoint
    
    # 处理微调模式
    if args.fine_tune:
        config_manager.merge_config({'training': {'finetune': {'enable': True}}})
    
    # 处理GPU设置
    if args.gpu:
        config_manager.merge_config({'device': {'num_gpus': len(args.gpu), 'device_ids': args.gpu}})
    
    # 处理详细日志
    if args.verbose:
        config_manager.merge_config({'logging': {'level': logging.DEBUG}})
    
    # 处理配置覆盖
    if args.config_override:
        override_config = {}
        for override in args.config_override:
            if '=' in override:
                key, value = override.split('=', 1)
                # 支持嵌套键，如 training.epochs=100
                keys = key.split('.')
                current = override_config
                for k in keys[:-1]:
                    if k not in current:
                        current[k] = {}
                    current = current[k]
                # 尝试转换数值类型
                try:
                    if isinstance(value, str) and value.lower() in ("true", "false"):
                        value = value.lower() == "true"
                    elif isinstance(value, str) and value.lower() in ("none", "null"):
                        value = None
                    elif isinstance(value, str) and '.' in value:
                        value = float(value)
                    else:
                        value = int(value)
                except ValueError:
                    # 如果转换失败，保持字符串类型
                    pass
                current[keys[-1]] = value
        if override_config:
            config_manager.merge_config(override_config)
    
    # 验证配置完整性
    required_keys = [
        'paths', 'device', 'gnn_ppi', 'training', 'optimizer', 'lr_scheduler',
        'loss', 'dataset', 'preprocessing'
    ]
    if not config_manager.validate_config(required_keys):
        sys.exit(1)
    
    config = config_manager.config
    config = apply_dataset_defaults(config)
    
    # 设置日志
    log_dir = config.get('paths', {}).get('logs_dir', os.path.join(project_root, 'artifacts', 'logs'))
    log_name = None # 使用根记录器以捕获所有模块的日志
    log_level = utils.logger.Logger.get_log_level(config['logging']['level'])
    setup_logger(log_dir, log_name, log_level)
    logging.info("启动训练脚本")
    
    # 设置随机种子
    setup_seed(config['seed']['random_seed'])
    
    # 初始化训练器并启动
    if config.get('distributed', {}).get('use_distributed', False):
        logging.info("检测到分布式配置，启动多GPU训练...")
        from training.trainer import DistributedExploratoryPPITrainer
        DistributedExploratoryPPITrainer.launch(config)
        # 注意：分布式模式下，主进程在这里等待所有子进程结束
        # 分布式模式下的评估可能需要单独处理
        return None, {}
    else:
        logging.info("初始化单GPU训练器...")
        trainer = ExploratoryPPITrainer(config)
        
        # 准备数据 (单GPU模式)
        train_data, val_data, test_data, graph_builder = prepare_data(config)

        resume_checkpoint = args.checkpoint
        
        # 设置微调模式
        if args.mode == 'finetune':
            config['training']['finetune']['enable'] = True
            logging.info("启用微调模式")
        
        # 训练循环
        logging.info("开始训练循环...")
        start_time = time.time()
        
        try:
            # 训练
            trainer.train(train_data, val_data, resume_checkpoint=resume_checkpoint)
            
            # 获取训练好的模型 (COT Generator)
            best_model = trainer.cot_generator
            
            # 保存最终模型
            final_model_path = os.path.join(config['paths']['checkpoints_dir'], 'final_model.pth')
            os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
            torch.save({
                'model_state_dict': best_model.state_dict(),
                'llm_classifier_state_dict': trainer.llm_wrapper.relation_classifier.state_dict() if getattr(trainer, 'llm_wrapper', None) is not None else None,
                'llm_state_dict': trainer.llm_wrapper.model.state_dict()
                    if (
                        getattr(trainer, 'llm_wrapper', None) is not None
                        and getattr(trainer.llm_wrapper, 'model', None) is not None
                        and bool((config.get('llm', {}) or {}).get('train_backbone', False))
                    )
                    else None,
                'config': config_manager.config
            }, final_model_path)
            logging.info(f"最终模型已保存到: {final_model_path}")
            
            # 在测试集上评估最终模型
            test_metrics = {}
            if config.get('training', {}).get('run_test_evaluation', False):
                logging.info("在测试集上评估最终模型...")
                if hasattr(trainer, 'evaluate'):
                    test_metrics = trainer.evaluate(test_data, best_model)
                    logging.info(f"测试集评估结果: {test_metrics}")
                else:
                    logging.warning("Trainer没有evaluate方法，跳过测试集评估")

        except KeyboardInterrupt:
            logging.info("训练被用户中断")
            sys.exit(1)
        except Exception as e:
            logging.error(f"训练过程中发生错误: {str(e)}", exc_info=True)
            sys.exit(1)
        
        end_time = time.time()
        training_time = end_time - start_time
        logging.info(f"训练完成，总耗时: {training_time:.2f}秒 ({training_time/3600:.2f}小时)")
        
        return final_model_path, test_metrics

if __name__ == "__main__":
    import sys
    train_main(sys.argv[1:])
