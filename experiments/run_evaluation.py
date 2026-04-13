import os
import sys

# 设置 Hugging Face 镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import argparse
import logging
import torch
import numpy as np
import pandas as pd
from pathlib import Path

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from configs.config_manager import ConfigManager
from evaluation.evaluator import PPIEvaluator
from utils.logger import setup_logger
import utils

def resolve_path(project_root: str, p: str) -> str:
    if not p:
        return ""
    if os.path.isabs(p):
        return p
    cand = os.path.join(project_root, p)
    return cand

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

def find_seq_dict_path(config) -> str:
    candidates = []
    data_dir = config.get('paths', {}).get('data_dir', 'data')
    processed_dir = config.get('paths', {}).get('processed_data_dir')
    if not processed_dir:
        processed_dir = os.path.join(data_dir, 'processed')

    candidates.extend([
        os.path.join(processed_dir, 'protein_id_seq.tsv'),
        os.path.join(processed_dir, 'processed', 'protein_id_seq.tsv'),
        os.path.join(processed_dir, 'protein.SHS27k.sequences.dictionary.tsv'),
        os.path.join(processed_dir, 'protein.STRING.sequences.dictionary.tsv'),
        os.path.join(data_dir, 'processed', 'protein.SHS27k.sequences.dictionary.tsv'),
        os.path.join(data_dir, 'processed', 'protein.STRING.sequences.dictionary.tsv'),
        os.path.join(data_dir, 'raw', 'shs27k', 'extracted', 'raw_data', 'protein.SHS27k.sequences.dictionary.tsv'),
        os.path.join(data_dir, 'raw', 'shs27k', 'extracted', 'raw_data', 'protein.STRING.sequences.dictionary.tsv'),
    ])

    for p in candidates:
        if p and os.path.exists(p):
            return p
    return ""

def setup_environment(config):
    """
    设置评估环境
    """
    # 设置日志
    log_dir = config.get('paths', {}).get('logs_dir', os.path.join(project_root, 'artifacts', 'logs'))
    log_name = 'evaluation'
    log_level = utils.logger.Logger.get_log_level(config['logging']['level'])
    setup_logger(log_dir, log_name, log_level)
    
    # 设置设备
    device = config['device']['device_type']
    if device == 'cuda' and not torch.cuda.is_available():
        logging.warning("CUDA不可用，将使用CPU")
        device = 'cpu'
        config['device']['device_type'] = device
    
    return device

def evaluate_main(args_list):
    parser = argparse.ArgumentParser(description="评估 PGPR 模型")
    parser.add_argument('--config', type=str, required=True, help="配置文件路径")
    parser.add_argument('--checkpoint', type=str, required=True, help="模型检查点路径")
    parser.add_argument('--data-file', type=str, help="测试数据文件路径")
    parser.add_argument('--output-dir', type=str, default='artifacts/metrics/evaluation', help="输出目录")
    parser.add_argument('--mode', type=str, choices=['standard', 'new_protein', 'zero_shot'], default='standard', help="评估模式")
    parser.add_argument('--batch-size', type=int, help="评估批次大小")
    parser.add_argument('--metrics', type=str, help="评估指标，多个指标用逗号分隔")
    parser.add_argument('--verbose', action='store_true', help="显示详细日志")
    parser.add_argument('--config-override', type=str, nargs='*', help='覆盖配置文件中的参数')
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
    config_manager.load_multiple_configs(config_files)
    
    # 覆盖配置
    if args.batch_size:
        if 'evaluation' not in config_manager.config:
            config_manager.config['evaluation'] = {}
        config_manager.config['evaluation']['batch_size'] = args.batch_size
    
    if args.metrics:
        if 'evaluation' not in config_manager.config:
            config_manager.config['evaluation'] = {}
        config_manager.config['evaluation']['metrics'] = args.metrics.split(',')

    if args.config_override:
        config_manager.override_config(args.config_override)

    config = config_manager.config
    config = apply_dataset_defaults(config)

    # 设置环境
    device = setup_environment(config)
    
    # 初始化评估器
    logging.info("初始化评估器...")
    evaluator = PPIEvaluator(config)
    
    # 加载模型
    logging.info(f"正在从 {args.checkpoint} 加载模型...")
    evaluator.load_model(args.checkpoint)
    
    # 准备数据
    from data.dataset import PPIDataset
    if args.data_file:
        data_path = args.data_file
    else:
        test_file = (config.get('dataset', {}) or {}).get('test_file')
        data_path = ""
        if test_file:
            cand = test_file
            if not os.path.isabs(cand):
                cand = resolve_path(project_root, cand)
            if os.path.exists(cand):
                data_path = cand
            else:
                alt = os.path.join(config.get('paths', {}).get('data_dir', 'data'), test_file)
                if os.path.exists(alt):
                    data_path = alt
        if not data_path:
            data_path = os.path.join(config.get('paths', {}).get('data_dir', 'data'), 'processed', str(test_file or ""))
    
    seq_dict_path = find_seq_dict_path(config)

    logging.info(f"加载测试数据: {data_path}")
    test_dataset = PPIDataset(data_path=data_path, seq_dict_path=seq_dict_path if seq_dict_path else None)
    
    # 如果需要，构建图
    # 注意：PPIEvaluator.evaluate 现在会自动处理图构建
    
    # 运行评估
    logging.info(f"开始运行评估 (模式: {args.mode})...")
    results = evaluator.evaluate(test_dataset, evaluation_mode=args.mode)
    
    # 输出结果
    logging.info("评估完成！结果如下：")
    for metric, value in results.items():
        if isinstance(value, (int, float)):
            logging.info(f"  {metric}: {value:.4f}")
    
    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    import json
    results_path = os.path.join(args.output_dir, f'evaluation_results_{args.mode}.json')
    
    # 处理不能直接序列化的结果（如混淆矩阵）
    serializable_results = {}
    for k, v in results.items():
        if isinstance(v, (int, float, str, list, dict)):
            serializable_results[k] = v
        elif isinstance(v, torch.Tensor):
            serializable_results[k] = v.tolist()
        elif isinstance(v, np.ndarray):
            serializable_results[k] = v.tolist()
            
    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=4)
    logging.info(f"评估结果已保存到: {results_path}")

if __name__ == "__main__":
    evaluate_main(sys.argv[1:])
