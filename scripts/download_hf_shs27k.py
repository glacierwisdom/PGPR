#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从Hugging Face镜像下载SHS27k数据集的脚本
"""

import os
import sys
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_datasets_installed():
    """检查datasets库是否已安装"""
    try:
        import datasets
        logger.info(f"datasets库版本: {datasets.__version__}")
        return True
    except ImportError:
        logger.error("datasets库未安装，请先安装: pip install datasets")
        return False

def download_shs27k(dataset_name="Synthyra/SHS27k", save_dir=None):
    """
    从Hugging Face镜像下载SHS27k数据集
    
    Args:
        dataset_name: 数据集名称
        save_dir: 保存目录
    
    Returns:
        bool: 下载是否成功
    """
    if not check_datasets_installed():
        return False
    
    from datasets import load_dataset
    
    # 设置保存目录
    if save_dir is None:
        save_dir = os.path.join("data", "raw", "shs27k", "hf_dataset")
    
    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"数据集将保存到: {save_dir}")
    
    try:
        # 加载数据集
        logger.info(f"正在加载数据集: {dataset_name}")
        
        # 设置Hugging Face镜像
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        
        dataset = load_dataset(dataset_name)
        
        logger.info(f"成功加载数据集: {dataset_name}")
        logger.info(f"数据集结构: {dataset}")
        
        # 保存数据集到本地
        logger.info(f"正在保存数据集到本地目录: {save_dir}")
        dataset.save_to_disk(save_dir)
        
        logger.info("数据集保存成功！")
        
        # 查看数据集信息
        for split in dataset.keys():
            logger.info(f"\n{split} 集信息:")
            logger.info(f"  样本数量: {len(dataset[split])}")
            logger.info(f"  特征: {dataset[split].features}")
            
            # 显示前几个样本
            logger.info(f"  前2个样本:")
            for i in range(min(2, len(dataset[split]))):
                sample = dataset[split][i]
                logger.info(f"    样本 {i+1}:")
                for key, value in sample.items():
                    if isinstance(value, str) and len(value) > 50:
                        logger.info(f"      {key}: {value[:50]}... (长度: {len(value)})")
                    else:
                        logger.info(f"      {key}: {value}")
        
        return True
        
    except Exception as e:
        logger.error(f"下载数据集失败: {str(e)}")
        logger.exception("详细错误信息:")
        return False

def main():
    """主函数"""
    logger.info("=== 开始下载SHS27k数据集 ===")
    
    # 检查Python版本
    if sys.version_info < (3, 7):
        logger.error("需要Python 3.7或更高版本")
        return False
    
    # 下载数据集
    success = download_shs27k()
    
    if success:
        logger.info("\n=== SHS27k数据集下载完成！ ===")
        logger.info("数据集位置: data/raw/shs27k/hf_dataset")
        return True
    else:
        logger.error("\n=== SHS27k数据集下载失败！ ===")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
