#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SHS27k数据集验证脚本
用于检查手动下载的SHS27k数据集是否正确放置和格式是否正确
"""

import os
import sys
import logging
import pandas as pd
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """主函数"""
    logger.info("=== SHS27k数据集验证脚本 ===")
    
    # 检查Python版本
    if sys.version_info < (3, 6):
        logger.error("需要Python 3.6或更高版本")
        return False
    
    # 数据集文件路径
    project_root = Path(__file__).parent.parent
    expected_path = project_root / "data" / "raw" / "shs27k" / "SHS27k.csv"
    
    logger.info(f"检查数据集文件: {expected_path}")
    
    # 检查文件是否存在
    if not expected_path.exists():
        logger.error(f"❌ 文件不存在: {expected_path}")
        logger.error("请按照以下步骤操作:")
        logger.error("1. 手动下载SHS27k数据集")
        logger.error("2. 将SHS27k.csv文件放置到: {expected_path}")
        logger.error("3. 再次运行此验证脚本")
        return False
    
    logger.info("✅ 文件存在")
    
    # 检查文件大小
    file_size = expected_path.stat().st_size
    logger.info(f"文件大小: {file_size / (1024 * 1024):.2f} MB")
    
    if file_size < 100 * 1024:  # 100KB
        logger.warning("⚠️  文件大小可能过小，可能下载不完整")
    
    # 读取数据集并验证格式
    try:
        logger.info("正在验证文件格式...")
        df = pd.read_csv(expected_path, header=None)
        
        logger.info(f"✅ 成功读取数据集")
        logger.info(f"数据集形状: {df.shape}")
        logger.info(f"样本数量: {len(df)}")
        
        # 验证列数
        if df.shape[1] == 3:
            logger.info("✅ 列数正确 (3列)")
            logger.info("列含义: [protein1, protein2, interaction]")
        else:
            logger.error(f"❌ 列数不正确，期望3列，实际{df.shape[1]}列")
            logger.error("数据集格式可能有问题")
            return False
        
        # 显示数据前几行
        logger.info("\n数据前5行示例:")
        logger.info(df.head().to_string(index=False, header=False))
        
        # 检查交互类型分布
        if df.shape[1] >= 3:
            interaction_counts = df.iloc[:, 2].value_counts()
            logger.info("\n交互类型分布:")
            for value, count in interaction_counts.items():
                logger.info(f"  {value}: {count}  ({count/len(df)*100:.2f}%)")
        
        # 检查是否有缺失值
        missing_values = df.isnull().sum().sum()
        if missing_values == 0:
            logger.info("✅ 无缺失值")
        else:
            logger.warning(f"⚠️  发现{missing_values}个缺失值")
        
        # 检查蛋白质ID格式
        protein1 = df.iloc[:, 0].tolist()[:10]
        protein2 = df.iloc[:, 1].tolist()[:10]
        
        logger.info("\n蛋白质ID示例:")
        logger.info(f"  Protein1前10个: {protein1}")
        logger.info(f"  Protein2前10个: {protein2}")
        
        # 检查蛋白质ID是否为字符串
        if all(isinstance(p, str) for p in protein1[:5] + protein2[:5]):
            logger.info("✅ 蛋白质ID格式正确")
        else:
            logger.warning("⚠️  蛋白质ID可能不是字符串格式")
        
    except Exception as e:
        logger.error(f"❌ 读取文件时出错: {str(e)}")
        logger.error("文件格式可能有问题")
        return False
    
    logger.info("\n=== 验证完成 ===")
    logger.info("✅ SHS27k数据集验证成功！")
    logger.info("\n下一步操作建议:")
    logger.info("1. 运行数据集预处理脚本:")
    logger.info("   python scripts/preprocess_shs27k.py")
    logger.info("\n2. 或者运行完整的数据准备流程:")
    logger.info("   python scripts/setup_data.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)