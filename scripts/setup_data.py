#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据准备脚本 - 使用Python实现，避免PowerShell编码问题
支持SHS27k/SHS148k数据集的预处理与划分（random/bfs/dfs）
"""

import os
import sys
import urllib.request
import logging
from pathlib import Path
import argparse

# 先创建logs目录
def create_logs_dir():
    """创建logs目录"""
    logs_dir = Path("artifacts") / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir

# 创建logs目录
create_logs_dir()

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('artifacts/logs/setup_data.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

def create_directories(dataset: str):
    """创建必要的目录结构"""
    # 获取项目根目录
    project_root = Path(__file__).parent.parent
    logger.info(f"项目根目录: {project_root}")
    
    # 定义目录路径
    data_dir = project_root / "data"
    raw_dir = data_dir / f"{dataset}_llapa" / "raw"
    processed_dir = data_dir / f"{dataset}_llapa" / "processed"
    
    # 创建目录
    dirs_to_create = [data_dir, raw_dir, processed_dir]
    for dir_path in dirs_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"创建目录: {dir_path}")
    
    return {
        "project_root": project_root,
        "raw_dir": raw_dir,
        "processed_dir": processed_dir,
    }

def download_shs27k(shs27k_raw_dir):
    """下载SHS27k数据集"""
    url = "https://github.com/luoyunan/DNN-PPI/raw/master/dataset/yeast/SHS27k/SHS27k.csv"
    save_path = shs27k_raw_dir / "SHS27k.csv"
    
    if save_path.exists():
        logger.info(f"SHS27k数据集已存在，跳过下载: {save_path}")
        return save_path
    
    logger.info(f"下载SHS27k数据集: {url}")
    
    try:
        # 使用urllib下载文件
        with urllib.request.urlopen(url) as response, open(save_path, 'wb') as out_file:
            # 获取文件大小
            file_size = int(response.getheader('Content-Length', 0))
            downloaded = 0
            block_size = 8192  # 8KB
            
            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                downloaded += len(buffer)
                out_file.write(buffer)
                
                # 显示下载进度
                if file_size > 0:
                    progress = downloaded / file_size * 100
                    sys.stdout.write(f"\r下载进度: {progress:.1f}% ({downloaded:,}/{file_size:,} bytes)")
                    sys.stdout.flush()
        
        sys.stdout.write('\n')  # 换行
        logger.info(f"成功下载SHS27k数据集到: {save_path}")
        return save_path
        
    except Exception as e:
        logger.error(f"下载SHS27k数据集失败: {e}")
        raise

def run_preprocess(project_root: Path, dataset: str, raw_file: Path):
    logger.info("运行预处理脚本...")
    
    try:
        import subprocess
        subprocess.run(
            [
                sys.executable,
                str(project_root / "scripts" / "preprocess_shs27k.py"),
                "--dataset",
                dataset,
                "--raw-file",
                str(raw_file),
            ],
            cwd=project_root,
            check=True,
            capture_output=False
        )
        logger.info(f"{dataset} 数据集预处理完成")
    except subprocess.CalledProcessError as e:
        logger.error(f"运行预处理脚本失败，返回码: {e.returncode}")
        logger.error(f"错误输出: {e.stderr.decode() if e.stderr else '无'}")
        raise
    except Exception as e:
        logger.error(f"运行预处理脚本失败: {e}")
        raise

def main():
    """主函数"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="shs27k", choices=["shs27k", "shs148k"])
    parser.add_argument("--raw-file", type=str, default=None)
    args = parser.parse_args()

    dataset = args.dataset.lower()
    logger.info(f"=== 开始 {dataset} 数据集准备流程 ===")
    
    try:
        # 创建目录结构
        dirs = create_directories(dataset)
        
        raw_file = Path(args.raw_file) if args.raw_file else None
        if dataset == "shs27k":
            if raw_file is None:
                raw_file = download_shs27k(dirs["raw_dir"])
            else:
                raw_file.parent.mkdir(parents=True, exist_ok=True)
                if not raw_file.exists():
                    download_shs27k(raw_file.parent)
        else:
            if raw_file is None:
                raw_file = dirs["raw_dir"] / "SHS148k.csv"
            if not raw_file.exists():
                raise FileNotFoundError(f"找不到原始数据文件: {raw_file}")
        
        run_preprocess(dirs["project_root"], dataset=dataset, raw_file=raw_file)
        
        logger.info(f"=== {dataset} 数据集准备流程完成！ ===")
        logger.info(f"原始数据目录: {dirs['raw_dir']}")
        logger.info(f"处理后数据目录: {dirs['processed_dir']}")
        
        print("\n" + "="*50)
        print("SHS27k数据集准备完成！")
        print(f"原始数据目录: {dirs['raw_dir']}")
        print(f"处理后数据目录: {dirs['processed_dir']}")
        print("详细日志请查看: artifacts/logs/setup_data.log")
        print("="*50)
        
    except Exception as e:
        logger.error(f"数据集准备流程失败: {e}")
        print(f"\n错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
