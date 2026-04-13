#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从Google Drive下载SHS27k数据集的脚本
"""

import os
import sys
import logging
import subprocess
import importlib.util
import time

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def install_gdown():
    """安装gdown库"""
    try:
        logger.info("正在安装gdown库...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
        logger.info("gdown库安装成功！")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"gdown库安装失败: {str(e)}")
        return False

def check_gdown_installed():
    """检查gdown库是否已安装"""
    spec = importlib.util.find_spec("gdown")
    return spec is not None

def download_from_google_drive(file_id, output_path, retry_times=3, retry_delay=5, disable_proxy=False):
    """
    从Google Drive下载文件
    
    Args:
        file_id: Google Drive文件ID
        output_path: 输出文件路径
        retry_times: 重试次数
        retry_delay: 重试间隔（秒）
        disable_proxy: 是否禁用代理
    
    Returns:
        bool: 下载是否成功
    """
    try:
        import gdown
        import requests
        
        logger.info(f"正在从Google Drive下载文件: {file_id}")
        logger.info(f"保存路径: {output_path}")
        
        # 创建输出目录
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"创建输出目录: {output_dir}")
        
        # 配置会话
        session = requests.Session()
        
        # 禁用代理（如果需要）
        if disable_proxy:
            session.trust_env = False
            logger.info("已禁用系统代理设置")
        
        # 使用gdown下载文件，带重试机制
        for i in range(retry_times):
            try:
                logger.info(f"下载尝试 {i+1}/{retry_times}")
                gdown.download(id=file_id, output=output_path, quiet=False, fuzzy=False, resume=False, proxy=None if disable_proxy else None)
                
                # 验证文件是否下载成功
                if os.path.exists(output_path):
                    file_size = os.path.getsize(output_path)
                    logger.info(f"文件下载成功！")
                    logger.info(f"文件大小: {file_size / (1024 * 1024):.2f} MB")
                    return True
                else:
                    logger.error(f"下载尝试 {i+1} 失败: 未找到下载的文件")
                    
            except Exception as e:
                logger.error(f"下载尝试 {i+1} 失败: {str(e)}")
                if i < retry_times - 1:
                    logger.info(f"{retry_delay}秒后重试...")
                    time.sleep(retry_delay)
                else:
                    logger.error("已达到最大重试次数")
        
        return False
            
    except ImportError:
        logger.error("gdown库未安装")
        return False
    except Exception as e:
        logger.error(f"下载失败: {str(e)}")
        logger.exception("详细错误信息:")
        return False

def extract_google_drive_id(url):
    """
    从Google Drive共享链接中提取文件ID
    
    Args:
        url: Google Drive共享链接
    
    Returns:
        str: 文件ID
    """
    # 处理不同格式的Google Drive链接
    if "id=" in url:
        # 格式: https://drive.google.com/file/d/1hJVrQXddB9JK68z7jlIcLfd9AmTWwgJr/view?usp=sharing
        return url.split("id=")[1].split("&")[0]
    elif "/d/" in url:
        # 格式: https://drive.google.com/file/d/1hJVrQXddB9JK68z7jlIcLfd9AmTWwgJr/view
        return url.split("/d/")[1].split("/")[0]
    else:
        logger.error(f"无法从链接中提取文件ID: {url}")
        return None

def main():
    """主函数"""
    logger.info("=== 开始下载Google Drive上的SHS27k数据集 ===")
    
    # 检查Python版本
    if sys.version_info < (3, 6):
        logger.error("需要Python 3.6或更高版本")
        return False
    
    # Google Drive链接
    google_drive_url = "https://drive.google.com/file/d/1hJVrQXddB9JK68z7jlIcLfd9AmTWwgJr/view?usp=sharing"
    
    # 提取文件ID
    file_id = extract_google_drive_id(google_drive_url)
    if not file_id:
        logger.error("无法提取Google Drive文件ID")
        return False
    
    logger.info(f"提取到的文件ID: {file_id}")
    
    # 检查并安装gdown库
    if not check_gdown_installed():
        logger.info("gdown库未安装，正在安装...")
        if not install_gdown():
            logger.error("无法安装gdown库，无法继续下载")
            return False
    
    # 下载文件（禁用代理以解决连接问题）
    output_path = os.path.join("data", "raw", "shs27k", "SHS27k.zip")
    success = download_from_google_drive(
        file_id=file_id,
        output_path=output_path,
        retry_times=5,
        retry_delay=10,
        disable_proxy=True  # 禁用代理以解决连接问题
    )
    
    if success:
        logger.info("\n=== SHS27k数据集下载完成！ ===")
        logger.info(f"数据集位置: {output_path}")
        return True
    else:
        logger.error("\n=== SHS27k数据集下载失败！ ===")
        logger.error("建议尝试手动下载:")
        logger.error(f"1. 访问链接: {google_drive_url}")
        logger.error(f"2. 下载文件到: {output_path}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)