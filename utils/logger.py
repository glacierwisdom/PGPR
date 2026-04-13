import logging
import os
import sys
from pathlib import Path
import datetime
from logging.handlers import RotatingFileHandler


def setup_logger(log_dir, log_name=None, level=logging.INFO,
                 max_bytes=10*1024*1024, backup_count=5,
                 format_str=None):
    """
    设置日志记录器
    
    Args:
        log_dir (str or Path): 日志目录路径
        log_name (str): 日志名称，如果为None则设置根记录器
        level (int): 日志级别
        max_bytes (int): 日志文件最大大小（字节），默认10MB
        backup_count (int): 保留的旧日志文件数量，默认5个
        format_str (str): 自定义日志格式，None表示使用默认格式
        
    Returns:
        logging.Logger: 配置好的日志记录器
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建日志文件名称（包含时间戳）
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_log_name = log_name if log_name else "root"
    log_file = log_dir / f"{file_log_name}_{timestamp}.log"
    
    # 创建日志记录器
    if log_name:
        logger = logging.getLogger(log_name)
        logger.propagate = False  # 避免重复记录到根记录器
    else:
        logger = logging.getLogger()
    
    logger.setLevel(level)
    
    # 清除现有的处理器
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 设置默认日志格式
    if format_str is None:
        format_str = "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    
    formatter = logging.Formatter(format_str)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 创建文件处理器（带轮转功能）
    file_handler = RotatingFileHandler(
        log_file, 
        maxBytes=max_bytes, 
        backupCount=backup_count, 
        encoding='utf-8'
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


class Logger:
    """日志记录器类"""
    
    def __init__(self, log_dir, log_name="ppi_prediction", level=logging.INFO,
                 max_bytes=10*1024*1024, backup_count=5,
                 format_str=None):
        """
        初始化日志记录器
        
        Args:
            log_dir (str or Path): 日志目录路径
            log_name (str): 日志名称
            level (int): 日志级别
            max_bytes (int): 日志文件最大大小（字节），默认10MB
            backup_count (int): 保留的旧日志文件数量，默认5个
            format_str (str): 自定义日志格式，None表示使用默认格式
        """
        self.logger = setup_logger(
            log_dir, log_name, level, 
            max_bytes=max_bytes, backup_count=backup_count,
            format_str=format_str
        )
        self.log_name = log_name
        
    def info(self, message):
        """记录信息级别日志"""
        self.logger.info(message)
    
    def debug(self, message):
        """记录调试级别日志"""
        self.logger.debug(message)
    
    def warning(self, message):
        """记录警告级别日志"""
        self.logger.warning(message)
    
    def error(self, message):
        """记录错误级别日志"""
        self.logger.error(message)
    
    def critical(self, message):
        """记录严重错误级别日志"""
        self.logger.critical(message)
    
    def exception(self, message):
        """记录异常信息"""
        self.logger.exception(message)
    
    def set_level(self, level):
        """设置日志级别"""
        self.logger.setLevel(level)
        for handler in self.logger.handlers:
            handler.setLevel(level)
    
    @staticmethod
    def get_logger(name):
        """获取已存在的日志记录器"""
        return logging.getLogger(name)
    
    @staticmethod
    def get_log_level(level_str):
        """将日志级别字符串转换为logging模块的日志级别
        
        Args:
            level_str (str or int): 日志级别字符串 (debug, info, warning, error, critical) 或 整数级别
            
        Returns:
            int: logging模块的日志级别
        """
        if isinstance(level_str, int):
            return level_str
            
        level_map = {
            'debug': logging.DEBUG,
            'info': logging.INFO,
            'warning': logging.WARNING,
            'error': logging.ERROR,
            'critical': logging.CRITICAL
        }
        if not isinstance(level_str, str):
            return logging.INFO
            
        return level_map.get(level_str.lower(), logging.INFO)