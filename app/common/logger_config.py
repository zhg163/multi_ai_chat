"""
日志配置 - 统一日志格式和配置
"""

import logging
import logging.handlers
import os
from pathlib import Path

def setup_logger(name=None, log_level=logging.INFO, log_file=None):
    """
    配置并返回一个日志记录器
    
    Parameters:
        name (str): 日志记录器名称，默认为None（使用根日志记录器）
        log_level (int): 日志级别，默认为INFO
        log_file (str): 日志文件路径，默认为None（不使用文件处理器）
        
    Returns:
        logging.Logger: 配置好的日志记录器
    """
    # 获取日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # 避免重复添加处理器
    if logger.handlers:
        return logger
    
    # 创建格式化器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器（如果需要）
    if log_file:
        # 确保目录存在
        log_dir = os.path.dirname(log_file)
        if log_dir:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            
        # 创建文件处理器，每天一个文件，保留30天
        file_handler = logging.handlers.TimedRotatingFileHandler(
            log_file, 
            when='midnight',
            interval=1,
            backupCount=30,
            encoding='utf-8'
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_logger(name, log_level=logging.INFO, log_file=None):
    """
    获取已配置的日志记录器
    
    这是获取日志记录器的推荐方法
    
    Parameters:
        name (str): 日志记录器名称
        log_level (int): 日志级别，默认为INFO
        log_file (str): 日志文件路径，默认为None
        
    Returns:
        logging.Logger: 配置好的日志记录器
    """
    return setup_logger(name, log_level, log_file) 