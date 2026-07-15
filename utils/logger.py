"""
日志工具
"""
import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(name: str = "scheduler", log_dir: str = "logs", level: int = logging.INFO) -> logging.Logger:
    """
    设置日志记录器

    Args:
        name: 日志记录器名称
        log_dir: 日志目录
        level: 日志级别

    Returns:
        配置好的Logger
    """
    # 创建日志目录
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # 创建logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 避免重复添加handler
    if logger.handlers:
        return logger

    # 日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 文件handler
    timestamp = datetime.now().strftime("%Y%m%d")
    log_file = Path(log_dir) / f"{name}_{timestamp}.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 控制台handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def get_logger(name: str = "scheduler") -> logging.Logger:
    """获取已配置的logger"""
    return logging.getLogger(name)
