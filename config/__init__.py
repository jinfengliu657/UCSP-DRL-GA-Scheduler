"""配置模块初始化"""
import yaml
from pathlib import Path
from typing import Dict


def load_config(config_path: str = 'config/config.yaml') -> Dict:
    """
    加载配置文件

    Args:
        config_path: 配置文件路径

    Returns:
        配置字典
    """
    path = Path(config_path)
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return get_default_config()


def get_default_config() -> Dict:
    """获取默认配置"""
    return {
        'database': {
            'host': 'localhost',
            'port': 3306,
            'user': 'root',
            'password': '',
            'database': 'test3'
        },
        'ga': {
            'max_generations': 500,
            'tabu_search_ratio': 0.2,
            'use_tabu_search': True
        },
        'dl_model_path': 'models/parameter_tuner.pth',
        'training_data': 'data/experiment_data.json',
        'dl_epochs': 100,
        'dl_batch_size': 32,
        'fitness_weights': {
            'w1': 0.4,
            'w2': 0.3,
            'w3': 0.3
        }
    }
