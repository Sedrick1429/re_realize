import yaml
import argparse
from pathlib import Path
from typing import Dict, Any


class Config:
    """配置管理类"""

    def __init__(self, config_path: str = None):
        self.config_dict = {}
        if config_path:
            self.load_yaml(config_path)

    def load_yaml(self, config_path: str):
        """加载YAML配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config_dict = yaml.safe_load(f)

    def load_dict(self, config_dict: Dict[str, Any]):
        """加载字典配置"""
        self.config_dict = config_dict

    def get(self, key: str, default=None):
        """获取配置值"""
        keys = key.split('.')
        value = self.config_dict
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value

    def set(self, key: str, value: Any):
        """设置配置值"""
        keys = key.split('.')
        d = self.config_dict
        for k in keys[:-1]:
            if k not in d:
                d[k] = {}
            d = d[k]
        d[keys[-1]] = value

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return self.config_dict

    def __repr__(self):
        return yaml.dump(self.config_dict, default_flow_style=False)


def merge_configs(base_config: Dict, override_config: Dict) -> Dict:
    """合并两个配置字典"""
    result = base_config.copy()
    for key, value in override_config.items():
        if isinstance(value, dict) and key in result:
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result


def load_config_from_args():
    """从命令行参数加载配置"""
    parser = argparse.ArgumentParser(description='加载配置')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='配置文件路径')
    parser.add_argument('--data-dir', type=str, help='数据目录')
    parser.add_argument('--batch-size', type=int, help='批大小')
    parser.add_argument('--epochs', type=int, help='训练轮数')
    parser.add_argument('--lr', type=float, help='学习率')
    parser.add_argument('--device', type=str, help='设备')

    args = parser.parse_args()

    # 加载基础配置
    config = Config(args.config)

    # 覆盖命令行参数
    if args.data_dir:
        config.set('data.data_dir', args.data_dir)
    if args.batch_size:
        config.set('data.batch_size', args.batch_size)
    if args.epochs:
        config.set('train.epochs', args.epochs)
    if args.lr:
        config.set('train.lr', args.lr)
    if args.device:
        config.set('device.device_type', args.device)

    return config
