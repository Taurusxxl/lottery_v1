# 系统配置管理\config_manager.py你好1110000
import os
import json
import logging
from typing import Dict, Any, Optional

class ConfigManager:
    """配置管理器"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            # 基础路径配置
            self.BASE_DIR = 'D:\\JupyterWork'
            self.LOG_DIR = os.path.join(self.BASE_DIR, 'logs')
            self.MODEL_DIR = os.path.join(self.BASE_DIR, 'models')
            self.DATA_DIR = os.path.join(self.BASE_DIR, 'data')
            self.CHECKPOINT_DIR = os.path.join(self.BASE_DIR, 'checkpoints')
            
            # 创建必要的目录
            for dir_path in [self.LOG_DIR, self.MODEL_DIR, self.DATA_DIR, self.CHECKPOINT_DIR]:
                os.makedirs(dir_path, exist_ok=True)
            
            # 数据库配置
            self.DB_CONFIG: dict = {
                'host': 'localhost',
                'port': 5432,
                'user': 'ace_user'
            }
            
            # 训练配置
            self.TRAINING_CONFIG: dict = {
                'max_epochs': 50,
                'batch_size': 32
            }
            
            # 系统配置
            self.SYSTEM_CONFIG = {
                'memory_limit': 8000,  # MB
                'gpu_memory_limit': 4000,  # MB
                'cleanup_interval': 300,  # seconds
                'log_retention_days': 7,
                'check_interval': 60,  # 检查新期号的间隔
                'AUTO_TUNING': {
                    'enable_per_sample': True,  # 启用逐样本调整
                    'adjustment_steps': 5,      # 每个样本最大调整次数
                    'learning_rate_range': (1e-5, 1e-2)  # 学习率调整范围
                },
                'DATA_CONFIG': {
                    'cache_size': 10000,
                    'normalize_range': (-1, 1)
                },
                'max_sequence_gap': 1,          # 允许的最大期号间隔
                'max_threads': 8,  # 留出4线程给系统
                'base_batch_size': 16,  # 初始批次
                'gpu_mem_limit': 1536,  # MB (保留500MB给系统)
                'cpu_util_threshold': 70,  # CPU使用率阈值
                'SAMPLE_CONFIG': {
                    'input_length': 144000,  # 修改这里
                    'target_length': 2880,   # 修改这里
                    'total_fetch': lambda: (  # 自动计算总获取量
                        self.SYSTEM_CONFIG['SAMPLE_CONFIG']['input_length'] 
                        + self.SYSTEM_CONFIG['SAMPLE_CONFIG']['target_length']
                    )
                }
            }
            
            self.initialized = True
    
    def get_db_config(self) -> Dict[str, str]:
        """获取数据库配置"""
        return self.DB_CONFIG.copy()
    
    def get_training_config(self) -> Dict[str, Any]:
        """获取训练配置"""
        return self.TRAINING_CONFIG.copy()
    
    def get_system_config(self) -> Dict[str, Any]:
        """获取系统配置"""
        return self.SYSTEM_CONFIG.copy()
    
    def update_config(self, config_name: str, updates: Dict[str, Any]) -> bool:
        """更新指定配置"""
        try:
            config = getattr(self, f'{config_name}_CONFIG')
            config.update(updates)
            return True
        except AttributeError:
            logging.error(f"配置 {config_name} 不存在")
            return False
    
    def save_config(self, config_name: str) -> bool:
        """保存配置到文件"""
        try:
            config = getattr(self, f'{config_name}_CONFIG')
            save_path = os.path.join(self.BASE_DIR, 'configs', f'{config_name.lower()}_config.json')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
            return True
        except Exception as e:
            logging.error(f"保存配置失败: {str(e)}")
            return False
    
    def load_config(self, config_name: str) -> bool:
        """从文件加载配置"""
        try:
            load_path = os.path.join(self.BASE_DIR, 'configs', f'{config_name.lower()}_config.json')
            if not os.path.exists(load_path):
                return False
            
            with open(load_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            setattr(self, f'{config_name}_CONFIG', config)
            return True
        except Exception as e:
            logging.error(f"加载配置失败: {str(e)}")
            return False

class ConfigValidator:
    """配置验证器"""
    @staticmethod
    def validate_db_config(config: Dict[str, str]) -> bool:
        """验证数据库配置"""
        required_fields = ['host', 'user', 'password', 'database', 'charset']
        return all(field in config for field in required_fields)
    
    @staticmethod
    def validate_training_config(config: Dict[str, Any]) -> bool:
        """验证训练配置"""
        try:
            assert config['batch_size'] > 0
            assert 0 < config['learning_rate'] < 1
            assert config['epochs'] > 0
            return True
        except (AssertionError, KeyError):
            return False
    
    @staticmethod
    def validate_system_config(config: Dict[str, Any]) -> bool:
        """验证系统配置"""
        try:
            assert config['memory_limit'] > 0
            assert config['gpu_memory_limit'] > 0
            assert config['cleanup_interval'] > 0
            assert config['log_retention_days'] > 0
            return True
        except (AssertionError, KeyError):
            return False

# 创建全局实例
config_instance = ConfigManager()

# 导出常用配置变量
BASE_DIR = config_instance.BASE_DIR
LOG_DIR = config_instance.LOG_DIR
MODEL_DIR = config_instance.MODEL_DIR
DATA_DIR = config_instance.DATA_DIR
CHECKPOINT_DIR = config_instance.CHECKPOINT_DIR

if __name__ == "__main__":
    print("配置验证：")
    print(f"数据库配置: {config_instance.DB_CONFIG}")
    print(f"训练配置: {config_instance.TRAINING_CONFIG}")