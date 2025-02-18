#1 Core Configuration Manager / 核心配置管理器
import os
import json
import logging
import logging.handlers
from datetime import datetime
from collections import deque
from IPython.display import clear_output
from pathlib import Path
from typing import Dict, Any, Optional
from cell3_monitor import (
    DailyRotatingFileHandler, 
    CustomFormatter,
    ProgressHandler, 
    LogDisplayManager
)
import numpy as np  # 确保所有使用np的地方都有导入

class CoreManager:
    """核心管理器 - 整合配置和日志管理"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            # 初始化基础目录结构
            self._init_directories()
            # 初始化日志系统
            self._init_logging_system()
            # 初始化配置系统（需要最先完成）
            self._init_config_system()
            
            # 添加配置验证规则
            self.validation_rules = {
                'learning_rate': lambda x: 0 < x < 1,
                'batch_size': lambda x: x > 0 and x & (x-1) == 0
            }
            
            # 初始化时执行配置测试
            self._test_config_system()
            
            # 最后设置初始化完成标记
            self.initialized = True
            self.logger.info("核心管理器初始化完成")

    def _init_directories(self):
        """初始化目录结构"""
        # 基础路径配置
        self.BASE_DIR = 'D:\\JupyterWork'
        self.LOG_DIR = os.path.join(self.BASE_DIR, 'logs')
        self.MODEL_DIR = os.path.join(self.BASE_DIR, 'models')
        self.DATA_DIR = os.path.join(self.BASE_DIR, 'data')
        self.CHECKPOINT_DIR = os.path.join(self.BASE_DIR, 'checkpoints')
        
        # 创建必要的目录
        for dir_path in [self.LOG_DIR, self.MODEL_DIR, self.DATA_DIR, self.CHECKPOINT_DIR]:
            os.makedirs(dir_path, exist_ok=True)

    def _init_logging_system(self):
        """初始化日志系统"""
        # 创建主日志处理器
        self.logger = logging.getLogger('ACE_System')
        self.logger.setLevel(logging.INFO)
        
        # 日常日志处理器
        daily_handler = self._create_daily_rotating_handler()
        self.logger.addHandler(daily_handler)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(self._create_custom_formatter())
        self.logger.addHandler(console_handler)
        
        # 进度条处理器
        progress_handler = self._create_progress_handler()
        self.logger.addHandler(progress_handler)

    def _init_config_system(self):
        """初始化配置系统"""
        # 系统配置
        self.SYSTEM_CONFIG = {
            'TRAINING_CONFIG': {  # 添加训练配置
                'batch_size': 64,
                'max_epochs': 100,
                'early_stopping_patience': 10,
                'learning_rate': 0.001,
                'model_checkpoint_interval': 5
            },
            'DATA_CONFIG': {
                'cache_size': 10000,
                'min_sequence_length': 14400,
                'normalize_range': (-1, 1)
            },
            'SAMPLE_CONFIG': {
                'input_length': 14400,
                'target_length': 2880
            }
        }
        
        # 单独的训练配置
        self.TRAINING_CONFIG = self.SYSTEM_CONFIG['TRAINING_CONFIG']
        
        # 数据库配置
        self.DB_CONFIG = {
            'host': 'localhost',
            'port': 3306,
            'user': 'root',
            'password': 'tt198803',
            'database': 'admin_data',
            'charset': 'utf8mb4'
        }
        
        # 优化器配置
        self.OPTIMIZER_CONFIG = {
            'learning_rate': 0.001,
            'beta_1': 0.9,
            'beta_2': 0.999,
            'epsilon': 1e-07
        }
        
        # 日志配置
        self.LOG_CONFIG = {
            'log_level': 'INFO',
            'log_format': '%(asctime)s [%(levelname)s] %(name)s - %(message)s',
            'log_file': 'system.log',
            'log_retention_days': 7
        }
        
        # 模型配置
        self.MODEL_CONFIG = {
            'model_type': 'transformer',
            'num_layers': 6,
            'num_heads': 8,
            'd_model': 512,
            'dff': 2048,
            'dropout_rate': 0.1
        }
        
        self.logger.info("配置系统初始化完成")

    def update_config(self, config_name: str, updates: Dict[str, Any]) -> bool:
        """更新指定配置"""
        try:
            config = getattr(self, f'{config_name}_CONFIG')
            config.update(updates)
            return True
        except AttributeError:
            self.logger.error(f"配置 {config_name} 不存在")
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
            self.logger.error(f"保存配置失败: {str(e)}")
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
            self.logger.error(f"加载配置失败: {str(e)}")
            return False

    def validate_config(self, config_name: str) -> bool:
        """验证配置有效性"""
        try:
            if config_name == 'DB':
                return self._validate_db_config(self.DB_CONFIG)
            elif config_name == 'SYSTEM':
                return self._validate_system_config(self.SYSTEM_CONFIG)
            elif config_name == 'TRAINING':
                return self._validate_training_config(self.TRAINING_CONFIG)
            else:
                self.logger.error(f"未知的配置类型: {config_name}")
                return False
        except Exception as e:
            self.logger.error(f"验证配置失败: {str(e)}")
            return False

    def _validate_db_config(self, config: Dict[str, str]) -> bool:
        """验证数据库配置"""
        required_fields = ['host', 'user', 'password', 'database', 'charset']
        return all(field in config for field in required_fields)

    def _validate_training_config(self, config: Dict[str, Any]) -> bool:
        """验证训练配置"""
        try:
            assert config['batch_size'] > 0
            assert config['max_epochs'] > 0
            return True
        except (AssertionError, KeyError):
            return False

    def _validate_system_config(self, config: Dict[str, Any]) -> bool:
        """验证系统配置"""
        try:
            assert config['memory_limit'] > 0
            assert config['gpu_memory_limit'] > 0
            assert config['cleanup_interval'] > 0
            assert config['log_retention_days'] > 0
            return True
        except (AssertionError, KeyError):
            return False

    def _create_daily_rotating_handler(self):
        """创建每日轮转的日志处理器"""
        handler = DailyRotatingFileHandler(
            base_dir=self.LOG_DIR,
            prefix='system'
        )
        handler.setFormatter(self._create_custom_formatter())
        return handler
        
    # 添加从cell4中的LogManager类的功能
    def setup_continuous_logging(self):
        """设置持续训练的日志系统"""
        self.continuous_logger = logging.getLogger('ContinuousTraining')
        self.continuous_logger.setLevel(logging.INFO)
        self.continuous_log_buffer = deque(maxlen=100)
        
        # 添加持续训练的文件处理器
        continuous_handler = DailyRotatingFileHandler(
            base_dir=self.LOG_DIR,
            prefix='continuous_training'
        )
        formatter = logging.Formatter(
            '[%(asctime)s] %(levellevel)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        continuous_handler.setFormatter(formatter)
        self.continuous_logger.addHandler(continuous_handler)
        
    def update_training_progress(self, model_idx: int, progress: float):
        """更新训练进度"""
        if hasattr(self, 'display_manager'):
            self.display_manager.progress_bars[model_idx] = progress
            self.display_manager._display_logs()
            
    def add_training_log(self, message: str):
        """添加训练日志"""
        if hasattr(self, 'continuous_log_buffer'):
            self.continuous_log_buffer.append(message)
            if hasattr(self, 'display_manager'):
                self.display_manager.log_buffer.append(message)
                self.display_manager._display_logs()

    # 添加从cell1的ConfigValidator的功能
    def validate_all_configs(self) -> Dict[str, bool]:
        """验证所有配置的有效性"""
        results = {}
        
        # 数据库配置验证
        results['db_config'] = self._extended_validate_db_config()
        
        # 训练配置验证
        results['training_config'] = self._extended_validate_training_config()
        
        # 系统配置验证
        results['system_config'] = self._extended_validate_system_config()
        
        return results
        
    def _extended_validate_db_config(self) -> bool:
        """扩展的数据库配置验证"""
        try:
            config = self.DB_CONFIG
            
            # 基本字段验证
            required_fields = ['host', 'user', 'password', 'database', 'charset']
            if not all(field in config for field in required_fields):
                self.logger.error("数据库配置缺少必要字段")
                return False
                
            # 端口验证
            if not isinstance(config.get('port', 3306), int):
                self.logger.error("数据库端口必须是整数")
                return False
                
            # 字符集验证
            if config.get('charset') not in ['utf8', 'utf8mb4']:
                self.logger.error("不支持的字符集")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"验证数据库配置时出错: {str(e)}")
            return False
            
    def _extended_validate_training_config(self) -> bool:
        """扩展的训练配置验证"""
        try:
            config = self.TRAINING_CONFIG
            
            # 参数范围验证
            validations = [
                config.get('batch_size', 0) > 0,
                config.get('max_epochs', 0) > 0,
                0 <= config.get('save_frequency', 100) <= 1000,
                0 <= config.get('eval_frequency', 50) <= 500,
                0 < config.get('min_improvement', 0.001) < 1
            ]
            
            if not all(validations):
                self.logger.error("训练配置参数范围无效")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"验证训练配置时出错: {str(e)}")
            return False
            
    def _extended_validate_system_config(self) -> bool:
        """扩展的系统配置验证"""
        try:
            config = self.SYSTEM_CONFIG
            
            # 资源限制验证
            if not all([
                config.get('memory_limit', 0) >= 1000,
                config.get('gpu_memory_limit', 0) >= 1000,
                config.get('max_threads', 0) > 0
            ]):
                self.logger.error("系统资源限制配置无效")
                return False
            
            # 采样配置验证
            sample_config = config.get('SAMPLE_CONFIG', {})
            if not all([
                sample_config.get('input_length', 0) > 0,
                sample_config.get('target_length', 0) > 0
            ]):
                self.logger.error("采样配置无效")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"验证系统配置时出错: {str(e)}")
            return False

    def export_configs(self, export_path: str = None) -> bool:
        """导出所有配置"""
        if export_path is None:
            export_path = os.path.join(self.BASE_DIR, 'configs', 'all_configs.json')
            
        try:
            configs = {
                'DB_CONFIG': self.DB_CONFIG,
                'TRAINING_CONFIG': self.TRAINING_CONFIG,
                'SYSTEM_CONFIG': self.SYSTEM_CONFIG,
                'OPTUNA_CONFIG': self.OPTUNA_CONFIG
            }
            
            os.makedirs(os.path.dirname(export_path), exist_ok=True)
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(configs, f, indent=4, ensure_ascii=False)
                
            self.logger.info(f"配置已导出到: {export_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"导出配置失败: {str(e)}")
            return False

    def _create_custom_formatter(self):
        """创建自定义格式化器"""
        return CustomFormatter()

    def _create_progress_handler(self):
        """创建进度处理器"""
        return ProgressHandler()

    # 从cell4的LoggingManager类补充
    def _configure_logging(self):
        """配置日志系统"""
        # 配置根日志记录器
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.LOG_DIR, "system.log")),
                logging.StreamHandler()
            ]
        )
        # 捕获警告信息
        logging.captureWarnings(True)

    # 从cell4的LogManager类补充
    def get_logger(self):
        """获取logger实例"""
        return self.logger

    def get_continuous_logger(self):
        """获取持续训练的logger实例"""
        if not hasattr(self, 'continuous_logger'):
            self.setup_continuous_logging()
        return self.continuous_logger

    # 从cell1的ConfigManager补充
    def get_config(self, config_name: str) -> Dict[str, Any]:
        """获取指定配置
        Args:
            config_name: 配置名称 ('DB', 'TRAINING', 'SYSTEM', 'OPTUNA')
        """
        try:
            return getattr(self, f'{config_name}_CONFIG').copy()
        except AttributeError:
            self.logger.error(f"配置 {config_name} 不存在")
            return {}

    def clear_log_buffers(self):
        """清理日志缓冲区"""
        if hasattr(self, 'continuous_log_buffer'):
            self.continuous_log_buffer.clear()
        if hasattr(self, 'display_manager'):
            self.display_manager.log_buffer.clear()
            self.display_manager._display_logs()

    def cleanup_logs(self, days: int = None):
        """清理旧日志文件
        Args:
            days: 保留天数，默认使用配置中的值
        """
        if days is None:
            days = self.SYSTEM_CONFIG.get('log_retention_days', 7)
            
        try:
            current_time = datetime.now()
            for file in os.listdir(self.LOG_DIR):
                if file.endswith('.log'):
                    file_path = os.path.join(self.LOG_DIR, file)
                    file_time = datetime.fromtimestamp(os.path.getctime(file_path))
                    if (current_time - file_time).days > days:
                        os.remove(file_path)
                        self.logger.info(f"已删除旧日志文件: {file}")
                        
        except Exception as e:
            self.logger.error(f"清理日志文件时出错: {str(e)}")

    def _configure_metrics(self):
        """配置性能指标记录"""
        self.metrics = {
            'training_loss': deque(maxlen=1000),
            'validation_loss': deque(maxlen=1000),
            'learning_rates': deque(maxlen=1000),
            'batch_times': deque(maxlen=100)
        }

    def log_metric(self, metric_name: str, value: float):
        """记录性能指标"""
        if hasattr(self, 'metrics') and metric_name in self.metrics:
            self.metrics[metric_name].append(value)

    def get_metrics_summary(self):
        """获取性能指标摘要"""
        if not hasattr(self, 'metrics'):
            return {}
            
        return {
            name: {
                'mean': np.mean(values),
                'min': np.min(values),
                'max': np.max(values)
            }
            for name, values in self.metrics.items()
            if values
        }

    def _test_config_system(self):
        """测试配置系统完整性"""
        try:
            required_keys = ['host', 'port', 'user', 'password'] 
            assert all(key in self.DB_CONFIG for key in required_keys), "数据库配置缺失必要参数"
            self.logger.info("配置系统测试通过")
        except AssertionError as e:
            self.logger.error(f"配置系统测试失败: {str(e)}")
            raise

    def validate_config_values(self, config: dict) -> bool:
        """验证配置参数值 (from ConfigValidator)"""
        valid = True
        for key, rule in self.validation_rules.items():
            if key in config:
                if not rule(config[key]):
                    self.logger.warning(f"参数 {key} 的值 {config[key]} 无效")
                    valid = False
        return valid

class DailyRotatingFileHandler(logging.FileHandler):
    """每日自动分文件的日志处理器"""
    def __init__(self, base_dir, prefix='log', max_bytes=50*1024*1024):
        self.base_dir = base_dir
        self.prefix = prefix
        self.max_bytes = max_bytes
        self.current_date = None
        self.current_file = None
        self.current_size = 0
        self.file_count = 1
        
        os.makedirs(base_dir, exist_ok=True)
        self._init_file()
        super().__init__(self.current_file, mode='a', encoding='utf-8')
    
    def _init_file(self):
        """初始化日志文件"""
        self.current_file = self._get_file_path()
        self.current_date = datetime.now().strftime('%Y%m%d')
        if os.path.exists(self.current_file):
            self.current_size = os.path.getsize(self.current_file)
        else:
            self.current_size = 0
    
    def _get_file_path(self):
        """获取当前日志文件路径"""
        today = datetime.now().strftime('%Y%m%d')
        if self.current_size >= self.max_bytes:
            self.file_count += 1
            return os.path.join(self.base_dir, f'{self.prefix}_{today}_{self.file_count}.log')
        elif today != self.current_date:
            self.file_count = 1
            self.current_date = today
            return os.path.join(self.base_dir, f'{self.prefix}_{today}.log')
        return self.current_file
    
    def emit(self, record):
        """重写emit方法，在写入日志前检查文件状态"""
        try:
            new_file = self._get_file_path()
            if new_file != self.current_file:
                if self.stream:
                    self.stream.close()
                self.current_file = new_file
                self.baseFilename = new_file
                self.current_size = 0
                self.stream = self._open()
            
            msg = self.format(record) + '\n'
            self.stream.write(msg)
            self.stream.flush()
            self.current_size += len(msg.encode('utf-8'))
            
        except Exception as e:
            self.handleError(record)

class ProgressHandler(logging.Handler):
    """进度条处理器"""
    def __init__(self):
        super().__init__()
        self.progress = 0
        
    def emit(self, record):
        if hasattr(record, 'progress'):
            print('\r' + ' ' * 80, end='\r')  # 清除当前行
            progress = int(record.progress * 50)
            print(f'\rTraining Progress: [{"="*progress}{" "*(50-progress)}] {record.progress*100:.1f}%', end='')

class CustomFormatter(logging.Formatter):
    """自定义日志格式化器"""
    def __init__(self):
        super().__init__()
        self.formatters = {
            logging.DEBUG: logging.Formatter(
                '[%(asctime)s] %(name)s - %(levelname)s - %(message)s'
            ),
            logging.INFO: logging.Formatter(
                '[%(asctime)s] %(levelname)s - %(message)s'
            ),
            logging.WARNING: logging.Formatter(
                '[%(asctime)s] WARNING - %(message)s'
            ),
            logging.ERROR: logging.Formatter(
                '[%(asctime)s] ERROR - %(message)s\n\tat %(pathname)s:%(lineno)d'
            ),
            logging.CRITICAL: logging.Formatter(
                '[%(asctime)s] CRITICAL - %(message)s\n\tat %(pathname)s:%(lineno)d\n%(exc_info)s'
            )
        }
    
    def format(self, record):
        formatter = self.formatters.get(record.levelno)
        return formatter.format(record)

class LogDisplayManager:
    """日志显示管理器"""
    def __init__(self, max_lines=10):
        self.max_lines = max_lines
        self.log_buffer = []
        self.progress_bars = {i: 0.0 for i in range(6)}
        self._clear_output()
    
    def _clear_output(self):
        """清空输出"""
        clear_output(wait=True)
    
    def _display_logs(self):
        """显示日志"""
        self._clear_output()
        
        start_idx = max(0, len(self.log_buffer) - self.max_lines)
        for log in self.log_buffer[start_idx:]:
            if log.strip():
                print(log)
        
        print('-' * 80)
        
        for model_idx, progress in self.progress_bars.items():
            bar_length = 50
            filled = int(progress * bar_length)
            bar = f"Model {model_idx + 1}: [{'='*filled}{' '*(bar_length-filled)}] {progress*100:.1f}%"
            print(bar)

# 创建全局实例
core_manager = CoreManager()

# 导出常用变量
logger = core_manager.logger
config = core_manager
BASE_DIR = core_manager.BASE_DIR
LOG_DIR = core_manager.LOG_DIR
MODEL_DIR = core_manager.MODEL_DIR
DATA_DIR = core_manager.DATA_DIR
CHECKPOINT_DIR = core_manager.CHECKPOINT_DIR
