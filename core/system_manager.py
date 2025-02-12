# 系统健康检查\system_manager.py
import os
import psutil
import logging
import time
import gc
import shutil
import tensorflow as tf
import subprocess
from datetime import datetime

# 获取logger实例
logger = logging.getLogger(__name__)

class SystemManager:
    """系统管理器 - 单例模式"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, config):
        if not hasattr(self, 'initialized'):
            # 初始化监控器
            self.memory_monitor = MemoryMonitor()
            self.system_monitor = SystemMonitor()
            self.system_cleaner = SystemCleaner()
            
            # 初始化阈值
            self.memory_warning_threshold = 0.85  # 85%内存使用率警告
            self.memory_critical_threshold = 0.95  # 95%内存使用率危险
            self.cpu_warning_threshold = 0.85  # 85% CPU使用率警告
            
            # 初始化状态
            self.last_cleanup_time = time.time()
            self.cleanup_interval = 300  # 5分钟执行一次清理
            self.initialized = True
            self.compatibility_checked = False
            
            self.config = config
            self.logger = logging.getLogger(__name__)
            
            logger.info("系统管理器初始化完成")
    
    def check_system_health(self):
        """检查系统健康状态"""
        try:
            # 检查内存使用
            memory_info = self.memory_monitor.check_memory()
            if not memory_info['healthy']:
                self.handle_memory_warning(memory_info)
            
            # 检查系统状态
            system_status = self.system_monitor.check_system_status()
            if not system_status['healthy']:
                self.handle_system_warning(system_status)
            
            # 定期清理
            self._perform_periodic_cleanup()
            
            return memory_info['healthy'] and system_status['healthy']
            
        except Exception as e:
            logger.error(f"检查系统健康状态时出错: {str(e)}")
            return False
    
    def handle_memory_warning(self, memory_info):
        """处理内存警告"""
        try:
            if memory_info['usage_percent'] > self.memory_critical_threshold:
                logger.critical("内存使用率超过临界值，执行紧急清理")
                self.system_cleaner._emergency_cleanup()
            elif memory_info['usage_percent'] > self.memory_warning_threshold:
                logger.warning("内存使用率较高，执行常规清理")
                self.system_cleaner._regular_cleanup()
                
        except Exception as e:
            logger.error(f"处理内存警告时出错: {str(e)}")
    
    def handle_system_warning(self, status):
        """处理系统警告"""
        try:
            if status.get('cpu_warning'):
                logger.warning(f"CPU使用率过高: {status['cpu_percent']}%")
            if status.get('gpu_warning'):
                logger.warning(f"GPU使用率过高: {status['gpu_usage']}%")
            if status.get('disk_warning'):
                logger.warning(f"磁盘使用率过高: {status['disk_percent']}%")
                
        except Exception as e:
            logger.error(f"处理系统警告时出错: {str(e)}")
    
    def _perform_periodic_cleanup(self):
        """执行定期清理"""
        current_time = time.time()
        if current_time - self.last_cleanup_time > self.cleanup_interval:
            try:
                self.system_cleaner.check_and_cleanup()
                self.last_cleanup_time = current_time
            except Exception as e:
                logger.error(f"执行定期清理时出错: {str(e)}")
    
    def get_system_metrics(self):
        """获取系统指标"""
        try:
            return {
                'memory': self.memory_monitor.get_memory_usage(),
                'cpu': psutil.cpu_percent(),
                'disk': psutil.disk_usage('/').percent,
                'gpu': self._get_gpu_metrics()
            }
        except Exception as e:
            logger.error(f"获取系统指标时出错: {str(e)}")
            return {}
    
    def _get_gpu_metrics(self):
        """获取GPU指标"""
        try:
            if not tf.config.list_physical_devices('GPU'):
                return None
                
            result = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=memory.used,memory.total,temperature.gpu', 
                 '--format=csv,nounits,noheader'],
                encoding='utf-8'
            )
            used, total, temp = map(int, result.strip().split(','))
            return {
                'memory_used': used,
                'memory_total': total,
                'temperature': temp,
                'utilization': used / total * 100
            }
        except Exception as e:
            logger.error(f"获取GPU指标时出错: {str(e)}")
            return None
    
    def check_system_compatibility(self):
        # 实现系统检查逻辑
        pass

    def check_dependencies(self):
        """系统启动时自动调用"""
        from ..notebooks.Untitled import check_requirements
        need_install = check_requirements(requirements)
        if need_install:
            self.install_dependencies(need_install)

    def install_dependencies(self, packages):
        """受控安装方法"""
        # 记录安装日志
        logger.info(f"自动安装依赖: {packages}")
        # 调用安装逻辑
        # ...

    def initialize_data_pipeline(self):
        """初始化数据管道"""
        from core.data_manager import DataPipeline
        self.data_pipeline = DataPipeline(self.config['data_path'])
        self.logger.info("数据管道初始化完成")

    def build_model_ensemble(self, model_type):
        """构建模型集合"""
        from models.model_ensemble import ModelEnsemble
        self.ensemble = ModelEnsemble(
            model_type=model_type,
            config=self.config['models']
        )
        self.logger.info(f"{model_type}模型集合构建完成")

    def start_training(self):
        """启动训练流程"""
        from optimizers.training_optimizer import TrainingOptimizer
        optimizer = TrainingOptimizer(
            model=self.ensemble,
            config=self.config['training']
        )
        optimizer.train(self.data_pipeline)
        self.logger.info("训练完成")

    def run_prediction(self):
        """执行预测"""
        realtime_data = self.data_pipeline.load_realtime()
        return self.ensemble.predict(realtime_data)

class MemoryMonitor:
    def check_memory(self):
        return {'healthy': True}  # 示例实现

class SystemMonitor:
    def check_system_status(self):
        return {'healthy': True}  # 示例实现

class SystemCleaner:
    def _emergency_cleanup(self):
        gc.collect()
        tf.keras.backend.clear_session()

# 创建全局实例
sys_manager = SystemManager()