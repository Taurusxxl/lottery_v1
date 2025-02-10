# 状态管理\state_manager.py
import os
import signal
import logging
import threading
from typing import Optional, Any, Dict
from collections import deque
from datetime import datetime
import torch

# 获取logger实例
logger = logging.getLogger(__name__)

class StateManager:
    """全局状态管理器 - 单例模式"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            # 训练相关状态
            self.trainer = None  # 全局trainer实例
            self.training_state = 'idle'  # 训练状态: idle/training/paused/stopped
            self.current_epoch = 0
            self.current_batch = 0
            self.best_performance = float('inf')
            
            # 显示相关状态
            self.display_running = True  # 显示线程运行标志
            self.display_thread = None  # 显示线程实例
            self.log_buffer = deque(maxlen=100)  # 日志缓冲区
            self.show_print = False  # 控制是否显示打印信息
            
            # 性能监控状态
            self.performance_metrics = {
                'loss': deque(maxlen=1000),
                'accuracy': deque(maxlen=1000),
                'learning_rate': 0.001
            }
            
            # 资源监控状态
            self.resource_metrics = {
                'memory_usage': 0,
                'cpu_usage': 0,
                'gpu_usage': 0
            }
            
            # 注册信号处理器
            self._register_signal_handlers()
            
            self.initialized = True
            logger.info("状态管理器初始化完成")
    
    def _register_signal_handlers(self):
        """注册信号处理器"""
        signal.signal(signal.SIGINT, self._signal_handler)   # Ctrl+C
        signal.signal(signal.SIGTERM, self._signal_handler)  # 终止信号
    
    def _signal_handler(self, signum, frame):
        """信号处理器"""
        logger.info(f"接收到信号: {signum}, 开始保存进度...")
        self.save_all_progress()
        import sys
        sys.exit(0)
    
    def save_all_progress(self):
        """保存所有进度和参数"""
        if self.trainer:
            try:
                # 保存训练进度
                self.trainer.save_training_progress()
                # 保存模型参数
                self.trainer.save_model_weights()
                # 保存性能指标
                self.save_performance_metrics()
                logger.info("所有进度和参数已保存")
            except Exception as e:
                logger.error(f"保存进度时出错: {str(e)}")
    
    def update_training_state(self, new_state: str):
        """更新训练状态"""
        valid_states = {'idle', 'training', 'paused', 'stopped'}
        if new_state not in valid_states:
            logger.error(f"无效的训练状态: {new_state}")
            return
            
        old_state = self.training_state
        self.training_state = new_state
        logger.info(f"训练状态从 {old_state} 变更为 {new_state}")
    
    def update_performance_metrics(self, metrics: Dict[str, float]):
        """更新性能指标"""
        try:
            for key, value in metrics.items():
                if key in self.performance_metrics:
                    self.performance_metrics[key].append(value)
                    
            # 更新最佳性能
            if 'loss' in metrics and metrics['loss'] < self.best_performance:
                self.best_performance = metrics['loss']
                logger.info(f"更新最佳性能: {self.best_performance:.4f}")
        except Exception as e:
            logger.error(f"更新性能指标时出错: {str(e)}")
    
    def update_resource_metrics(self, metrics: Dict[str, float]):
        """更新资源使用指标"""
        try:
            self.resource_metrics.update(metrics)
            # 检查资源使用是否超过警戒线
            if metrics.get('memory_usage', 0) > 90:
                logger.warning("内存使用率超过90%!")
            if metrics.get('gpu_usage', 0) > 90:
                logger.warning("GPU使用率超过90%!")
        except Exception as e:
            logger.error(f"更新资源指标时出错: {str(e)}")
    
    def set_trainer(self, trainer: Any):
        """设置trainer实例"""
        self.trainer = trainer
    
    def set_display_thread(self, thread: threading.Thread):
        """设置显示线程"""
        self.display_thread = thread
    
    def stop_display(self):
        """停止显示线程"""
        self.display_running = False
        if self.display_thread and self.display_thread.is_alive():
            self.display_thread.join()
            logger.info("显示线程已停止")
    
    def save_performance_metrics(self):
        """保存性能指标到文件"""
        try:
            metrics_file = os.path.join('logs', f'metrics_{datetime.now():%Y%m%d_%H%M%S}.json')
            import json
            with open(metrics_file, 'w') as f:
                # 将deque转换为list后保存
                metrics_to_save = {
                    k: list(v) if isinstance(v, deque) else v 
                    for k, v in self.performance_metrics.items()
                }
                json.dump(metrics_to_save, f, indent=4)
            logger.info(f"性能指标已保存到: {metrics_file}")
        except Exception as e:
            logger.error(f"保存性能指标时出错: {str(e)}")

    def save_training_state(self):
        """保存训练状态"""
        torch.save({
            'model_states': [m.get_weights() for m in model_ensemble.models],
            'optimizer_states': training_optimizer.get_states()
        }, 'training_state.pt')

    def restore_training_state(self):
        """恢复训练状态"""
        if os.path.exists('training_state.pt'):
            state = torch.load('training_state.pt')
            # 恢复模型参数
            for model, weights in zip(model_ensemble.models, state['model_states']):
                model.set_weights(weights)
            # 恢复优化器状态
            training_optimizer.set_states(state['optimizer_states'])

# 创建全局实例
state_manager = StateManager()