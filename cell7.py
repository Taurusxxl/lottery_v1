#7 资源监控器\resource_monitor.py
import psutil
import logging
import threading
import time
import numpy as np
from collections import deque
from datetime import datetime

# 获取logger实例
logger = logging.getLogger(__name__)

class ResourceMonitor:
    """资源监控器 - 负责监控CPU、内存、GPU等资源使用情况"""
    
    def __init__(self, window_size=100, check_interval=5):
        self.window_size = window_size
        self.check_interval = check_interval
        self.lock = threading.Lock()
        
        # 监控指标存储
        self.metrics = {
            'cpu_usage': deque(maxlen=window_size),
            'memory_usage': deque(maxlen=window_size),
            'disk_usage': deque(maxlen=window_size),
            'gpu_usage': None,
            'gpu_memory': None
        }
        
        # 警报阈值
        self.thresholds = {
            'cpu_usage': 90,    # CPU使用率超过90%
            'memory_usage': 90,  # 内存使用率超过90%
            'disk_usage': 90,    # 磁盘使用率超过90%
            'gpu_usage': 90,     # GPU使用率超过90%
            'gpu_memory': 90     # GPU内存使用率超过90%
        }
        
        # 监控线程
        self.monitor_thread = None
        self.is_running = False
        
        # 警报历史
        self.alerts = []
        
        self.memory_usage = 0.0  # 确保有此属性初始化
        self.cpu_usage = 0.0
        self.gpu_usage = 0.0
        
        logger.info("资源监控器初始化完成")
    
    def start(self):
        """启动资源监控"""
        if self.monitor_thread and self.monitor_thread.is_alive():
            logger.warning("资源监控器已在运行")
            return
            
        self.is_running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        logger.info("资源监控器已启动")
    
    def stop(self):
        """停止资源监控"""
        self.is_running = False
        if self.monitor_thread:
            self.monitor_thread.join()
        logger.info("资源监控器已停止")
    
    def _monitor_loop(self):
        """监控循环"""
        while self.is_running:
            try:
                self._collect_metrics()
                self._check_alerts()
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"资源监控循环出错: {str(e)}")
    
    def _collect_metrics(self):
        """收集资源指标"""
        with self.lock:
            # CPU使用率
            self.cpu_usage = psutil.cpu_percent()
            self.metrics['cpu_usage'].append(self.cpu_usage)
            
            # 内存使用率
            mem = psutil.virtual_memory()
            self.memory_usage = mem.percent  # 确保有此属性更新
            self.metrics['memory_usage'].append(self.memory_usage)
            
            # 磁盘使用率
            disk = psutil.disk_usage('/')
            self.metrics['disk_usage'].append(disk.percent)
    
    def _check_alerts(self):
        """检查警报"""
        with self.lock:
            for metric, values in self.metrics.items():
                if not values:
                    continue
                current = values[-1]
                if current > self.thresholds[metric]:
                    alert = {
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'metric': metric,
                        'value': current,
                        'threshold': self.thresholds[metric]
                    }
                    self.alerts.append(alert)
                    logger.warning(f"资源警报: {metric} = {current}% (阈值: {self.thresholds[metric]}%)")

# 创建全局实例
resource_monitor = ResourceMonitor()