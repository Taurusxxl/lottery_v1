#8 性能监控器\performance_monitor.py
import logging
import numpy as np
import time
from datetime import datetime
import json
import os
from collections import deque
import threading

# 获取logger实例
logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """性能监控器 - 负责收集和分析性能指标"""
    def __init__(self, save_dir: str, window_size=100):
        self.save_dir = save_dir
        self.window_size = window_size
        self.lock = threading.Lock()
        
        # 性能指标存储
        self.metrics = {
            'loss': deque(maxlen=window_size),
            'accuracy': deque(maxlen=window_size),
            'training_time': deque(maxlen=window_size),
            'prediction_time': deque(maxlen=window_size),
            'memory_usage': deque(maxlen=window_size),
            'gpu_usage': deque(maxlen=window_size),  # 新增GPU使用率指标
            'batch_time': deque(maxlen=window_size)  # 新增批次处理时间指标
        }
        
        # 警报阈值
        self.thresholds = {
            'loss_increase': 0.2,        # 损失增加超过20%
            'accuracy_drop': 0.1,        # 准确率下降超过10%
            'training_time_increase': 0.5,  # 训练时间增加超过50%
            'memory_usage': 0.9,         # 内存使用率超过90%
            'gpu_usage': 0.95,           # GPU使用率超过95%
            'batch_time_increase': 0.3   # 批次时间增加超过30%
        }
        
        # 性能趋势分析
        self.trend_window = 10  # 趋势分析窗口
        self.trend_threshold = 0.05  # 趋势判定阈值
        
        # 初始化性能日志文件
        self._init_log_file()
        
        os.makedirs(self.save_dir, exist_ok=True)
        
        self._running = False
        self._thread = threading.Thread(target=self._monitor_loop)
        
    def _init_log_file(self):
        """初始化性能日志文件"""
        try:
            self.log_file = os.path.join(
                self.save_dir, 
                f'performance_{datetime.now().strftime("%Y%m%d")}.json'
            )
            os.makedirs(self.save_dir, exist_ok=True)
            
            if not os.path.exists(self.log_file):
                with open(self.log_file, 'w') as f:
                    json.dump([], f)
                    
        except Exception as e:
            logger.error(f"初始化性能日志文件失败: {str(e)}")
    
    def update_metrics(self, metrics_dict):
        """更新性能指标"""
        try:
            with self.lock:
                # 更新指标
                for metric_name, value in metrics_dict.items():
                    if metric_name in self.metrics:
                        self.metrics[metric_name].append(value)
                
                # 记录到日志文件
                self._log_metrics(metrics_dict)
                
                # 检查警报
                self._check_alerts()
                
                # 分析趋势
                self._analyze_trends()
                
        except Exception as e:
            logger.error(f"更新性能指标失败: {str(e)}")
    
    def _log_metrics(self, metrics_dict):
        """记录性能指标到日志文件"""
        try:
            log_entry = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'metrics': metrics_dict
            }
            
            # 读取现有日志
            with open(self.log_file, 'r') as f:
                logs = json.load(f)
                
            # 添加新记录
            logs.append(log_entry)
            
            # 写回文件
            with open(self.log_file, 'w') as f:
                json.dump(logs, f, indent=4)
                
        except Exception as e:
            logger.error(f"记录性能指标失败: {str(e)}")
    
    def _check_alerts(self):
        """检查性能警报"""
        try:
            with self.lock:
                # 检查损失值增加
                if len(self.metrics['loss']) >= 2:
                    loss_increase = (self.metrics['loss'][-1] - self.metrics['loss'][-2]) / self.metrics['loss'][-2]
                    if loss_increase > self.thresholds['loss_increase']:
                        logger.warning(f"损失值显著增加: {loss_increase:.2%}")
                
                # 检查准确率下降
                if len(self.metrics['accuracy']) >= 2:
                    acc_drop = (self.metrics['accuracy'][-2] - self.metrics['accuracy'][-1]) / self.metrics['accuracy'][-2]
                    if acc_drop > self.thresholds['accuracy_drop']:
                        logger.warning(f"准确率显著下降: {acc_drop:.2%}")
                
                # 检查训练时间增加
                if len(self.metrics['training_time']) >= 2:
                    time_increase = (self.metrics['training_time'][-1] - self.metrics['training_time'][-2]) / self.metrics['training_time'][-2]
                    if time_increase > self.thresholds['training_time_increase']:
                        logger.warning(f"训练时间显著增加: {time_increase:.2%}")
                
                # 检查内存使用
                if self.metrics['memory_usage'] and self.metrics['memory_usage'][-1] > self.thresholds['memory_usage']:
                    logger.warning(f"内存使用率过高: {self.metrics['memory_usage'][-1]:.1%}")
                
                # 检查GPU使用
                if self.metrics['gpu_usage'] and self.metrics['gpu_usage'][-1] > self.thresholds['gpu_usage']:
                    logger.warning(f"GPU使用率过高: {self.metrics['gpu_usage'][-1]:.1%}")
                
        except Exception as e:
            logger.error(f"检查性能警报失败: {str(e)}")
    
    def _analyze_trends(self):
        """分析性能趋势"""
        try:
            with self.lock:
                for metric_name, values in self.metrics.items():
                    if len(values) >= self.trend_window:
                        # 计算趋势斜率
                        x = np.arange(self.trend_window)
                        y = np.array(list(values)[-self.trend_window:])
                        slope = np.polyfit(x, y, 1)[0]
                        
                        # 判断趋势
                        if abs(slope) > self.trend_threshold:
                            trend = "上升" if slope > 0 else "下降"
                            logger.info(f"{metric_name}指标呈{trend}趋势")
                            
        except Exception as e:
            logger.error(f"分析性能趋势失败: {str(e)}")
    
    def get_metric_history(self, metric_name, window=None):
        """获取指定指标的历史数据
        Args:
            metric_name: 指标名称
            window: 获取最近window个数据点，None表示获取全部
        Returns:
            list: 指标历史数据
        """
        try:
            with self.lock:
                if metric_name not in self.metrics:
                    logger.warning(f"未找到指标: {metric_name}")
                    return []
                    
                values = list(self.metrics[metric_name])
                if window is not None:
                    values = values[-window:]
                return values
                
        except Exception as e:
            logger.error(f"获取指标历史数据失败: {str(e)}")
            return []
    
    def set_threshold(self, metric_name, value):
        """设置警报阈值
        Args:
            metric_name: 指标名称
            value: 阈值
        """
        try:
            with self.lock:
                if metric_name not in self.thresholds:
                    logger.warning(f"未找到阈值配置: {metric_name}")
                    return
                    
                self.thresholds[metric_name] = value
                logger.info(f"已更新{metric_name}的警报阈值为: {value}")
                
        except Exception as e:
            logger.error(f"设置警报阈值失败: {str(e)}")
    
    def export_metrics(self, start_time=None, end_time=None):
        """导出指定时间范围的性能指标
        Args:
            start_time: 开始时间，datetime对象
            end_time: 结束时间，datetime对象
        Returns:
            dict: 导出的性能指标
        """
        try:
            with open(self.log_file, 'r') as f:
                logs = json.load(f)
                
            # 过滤时间范围
            if start_time or end_time:
                filtered_logs = []
                for log in logs:
                    log_time = datetime.strptime(log['timestamp'], '%Y-%m-%d %H:%M:%S')
                    if start_time and log_time < start_time:
                        continue
                    if end_time and log_time > end_time:
                        continue
                    filtered_logs.append(log)
                logs = filtered_logs
                
            return logs
            
        except Exception as e:
            logger.error(f"导出性能指标失败: {str(e)}")
            return []
    
    def get_summary(self):
        """获取性能总结"""
        try:
            with self.lock:
                summary = {}
                for metric_name, values in self.metrics.items():
                    if values:
                        summary[metric_name] = {
                            'current': values[-1],
                            'mean': np.mean(values),
                            'std': np.std(values),
                            'min': np.min(values),
                            'max': np.max(values)
                        }
                
                # 添加趋势分析
                trends = {}
                for metric_name, values in self.metrics.items():
                    if len(values) >= self.trend_window:
                        x = np.arange(self.trend_window)
                        y = np.array(list(values)[-self.trend_window:])
                        slope = np.polyfit(x, y, 1)[0]
                        
                        if abs(slope) > self.trend_threshold:
                            trend = "上升" if slope > 0 else "下降"
                        else:
                            trend = "稳定"
                            
                        trends[metric_name] = {
                            'trend': trend,
                            'slope': slope
                        }
                
                summary['trends'] = trends
                return summary
                
        except Exception as e:
            logger.error(f"获取性能总结失败: {str(e)}")
            return {}
    
    def reset(self):
        """重置性能监控器"""
        try:
            with self.lock:
                for metric_name in self.metrics:
                    self.metrics[metric_name].clear()
                logger.info("性能监控器已重置")
                
        except Exception as e:
            logger.error(f"重置性能监控器失败: {str(e)}")
    
    def get_alerts_history(self):
        """获取警报历史"""
        try:
            with open(self.log_file, 'r') as f:
                logs = json.load(f)
            
            alerts = []
            for log in logs:
                if 'alerts' in log:
                    alerts.extend(log['alerts'])
            return alerts
            
        except Exception as e:
            logger.error(f"获取警报历史失败: {str(e)}")
            return []
    
    def get_performance_report(self, start_time=None, end_time=None):
        """生成性能报告
        Args:
            start_time: 开始时间，datetime对象
            end_time: 结束时间，datetime对象
        Returns:
            dict: 性能报告
        """
        try:
            # 获取指定时间范围的指标
            metrics = self.export_metrics(start_time, end_time)
            
            # 获取警报历史
            alerts = self.get_alerts_history()
            
            # 获取当前性能总结
            summary = self.get_summary()
            
            report = {
                'time_range': {
                    'start': start_time.strftime('%Y-%m-%d %H:%M:%S') if start_time else None,
                    'end': end_time.strftime('%Y-%m-%d %H:%M:%S') if end_time else None
                },
                'metrics': metrics,
                'alerts': alerts,
                'summary': summary,
                'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            return report
            
        except Exception as e:
            logger.error(f"生成性能报告失败: {str(e)}")
            return {}

    def analyze_performance_trend(self, match_history):
        """分析性能趋势"""
        try:
            # 1. 计算移动平均
            window_size = 100
            moving_avg = np.convolve(match_history, np.ones(window_size)/window_size, mode='valid')
            
            # 2. 判断趋势
            if len(moving_avg) >= 2:
                trend = moving_avg[-1] - moving_avg[-2]
                
                if trend > 0.1:
                    return "IMPROVING"
                elif trend < -0.1:
                    return "DEGRADING"
                else:
                    return "STABLE"
                    
            return "INSUFFICIENT_DATA"
            
        except Exception as e:
            logger.error(f"分析性能趋势时出错: {str(e)}")
            return None

    def start(self):
        """启动监控线程"""
        self._running = True
        self._thread.start()
        logger.info("性能监控器已启动")
    
    def stop(self):
        """停止监控线程"""
        self._running = False
        self._thread.join()
        logger.info("性能监控器已停止")
    
    def _monitor_loop(self):
        """监控主循环"""
        while self._running:
            # 这里添加实际的监控逻辑
            time.sleep(1)  # 每秒采集一次数据

# 创建全局实例
performance_monitor = PerformanceMonitor(save_dir='logs/performance')