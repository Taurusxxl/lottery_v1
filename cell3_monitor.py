#3 System Monitor / 系统监控模块
import psutil
import logging
import threading
import time
import gc
import shutil
import tensorflow as tf
import subprocess
import numpy as np
import json
import os
from collections import deque
from datetime import datetime

# 获取logger实例
logger = logging.getLogger(__name__)

# 从cell1移入的日志处理器类
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
        """重写emit方法,在写入日志前检查文件状态"""
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
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ),
            logging.INFO: logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            ),
            logging.WARNING: logging.Formatter(
                '%(asctime)s - %(levelname)s - WARNING: %(message)s'
            ),
            logging.ERROR: logging.Formatter(
                '%(asctime)s - %(levelname)s - ERROR: %(message)s\n%(pathname)s:%(lineno)d'
            ),
            logging.CRITICAL: logging.Formatter(
                '%(asctime)s - %(levelname)s - CRITICAL: %(message)s\n%(pathname)s:%(lineno)d\n%(exc_info)s'
            )
        }
    
    def format(self, record):
        formatter = self.formatters.get(record.levelno)
        return formatter.format(record)

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
        
        # 添加缺失的警报阈值初始化
        self.thresholds = {
            'cpu_usage': 90,    # CPU使用率超过90%
            'memory_usage': 90,  # 内存使用率超过90%
            'disk_usage': 90,    # 磁盘使用率超过90%
            'gpu_usage': 90,     # GPU使用率超过90%
            'gpu_memory': 90     # GPU内存使用率超过90%
        }
        
        # 添加警报历史
        self.alerts = []
        
        # 确保基础属性初始化
        self._memory_usage = 0.0
        self.cpu_usage = 0.0
        self.gpu_usage = 0.0

        # 添加线程控制
        self._running = False
        self._thread = threading.Thread(target=self._monitor_loop)

        # 确保初始化时收集初始指标
        self._monitor_loop()  # 添加初始数据收集
        self.start()  # 启动监控线程

    def _monitor_loop(self):
        """监控循环的完整实现"""
        while self._running:
            try:
                self._collect_metrics()
                self._check_alerts()
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"资源监控循环出错: {str(e)}")
                
    def _check_alerts(self):
        """完整的警报检查逻辑"""
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

    def _collect_metrics(self):
        """收集资源指标"""
        try:
            with self.lock:
                # CPU使用率
                self.cpu_usage = psutil.cpu_percent()
                self.metrics['cpu_usage'].append(self.cpu_usage)
                
                # 内存使用率
                mem = psutil.virtual_memory()
                self._memory_usage = mem.percent
                self.metrics['memory_usage'].append(self._memory_usage)
                
                # 磁盘使用率
                disk = psutil.disk_usage('/')
                self.metrics['disk_usage'].append(disk.percent)
                
        except Exception as e:
            logger.error(f"收集资源指标时出错: {str(e)}")

    def get_memory_usage(self):
        """获取当前内存使用率"""
        return self._memory_usage

    def start(self):
        """启动资源监控"""
        if not self._running:
            self._running = True
            self._thread.start()
            logger.info("资源监控已启动")
    
    def stop(self):
        """停止资源监控"""
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=5)
        logger.info("资源监控已停止")

class PerformanceMonitor:
    """性能监控器 - 负责收集和分析性能指标"""
    def __init__(self, save_dir='logs/performance', window_size=1000):
        self.save_dir = save_dir
        self.window_size = window_size
        self.metrics = {
            'cpu_usage': deque(maxlen=window_size),
            'memory_usage': deque(maxlen=window_size),
            'gpu_usage': deque(maxlen=window_size),
            'loss': deque(maxlen=window_size),
            'accuracy': deque(maxlen=window_size)
        }
        self._running = False
        self._thread = threading.Thread(target=self._monitor_loop)
        
        # 添加内存监控器初始化
        from cell2_utils import memory_manager
        self.memory_monitor = memory_manager

    def _monitor_loop(self):
        """性能监控主循环"""
        while self._running:
            try:
                # 收集CPU使用率
                cpu_usage = psutil.cpu_percent()
                self.metrics['cpu_usage'].append(cpu_usage)
                
                # 收集内存使用率
                mem_usage = psutil.virtual_memory().percent
                self.metrics['memory_usage'].append(mem_usage)
                
                # 收集GPU使用率（需要安装GPU监控库）
                gpu_usage = 0  # 这里需要根据实际GPU监控库实现
                self.metrics['gpu_usage'].append(gpu_usage)
                
                # 保存指标到文件
                self._save_metrics()
                
                time.sleep(1)  # 每秒收集一次
                
            except Exception as e:
                logger.error(f"性能监控出错: {str(e)}")
                break

    def start(self):
        """启动监控线程"""
        if not self._running:
            self._running = True
            self._thread.start()
            logger.info("性能监控已启动")

    def stop(self):
        """停止监控线程"""
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=5)
        logger.info("性能监控已停止")

    def _save_metrics(self):
        """保存指标到文件"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.save_dir, f"metrics_{timestamp}.json")
            
            metrics_to_save = {
                k: list(v) for k, v in self.metrics.items()
            }
            
            with open(filename, 'w') as f:
                json.dump(metrics_to_save, f)
                
        except Exception as e:
            logger.error(f"保存性能指标失败: {str(e)}")

class SystemManager:
    """系统管理器 - 单例模式"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            # 初始化各个监控器
            self.resource_monitor = ResourceMonitor()
            self.performance_monitor = PerformanceMonitor(save_dir='logs/performance')
            
            # 初始化其他组件
            self.memory_monitor = MemoryMonitor()
            self.system_cleaner = SystemCleaner()
            
            # 添加缺失的阈值初始化
            self.memory_warning_threshold = 0.85  # 85%内存使用率警告
            self.memory_critical_threshold = 0.95  # 95%内存使用率危险
            self.cpu_warning_threshold = 0.85  # 85% CPU使用率警告
            
            # 添加缺失的状态初始化
            self.last_cleanup_time = time.time()
            self.cleanup_interval = 300  # 5分钟执行一次清理
            self.compatibility_checked = False
            
            # 添加系统状态监控配置
            self.status_check_interval = 60  # 60秒检查一次系统状态
            self.last_status_check = time.time()
            self.status_history = deque(maxlen=1000)  # 存储最近1000次状态检查结果
            
            # 添加资源监控配置
            self.resource_warning_count = 0
            self.max_warning_threshold = 5  # 最大警告次数，超过后采取行动
            
            # 添加日志配置
            self.log_dir = 'logs/system'
            os.makedirs(self.log_dir, exist_ok=True)
            
            # 添加系统恢复机制
            self.recovery_attempts = 0
            self.max_recovery_attempts = 3
            
            # 启动基础监控
            self.start_basic_monitoring()
            
            self.initialized = True
            
            logger.info("系统管理器初始化完成")

    def check_system_compatibility(self):
        """检查系统兼容性"""
        try:
            compatibility = {
                'python_version': sys.version,
                'tensorflow_version': tf.__version__,
                'gpu_available': bool(tf.config.list_physical_devices('GPU')),
                'memory_sufficient': psutil.virtual_memory().total >= 8 * (1024 ** 3),  # 最少8GB内存
                'disk_sufficient': psutil.disk_usage('/').free >= 10 * (1024 ** 3)  # 最少10GB可用空间
            }
            
            self.compatibility_checked = True
            return compatibility
        except Exception as e:
            logger.error(f"检查系统兼容性时出错: {str(e)}")
            return None

    def get_system_status(self):
        """获取完整的系统状态报告"""
        try:
            status = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'system_metrics': self.get_system_metrics(),
                'performance_metrics': self.performance_monitor.get_summary(),
                'memory_status': self.memory_monitor.check_memory(),
                'compatibility': self.check_system_compatibility() if not self.compatibility_checked else None
            }
            return status
        except Exception as e:
            logger.error(f"获取系统状态报告时出错: {str(e)}")
            return None

    def check_system_health(self):
        """检查系统健康状态"""
        try:
            # 检查内存使用
            memory_info = self.memory_monitor.check_memory()
            if not memory_info['healthy']:
                self.handle_memory_warning(memory_info)
            
            # 检查系统状态
            system_status = self.get_system_metrics()
            if any(status > self.cpu_warning_threshold for status in [
                system_status.get('cpu', 0),
                system_status.get('memory', 0),
                system_status.get('disk', 0)
            ]):
                self.handle_system_warning(system_status)
            
            # 定期清理
            self._perform_periodic_cleanup()
            
            return memory_info['healthy'] and all(
                status < self.cpu_warning_threshold for status in [
                    system_status.get('cpu', 0),
                    system_status.get('memory', 0),
                    system_status.get('disk', 0)
                ]
            )
            
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
            if status.get('cpu', 0) > self.cpu_warning_threshold:
                logger.warning(f"CPU使用率过高: {status['cpu']}%")
            if status.get('memory', 0) > self.memory_warning_threshold:
                logger.warning(f"内存使用率过高: {status['memory']}%")
            if status.get('disk', 0) > 90:  # 磁盘空间阈值固定为90%
                logger.warning(f"磁盘使用率过高: {status['disk']}%")
                
        except Exception as e:
            logger.error(f"处理系统警告时出错: {str(e)}")

    def check_dependencies(self):
        """系统启动时自动调用"""
        try:
            from ..notebooks.Untitled import check_requirements
            need_install = check_requirements(requirements)
            if need_install:
                self.install_dependencies(need_install)
        except Exception as e:
            logger.error(f"检查依赖时出错: {str(e)}")

    def install_dependencies(self, packages):
        """受控安装方法"""
        try:
            logger.info(f"自动安装依赖: {packages}")
            # 这里可以添加实际的包安装逻辑
            # pip.main(['install'] + packages)
        except Exception as e:
            logger.error(f"安装依赖时出错: {str(e)}")

    def get_system_metrics(self):
        """获取系统指标"""
        try:
            return {
                'memory': self.memory_monitor.check_memory(),
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

    def _perform_periodic_cleanup(self):
        """执行定期清理"""
        current_time = time.time()
        if current_time - self.last_cleanup_time > self.cleanup_interval:
            try:
                self.system_cleaner.check_and_cleanup()
                self.last_cleanup_time = current_time
            except Exception as e:
                logger.error(f"执行定期清理时出错: {str(e)}")

    def analyze_system_performance(self):
        """分析系统整体性能"""
        try:
            perf_summary = self.performance_monitor.get_summary()
            sys_metrics = self.get_system_metrics()
            
            analysis = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'performance_status': perf_summary,
                'system_status': sys_metrics,
                'health_check': self.check_system_health(),
                'recommendations': self._generate_recommendations(perf_summary, sys_metrics)
            }
            
            return analysis
        except Exception as e:
            logger.error(f"分析系统性能时出错: {str(e)}")
            return None

    def _generate_recommendations(self, perf_summary, sys_metrics):
        """生成系统优化建议"""
        recommendations = []
        
        # 检查内存使用
        if sys_metrics.get('memory', 0) > self.memory_warning_threshold:
            recommendations.append("建议清理内存或增加内存容量")
            
        # 检查CPU使用
        if sys_metrics.get('cpu', 0) > self.cpu_warning_threshold:
            recommendations.append("建议优化计算密集型任务或增加CPU资源")
            
        # 检查GPU使用
        gpu_metrics = sys_metrics.get('gpu')
        if gpu_metrics and gpu_metrics.get('utilization', 0) > 90:
            recommendations.append("建议优化GPU使用效率或考虑增加GPU资源")
            
        return recommendations

    def _handle_resource_warnings(self):
        """处理资源警告的升级机制"""
        self.resource_warning_count += 1
        if self.resource_warning_count >= self.max_warning_threshold:
            logger.critical("资源警告次数过多，执行紧急清理")
            self.system_cleaner._emergency_cleanup()
            self.resource_warning_count = 0

    def _perform_system_recovery(self):
        """系统恢复机制"""
        try:
            if self.recovery_attempts >= self.max_recovery_attempts:
                logger.critical("系统恢复次数超过限制，需要人工干预")
                return False
                
            logger.warning(f"尝试系统恢复，第{self.recovery_attempts + 1}次")
            
            # 执行恢复步骤
            self.system_cleaner._emergency_cleanup()
            tf.keras.backend.clear_session()
            gc.collect()
            
            self.recovery_attempts += 1
            return True
            
        except Exception as e:
            logger.error(f"执行系统恢复时出错: {str(e)}")
            return False
            
    def reset_recovery_count(self):
        """重置恢复计数"""
        self.recovery_attempts = 0

    def get_system_summary(self):
        """获取系统综合报告"""
        try:
            return {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'system_health': self.check_system_health(),
                'performance_metrics': self.performance_monitor.get_summary(),
                'resource_metrics': self.resource_monitor.metrics,
                'memory_status': self.memory_monitor.check_memory(),
                'recovery_attempts': self.recovery_attempts,
                'last_cleanup': datetime.fromtimestamp(self.last_cleanup_time).strftime('%Y-%m-%d %H:%M:%S'),
                'warnings_count': self.resource_warning_count
            }
        except Exception as e:
            logger.error(f"获取系统综合报告时出错: {str(e)}")
            return {}

    def _test_config_system(self):
        """测试配置系统完整性"""
        try:
            required_keys = ['host', 'port', 'user', 'password'] 
            assert all(key in self.DB_CONFIG for key in required_keys), "数据库配置缺失必要参数"
            logger.info("配置系统测试通过")
        except AssertionError as e:
            logger.error(f"配置系统测试失败: {str(e)}")
            raise

    def start_basic_monitoring(self):
        """启动基础监控"""
        try:
            # 启动资源监控
            if hasattr(self.resource_monitor, 'start'):
                self.resource_monitor.start()
            
            # 启动性能监控
            if hasattr(self.performance_monitor, 'start'):
                self.performance_monitor.start()
            
            # 启动内存监控
            if hasattr(self.memory_monitor, 'start'):
                self.memory_monitor.start()
            
            logger.info("基础监控已启动")
            
        except Exception as e:
            logger.error(f"启动基础监控失败: {str(e)}")
            
    def stop_basic_monitoring(self):
        """停止基础监控"""
        try:
            # 停止资源监控
            if hasattr(self.resource_monitor, 'stop'):
                self.resource_monitor.stop()
            
            # 停止性能监控
            if hasattr(self.performance_monitor, 'stop'):
                self.performance_monitor.stop()
            
            # 停止内存监控
            if hasattr(self.memory_monitor, 'stop'):
                self.memory_monitor.stop()
            
            logger.info("基础监控已停止")
            
        except Exception as e:
            logger.error(f"停止基础监控失败: {str(e)}")

# 从cell1移入的LogDisplayManager
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

# 辅助类定义
class MemoryMonitor:
    """内存监控类"""
    def check_memory(self):
        """检查内存状态"""
        try:
            memory = psutil.virtual_memory()
            memory_info = {
                'total': memory.total,
                'available': memory.available,
                'used': memory.used,
                'usage_percent': memory.percent,
                'healthy': memory.percent < 90,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            return memory_info
        except Exception as e:
            logger.error(f"检查内存状态时出错: {str(e)}")
            return {'healthy': False, 'error': str(e)}

    def get_memory_usage(self):
        """获取当前内存使用情况"""
        try:
            return psutil.virtual_memory().percent
        except Exception as e:
            logger.error(f"获取内存使用情况时出错: {str(e)}")
            return None

class SystemCleaner:
    """系统清理类"""
    def __init__(self):
        self.temp_dirs = ['/tmp', os.path.expanduser('~/.cache')]
        self.last_cleanup = time.time()
        self.cleanup_interval = 3600  # 1小时

    def check_and_cleanup(self):
        """检查并执行清理"""
        try:
            current_time = time.time()
            if current_time - self.last_cleanup >= self.cleanup_interval:
                self._regular_cleanup()
                self.last_cleanup = current_time
        except Exception as e:
            logger.error(f"执行定期清理时出错: {str(e)}")

    def _regular_cleanup(self):
        """常规清理"""
        try:
            # 清理Python缓存
            gc.collect()
            
            # 清理TensorFlow会话
            tf.keras.backend.clear_session()
            
            # 清理临时文件
            self._cleanup_temp_files()
            
            logger.info("完成常规清理")
        except Exception as e:
            logger.error(f"执行常规清理时出错: {str(e)}")

    def _emergency_cleanup(self):
        """紧急清理"""
        try:
            # 强制垃圾回收
            gc.collect()
            
            # 清理TensorFlow会话
            tf.keras.backend.clear_session()
            
            # 清理临时文件
            self._cleanup_temp_files(emergency=True)
            
            logger.warning("完成紧急清理")
        except Exception as e:
            logger.error(f"执行紧急清理时出错: {str(e)}")

    def _cleanup_temp_files(self, emergency=False):
        """清理临时文件
        Args:
            emergency: 是否为紧急清理
        """
        for temp_dir in self.temp_dirs:
            if os.path.exists(temp_dir):
                try:
                    if emergency:
                        shutil.rmtree(temp_dir)
                        os.makedirs(temp_dir)
                    else:
                        # 只删除超过1天的文件
                        for root, dirs, files in os.walk(temp_dir):
                            for f in files:
                                path = os.path.join(root, f)
                                if time.time() - os.path.getmtime(path) > 86400:
                                    os.remove(path)
                except Exception as e:
                    logger.error(f"清理临时文件时出错: {str(e)}")

# 创建资源监控器实例（确保在类定义之后）
resource_monitor = ResourceMonitor()
# 创建性能监控器实例
performance_monitor = PerformanceMonitor()

# 保持原有全局实例
monitor_system = SystemManager()

def init_monitoring():
    """初始化监控系统"""
    monitor_system.resource_monitor.start()
    monitor_system.performance_monitor.start()
    logger.info("监控系统已启动")

def stop_monitoring():
    """停止监控系统"""
    monitor_system.resource_monitor.stop()
    monitor_system.performance_monitor.stop()
    logger.info("监控系统已停止")

class TrainingMonitor(PerformanceMonitor):
    """训练专用监控"""
    def __init__(self):
        super().__init__()
        self.metrics.update({
            'model_loss': [deque(maxlen=1000) for _ in range(6)],
            'param_values': deque(maxlen=1000)
        })
        
    def log_training_metrics(self, model_idx, loss, params):
        """记录训练指标"""
        self.metrics['model_loss'][model_idx].append(loss)
        self.metrics['param_values'].append(params)
