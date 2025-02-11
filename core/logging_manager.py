# 日志系统初始化\logging_manager.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 添加项目根目录到路径
import logging
import logging.handlers
from datetime import datetime
from collections import deque
from IPython.display import clear_output
from pathlib import Path

# 从配置管理器获取基础配置
from .config_manager import LOG_DIR

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
    
    def _init_file(self):
        """初始化日志文件"""
        self.current_file = self._get_file_path()
        self.current_date = datetime.now().strftime('%Y%m%d')
        if os.path.exists(self.current_file):
            self.current_size = os.path.getsize(self.current_file)
        else:
            self.current_size = 0
    
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
            print('\r' + ' ' * 80, end='\r')
            progress = int(record.progress * 50)
            print(f'\rTraining Progress: [{"="*progress}{" "*(50-progress)}] {record.progress*100:.1f}%', end='')

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

class LoggingManager:
    """日志管理器 - 单例模式"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._configure_logging()
        return cls._instance
    
    def _configure_logging(self):
        """配置日志系统"""
        BASE_DIR = Path(__file__).parent.parent
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
            handlers=[
                logging.FileHandler(BASE_DIR / "system.log"),
                logging.StreamHandler()
            ]
        )
        logging.captureWarnings(True)
        self.logger = logging.getLogger('ACE_System')

class LogManager:
    """日志管理器"""
    def __init__(self):
        self.logger = logging.getLogger('ContinuousTraining')
        self.logger.setLevel(logging.INFO)
        self.log_buffer = deque(maxlen=100)
        self.display_manager = LogDisplayManager()
        self._setup_handlers()
    
    def _setup_handlers(self):
        """设置日志处理器"""
        file_handler = DailyRotatingFileHandler(
            base_dir=LOG_DIR,
            prefix='continuous_training'
        )
        
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        
        progress_handler = ProgressHandler()
        progress_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(progress_handler)
    
    def get_logger(self):
        """获取logger实例"""
        return self.logger
    
    def update_progress(self, model_idx: int, progress: float):
        """更新进度条"""
        self.display_manager.progress_bars[model_idx] = progress
        self.display_manager._display_logs()
    
    def add_log(self, message: str):
        """添加日志到缓冲区"""
        self.log_buffer.append(message)
        self.display_manager.log_buffer.append(message)
        self.display_manager._display_logs()

# 创建全局实例
logger = LoggingManager().logger
