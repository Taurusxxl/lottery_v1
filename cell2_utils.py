#2 Utility Functions / 工具函数模块
import os
import gc
import re
import time
import psutil
import logging
import tensorflow as tf
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
import shutil

# 获取logger
logger = logging.getLogger(__name__)

class DateUtils:
    """日期处理工具类"""
    
    @staticmethod
    def parse_issue(issue_str: str) -> Tuple[str, int]:
        """解析期号字符串"""
        match = re.match(r"(\d{8})-(\d{4})", issue_str)
        if not match:
            raise ValueError("无效的期号格式")
        return match.group(1), int(match.group(2))
    
    @staticmethod
    def get_next_issue(current_issue: str) -> str:
        """获取下一期号"""
        date_str, period = DateUtils.parse_issue(current_issue)
        date = datetime.strptime(date_str, "%Y%m%d")
        
        if period == 1440:
            new_date = date + timedelta(days=1)
            new_period = 1
        else:
            new_date = date
            new_period = period + 1
        
        return f"{new_date.strftime('%Y%m%d')}-{new_period:04d}"

class MemoryManager:
    """内存管理工具类"""
    
    def __init__(self, 
                warning_threshold_mb: int = 8000,
                critical_threshold_mb: int = 10000,
                cleanup_interval: int = 300,
                full_cleanup_interval: int = 14400):
        """
        初始化内存管理器
        Args:
            warning_threshold_mb: 警告阈值(MB)
            critical_threshold_mb: 临界阈值(MB)
            cleanup_interval: 常规清理间隔(秒)
            full_cleanup_interval: 全面清理间隔(秒)
        """
        self.warning_threshold = warning_threshold_mb * 1024 * 1024  # 转换为字节
        self.critical_threshold = critical_threshold_mb * 1024 * 1024
        self.cleanup_interval = cleanup_interval
        self.full_cleanup_interval = full_cleanup_interval
        
        self.last_cleanup_time = time.time()
        self.last_full_cleanup_time = time.time()
        
        logger.info(f"内存管理器初始化完成 - 警告阈值:{warning_threshold_mb}MB, 临界阈值:{critical_threshold_mb}MB")

    def check_memory_status(self) -> bool:
        """
        检查内存状态并在必要时执行清理
        Returns:
            bool: 内存状态是否正常
        """
        try:
            current_usage = self.get_memory_usage()
            current_time = time.time()

            # 检查是否需要执行清理
            if current_time - self.last_cleanup_time > self.cleanup_interval:
                self._regular_cleanup()
                self.last_cleanup_time = current_time

            if current_time - self.last_full_cleanup_time > self.full_cleanup_interval:
                self._full_cleanup()
                self.last_full_cleanup_time = current_time

            # 检查内存使用是否超过阈值
            if current_usage > self.critical_threshold:
                logger.warning(f"内存使用超过临界值: {current_usage/1024/1024:.1f}MB")
                self._emergency_cleanup()
                return False
            elif current_usage > self.warning_threshold:
                logger.warning(f"内存使用超过警告值: {current_usage/1024/1024:.1f}MB")
                self._optimize_memory()

            return True

        except Exception as e:
            logger.error(f"检查内存状态时出错: {str(e)}")
            return False

    def get_memory_usage(self) -> int:
        """
        获取当前进程的内存使用量(字节)
        Returns:
            int: 内存使用量(字节)
        """
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss
        except Exception as e:
            logger.error(f"获取内存使用量时出错: {str(e)}")
            return 0

    def get_memory_info(self) -> Dict[str, Any]:
        """
        获取详细的内存使用信息
        Returns:
            Dict: 内存使用信息
        """
        try:
            memory = psutil.virtual_memory()
            process = psutil.Process(os.getpid())
            
            return {
                'total': memory.total,
                'available': memory.available,
                'used': memory.used,
                'free': memory.free,
                'percent': memory.percent,
                'process_usage': process.memory_info().rss,
                'process_percent': process.memory_percent()
            }
        except Exception as e:
            logger.error(f"获取内存信息时出错: {str(e)}")
            return {}

    def _regular_cleanup(self):
        """执行常规清理"""
        try:
            # 1. 清理Python垃圾
            gc.collect()
            
            # 2. 清理TF会话
            tf.keras.backend.clear_session()
            
            # 3. 清理不用的变量
            for name in list(globals().keys()):
                if name.startswith('_temp_'):
                    del globals()[name]
                    
            logger.info("完成常规内存清理")
            
        except Exception as e:
            logger.error(f"常规清理时出错: {str(e)}")

    def _full_cleanup(self):
        """执行全面清理"""
        try:
            # 1. 执行常规清理
            self._regular_cleanup()
            
            # 2. 重置TensorFlow状态
            gpus = tf.config.list_physical_devices('GPU')
            for gpu in gpus:
                tf.config.experimental.reset_memory_stats(gpu.name)
            
            # 3. 清理模型缓存
            self._cleanup_model_cache()
            
            logger.info("完成全面内存清理")
            
        except Exception as e:
            logger.error(f"全面清理时出错: {str(e)}")

    def _emergency_cleanup(self):
        """执行紧急清理"""
        try:
            # 三级清理策略
            gc.collect(2)  # 强制回收老年代内存
            tf.keras.backend.clear_session()
            
            # 释放多GPU内存
            for gpu in tf.config.list_physical_devices('GPU'):
                try:
                    tf.config.experimental.reset_memory_stats(gpu.name)
                except RuntimeError as e:
                    logger.warning(f"GPU内存重置失败: {str(e)}")

            # 清理临时文件缓存
            self._clean_temp_files()
            
            logger.warning("执行紧急内存清理")
            
        except RuntimeError as e:
            logger.error(f"运行时错误: {str(e)}")
        except MemoryError as e:
            logger.critical("内存严重不足，无法完成清理！")
        except IOError as e:
            logger.error(f"文件清理失败: {str(e)}")
        except Exception as e:
            logger.error(f"未预期的清理错误: {str(e)}", exc_info=True)

    def _clean_temp_files(self) -> None:
        """清理临时文件"""
        try:
            temp_dir = os.path.join(core_manager.BASE_DIR, 'temp')
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                os.makedirs(temp_dir, exist_ok=True)
        except Exception as e:
            logger.warning(f"临时文件清理失败: {str(e)}")

    def _optimize_memory(self):
        """优化内存使用"""
        try:
            # 1. 检查并清理大对象
            for obj in gc.get_objects():
                if hasattr(obj, 'nbytes') and getattr(obj, 'nbytes', 0) > 1e8:  # >100MB
                    del obj
            
            # 2. 执行垃圾回收
            gc.collect()
            
            logger.info("完成内存优化")
            
        except Exception as e:
            logger.error(f"内存优化时出错: {str(e)}")

    def _cleanup_model_cache(self):
        """清理模型缓存"""
        try:
            # 清理Keras后端缓存
            tf.keras.backend.clear_session()
            
            # 清理模型检查点文件
            checkpoint_dir = os.path.join(os.getcwd(), 'checkpoints')
            if os.path.exists(checkpoint_dir):
                for item in os.listdir(checkpoint_dir):
                    if item.endswith('.temp'):
                        os.remove(os.path.join(checkpoint_dir, item))
                        
            # 释放TensorFlow占用的缓存
            gpus = tf.config.list_physical_devices('GPU')
            for gpu in gpus:
                tf.config.experimental.reset_memory_stats(gpu.name)
            
            logger.info("完成模型缓存清理")
            
        except Exception as e:
            logger.error(f"清理模型缓存时出错: {str(e)}")

    def optimize_for_large_data(self):
        """针对大样本的优化策略"""
        # 新增大样本优化策略
        self.enable_memmap = True  # 启用内存映射
        self.chunk_size = 10000    # 分块加载
        tf.keras.backend.set_floatx('float16')  # 压缩精度
        logger.info("已启用大数据优化策略")

    def optimize_for_hardware(self) -> bool:
        """硬件定制优化"""
        try:
            # 1. 限制TensorFlow内存使用
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                tf.config.set_logical_device_configuration(
                    gpus[0],
                    [tf.config.LogicalDeviceConfiguration(memory_limit=1536)]
                )
            
            # 2. 配置CPU并行线程
            tf.config.threading.set_intra_op_parallelism_threads(6)
            tf.config.threading.set_inter_op_parallelism_threads(4)
            
            # 3. 启用内存映射
            self.enable_memmap = True
            logger.info("已完成硬件优化配置")
            return True
        except RuntimeError as e:
            logger.error(f"运行时配置错误: {str(e)}")
            return False
        except ValueError as e:
            logger.error(f"无效的配置参数: {str(e)}")
            return False

# 创建全局实例
date_utils = DateUtils()
memory_manager = MemoryManager()
