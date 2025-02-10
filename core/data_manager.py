# 数据管道构建\data_manager.py
import os
import numpy as np
import pandas as pd
import logging
import pymysql
from datetime import datetime, timedelta
import threading
from collections import deque
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from core.database_manager import db_manager  # 导入数据库管理器实例
from core.config_manager import config_instance  # 导入配置管理器实例

# 获取logger实例
logger = logging.getLogger(__name__)

class DataManager:
    """数据管理器 - 单例模式"""
    _instance = None
    
    def __new__(cls, db_config=None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, db_config=None):
        if not hasattr(self, 'initialized'):
            # 初始化组件
            self.data_pool = DataPool()
            self.data_processor = DataProcessor()
            self.data_validator = DataValidator()
            
            # 数据缓存参数
            self.cache_size = config_instance.SYSTEM_CONFIG['DATA_CONFIG']['cache_size']
            self.batch_size = 32
            self.sequence_length = 14400
            
            # 线程锁
            self.lock = threading.Lock()
            
            # 新增配置项
            self.comparison_dir = os.path.join(config_instance.BASE_DIR, 'comparison')
            os.makedirs(self.comparison_dir, exist_ok=True)
            self.issue_file = os.path.join(self.comparison_dir, 'issue_number.txt')
            
            # 标记初始化完成
            self.initialized = True
            logger.info("数据管理器初始化完成")
            
            # 在DataManager中增加锁
            self.issue_lock = threading.Lock()
            
            # 获取配置参数
            self.normalize_range = config_instance.SYSTEM_CONFIG['DATA_CONFIG']['normalize_range']
    
    def get_training_batch(self, batch_size=None):
        """获取训练批次"""
        try:
            batch_size = batch_size or self.batch_size
            with self.lock:
                return self.data_processor.get_training_batch(
                    self.data_pool.get_latest_data(),
                    batch_size
                )
        except Exception as e:
            logger.error(f"获取训练批次时出错: {str(e)}")
            return None, None
    
    def update_data(self):
        """更新数据"""
        try:
            new_data = self._fetch_new_data()
            if new_data is not None:
                with self.lock:
                    self.data_pool.update_data(new_data)
                logger.info(f"数据更新成功，当前数据量: {len(self.data_pool.data)}")
                return True
            return False
        except Exception as e:
            logger.error(f"更新数据时出错: {str(e)}")
            return False
    
    def _get_next_issue(self, current_issue):
        """计算下一期号"""
        date_str, period = current_issue.split('-')
        date = datetime.strptime(date_str, '%Y%m%d')
        period = int(period)
        
        if period == 1440:
            next_date = date + timedelta(days=1)
            next_period = 1
        else:
            next_date = date
            next_period = period + 1
            
        return f"{next_date.strftime('%Y%m%d')}-{next_period:04d}"

    def _fetch_new_data(self):
        """根据144000+2880的数据需求获取样本"""
        try:
            with self.issue_lock:
                with open(self.issue_file, 'r+') as f:
                    last_issue = f.read().strip()
                    
                    # 使用配置获取总数
                    total = config_instance.SYSTEM_CONFIG['SAMPLE_CONFIG']['total_fetch']()
                    
                    # 构建精确查询
                    query = f"""
                        SELECT date_period, number 
                        FROM admin_tab 
                        WHERE date_period {'>' if last_issue else ''}= '{last_issue}'
                        ORDER BY date_period 
                        LIMIT {total}
                    """
                    
                    records = db_manager.execute_query(query)
                    
                    # 验证数据连续性
                    if not self._validate_sequence(records):
                        raise ValueError("数据存在断层")
                    
                    # 更新期号文件
                    new_last = records[-1]['date_period']
                    f.seek(0)
                    f.write(new_last)
                    
                    return self._process_numbers(records)  # 处理五位号码
                    
        except Exception as e:
            logger.error(f"数据获取失败: {str(e)}")
            return None

    def _process_numbers(self, records):
        """处理五位数字号码"""
        processed = []
        for r in records:
            # 将"00236"转换为[0,0,2,3,6]
            numbers = [int(d) for d in r['number'].zfill(5)]
            processed.append({
                'date_period': r['date_period'],
                'numbers': numbers,
                'time_features': self.time_feature_extractor.extract_features(r['date_period'])
            })
        return processed

    def validate_data(self, data):
        """验证数据有效性"""
        return self.data_validator.validate(data)
    
    def get_data_stats(self):
        """获取数据统计信息"""
        try:
            with self.lock:
                return {
                    'total_samples': len(self.data_pool.data),
                    'cache_size': self.data_pool.get_cache_size(),
                    'last_update': self.data_pool.last_update_time
                }
        except Exception as e:
            logger.error(f"获取数据统计信息时出错: {str(e)}")
            return None

    def get_data_by_period(self, start_period, end_period):
        """获取指定期间的数据"""
        try:
            query = """
                SELECT date_period, number 
                FROM admin_tab 
                WHERE date_period BETWEEN %s AND %s
                ORDER BY date_period
            """
            records = db_manager.execute_query(
                query, 
                params=(start_period, end_period),
                use_cache=True
            )
            
            if not records:
                return None
                
            return self.data_processor.process_records(records)
            
        except Exception as e:
            logger.error(f"获取期间数据时出错: {str(e)}")
            return None

    def clear_data_cache(self):
        """清理数据缓存"""
        try:
            with self.lock:
                self.data_pool.clear_cache()
                db_manager.clear_cache()
            logger.info("数据缓存已清理")
        except Exception as e:
            logger.error(f"清理数据缓存时出错: {str(e)}")

    def get_training_sample(self):
        """获取完整训练样本"""
        # 修改为返回单一样本
        sample = self.data_pool.get_data('latest')
        if self._validate_sample(sample):
            return sample
        return None

    def _validate_sample(self, sample):
        """验证样本完整性"""
        cfg = config_instance.SYSTEM_CONFIG['SAMPLE_CONFIG']
        return (
            len(sample['input']) == cfg['input_length'] and 
            len(sample['target']) == cfg['target_length']
        )

class DataPool:
    """数据池 - 负责数据缓存和管理"""
    def __init__(self, max_size=10000):
        from collections import OrderedDict
        self.data = []
        self.cache = OrderedDict()
        self.max_size = max_size
        self.lock = threading.Lock()
        self.last_update_time = None
        
        # 初始化数据缩放器
        self.scaler = MinMaxScaler()
        self.is_scaler_fitted = False
    
    def update_data(self, new_data):
        """更新数据"""
        with self.lock:
            self.data.extend(new_data)
            self.last_update_time = datetime.now()
            
            # 如果还没有拟合scaler，进行拟合
            if not self.is_scaler_fitted and len(self.data) > 0:
                self.scaler.fit(np.array(self.data))
                self.is_scaler_fitted = True
    
    def get_latest_data(self, n=1000):
        """获取最新的n条数据"""
        with self.lock:
            return self.data[-n:] if self.data else []
    
    def get_cache_size(self):
        """获取缓存大小"""
        return len(self.cache)

    def clear_cache(self):
        """清理数据缓存"""
        with self.lock:
            self.cache.clear()

    def add_batch(self, batch):
        # 添加对齐检查
        aligned = self._align_sequences(batch)
        self.data.extend(aligned)
        
    def _align_sequences(self, batch):
        max_len = max(len(item['input']) for item in batch)
        for item in batch:
            item['input'] = np.pad(item['input'], (0, max_len - len(item['input'])))
        return batch

    def add_data(self, key, data):
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            self.cache[key] = data
            if len(self.cache) > self.max_size:
                self.cache.popitem(last=False)  # LRU淘汰
                
    def get_data(self, version='latest'):
        if version == 'latest':
            return self.cache[next(reversed(self.cache))]
        else:
            return self.cache.get(version, None)

class DataProcessor:
    """数据处理器 - 负责数据预处理和批次生成"""
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.lock = threading.Lock()
        # 添加时间特征提取器
        self.time_feature_extractor = TimeFeatureExtractor()
    
    def process_records(self, records):
        cleaned = self._remove_nan(records)
        normalized = self._normalize(cleaned)
        windowed = self._window_sampling(normalized)
        return windowed

    def _remove_nan(self, data):
        return [d for d in data if not np.isnan(d['numbers']).any()]
        
    def _normalize(self, data):
        numbers = np.array([d['numbers'] for d in data])
        self.scaler.fit(numbers)
        return self.scaler.transform(numbers)
        
    def _window_sampling(self, data):
        cfg = config_instance.SYSTEM_CONFIG['SAMPLE_CONFIG']
        return {
            'input': data[:cfg['input_length']],
            'target': data[cfg['input_length'] : cfg['input_length']+cfg['target_length']]
        }

    def get_training_batch(self, data, batch_size):
        """生成训练批次"""
        try:
            if not data or len(data) < batch_size:
                return None, None
            
            # 随机选择批次
            indices = np.random.choice(len(data), batch_size)
            batch_data = np.array([data[i] for i in indices])
            
            # 分割输入和目标
            X = batch_data[:, :-1]  # 所有特征除了最后一个
            y = batch_data[:, -1]   # 最后一个特征作为目标
            
            return X, y
            
        except Exception as e:
            logger.error(f"生成训练批次时出错: {str(e)}")
            return None, None

class DataValidator:
    """数据验证器 - 负责数据有效性检查"""
    def __init__(self):
        self.lock = threading.Lock()
    
    def validate(self, data):
        """验证数据有效性"""
        return all([
            self._check_nan(data),
            self._check_range(data),
            self._check_sequence(data)
        ])
        
    def _check_range(self, data):
        return np.all((data >= 1) & (data <= 33))
        
    def _check_sequence(self, data):
        diffs = np.diff([d['date_period'] for d in data])
        return np.all(diffs == 600)  # 假设期号间隔10分钟

    def _check_number(self, number_array):
        """验证五位数字是否合法"""
        return np.all((number_array >= 0) & (number_array <= 9))
        
    def _check_sequence_gap(self, diffs):
        """验证期号间隔是否为1期(10分钟)"""
        return np.all(diffs == 1)  # 假设期号连续递增1

class TimeFeatureExtractor:
    """时间特征提取器"""
    def __init__(self):
        self.use_weekday = True
        self.use_month = True
        self.use_day = True
        self.use_period = True
    
    def extract_features(self, date_period):
        """提取时间特征
        Args:
            date_period: 期号字符串 (格式: YYYYMMDD-XXXX)
        Returns:
            features: 时间特征列表
        """
        try:
            # 解析日期和期号
            date_str, period = date_period.split('-')
            date = datetime.strptime(date_str, '%Y%m%d')
            period_num = int(period)
            
            features = []
            
            # 添加月份特征
            if self.use_month:
                features.append(date.month / 12)  # 归一化到0-1
            
            # 添加日期特征    
            if self.use_day:
                features.append(date.day / 31)  # 归一化到0-1
            
            # 添加星期特征
            if self.use_weekday:
                features.append(date.weekday() / 6)  # 归一化到0-1
            
            # 添加期号特征
            if self.use_period:
                features.append((period_num - 1) / 1440)  # 归一化到0-1
            
            return features
            
        except Exception as e:
            logger.error(f"提取时间特征时出错: {str(e)}")
            return [0] * 4  # 返回全零特征

# 创建全局实例
data_manager = DataManager()