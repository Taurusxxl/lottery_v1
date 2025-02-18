#4 Data Management System / 数据管理系统
import os
import numpy as np
import pandas as pd
import logging
import pymysql
import threading
from datetime import datetime, timedelta
from collections import deque, OrderedDict
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from sqlalchemy.pool import QueuePool
from pymysql.cursors import DictCursor
from cell1_core import core_manager
import pickle
import time

# 获取logger实例
logger = logging.getLogger(__name__)

class DataManager:
    """统一数据管理器 - 整合数据库管理和数据管道功能"""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not core_manager.initialized:
            raise RuntimeError("核心配置未初始化！请先运行cell1_core.py")
        
        # 获取配置
        try:
            # 直接从SYSTEM_CONFIG中获取DATA_CONFIG
            data_config = core_manager.SYSTEM_CONFIG['DATA_CONFIG']
        except (KeyError, AttributeError):
            logger.error("数据配置缺失，使用默认值")
            data_config = {
                'cache_size': 10000,
                'min_sequence_length': 14400,
                'normalize_range': (-1, 1)
            }
        
        # 设置数据管理器属性
        self.cache_size = data_config['cache_size']
        self.normalize_range = data_config['normalize_range']
        
        # 初始化数据库配置
        self.DB_CONFIG = core_manager.DB_CONFIG
        
        # 初始化连接池
        self.pool = self._create_pool()
        
        # 初始化数据处理组件
        self.data_pool = DataPool()
        self.data_processor = DataProcessor()
        self.data_validator = DataValidator()
        self.time_feature_extractor = TimeFeatureExtractor()
        
        # 初始化数据缓存
        self.query_cache = {}
        self.cache_timeout = 300  # 5分钟缓存超时
        
        # 初始化数据配置
        self.sequence_length = data_config['min_sequence_length']
        
        # 初始化目录
        self._init_directories()
        
        # 修改数据库连接初始化
        self.connection = None
        self._init_connection()
        
        logger.info("数据管理器初始化完成")

    def _init_db_config(self):
        """初始化数据库配置"""
        db_config = config_instance.get_db_config()
        db_config.update({
            'database': 'admin_data',
            'charset': 'utf8mb4'
        })
        return db_config

    def _init_directories(self):
        """初始化目录结构"""
        self.comparison_dir = os.path.join(core_manager.BASE_DIR, 'comparison')
        os.makedirs(self.comparison_dir, exist_ok=True)
        self.issue_file = os.path.join(self.comparison_dir, 'issue_number.txt')

    def _create_pool(self):
        """创建数据库连接池"""
        try:
            pool = QueuePool(
                creator=lambda: pymysql.connect(**self.DB_CONFIG),
                pool_size=10,
                max_overflow=20,
                timeout=30
            )
            logger.info("数据库连接池创建成功")
            return pool
        except Exception as e:
            logger.error(f"创建数据库连接池失败: {str(e)}")
            raise

    def execute_query(self, query, params=None, retry=3, use_cache=False):
        """执行SQL查询（增加数据量检查）"""
        for attempt in range(retry):
            try:
                # 检查缓存
                if use_cache:
                    cache_key = f"{query}_{str(params)}"
                    cached_result = self._get_from_cache(cache_key)
                    if cached_result is not None:
                        logger.info(f"成功获取数据 {len(cached_result)} 条")
                        return cached_result
                
                # 获取连接和游标
                connection = self.pool.connect()
                try:
                    cursor = connection.cursor(DictCursor)
                    cursor.execute(query, params)
                    result = cursor.fetchall()
                    
                    if use_cache:
                        self._update_cache(cache_key, result)
                    logger.info(f"成功获取数据 {len(result)} 条")
                    return result
                finally:
                    cursor.close()
                    connection.close()
                
            except pymysql.OperationalError as e:
                if attempt < retry - 1:
                    logger.warning(f"数据库连接失败，尝试重连({attempt+1}/{retry})")
                    self.pool = self._create_pool()  # 重建连接池
                    time.sleep(2 ** attempt)
                    continue
                raise
        return None

    def execute_batch(self, query, params_list):
        """批量执行查询"""
        try:
            connection = self.pool.connect()
            cursor = connection.cursor()
            
            try:
                cursor.executemany(query, params_list)
                connection.commit()
                return True
            finally:
                cursor.close()
                connection.close()
                
        except Exception as e:
            logger.error(f"批量执行查询失败: {str(e)}")
            return False

    def get_records_by_issue(self, start_issue, limit):
        """按期号范围获取记录"""
        query = f"""
            SELECT * FROM admin_tab 
            WHERE date_period >= %s
            ORDER BY date_period ASC
            LIMIT %s
        """
        return self.execute_query(query, (start_issue, limit))

    def close_all(self):
        """关闭所有数据库连接"""
        if self.pool:
            self.pool.dispose()
            logger.info("已关闭所有数据库连接")

    def get_data_stats(self):
        """获取数据统计信息"""
        try:
            with self.lock:
                return {
                    'total_samples': len(self.data_pool.data),
                    'cache_size': self.data_pool.get_cache_size(),
                    'last_update': self.data_pool.last_update_time,
                    'memory_usage': self._get_memory_usage(),
                    'database_connections': self.pool.size()
                }
        except Exception as e:
            logger.error(f"获取数据统计信息时出错: {str(e)}")
            return None

    def _get_memory_usage(self):
        """获取内存使用情况"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)  # 转换为MB
        except:
            return None

    def get_training_batch(self, batch_size=None):
        """获取训练批次"""
        try:
            batch_size = batch_size or self.batch_size
            with self._lock:
                # 获取原始数据
                data = self.data_pool.get_latest_data()
                # 处理数据
                processed = self.data_processor.process_records(data)
                # 验证数据
                if not self.data_validator.validate(processed):
                    raise ValueError("数据验证失败")
                return processed
        except Exception as e:
            logger.error(f"获取训练批次时出错: {str(e)}")
            return None

    def update_data(self):
        """更新数据"""
        try:
            new_data = self._fetch_new_data()
            if new_data is not None:
                with self._lock:
                    self.data_pool.update_data(new_data)
                logger.info(f"数据更新成功，当前数据量: {len(self.data_pool.data)}")
                return True
            return False
        except Exception as e:
            logger.error(f"更新数据时出错: {str(e)}")
            return False

    def _fetch_new_data(self):
        """获取新数据"""
        try:
            with self.issue_lock:
                # 读取最后一期期号
                with open(self.issue_file, 'r+') as f:
                    last_issue = f.read().strip()
                    
                    # 获取总数据量
                    total = config_instance.SYSTEM_CONFIG['SAMPLE_CONFIG']['total_fetch']()
                    
                    # 构建查询
                    query = f"""
                        SELECT date_period, number 
                        FROM admin_tab 
                        WHERE date_period {'>' if last_issue else ''}= '{last_issue}'
                        ORDER BY date_period 
                        LIMIT {total}
                    """
                    
                    records = self.execute_query(query)
                    
                    # 验证数据连续性
                    if not self._validate_sequence(records):
                        raise ValueError("数据存在断层")
                    
                    # 更新期号文件
                    if records:
                        new_last = records[-1]['date_period']
                        f.seek(0)
                        f.write(new_last)
                        f.truncate()
                    
                    return self._process_numbers(records)
                    
        except Exception as e:
            logger.error(f"获取新数据失败: {str(e)}")
            return None

    def _validate_sequence(self, records):
        """验证数据序列连续性"""
        if not records or len(records) < 2:
            return True
            
        max_gap = config_instance.SYSTEM_CONFIG['max_sequence_gap']
        
        for i in range(1, len(records)):
            current = records[i]['date_period']
            previous = records[i-1]['date_period']
            if not self._is_consecutive_periods(previous, current, max_gap):
                return False
        return True

    def _is_consecutive_periods(self, prev_period, curr_period, max_gap):
        """检查两个期号是否连续"""
        try:
            prev_date, prev_num = prev_period.split('-')
            curr_date, curr_num = curr_period.split('-')
            
            prev_dt = datetime.strptime(prev_date, '%Y%m%d')
            curr_dt = datetime.strptime(curr_date, '%Y%m%d')
            
            if prev_dt == curr_dt:
                return int(curr_num) - int(prev_num) <= max_gap
            
            if curr_dt - prev_dt == timedelta(days=1):
                return int(prev_num) == 1440 and int(curr_num) == 1
                
            return False
            
        except Exception as e:
            logger.error(f"检查期号连续性时出错: {str(e)}")
            return False

    def _process_numbers(self, records):
        """处理数字号码"""
        try:
            processed = []
            for r in records:
                numbers = [int(d) for d in r['number'].zfill(5)]
                processed.append({
                    'date_period': r['date_period'],
                    'numbers': numbers,
                    'time_features': self.time_feature_extractor.extract_features(r['date_period'])
                })
            return processed
        except Exception as e:
            logger.error(f"处理号码时出错: {str(e)}")
            return None

    def _get_from_cache(self, key):
        """从缓存获取数据"""
        if key in self.query_cache:
            timestamp, data = self.query_cache[key]
            if datetime.now() - timestamp < timedelta(seconds(self.cache_timeout)):
                return data
            del self.query_cache[key]
        return None

    def _update_cache(self, key, data):
        """更新缓存"""
        self.query_cache[key] = (datetime.now(), data)

    def clear_cache(self):
        """清空所有数据缓存"""
        with self.lock:  # 使用锁保证线程安全
            self.query_cache.clear()
            self.data_pool.data.clear()
            logger.info("已清空所有数据缓存")

    def get_data_by_date_range(self, start_date, end_date):
        """按日期范围获取数据"""
        try:
            query = """
                SELECT * FROM admin_tab 
                WHERE DATE(SUBSTRING_INDEX(date_period, '-', 1)) 
                BETWEEN %s AND %s
                ORDER BY date_period
            """
            return self.execute_query(query, (start_date, end_date))
        except Exception as e:
            logger.error(f"获取日期范围数据失败: {str(e)}")
            return None

    def check_data_continuity(self, data):
        """检查数据连续性"""
        try:
            if not data or len(data) < 2:
                return True
                
            periods = [d['date_period'] for d in data]
            for i in range(1, len(periods)):
                curr_period = periods[i]
                prev_period = periods[i-1]
                
                if not self._is_consecutive_periods(prev_period, curr_period):
                    logger.warning(f"数据不连续: {prev_period} -> {curr_period}")
                    return False
            return True
            
        except Exception as e:
            logger.error(f"检查数据连续性失败: {str(e)}")
            return False

    def generate_test_batch(self, size=1000):
        """生成测试批次"""
        try:
            with self._lock:
                latest_data = self.data_pool.get_latest_data(size)
                if not latest_data:
                    return None
                    
                test_data = self.data_processor.process_records(latest_data)
                if not self.data_validator.validate(test_data):
                    raise ValueError("测试数据验证失败")
                    
                return test_data
                
        except Exception as e:
            logger.error(f"生成测试批次失败: {str(e)}")
            return None

    def save_cache_to_disk(self, cache_path=None):
        """保存缓存到磁盘"""
        try:
            if cache_path is None:
                cache_path = os.path.join(config_instance.BASE_DIR, 'cache', 'data_cache.pkl')
                
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            
            cache_data = {
                'pool_data': self.data_pool.data,
                'query_cache': self.query_cache,
                'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
                
            logger.info(f"缓存已保存到: {cache_path}")
            return True
            
        except Exception as e:
            logger.error(f"保存缓存失败: {str(e)}")
            return False

    def load_training_samples(self, limit=1000):
        """加载训练样本"""
        try:
            # 修正字段名
            query = "SELECT number, date_period FROM admin_tab ORDER BY date_period DESC LIMIT %s"
            raw_data = self.execute_query(query, (limit,))
            
            if not raw_data:
                logger.error("未查询到任何数据")
                return None
                
            processed = self.data_processor.process_training_data(raw_data)
            if processed is None:
                return None
                
            sequences = self._create_sequences(processed, self.sequence_length)
            if sequences is None or len(sequences) == 0:
                logger.error("无法创建有效序列，可能原因：\n"
                             "1. 输入数据长度不足\n"
                             f"2. 序列长度设置过大（当前设置：{self.sequence_length}）\n"
                             "3. 数据预处理失败")
                return None
                
            logger.info(f"成功加载 {len(sequences)} 个训练样本")
            return sequences
            
        except Exception as e:
            logger.error(f"加载训练样本失败: {str(e)}")
            return None

    def _create_sequences(self, data, seq_length):
        """创建时间序列数据"""
        try:
            sequences = []
            for i in range(len(data) - seq_length + 1):  # 修正循环范围
                seq = data[i:i+seq_length]
                sequences.append(seq)
            return np.array(sequences)
        except Exception as e:
            logger.error(f"创建序列失败: {str(e)}")
            return None

    def process_training_data(self, raw_data):
        """处理原始训练数据"""
        try:
            if not raw_data:
                logger.error("原始数据为空")
                return None
                
            processed = []
            for idx, record in enumerate(raw_data):
                try:
                    # 验证必要字段存在
                    if 'number' not in record or 'date_period' not in record:
                        logger.warning(f"记录{idx}缺失必要字段，已跳过")
                        continue
                        
                    # 处理数字格式：将字符串拆分为单个字符
                    number_str = str(record['number']).strip()
                    if len(number_str) != 5:
                        logger.warning(f"记录{idx}号码长度错误: {number_str}")
                        continue
                        
                    numbers = [int(c) for c in number_str if c.isdigit()]
                    if len(numbers) != 5:
                        logger.warning(f"记录{idx}包含非数字字符: {number_str}")
                        continue
                        
                    time_feat = self._parse_time_features(str(record['date_period']))
                    processed.append(numbers + time_feat)
                    
                except Exception as e:
                    logger.warning(f"处理记录{idx}时出错: {str(e)}，已跳过该记录")

            if not processed:
                logger.error("无有效数据可处理")
                return None

            scaler = MinMaxScaler()
            return scaler.fit_transform(processed)
            
        except Exception as e:
            logger.error(f"数据处理失败: {str(e)}")
            return None

    def _parse_time_features(self, date_period):
        """解析时间特征"""
        date_str, period = date_period.split('-')
        dt = datetime.strptime(date_str, "%Y%m%d")
        return [
            dt.hour/24.0,  # 小时归一化
            dt.weekday()/7.0,  # 星期归一化
            int(period)/1440.0  # 期号归一化
        ]

    def check_connection(self):
        """检查数据库连接状态"""
        try:
            result = self.execute_query("SELECT 1", retry=1)
            return bool(result)
        except Exception as e:
            logger.error(f"数据库连接检查失败: {str(e)}")
            return False

    def get_sequence(self, start_issue, end_issue):
        """获取指定期号范围的序列数据"""
        try:
            # 确保连接有效
            if not self.connection or not self.connection.open:
                self._init_connection()
            
            # 使用新的连接执行查询
            with self.connection.cursor() as cursor:
                query = """
                    SELECT number, date_period 
                    FROM admin_tab 
                    WHERE date_period >= %s AND date_period <= %s
                    ORDER BY date_period ASC
                """
                cursor.execute(query, (start_issue, end_issue))
                results = cursor.fetchall()
                
                if not results:
                    logger.warning(f"未找到期号范围 {start_issue} 到 {end_issue} 的数据")
                    return None
                    
                # 转换数据格式
                sequence = []
                for row in results:
                    numbers = [int(n) for n in str(row['number']).zfill(5)]
                    normalized = [(n - 4.5) / 4.5 for n in numbers]
                    sequence.append(normalized)
                    
                sequence = np.array(sequence, dtype=np.float32)
                logger.info(f"获取到序列数据 {len(sequence)} 条")
                return sequence
                
        except pymysql.Error as e:
            logger.error(f"数据库操作失败: {str(e)}")
            self._init_connection()  # 尝试重新连接
            return None
        except Exception as e:
            logger.error(f"获取序列数据失败: {str(e)}")
            return None

    def _init_connection(self):
        """初始化数据库连接"""
        try:
            if self.connection and self.connection.open:
                self.connection.close()
                
            # 重新创建连接
            self.connection = pymysql.connect(
                host=self.DB_CONFIG['host'],
                port=self.DB_CONFIG['port'],
                user=self.DB_CONFIG['user'],
                password=self.DB_CONFIG['password'],
                database=self.DB_CONFIG['database'],
                charset=self.DB_CONFIG['charset'],
                cursorclass=pymysql.cursors.DictCursor,
                autocommit=True,
                connect_timeout=60
            )
            logger.info("数据库连接初始化成功")
        except Exception as e:
            logger.error(f"数据库连接初始化失败: {str(e)}")
            self.connection = None
            raise

class DataPool:
    """数据池 - 负责数据缓存和管理"""
    def __init__(self, max_size=10000):
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
                self.scaler.fit(np.array([d['numbers'] for d in self.data]))
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
        """添加数据批次"""
        # 添加对齐检查
        aligned = self._align_sequences(batch)
        with self.lock:
            self.data.extend(aligned)
        
    def _align_sequences(self, batch):
        """对齐序列长度"""
        max_len = max(len(item['input']) for item in batch)
        aligned = []
        for item in batch:
            aligned_item = item.copy()
            aligned_item['input'] = np.pad(
                item['input'], 
                (0, max_len - len(item['input'])),
                'constant'
            )
            aligned.append(aligned_item)
        return aligned

    def get_training_data(self, sequence_length):
        """获取训练数据"""
        with self.lock:
            if len(self.data) < sequence_length:
                return None
            data = np.array([d['numbers'] for d in self.data])
            if self.is_scaler_fitted:
                data = self.scaler.transform(data)
            return data

    def get_data_window(self, start_idx, window_size):
        """获取指定窗口的数据"""
        with self.lock:
            if start_idx + window_size > len(self.data):
                return None
            return self.data[start_idx:start_idx + window_size]

    def get_latest_periods(self, n_periods):
        """获取最近n期数据"""
        with self.lock:
            return self.data[-n_periods:] if len(self.data) >= n_periods else None

    def preload_data(self, start_date, end_date):
        """预加载指定日期范围的数据"""
        try:
            query = """
                SELECT * FROM admin_tab 
                WHERE DATE(SUBSTRING_INDEX(date_period, '-', 1)) 
                BETWEEN %s AND %s
                ORDER BY date_period
            """
            records = data_manager.execute_query(query, (start_date, end_date))
            
            if records:
                self.update_data(records)
                return True
            return False
            
        except Exception as e:
            logger.error(f"预加载数据失败: {str(e)}")
            return False

class DataProcessor:
    """数据处理器 - 负责数据预处理和批次生成"""
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.lock = threading.Lock()
        # 添加时间特征提取器
        self.time_feature_extractor = TimeFeatureExtractor()
    
    def process_records(self, records):
        """处理数据记录"""
        try:
            # 1. 清理数据
            cleaned = self._remove_invalid_data(records)
            
            # 2. 特征工程
            features = self._extract_features(cleaned)
            
            # 3. 数据标准化
            normalized = self._normalize_features(features)
            
            # 4. 序列化处理
            sequences = self._create_sequences(normalized)
            
            return sequences
            
        except Exception as e:
            logger.error(f"处理数据记录时出错: {str(e)}")
            return None
    
    def _remove_invalid_data(self, records):
        """移除无效数据"""
        return [r for r in records if self._is_valid_record(r)]
    
    def _is_valid_record(self, record):
        """检查记录是否有效"""
        try:
            numbers = record['numbers']
            return (
                isinstance(numbers, list) and
                len(numbers) == 5 and
                all(isinstance(n, int) and 0 <= n <= 9 for n in numbers)
            )
        except:
            return False
    
    def _extract_features(self, records):
        """提取特征"""
        features = []
        for record in records:
            # 基础数字特征
            number_features = np.array(record['numbers'])
            # 时间特征
            time_features = self.time_feature_extractor.extract_features(record['date_period'])
            # 合并特征
            combined = np.concatenate([number_features, time_features])
            features.append(combined)
        return np.array(features)
    
    def _normalize_features(self, features):
        """标准化特征"""
        with self.lock:
            return self.scaler.fit_transform(features)
    
    def _create_sequences(self, data):
        """创建序列数据"""
        sequences = []
        for i in range(len(data) - config_instance.SYSTEM_CONFIG['SAMPLE_CONFIG']['total_fetch']()):
            seq = {
                'input': data[i:i+config_instance.SYSTEM_CONFIG['SAMPLE_CONFIG']['input_length']],
                'target': data[i+config_instance.SYSTEM_CONFIG['SAMPLE_CONFIG']['input_length']:
                              i+config_instance.SYSTEM_CONFIG['SAMPLE_CONFIG']['total_fetch']()]
            }
            sequences.append(seq)
        return sequences

    def process_with_time_features(self, data):
        """处理数据并添加时间特征"""
        try:
            processed = []
            for record in data:
                # 基础特征
                features = self._extract_base_features(record)
                # 时间特征
                time_features = self.time_feature_extractor.extract_features(record['date_period'])
                # 组合特征
                combined = np.concatenate([features, time_features])
                processed.append(combined)
            return np.array(processed)
        except Exception as e:
            logger.error(f"处理时间特征时出错: {str(e)}")
            return None

    def _extract_base_features(self, record):
        """提取基础特征"""
        try:
            numbers = np.array(record['numbers'])
            # 添加统计特征
            stats = [
                np.mean(numbers),
                np.std(numbers),
                np.max(numbers),
                np.min(numbers)
            ]
            return np.concatenate([numbers, stats])
        except Exception as e:
            logger.error(f"提取基础特征时出错: {str(e)}")
            return None

    def apply_feature_scaling(self, data, feature_range=(-1, 1)):
        """应用特征缩放"""
        try:
            self.scaler = MinMaxScaler(feature_range=feature_range)
            return self.scaler.fit_transform(data)
        except Exception as e:
            logger.error(f"特征缩放失败: {str(e)}")
            return None

    def create_sliding_windows(self, data, window_size, stride=1):
        """创建滑动窗口数据"""
        try:
            windows = []
            for i in range(0, len(data) - window_size + 1, stride):
                windows.append(data[i:i + window_size])
            return np.array(windows)
        except Exception as e:
            logger.error(f"创建滑动窗口失败: {str(e)}")
            return None

    def batch_normalize(self, batches):
        """批量数据标准化"""
        try:
            normalized_batches = []
            for batch in batches:
                normalized = self._normalize_features(batch)
                normalized_batches.append(normalized)
            return normalized_batches
            
        except Exception as e:
            logger.error(f"批量标准化失败: {str(e)}")
            return None

    def process_training_data(self, raw_data):
        """处理原始训练数据"""
        try:
            if not raw_data:
                logger.error("原始数据为空")
                return None
                
            processed = []
            for idx, record in enumerate(raw_data):
                try:
                    # 验证必要字段存在
                    if 'number' not in record or 'date_period' not in record:
                        logger.warning(f"记录{idx}缺失必要字段，已跳过")
                        continue
                        
                    # 处理数字格式：将字符串拆分为单个字符
                    number_str = str(record['number']).strip()
                    if len(number_str) != 5:
                        logger.warning(f"记录{idx}号码长度错误: {number_str}")
                        continue
                        
                    numbers = [int(c) for c in number_str if c.isdigit()]
                    if len(numbers) != 5:
                        logger.warning(f"记录{idx}包含非数字字符: {number_str}")
                        continue
                        
                    time_feat = self._parse_time_features(str(record['date_period']))
                    processed.append(numbers + time_feat)
                    
                except Exception as e:
                    logger.warning(f"处理记录{idx}时出错: {str(e)}，已跳过该记录")

            if not processed:
                logger.error("无有效数据可处理")
                return None

            scaler = MinMaxScaler()
            return scaler.fit_transform(processed)
            
        except Exception as e:
            logger.error(f"数据处理失败: {str(e)}")
            return None

    def _parse_time_features(self, date_period):
        """解析时间特征"""
        date_str, period = date_period.split('-')
        dt = datetime.strptime(date_str, "%Y%m%d")
        return [
            dt.hour/24.0,  # 小时归一化
            dt.weekday()/7.0,  # 星期归一化
            int(period)/1440.0  # 期号归一化
        ]

class DataValidator:
    """数据验证器 - 负责数据有效性检查"""
    def __init__(self):
        self.lock = threading.Lock()
        self.validation_rules = {
            'sequence_length': self._check_sequence_length,
            'number_range': self._check_number_range,
            'time_continuity': self._check_time_continuity,
            'feature_completeness': self._check_feature_completeness
        }
    
    def validate(self, data):
        """验证数据有效性"""
        try:
            with self.lock:
                return all(
                    rule(data) for rule in self.validation_rules.values()
                )
        except Exception as e:
            logger.error(f"数据验证时出错: {str(e)}")
            return False
    
    def _check_sequence_length(self, data):
        """检查序列长度"""
        required_length = config_instance.SYSTEM_CONFIG['SAMPLE_CONFIG']['total_fetch']()
        return len(data) >= required_length
    
    def _check_number_range(self, data):
        """检查数字范围"""
        try:
            numbers = np.array([d['numbers'] for d in data])
            return np.all((numbers >= 0) & (numbers <= 9))
        except:
            return False
    
    def _check_time_continuity(self, data):
        """检查时间连续性"""
        try:
            periods = [d['date_period'] for d in data]
            for i in range(1, len(periods)):
                if not self._is_consecutive_periods(periods[i-1], periods[i]):
                    return False
            return True
        except:
            return False
    
    def _check_feature_completeness(self, data):
        """检查特征完整性"""
        try:
            return all(
                'numbers' in d and 'time_features' in d 
                for d in data
            )
        except:
            return False
    
    def _is_consecutive_periods(self, prev, curr):
        """检查期号是否连续"""
        try:
            p_date, p_num = prev.split('-')
            c_date, c_num = curr.split('-')
            
            if p_date == c_date:
                return int(c_num) - int(p_num) == 1
            
            p_dt = datetime.strptime(p_date, '%Y%m%d')
            c_dt = datetime.strptime(c_date, '%Y%m%d')
            
            return (c_dt - p_dt).days == 1 and int(p_num) == 1440 and int(c_num) == 1
            
        except:
            return False

    def validate_data_completeness(self, data):
        """验证数据完整性"""
        try:
            # 检查数据结构
            if not isinstance(data, (list, np.ndarray)):
                return False
                
            # 检查数据量
            if len(data) < config_instance.SYSTEM_CONFIG['SAMPLE_CONFIG']['input_length']:
                return False
                
            # 检查每条记录的完整性
            for record in data:
                if not self._check_record_completeness(record):
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"验证数据完整性时出错: {str(e)}")
            return False
    
    def _check_record_completeness(self, record):
        """检查单条记录的完整性"""
        required_fields = ['date_period', 'numbers', 'time_features']
        return all(field in record for field in required_fields)

    def validate_batch_structure(self, batch):
        """验证批次数据结构"""
        try:
            if not isinstance(batch, dict):
                return False
                
            required_keys = ['input', 'target']
            if not all(key in batch for key in required_keys):
                return False
                
            input_shape = batch['input'].shape
            target_shape = batch['target'].shape
            
            if len(input_shape) != 3 or len(target_shape) != 3:
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"验证批次结构失败: {str(e)}")
            return False

    def _check_number_format(self, number_str):
        """验证号码格式是否正确"""
        try:
            parts = number_str.split(',')
            if len(parts) != 5:
                return False
            return all(n.strip().isdigit() and 0 <= int(n) <= 9 for n in parts)
        except:
            return False

    def process_training_data(self, raw_data):
        """处理原始训练数据"""
        try:
            if not raw_data:
                logger.error("原始数据为空")
                return None
                
            processed = []
            for idx, record in enumerate(raw_data):
                try:
                    # 验证必要字段存在
                    if 'number' not in record or 'date_period' not in record:
                        logger.warning(f"记录{idx}缺失必要字段，已跳过")
                        continue
                        
                    # 处理数字格式：将字符串拆分为单个字符
                    number_str = str(record['number']).strip()
                    if len(number_str) != 5:
                        logger.warning(f"记录{idx}号码长度错误: {number_str}")
                        continue
                        
                    numbers = [int(c) for c in number_str if c.isdigit()]
                    if len(numbers) != 5:
                        logger.warning(f"记录{idx}包含非数字字符: {number_str}")
                        continue
                        
                    time_feat = self._parse_time_features(str(record['date_period']))
                    processed.append(numbers + time_feat)
                    
                    if not self._check_number_format(str(record['number'])):
                        logger.warning(f"记录{idx}号码格式错误: {record['number']}")
                        continue
                    
                except Exception as e:
                    logger.warning(f"处理记录{idx}时出错: {str(e)}，已跳过该记录")

            if not processed:
                logger.error("无有效数据可处理")
                return None

            scaler = MinMaxScaler()
            return scaler.fit_transform(processed)
            
        except Exception as e:
            logger.error(f"数据处理失败: {str(e)}")
            return None

    def _parse_time_features(self, date_period):
        """解析时间特征"""
        date_str, period = date_period.split('-')
        dt = datetime.strptime(date_str, "%Y%m%d")
        return [
            dt.hour/24.0,  # 小时归一化
            dt.weekday()/7.0,  # 星期归一化
            int(period)/1440.0  # 期号归一化
        ]

class TimeFeatureExtractor:
    """时间特征提取器"""
    def __init__(self):
        self.periodic_features = {
            'hour_of_day': (24, lambda dt: dt.hour),
            'minute_of_hour': (60, lambda dt: dt.minute),
            'day_of_week': (7, lambda dt: dt.weekday()),
            'day_of_month': (31, lambda dt: dt.day - 1),
            'month_of_year': (12, lambda dt: dt.month - 1)
        }
    
    def extract_features(self, date_period):
        """提取时间特征"""
        try:
            # 解析日期和期号
            date_str, period = date_period.split('-')
            date = datetime.strptime(date_str, '%Y%m%d')
            period_num = int(period)
            
            features = []
            
            # 添加周期性特征
            for period, func in self.periodic_features.values():
                value = func(date)
                # 转换为sin和cos特征以保持周期性
                sin_value = np.sin(2 * np.pi * value / period)
                cos_value = np.cos(2 * np.pi * value / period)
                features.extend([sin_value, cos_value])
            
            # 添加期号特征
            period_feature = (period_num - 1) / 1440  # 归一化到0-1
            features.append(period_feature)
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"提取时间特征时出错: {str(e)}")
            return np.zeros(len(self.periodic_features) * 2 + 1)  # 返回全零特征

class EnhancedDataManager(DataManager):
    """增强型数据管理器，支持流式数据加载"""
    def __init__(self):
        super().__init__()
        self.data_buffer = deque(maxlen=14400*2)  # 双倍缓冲
        self.last_processed = None
        
    def stream_training_samples(self):
        """实时数据流生成器"""
        while True:
            # 获取最新14400+2880期数据
            latest = self.execute_query(
                "SELECT number, date_period FROM admin_tab "
                "ORDER BY date_period DESC LIMIT 17280"
            )
            if latest and latest != self.last_processed:
                processed = self.data_processor.process_streaming_data(latest)
                if processed is not None:
                    sequences = self._create_sequences(processed, 14400)
                    if sequences:
                        yield sequences[0]  # 取最新序列
                        self.last_processed = latest
                time.sleep(58)  # 每58秒检查一次

# 创建全局实例
data_manager = DataManager()
