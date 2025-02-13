#5 数据库管理器\database_manager.py
import pymysql
import logging
import threading
from pymysql.cursors import DictCursor
from sqlalchemy.pool import QueuePool  # 使用SQLAlchemy自带的连接池
from datetime import datetime, timedelta
from cell1 import ConfigManager  # 修改导入方式

# 获取logger实例
logger = logging.getLogger(__name__)

class DatabaseManager:
    """数据库管理器 - 单例模式"""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, db_config=None):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, db_config=None):
        if not hasattr(self, 'initialized'):
            self.DB_CONFIG = {
                'host': 'localhost',
                'user': 'root',
                'password': 'tt198803',  # 使用提供的密码
                'database': 'admin_data',
                'charset': 'utf8mb4'
            }
            
            # 创建数据库连接池
            self.pool = self._create_pool()
            
            # 初始化查询缓存
            self.query_cache = {}
            self.cache_timeout = 300  # 5分钟缓存超时
            
            # 标记初始化完成
            self.initialized = True
            logger.info("数据库管理器初始化完成")
    
    def _create_pool(self):
        """创建数据库连接池"""
        try:
            self.pool = QueuePool(
                creator=lambda: pymysql.connect(**self.DB_CONFIG),
                pool_size=10,
                max_overflow=20,
                timeout=30
            )
            logger.info("数据库连接池创建成功")
            return self.pool
        except Exception as e:
            logger.error(f"创建数据库连接池失败: {str(e)}")
            raise
    
    def execute_query(self, query, params=None, use_cache=False):
        """执行查询"""
        try:
            # 检查缓存
            if use_cache:
                cache_key = f"{query}_{str(params)}"
                cached_result = self._get_from_cache(cache_key)
                if cached_result is not None:
                    return cached_result
            
            # 获取连接和游标
            connection = self.pool.connect()
            try:
                cursor = connection.cursor(DictCursor)
                # 执行查询
                cursor.execute(query, params)
                result = cursor.fetchall()
                
                # 更新缓存
                if use_cache:
                    self._update_cache(cache_key, result)
                
                return result
            finally:
                cursor.close()
                connection.close()
                
        except Exception as e:
            logger.error(f"执行查询失败: {str(e)}")
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
    
    def _get_from_cache(self, key):
        """从缓存获取数据"""
        if key in self.query_cache:
            timestamp, data = self.query_cache[key]
            if datetime.now() - timestamp < timedelta(seconds=self.cache_timeout):
                return data
            else:
                del self.query_cache[key]
        return None
    
    def _update_cache(self, key, data):
        """更新缓存"""
        self.query_cache[key] = (datetime.now(), data)
    
    def clear_cache(self):
        """清理缓存"""
        self.query_cache.clear()
        logger.info("查询缓存已清理")

    def get_records_by_issue(self, start_issue, limit):
        """按期号范围获取记录"""
        query = f"""
            SELECT * FROM admin_tab 
            WHERE date_period >= %s
            ORDER BY date_period ASC
            LIMIT %s
        """
        return self.execute_query(query, (start_issue, limit))

# 创建全局实例
db_manager = DatabaseManager()