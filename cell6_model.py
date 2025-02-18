#6 Model Core System / 模型核心系统
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import logging
import os
import json
import time
from collections import deque
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MultiHeadAttention, LayerNormalization, Bidirectional, Add
from typing import Dict, Any, Optional
from cell1_core import core_manager
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import math

# 获取logger实例 
logger = logging.getLogger(__name__)

class ModelCore:
    """整合后的模型核心类"""
    
    # 1. 配置管理 (from model_config.py)
    def __init__(self, config_path: Optional[str] = None):
        """初始化模型核心类
        Args:
            config_path: 配置文件路径,如果为None则使用默认配置
        """
        # 从base_model.py继承的属性
        self.sequence_length = 14400
        self.feature_dim = 5
        self.prediction_range = 2880
        self.performance_history = []
        
        # 从model_config.py继承的属性
        self.config_path = config_path
        self.config = self._load_config()
        self.input_shape = core_manager.SYSTEM_CONFIG['SAMPLE_CONFIG']['input_length']
        
        # 添加预测相关的属性
        self.prediction_history = deque(maxlen=1000)  # 预测历史记录
        self.prediction_cache = {}  # 预测结果缓存
        self.cache_timeout = 300   # 缓存超时时间(秒)
        self.confidence_threshold = 0.8  # 置信度阈值

        # 确保在初始化时创建新图
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.default_config = {
                'filters': 64,
                'units': 128,
                'dense_units': 64
            }
            self.models = [self._build_model(self.default_config) for _ in range(6)]
        self.session = tf.compat.v1.Session(graph=self.graph)
        tf.compat.v1.keras.backend.set_session(self.session)
        
        # 初始化优化器
        self.optimizer = self._create_optimizer()
        
        logger.info("模型核心类初始化完成")

    def _load_config(self) -> Dict[str, Any]:
        """从model_config.py继承的配置加载方法"""
        try:
            if self.config_path and os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                logger.info(f"从{self.config_path}加载配置")
                return config
            
            # 使用默认配置
            return self._get_default_config()
            
        except Exception as e:
            logger.error(f"加载配置失败: {str(e)}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'optimizer_config': {
                'learning_rate': 0.001,
                'beta_1': 0.9,
                'beta_2': 0.999
            },
            'architecture_config': {
                'model_1': {'filters': 64, 'units': 128, 'dense_units': 64}
            }
        }

    def get_model_config(self, model_index: int) -> Dict[str, Any]:
        """获取指定模型配置 (from model_config.py)"""
        try:
            return self.config['architecture_config'][f'model_{model_index}']
        except KeyError:
            logger.error(f"未找到模型{model_index}的配置")
            return {}

    def validate_config(self) -> bool:
        """验证配置有效性 (from model_config.py)"""
        try:
            # 验证基础配置
            base_config = self.config['base_config']
            assert base_config['sequence_length'] > 0
            assert base_config['feature_dim'] > 0
            assert base_config['batch_size'] > 0
            
            # 验证优化器配置
            optimizer_config = self.config['optimizer_config']
            assert optimizer_config['learning_rate'] > 0
            assert 0 < optimizer_config['beta_1'] < 1
            assert 0 < optimizer_config['beta_2'] < 1
            
            # 验证模型架构配置
            for model_name, model_config in self.config['architecture_config'].items():
                assert all(value > 0 for value in model_config.values() if isinstance(value, (int, float)))
            
            return True
        except Exception as e:
            logger.error(f"配置验证失败: {str(e)}")
            return False

    def update_config(self, new_config: Dict[str, Any]) -> None:
        """更新配置 (from model_config.py)"""
        try:
            self.config.update(new_config)
            self._save_config()
            logger.info("配置更新成功")
        except Exception as e:
            logger.error(f"更新配置失败: {str(e)}")

    def _save_config(self) -> None:
        """保存配置到文件 (from model_config.py)"""
        try:
            if self.config_path:
                with open(self.config_path, 'w', encoding='utf-8') as f:
                    json.dump(self.config, f, indent=4)
                logger.info(f"配置已保存到{self.config_path}")
        except Exception as e:
            logger.error(f"保存配置失败: {str(e)}")

    def get_optimizer_config(self) -> Dict[str, Any]:
        """获取优化器配置"""
        return self.config['optimizer_config']
    
    def get_loss_config(self) -> Dict[str, Any]:
        """获取损失函数配置"""
        return self.config['loss_config']
    
    def get_training_config(self) -> Dict[str, Any]:
        """获取训练配置"""
        return self.config['training_config']
    
    def get_ensemble_config(self) -> Dict[str, Any]:
        """获取集成配置"""
        return self.config['ensemble_config']
    
    def get_monitor_config(self) -> Dict[str, Any]:
        """获取监控配置"""
        return self.config['monitor_config']

    def _validate_model_architecture(self, architecture: Dict[str, Any]) -> bool:
        """验证模型架构配置"""
        try:
            # 检查必需的层配置
            required_layers = ['lstm_units', 'attention_heads', 'dense_units']
            for layer in required_layers:
                assert any(layer in model for model in architecture.values())
            
            # 检查层参数范围
            for model_config in architecture.values():
                assert 32 <= model_config.get('lstm_units', 64) <= 512
                assert 2 <= model_config.get('attention_heads', 4) <= 16
                assert 16 <= model_config.get('dense_units', 32) <= 256
                
            return True
        except Exception as e:
            logger.error(f"模型架构验证失败: {str(e)}")
            return False

    def _validate_training_strategy(self, strategy: Dict[str, Any]) -> bool:
        """验证训练策略配置"""
        try:
            assert 0 < strategy['learning_rate'] < 1
            assert 16 <= strategy['batch_size'] <= 512
            assert 0 < strategy['dropout_rate'] < 1
            assert strategy['early_stopping_patience'] > 0
            return True
        except Exception as e:
            logger.error(f"训练策略验证失败: {str(e)}")
            return False

    def _validate_preprocessing_config(self) -> bool:
        """验证预处理配置"""
        try:
            preprocess_cfg = self.config['preprocessing_config']
            assert 'sequence_length' in preprocess_cfg
            assert 'sliding_window' in preprocess_cfg
            assert 'normalization' in preprocess_cfg
            return True
        except Exception as e:
            logger.error(f"预处理配置验证失败: {str(e)}")
            return False

    def _validate_ensemble_config(self) -> bool:
        """验证集成配置"""
        try:
            ensemble_cfg = self.config['ensemble_config']
            assert 'voting_method' in ensemble_cfg
            assert ensemble_cfg['voting_method'] in ['majority', 'weighted', 'average']
            assert 0 < ensemble_cfg['min_weight'] < ensemble_cfg['max_weight'] < 1
            return True
        except Exception as e:
            logger.error(f"集成配置验证失败: {str(e)}")
            return False

    # 2. 特征工程 (from base_model.py)
    def _build_basic_features(self, x):
        """构建基础特征 (from base_model.py)"""
        try:
            # 1. 时间特征
            time_features = self._add_positional_encoding(x)
            
            # 2. 多尺度特征
            multi_scale = self._build_multi_scale_features(x)
            
            # 3. 统计特征
            statistical = self._build_statistical_features(x)
            
            # 合并所有基础特征
            x = tf.concat([time_features, multi_scale, statistical], axis=-1)
            return x
            
        except Exception as e:
            logger.error(f"构建基础特征时出错: {str(e)}")
            return x

    def _add_positional_encoding(self, x):
        """时序位置编码 (修正序列长度计算)"""
        seq_len = tf.shape(x)[1]
        d_model = tf.shape(x)[-1]
        
        position = tf.range(seq_len, dtype=tf.float32)[:, tf.newaxis]
        
        # 修正：先计算常量部分
        log_value = tf.cast(-math.log(10000.0), tf.float32)
        d_model_float = tf.cast(d_model, tf.float32)
        angle_factor = log_value / d_model_float
        
        # 修正：计算正确的序列长度
        d_model_half = tf.cast(tf.math.floor(d_model_float / 2), tf.int32)
        total_length = seq_len * d_model_half
        
        # 生成角度基数
        angle_rads = tf.range(0, d_model_float, 2.0) * angle_factor
        div_term = tf.exp(angle_rads)
        
        pe = tf.zeros((seq_len, d_model))
        
        # 正弦部分 - 确保长度匹配
        indices_sin = tf.stack([
            tf.repeat(tf.range(seq_len), d_model_half),
            tf.tile(tf.range(0, d_model, 2)[:d_model_half], [seq_len])
        ], axis=1)
        
        updates_sin = tf.sin(tf.reshape(position, [-1, 1]) * div_term[:d_model_half])
        pe = tf.tensor_scatter_nd_update(pe, indices_sin, tf.reshape(updates_sin, [-1]))
        
        # 余弦部分 - 确保长度匹配
        indices_cos = tf.stack([
            tf.repeat(tf.range(seq_len), d_model_half),
            tf.tile(tf.range(1, d_model, 2)[:d_model_half], [seq_len])
        ], axis=1)
        
        updates_cos = tf.cos(tf.reshape(position, [-1, 1]) * div_term[:d_model_half])
        pe = tf.tensor_scatter_nd_update(pe, indices_cos, tf.reshape(updates_cos, [-1]))
        
        return x + pe[tf.newaxis, :, :]

    def _build_multi_scale_features(self, x):
        """多尺度特征提取 (from base_model.py)"""
        try:
            conv1 = Conv1D(32, kernel_size=3, padding='same')(x)
            conv2 = Conv1D(32, kernel_size=5, padding='same')(x)
            conv3 = Conv1D(32, kernel_size=7, padding='same')(x)
            dconv1 = Conv1D(32, kernel_size=3, dilation_rate=2, padding='same')(x)
            dconv2 = Conv1D(32, kernel_size=5, dilation_rate=2, padding='same')(x)
            return tf.keras.layers.Concatenate()([conv1, conv2, conv3, dconv1, dconv2])
        except Exception as e:
            logger.error(f"构建多尺度特征时出错: {str(e)}")
            return x

    def _build_statistical_features(self, x):
        """统计特征提取 (from base_model.py)"""
        try:
            mean = tf.reduce_mean(x, axis=1, keepdims=True)
            std = tf.math.reduce_std(x, axis=1, keepdims=True)
            kurtosis = tf.reduce_mean(tf.pow(x - mean, 4), axis=1, keepdims=True) / tf.pow(std, 4)
            skewness = tf.reduce_mean(tf.pow(x - mean, 3), axis=1, keepdims=True) / tf.pow(std, 3)
            return tf.keras.layers.Concatenate()([mean, std, kurtosis, skewness])
        except Exception as e:
            logger.error(f"构建统计特征时出错: {str(e)}")
            return x

    def _build_temporal_features(self, x):
        """构建时序特征"""
        try:
            # 1. 时间编码
            time_encoding = self._add_temporal_encoding(x)
            
            # 2. 周期特征
            periodic = self._build_periodic_features(x)
            
            # 3. 趋势特征
            trend = self._build_trend_features(x)
            
            # 4. 相关性特征
            correlation = self._build_correlation_features(x)
            
            return tf.concat([time_encoding, periodic, trend, correlation], axis=-1)
            
        except Exception as e:
            logger.error(f"构建时序特征时出错: {str(e)}")
            return x

    def _build_periodic_features(self, x, periods=[60, 120, 360, 720, 1440]):
        """构建周期性特征"""
        features = []
        for period in periods:
            pattern = self._extract_periodic_pattern(x, period)
            deviation = x - pattern
            strength = tf.reduce_mean(tf.abs(pattern), axis=-1, keepdims=True)
            features.extend([pattern, deviation, strength])
        return tf.keras.layers.Concatenate()(features)

    def _build_correlation_features(self, x):
        """构建相关性特征"""
        # 1. 位置间相关性
        correlations = []
        for i in range(5):
            for j in range(i+1, 5):
                corr = self._compute_correlation(x[..., i], x[..., j])
                correlations.append(corr)
        
        # 2. 滞后相关性
        lag_correlations = []
        for lag in [1, 2, 3, 5, 10]:
            lagged_corr = self._compute_lag_correlation(x, lag)
            lag_correlations.append(lagged_corr)
        
        return tf.concat([*correlations, *lag_correlations], axis=-1)

    def _compute_correlation(self, x1, x2):
        """计算相关系数"""
        x1_norm = (x1 - tf.reduce_mean(x1)) / tf.math.reduce_std(x1)
        x2_norm = (x2 - tf.reduce_mean(x2)) / tf.math.reduce_std(x2)
        return tf.reduce_mean(x1_norm * x2_norm, axis=-1, keepdims=True)

    def _compute_lag_correlation(self, x, lag):
        """计算滞后相关性"""
        x_current = x[:, lag:]
        x_lagged = x[:, :-lag]
        return self._compute_correlation(x_current, x_lagged)

    def _extract_periodic_pattern(self, x, period):
        """提取周期性模式"""
        try:
            batch_size = tf.shape(x)[0]
            length = tf.shape(x)[1]
            n_periods = length // period
            
            x_reshaped = tf.reshape(x[:, :n_periods*period], 
                               (batch_size, n_periods, period, -1))
            pattern = tf.reduce_mean(x_reshaped, axis=1)
            return pattern
        except Exception as e:
            logger.error(f"提取周期模式时出错: {str(e)}")
            return x

    def _add_temporal_encoding(self, x):
        """添加时间编码"""
        try:
            seq_len = tf.shape(x)[1]
            d_model = tf.shape(x)[-1]
            
            # 1. 位置编码
            position = tf.range(seq_len, dtype=tf.float32)[:, tf.newaxis]
            div_term = tf.exp(
                tf.range(0, d_model, 2, dtype=tf.float32) * 
                (-math.log(10000.0) / d_model)
            )
            
            pe = tf.zeros((seq_len, d_model))
            pe = tf.tensor_scatter_nd_update(
                pe,
                tf.stack([tf.range(seq_len), tf.range(0, d_model, 2)], axis=1),
                tf.sin(position * div_term)
            )
            pe = tf.tensor_scatter_nd_update(
                pe,
                tf.stack([tf.range(seq_len), tf.range(1, d_model, 2)], axis=1),
                tf.cos(position * div_term)
            )
            
            # 2. 添加时间周期特征
            day_in_week = tf.cast(tf.math.floormod(position, 7), tf.float32) / 7.0
            hour_in_day = tf.cast(tf.math.floormod(position, 24), tf.float32) / 24.0
            minute_in_hour = tf.cast(tf.math.floormod(position, 60), tf.float32) / 60.0
            
            time_features = tf.concat([
                pe,
                day_in_week,
                hour_in_day,
                minute_in_hour
            ], axis=-1)
            
            return time_features[tf.newaxis, :, :]
            
        except Exception as e:
            logger.error(f"添加时间编码时出错: {str(e)}")
            return x

    def _adjust_sequence_length(self, x):
        """调整序列长度"""
        current_length = x.shape[1]
        if current_length > self.prediction_range:
            x = tf.keras.layers.AveragePooling1D(
                pool_size=current_length // self.prediction_range)(x)
        elif current_length < self.prediction_range:
            x = tf.keras.layers.UpSampling1D(
                size=self.prediction_range // current_length)(x)
        return x

    def _build_pattern_features(self, x):
        """构建形态特征"""
        try:
            # 1. 连号分析
            consecutive = tf.reduce_sum(tf.cast(
                x[:, 1:] == x[:, :-1] + 1, tf.float32
            ), axis=-1, keepdims=True)
            
            # 2. 重复号分析
            unique_counts = tf.reduce_sum(
                tf.one_hot(tf.cast(x, tf.int32), 10),
                axis=-2
            )
            repeats = tf.reduce_sum(
                tf.cast(unique_counts > 1, tf.float32),
                axis=-1, keepdims=True
            )
            
            # 3. 号码分布特征
            distribution = self._analyze_number_distribution(x)
            
            # 4. 形态组合特征
            combinations = self._analyze_pattern_combinations(x)
            
            # 5. 形态周期规律
            periodicity = self._analyze_pattern_periodicity(x)
            
            return tf.concat([
                consecutive, 
                repeats, 
                distribution,
                combinations,
                periodicity
            ], axis=-1)
            
        except Exception as e:
            logger.error(f"构建形态特征时出错: {str(e)}")
            return x
    
    def _analyze_number_distribution(self, x):
        """分析号码分布特征"""
        try:
            # 1. 大小比例
            big_nums = tf.reduce_mean(tf.cast(x >= 5, tf.float32), axis=-1, keepdims=True)
            
            # 2. 奇偶比例
            odd_nums = tf.reduce_mean(tf.cast(x % 2 == 1, tf.float32), axis=-1, keepdims=True)
            
            # 3. 012路数分析
            mod_3 = tf.cast(x % 3, tf.float32)
            route_0 = tf.reduce_mean(tf.cast(mod_3 == 0, tf.float32), axis=-1, keepdims=True)
            route_1 = tf.reduce_mean(tf.cast(mod_3 == 1, tf.float32), axis=-1, keepdims=True)
            route_2 = tf.reduce_mean(tf.cast(mod_3 == 2, tf.float32), axis=-1, keepdims=True)
            
            # 4. 和值分析
            sum_value = tf.reduce_sum(x, axis=-1, keepdims=True)
            
            return tf.concat([
                big_nums, odd_nums,
                route_0, route_1, route_2,
                sum_value
            ], axis=-1)
        except Exception as e:
            logger.error(f"分析号码分布特征时出错: {str(e)}")
            return tf.zeros_like(x[..., :1])

    # 3. 模型构建 (from model_builder.py)
    def build_model(self, model_num=None, params=None):
        """整合后的模型构建方法"""
        try:
            inputs = tf.keras.Input(shape=(self.sequence_length, self.feature_dim))
            
            # 1. 基础特征提取 (from base_model.py)
            x = self._build_basic_features(inputs)
            
            # 2. 根据模型编号选择不同的特征提取方法 (from model_builder.py)
            if model_num is not None:
                x = self._build_model_specific_features(x, model_num, params or {})
            
            # 3. 预测输出头 (from base_model.py)
            outputs = self._build_prediction_head(x)
            
            model = Model(inputs=inputs, outputs=outputs)
            self.compile_model(model)
            return model
        except Exception as e:
            logger.error(f"构建模型时出错: {str(e)}")
            raise

    def _build_model_specific_features(self, x, model_num, params):
        """不同模型架构的特征构建 (from model_builder.py)"""
        if model_num == 1:
            return self._build_lstm_gru_attention(x, params)
        elif model_num == 2:
            return self._build_bilstm_residual(x, params)
        elif model_num == 3:
            return self._build_temporal_conv_lstm(x, params)
        elif model_num == 4:
            return self._build_transformer(x, params)
        elif model_num == 5:
            return self._build_gru_attention_skip(x, params)
        elif model_num == 6:
            return self._build_digit_correlation_model(x, params)
        elif model_num == 7:
            return self._build_probability_model(x, params)
        else:
            return self._build_lstm_cnn(x, params)

    def _build_lstm_gru_attention(self, x, params):
        """LSTM+GRU+注意力模型 (from model_builder.py)"""
        x = LSTM(params.get('lstm_units', 128), return_sequences=True)(x)
        x = GRU(params.get('gru_units', 64), return_sequences=True)(x)
        x = MultiHeadAttention(
            num_heads=params.get('attention_heads', 4),
            key_dim=params.get('key_dim', 16),
            value_dim=params.get('value_dim', 16)
        )(x, x)
        return x

    def _build_bilstm_residual(self, x, params):
        """BiLSTM残差网络 (from model_builder.py)"""
        try:
            main = Bidirectional(LSTM(params['lstm_units'], return_sequences=True))(x)
            residual = Conv1D(params['lstm_units']*2, kernel_size=1)(x)
            x = Add()([main, residual])
            return LayerNormalization()(x)
        except Exception as e:
            logger.error(f"构建BiLSTM残差网络时出错: {str(e)}")
            return x

    def _build_temporal_conv_lstm(self, x, params):
        """时空卷积LSTM (from model_builder.py)"""
        try:
            x = Conv1D(64, kernel_size=3, padding='same')(x)
            x = Conv1D(64, kernel_size=3, padding='causal', dilation_rate=2)(x)
            x = tf.keras.layers.PReLU()(x)
            x = LSTM(64, return_sequences=True)(x)
            return x
        except Exception as e:
            logger.error(f"构建时空卷积LSTM时出错: {str(e)}")
            return x

    def _build_transformer(self, x, params):
        """构建Transformer模型 (from model_builder.py)"""
        try:
            x = MultiHeadAttention(
                num_heads=params['num_heads'],
                key_dim=params['key_dim']
            )(x, x)
            return LayerNormalization()(x)
        except Exception as e:
            logger.error(f"构建Transformer时出错: {str(e)}")
            return x

    def _build_gru_attention_skip(self, x, params):
        """GRU + 自注意力 + 跳跃连接 (from model_builder.py)"""
        try:
            gru_out = GRU(params['gru_units'], return_sequences=True)(x)
            att = MultiHeadAttention(num_heads=4, key_dim=16)(gru_out, gru_out)
            x = Add()([att, x])
            return LayerNormalization()(x)
        except Exception as e:
            logger.error(f"构建GRU注意力网络时出错: {str(e)}")
            return x

    def _build_lstm_cnn(self, x, params):
        """LSTM + CNN模型 (from model_builder.py)"""
        try:
            x = LSTM(params['lstm_units'], return_sequences=True)(x)
            x = Conv1D(filters=16, kernel_size=3, padding='same')(x)
            return tf.keras.layers.BatchNormalization()(x)
        except Exception as e:
            logger.error(f"构建LSTM-CNN时出错: {str(e)}")
            return x

    def _build_digit_correlation_model(self, x, params):
        """数字关联分析模型"""
        try:
            # 1. 相邻数字关系
            adjacent_patterns = Conv1D(64, kernel_size=2, strides=1, padding='same')(x)
            
            # 2. 数字组合模式
            combination_patterns = []
            for window in [3, 5, 7]:
                pattern = Conv1D(32, kernel_size=window, padding='same')(x)
                combination_patterns.append(pattern)
            
            # 3. 合并所有模式
            x = tf.keras.layers.Concatenate()([adjacent_patterns, *combination_patterns])
            x = LSTM(128, return_sequences=True)(x)
            return x
        except Exception as e:
            logger.error(f"构建数字关联分析模型时出错: {str(e)}")
            return x

    def _build_probability_model(self, x, params):
        """概率分布学习模型"""
        try:
            # 1. 历史概率分布
            hist_probs = self._build_historical_probabilities(x)
            
            # 2. 条件概率特征
            cond_probs = self._build_conditional_probabilities(x)
            
            # 3. 组合概率模型
            x = tf.keras.layers.Concatenate()([hist_probs, cond_probs])
            x = Dense(256, activation='relu')(x)
            x = Dense(128, activation='relu')(x)
            return x
        except Exception as e:
            logger.error(f"构建概率模型时出错: {str(e)}")
            return x

    def _build_historical_probabilities(self, x):
        """构建历史概率分布特征"""
        try:
            # 1. 计算历史频率分布
            freqs = tf.zeros((10, 5))  # 10个数字在5个位置的频率
            for i in range(5):
                digit_freqs = tf.keras.layers.Lambda(
                    lambda x: tf.cast(
                        tf.histogram_fixed_width(x[..., i], [0, 9], nbins=10),
                        tf.float32
                    )
                )(x)
                freqs = tf.tensor_scatter_nd_update(
                    freqs,
                    [[j, i] for j in range(10)],
                    digit_freqs
                )
            
            # 2. 计算条件概率
            cond_probs = self._calculate_conditional_probs(x)
            
            return tf.concat([freqs, cond_probs], axis=-1)
        except Exception as e:
            logger.error(f"构建历史概率分布特征时出错: {str(e)}")
            return x

    def _build_conditional_probabilities(self, x):
        """构建条件概率特征"""
        try:
            # 1. 计算相邻位置条件概率
            adjacent_probs = []
            for i in range(4):
                curr = tf.cast(x[..., i], tf.int32)
                next_digit = tf.cast(x[..., i+1], tf.int32)
                probs = self._compute_transition_probs(curr, next_digit)
                adjacent_probs.append(probs)
            
            # 2. 计算跳跃位置条件概率
            skip_probs = []
            for i in range(3):
                curr = tf.cast(x[..., i], tf.int32)
                next_digit = tf.cast(x[..., i+2], tf.int32)
                probs = self._compute_transition_probs(curr, next_digit)
                skip_probs.append(probs)
            
            return tf.concat([*adjacent_probs, *skip_probs], axis=-1)
        except Exception as e:
            logger.error(f"构建条件概率特征时出错: {str(e)}")
            return x

    def _compute_transition_probs(self, curr_digits, next_digits):
        """计算转移概率矩阵"""
        try:
            # 创建10x10的转移矩阵
            transition_matrix = tf.zeros((10, 10))
            
            # 统计转移次数
            for i in range(10):
                for j in range(10):
                    mask_curr = tf.cast(curr_digits == i, tf.float32)
                    mask_next = tf.cast(next_digits == j, tf.float32)
                    count = tf.reduce_sum(mask_curr * mask_next)
                    transition_matrix = tf.tensor_scatter_nd_update(
                        transition_matrix,
                        [[i, j]],
                        [count]
                    )
            
            # 计算概率
            row_sums = tf.reduce_sum(transition_matrix, axis=1, keepdims=True)
            probs = transition_matrix / (row_sums + 1e-7)
            
            return probs
        except Exception as e:
            logger.error(f"计算转移概率时出错: {str(e)}")
            return tf.zeros((10, 10))

    def _build_combined_features(self, feature_list):
        """构建组合特征 (from model_builder.py)"""
        try:
            # 1. 特征连接
            x = tf.keras.layers.Concatenate()(feature_list)
            
            # 2. 非线性变换
            x = Dense(256, activation='relu')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(0.2)(x)
            
            # 3. 特征交互
            x = self._build_feature_interactions(x)
            
            return x
            
        except Exception as e:
            logger.error(f"构建组合特征时出错: {str(e)}")
            return tf.zeros_like(x)

    def _build_feature_interactions(self, x):
        """构建特征交互 (from model_builder.py)"""
        try:
            # 1. 自注意力交互
            att = MultiHeadAttention(
                num_heads=4,
                key_dim=32
            )(x, x)
            x = Add()([x, att])
            x = LayerNormalization()(x)
            
            # 2. 非线性特征组合
            x = Dense(128, activation='relu')(x)
            x = Dense(64, activation='relu')(x)
            
            return x
            
        except Exception as e:
            logger.error(f"构建特征交互时出错: {str(e)}")
            return tf.zeros_like(x)

    def _build_ensemble_predictor(self, x, params):
        """集成预测器"""
        # 1. 多模型特征
        features = []
        
        # 组合模式特征
        comb_features = self._build_combination_predictor(x, params)
        # 概率特征
        prob_features = self._build_probability_predictor(x, params)
        # 周期特征  
        period_features = self._build_periodic_predictor(x, params)
        # 趋势特征
        trend_features = self._build_trend_predictor(x, params)
        # 统计特征
        stat_features = self._build_statistical_predictor(x, params)
        
        features.extend([
            comb_features, prob_features, period_features,
            trend_features, stat_features
        ])
        
        # 2. 特征融合
        x = tf.keras.layers.Concatenate()(features)
        
        return x

    def _build_attention_residual_block(self, x, params):
        """注意力残差块"""
        # 多头自注意力
        att = MultiHeadAttention(
            num_heads=params.get('num_heads', 4),
            key_dim=params.get('key_dim', 32)
        )(x, x)
        
        # 残差连接
        x = Add()([x, att])
        x = LayerNormalization()(x)
        
        # FFN
        ffn = Dense(params.get('ffn_dim', 256), activation='relu')(x)
        ffn = Dense(x.shape[-1])(ffn)
        
        # 残差连接
        x = Add()([x, ffn])
        x = LayerNormalization()(x)
        
        return x

    def _build_combination_predictor(self, x, params):
        """组合数字预测器"""
        try:
            # 1. 局部组合模式
            local_patterns = []
            for window in [2, 3, 4]:
                pattern = Conv1D(32, kernel_size=window, padding='same')(x)
                local_patterns.append(pattern)
            
            # 2. 全局组合模式
            global_pattern = self._build_attention_residual_block(
                tf.concat(local_patterns, axis=-1),
                params
            )
            
            return global_pattern
        except Exception as e:
            logger.error(f"构建组合预测器出错: {str(e)}")
            return x

    def _build_trend_predictor(self, x, params):
        """趋势预测器"""
        try:
            # 1. 多尺度趋势分析 
            trends = []
            windows = [60, 360, 720, 1440]
            
            for window in windows:
                ma = tf.keras.layers.AveragePooling1D(
                    pool_size=window, strides=1, padding='same')(x)
                trend = tf.sign(x - ma)
                trends.append(trend)
            
            # 2. 趋势特征融合
            x = tf.keras.layers.Concatenate()(trends)
            x = Bidirectional(LSTM(64, return_sequences=True))(x)
            return x
        except Exception as e:
            logger.error(f"构建趋势预测器出错: {str(e)}")
            return x

    def _build_statistical_predictor(self, x, params):
        """统计模式预测器"""
        try:
            stats = []
            
            # 均值特征
            mean = tf.reduce_mean(x, axis=1, keepdims=True)
            # 标准差特征
            std = tf.math.reduce_std(x, axis=1, keepdims=True)
            # 峰度
            kurtosis = tf.reduce_mean(tf.pow(x - mean, 4), axis=1, keepdims=True) / tf.pow(std, 4)
            # 偏度
            skewness = tf.reduce_mean(tf.pow(x - mean, 3), axis=1, keepdims=True) / tf.pow(std, 3)
            
            stats.extend([mean, std, kurtosis, skewness])
            
            x = tf.keras.layers.Concatenate()(stats)
            x = Dense(128, activation='relu')(x)
            return x
        except Exception as e:
            logger.error(f"构建统计预测器出错: {str(e)}")
            return x

    # 4. 训练评估 (from base_model.py)
    def _build_prediction_head(self, x):
        """预测输出头 (from base_model.py)"""
        # 1. 每位数字的概率分布
        batch_size = tf.shape(x)[0]
        digit_predictions = []
        for i in range(5):
            digit_pred = Dense(10, activation='softmax', name=f'digit_{i}')(x)
            digit_predictions.append(digit_pred)

        # 2. 完整号码匹配概率
        match_prob = Dense(self.prediction_range, activation='sigmoid', name='match_prob')(x)

        # 3. 预测置信度
        confidence = Dense(1, activation='sigmoid', name='confidence')(x)
        
        return {
            'digits': digit_predictions,
            'match_prob': match_prob,
            'confidence': confidence
        }

    def enhanced_match_loss(self, y_true, y_pred):
        """增强型匹配损失函数 (from base_model.py)"""
        try:
            # 1. 预处理
            y_pred_expanded = tf.expand_dims(y_pred, axis=1)
            y_pred_rounded = tf.round(y_pred_expanded)
            
            # 2. 计算匹配情况
            matches = tf.cast(tf.equal(y_true, y_pred_rounded), tf.float32)
            match_counts = tf.reduce_sum(matches, axis=-1)
            best_match_indices = tf.argmax(match_counts, axis=1)
            best_targets = tf.gather(y_true, best_match_indices, batch_dims=1)
            best_match_counts = tf.reduce_max(match_counts, axis=1)
            
            # 3. 计算基础匹配损失
            base_loss = tf.reduce_mean(tf.abs(y_pred - best_targets), axis=1)
            
            # 4. 计算方向性损失
            direction_loss = self._calculate_direction_loss(y_pred, best_targets)
            
            # 5. 完全匹配时损失为0
            perfect_match = tf.cast(tf.equal(best_match_counts, 5.0), tf.float32)
            
            # 6. 组合损失(动态权重)
            direction_weight = tf.exp(-best_match_counts / 5.0) * 0.5
            total_loss = base_loss * (1.0 - perfect_match) + direction_weight * direction_loss
            
            return total_loss
        except Exception as e:
            logger.error(f"计算损失时出错: {str(e)}")
            return 5.0 * tf.ones_like(y_pred[:, 0])

    def compile_model(self, model):
        """编译模型 (from base_model.py)"""
        model.compile(
            optimizer=self.optimizer,
            loss=self.enhanced_match_loss,
            metrics=['accuracy']
        )

    def train_step(self, batch_data):
        """训练步骤 (from base_model.py)"""
        try:
            with tf.GradientTape() as tape:
                predictions = self.model(batch_data['input'], training=True)
                loss = self.enhanced_match_loss(batch_data['target'], predictions)
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            return loss
        except Exception as e:
            logger.error(f"训练步骤执行出错: {str(e)}")
            return None

    def validate_step(self, batch_data):
        """验证步骤 (from base_model.py)"""
        try:
            predictions = self.model(batch_data['input'], training=False)
            loss = self.enhanced_match_loss(batch_data['target'], predictions)
            matches = self._calculate_matches(predictions, batch_data['target'])
            return {
                'loss': loss,
                'accuracy': tf.reduce_mean(matches)
            }
        except Exception as e:
            logger.error(f"验证步骤执行出错: {str(e)}")
            return None

    def _calculate_matches(self, predictions, targets):
        """计算匹配程度 (from base_model.py)"""
        try:
            rounded_preds = tf.round(predictions)
            matches = tf.cast(tf.equal(rounded_preds, targets), tf.float32)
            full_matches = tf.reduce_all(matches, axis=-1)
            return full_matches
        except Exception as e:
            logger.error(f"计算匹配程度时出错: {str(e)}")
            return tf.zeros_like(predictions[..., 0])

    def _calculate_direction_loss(self, y_pred, best_targets):
        """计算方向性损失 (from base_model.py)"""
        try:
            value_diff = best_targets - y_pred
            direction_mask = tf.cast(
                tf.not_equal(tf.round(y_pred), best_targets),
                tf.float32
            )
            direction_factor = tf.sigmoid(value_diff * 2.0) * 2.0 - 1.0
            return tf.reduce_mean(
                direction_mask * direction_factor * tf.abs(value_diff),
                axis=-1
            )
        except Exception as e:
            logger.error(f"计算方向性损失时出错: {str(e)}")
            return tf.zeros_like(y_pred[:, 0])

    # 5. 模型保存加载 (from base_model.py)
    def save_model(self, path: str):
        """保存模型 (from base_model.py)"""
        try:
            self.model.save(path)
            logger.info(f"模型已保存到: {path}")
        except Exception as e:
            logger.error(f"保存模型时出错: {str(e)}")

    def load_model(self, path: str):
        """加载模型 (from base_model.py)"""
        try:
            self.model = tf.keras.models.load_model(
                path,
                custom_objects={'enhanced_match_loss': self.enhanced_match_loss}
            )
            logger.info(f"已加载模型: {path}")
        except Exception as e:
            logger.error(f"加载模型时出错: {str(e)}")

    def predict(self, input_data):
        """执行预测"""
        try:
            # 添加输入验证
            if input_data.shape[-1] != 8:  # 5个号码+3个时间特征
                logger.error(f"输入特征维度错误，预期8维，实际收到{input_data.shape[-1]}维")
                return None
            
            # 获取集成预测结果
            predictions = []
            confidences = []
            
            for i, model in enumerate(self.models):
                pred = model.predict(input_data)
                pred_value = pred['digits']
                confidence = pred['confidence']
                
                predictions.append(pred_value * self.weights[i])
                confidences.append(confidence)
            
            # 集成预测结果
            ensemble_pred = np.sum(predictions, axis=0)
            mean_confidence = np.mean(confidences)
            
            return {
                'prediction': ensemble_pred,
                'confidence': mean_confidence
            }
            
        except Exception as e:
            logger.error(f"预测失败: {str(e)}")
            return None

    def predict_with_cache(self, X):
        """带缓存的预测"""
        cache_key = hash(str(X))
        current_time = time.time()
        
        # 检查缓存
        if cache_key in self.prediction_cache:
            cached_result, cache_time = self.prediction_cache[cache_key]
            if current_time - cache_time < self.cache_timeout:
                return cached_result
        
        # 执行预测
        result = self.predict(X)
        
        # 更新缓存
        self.prediction_cache[cache_key] = (result, current_time)
        
        return result

    def get_prediction_history(self, start_time=None, end_time=None):
        """获取预测历史"""
        history = list(self.prediction_history)
        
        if start_time:
            history = [h for h in history if h['timestamp'] >= start_time]
        if end_time:
            history = [h for h in history if h['timestamp'] <= end_time]
            
        return history

    def analyze_prediction_accuracy(self):
        """分析预测准确率"""
        if not self.prediction_history:
            return None
            
        correct = 0
        total = 0
        
        for pred in self.prediction_history:
            if pred.get('actual') is not None:
                correct += int(np.array_equal(
                    pred['prediction'],
                    pred['actual']
                ))
                total += 1
        
        accuracy = correct / total if total > 0 else 0
        
        return {
            'accuracy': accuracy,
            'total_predictions': total,
            'correct_predictions': correct
        }

    def _record_prediction(self, prediction, confidence):
        """记录预测结果"""
        try:
            self.prediction_history.append({
                'prediction': prediction,
                'confidence': confidence,
                'timestamp': datetime.now()
            })
            logger.info(f"记录预测结果: {prediction}, 置信度: {confidence:.2f}")
        except Exception as e:
            logger.error(f"记录预测结果失败: {str(e)}")

    def _clean_prediction_cache(self):
        """清理过期的预测缓存"""
        try:
            current_time = time.time()
            expired_keys = [
                k for k, (_, cache_time) in self.prediction_cache.items()
                if current_time - cache_time > self.cache_timeout
            ]
            for k in expired_keys:
                del self.prediction_cache[k]
            if expired_keys:
                logger.info(f"清理了{len(expired_keys)}条过期预测缓存")
        except Exception as e:
            logger.error(f"清理预测缓存失败: {str(e)}")

    def get_prediction_stats(self) -> Dict[str, Any]:
        """获取预测统计信息"""
        try:
            if not self.prediction_history:
                return {}
                
            recent_predictions = list(self.prediction_history)[-100:]
            return {
                'total_predictions': len(self.prediction_history),
                'recent_avg_confidence': np.mean([p['confidence'] for p in recent_predictions]),
                'cache_hit_rate': self._calculate_cache_hit_rate(),
                'last_prediction_time': self.prediction_history[-1]['timestamp'],
                'accuracy_stats': self.analyze_prediction_accuracy()
            }
        except Exception as e:
            logger.error(f"获取预测统计失败: {str(e)}")
            return {}

    def _calculate_cache_hit_rate(self) -> float:
        """计算缓存命中率"""
        try:
            if not hasattr(self, '_cache_stats'):
                self._cache_stats = {'hits': 0, 'misses': 0}
            total = self._cache_stats['hits'] + self._cache_stats['misses']
            return self._cache_stats['hits'] / total if total > 0 else 0
        except Exception as e:
            logger.error(f"计算缓存命中率失败: {str(e)}")
            return 0.0

    def _build_model(self, config):
        """构建单个模型（修正输入形状）"""
        try:
            with self.graph.as_default():
                # 修改输入形状为(None, sequence_length, features)以支持批处理
                inputs = tf.keras.Input(shape=(None, 5))  # 动态序列长度
                x = Conv1D(config['filters'], 3, activation='relu')(inputs)
                x = LSTM(config['units'], return_sequences=True)(x)
                x = Dense(config['dense_units'], activation='relu')(x)
                outputs = Dense(5, activation='softmax')(x)
                model = tf.keras.Model(inputs=inputs, outputs=outputs)
                
                # 编译模型
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(
                        learning_rate=self.config['optimizer_config']['learning_rate']
                    ),
                    loss='mse',
                    metrics=['mae']
                )
                return model
            
        except KeyError as e:
            logger.error(f"配置缺失关键参数: {str(e)}")
            raise

    def _create_optimizer(self):
        """创建优化器"""
        try:
            opt_cfg = self.config['optimizer_config']
            return tf.keras.optimizers.Adam(
                learning_rate=opt_cfg['learning_rate'],
                beta_1=opt_cfg['beta_1'],
                beta_2=opt_cfg['beta_2']
            )
        except KeyError as e:
            logger.error(f"配置缺失关键参数: {str(e)}，使用默认优化器")
            return tf.keras.optimizers.Adam(learning_rate=0.001)

    def reset_models(self):
        """安全重置模型"""
        try:
            # 清理现有会话
            if self.session:
                self.session.close()
            # 创建新图和新会话
            self.graph = tf.Graph()
            with self.graph.as_default():
                self.models = [self._build_model(self.default_config) for _ in range(6)]
            self.session = tf.compat.v1.Session(graph=self.graph)
            tf.compat.v1.keras.backend.set_session(self.session)
            logger.info("模型重置成功")
        except Exception as e:
            logger.error(f"模型重置失败: {str(e)}")
            raise

# 创建全局模型核心实例
model_core = ModelCore()

# 生成测试数据
test_input = np.random.rand(1, 14400, 5)  # 批次大小1，序列长度14400，特征维度5

with model_core.session.as_default():
    with model_core.graph.as_default():
        # 创建输入占位符
        input_tensor = tf.placeholder(tf.float32, shape=(1, 14400, 5))
        # 获取编码输出
        encoded_output = model_core._add_positional_encoding(input_tensor)
        # 运行会话
        result = model_core.session.run(encoded_output, feed_dict={input_tensor: test_input})

print("输入形状:", test_input.shape)
print("编码后形状:", result.shape)
print("编码示例(前3个时间步):\n", result[0, :3, :5])
