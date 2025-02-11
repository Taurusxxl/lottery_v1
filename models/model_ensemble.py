# 集成模型实例化\model_ensemble.py
import numpy as np
import tensorflow as tf
import logging
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, GRU, Bidirectional, Conv1D
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Add
import os
import json
import threading
from core import config_manager
from optimizers import model_optimizer
from concurrent.futures import ThreadPoolExecutor
import torch

# 获取logger实例
logger = logging.getLogger(__name__)

class ModelEnsemble:
    """模型集成类"""
    
    def __init__(self, sequence_length=14400, feature_dim=5):
        """
        初始化模型集成
        Args:
            sequence_length: 输入序列长度
            feature_dim: 特征维度
        """
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.prediction_range = 2880  # 预测范围
        self.models = []  # 模型列表
        self.weights = np.ones(6) / 6  # 初始权重平均分配
        
        # 模型性能追踪
        self.performance_history = {i: [] for i in range(6)}
        
        # 新增训练同步机制
        self.training_lock = threading.Lock()
        self.finished_models = 0
        self.total_models = 6
        
        # 初始化模型
        self._build_models()
        logger.info("模型集成初始化完成")
        
    @staticmethod
    def enhanced_match_loss(y_true, y_pred):
        """
        增强型匹配损失函数 - 结合匹配度评判和方向性引导
        
        Args:
            y_true: 真实值 (batch_size, 2880, 5) - 2880期目标值
            y_pred: 预测值 (batch_size, 5) - 预测的一组五位数
            
        Returns:
            total_loss: 综合损失值,包含匹配损失和方向性损失
        """
        try:
            # 1. 预处理
            y_pred_expanded = tf.expand_dims(y_pred, axis=1)  # (batch_size, 1, 5)
            y_pred_rounded = tf.round(y_pred_expanded)
            
            # 2. 计算匹配情况
            matches = tf.cast(tf.equal(y_true, y_pred_rounded), tf.float32)  # (batch_size, 2880, 5)
            match_counts = tf.reduce_sum(matches, axis=-1)  # (batch_size, 2880)
            
            # 3. 找出最佳匹配的目标值
            best_match_indices = tf.argmax(match_counts, axis=1)  # (batch_size,)
            best_targets = tf.gather(y_true, best_match_indices, batch_dims=1)  # (batch_size, 5)
            best_match_counts = tf.reduce_max(match_counts, axis=1)  # (batch_size,)
            
            # 4. 计算基础匹配损失
            base_loss = 5.0 - best_match_counts  # (batch_size,)
            
            # 5. 计算方向性损失
            # 计算预测值与最佳匹配目标的差异
            value_diff = tf.squeeze(best_targets - y_pred, axis=1)  # (batch_size, 5)
            
            # 创建方向性掩码(只对未匹配的位置计算方向性损失)
            direction_mask = tf.cast(
                tf.not_equal(y_pred_rounded, tf.expand_dims(best_targets, axis=1)),
                tf.float32
            )  # (batch_size, 1, 5)
            
            # 使用sigmoid函数平滑方向性梯度
            direction_factor = tf.sigmoid(value_diff * 2.0) * 2.0 - 1.0  # 范围(-1, 1)
            
            # 计算方向性损失(差异越大,损失越大)
            direction_loss = tf.reduce_mean(
                direction_mask * direction_factor * tf.abs(value_diff),
                axis=-1
            )  # (batch_size,)
            
            # 6. 完全匹配时损失为0
            perfect_match = tf.cast(tf.equal(best_match_counts, 5.0), tf.float32)
            
            # 7. 组合损失(动态权重)
            # 匹配数越少,方向性损失权重越大
            direction_weight = tf.exp(-best_match_counts / 5.0) * 0.5  # 随匹配数增加呈指数衰减
            total_loss = base_loss * (1.0 - perfect_match) + direction_weight * direction_loss
            
            # 8. 添加调试信息
            tf.debugging.assert_all_finite(
                total_loss,
                "Loss computation resulted in invalid values"
            )
            
            return total_loss
            
        except Exception as e:
            logger.error(f"计算损失时出错: {str(e)}")
            return 5.0 * tf.ones_like(y_pred[:, 0])
        
    def _build_models(self):
        """构建所有模型"""
        try:
            model_params = {
                'model_1': {
                    'conv_filters': [32, 64, 128],
                    'lstm_units': 128,
                    'attention_heads': 8
                },
                'model_2': {
                    'conv_filters': [64, 128],
                    'lstm_units': 256,
                    'dropout': 0.2
                },
                'model_3': {
                    'gru_units': 128,
                    'dense_units': [256, 128],
                    'learning_rate': 0.001
                },
                'model_4': {
                    'num_heads': 8,
                    'key_dim': 64,
                    'ff_dim': 256
                },
                'model_5': {
                    'lstm_units': 128,
                    'attention_heads': 4,
                    'dropout': 0.1
                },
                'model_6': {
                    'conv_filters': [32, 64],
                    'gru_units': 128,
                    'dense_units': [128, 64]
                }
            }
            
            for i in range(6):
                logger.info(f"构建模型 {i+1}/6...")
                model = self._build_model(i+1, model_params[f'model_{i+1}'])
                self.compile_model(model)  # 使用新的compile_model方法
                self.models.append(model)
                
        except Exception as e:
            logger.error(f"构建模型时出错: {str(e)}")
            raise
            
    def _build_model(self, model_num, params):
        """构建增强版单个模型"""
        try:
            inputs = tf.keras.Input(shape=(self.sequence_length, self.feature_dim))
            
            # 1. 基础特征提取
            x = self._build_basic_features(inputs)
            
            # 2. 高级特征分析
            advanced_features = self._build_advanced_features(inputs)
            
            # 3. 特征融合
            x = tf.keras.layers.Concatenate()([x, advanced_features])
            
            # 4. 深度特征提取
            x = self._build_deep_features(x, params)
            
            # 5. 预测头
            outputs = self._build_prediction_head(x)
            
            return Model(inputs=inputs, outputs=outputs)
            
        except Exception as e:
            logger.error(f"构建模型 {model_num} 时出错: {str(e)}")
            raise

    def _add_positional_encoding(self, x):
        """添加位置编码"""
        seq_len = tf.shape(x)[1]
        d_model = tf.shape(x)[-1]
        
        # 生成位置编码矩阵
        position = tf.range(seq_len, dtype=tf.float32)[:, tf.newaxis]
        div_term = tf.exp(tf.range(0, d_model, 2, dtype=tf.float32) * -(tf.math.log(10000.0) / d_model))
        
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
        
        return x + pe[tf.newaxis, :, :]
        
    def _build_multi_scale_features(self, x):
        """多尺度特征提取"""
        # 不同kernel_size的并行卷积
        conv1 = Conv1D(32, kernel_size=3, padding='same')(x)
        conv2 = Conv1D(32, kernel_size=5, padding='same')(x)
        conv3 = Conv1D(32, kernel_size=7, padding='same')(x)
        
        # 空洞卷积捕获更大感受野
        dconv1 = Conv1D(32, kernel_size=3, dilation_rate=2, padding='same')(x)
        dconv2 = Conv1D(32, kernel_size=3, dilation_rate=4, padding='same')(x)
        
        # 特征融合
        x = tf.keras.layers.Concatenate()([conv1, conv2, conv3, dconv1, dconv2])
        x = tf.keras.layers.BatchNormalization()(x)
        return tf.keras.layers.Activation('relu')(x)
        
    def _build_multi_task_head(self, x):
        """多任务输出头"""
        # 主任务: 预测下一个数字
        main_output = Dense(5, activation='linear', name='main_output')(x)
        
        # 辅助任务1: 预测趋势
        trend = Dense(1, activation='tanh', name='trend')(x)
        
        # 辅助任务2: 预测波动性
        volatility = Dense(1, activation='sigmoid', name='volatility')(x)
        
        return {
            'main_output': main_output,
            'trend': trend,
            'volatility': volatility
        }
        
    def _build_bilstm_residual(self, x, params):
        """双向LSTM + 残差连接"""
        # 主路径
        main = Bidirectional(LSTM(params['lstm_units'], return_sequences=True))(x)
        
        # 残差路径
        residual = Conv1D(params['lstm_units']*2, kernel_size=1)(x)  # 匹配通道数
        
        # 残差连接
        x = Add()([main, residual])
        return LayerNormalization()(x)
        
    def _build_temporal_conv_lstm(self, x, params):
        """时空卷积 + LSTM"""
        # 时序卷积模块
        x = Conv1D(64, kernel_size=3, padding='same')(x)
        x = tf.keras.layers.PReLU()(x)
        x = tf.keras.layers.SpatialDropout1D(0.1)(x)
        
        # 因果卷积
        x = Conv1D(64, kernel_size=3, padding='causal', dilation_rate=2)(x)
        x = tf.keras.layers.PReLU()(x)
        
        return LSTM(params['lstm_units'], return_sequences=True)(x)
        
    def _build_gru_attention_skip(self, x, params):
        """GRU + 自注意力 + 跳跃连接"""
        # GRU层
        gru_out = GRU(params['gru_units'], return_sequences=True)(x)
        
        # 自注意力
        att = MultiHeadAttention(num_heads=4, key_dim=16)(gru_out, gru_out)
        
        # 跳跃连接
        x = Add()([att, x])
        return LayerNormalization()(x)
        
    def _adjust_sequence_length(self, x):
        """调整序列长度到预测范围"""
        # 获取当前序列长度
        current_length = x.shape[1]
        
        if current_length > self.prediction_range:
            # 如果当前长度大于目标长度，使用池化层减少长度
            pool_size = current_length // self.prediction_range
            x = tf.keras.layers.AveragePooling1D(pool_size=pool_size)(x)
        elif current_length < self.prediction_range:
            # 如果当前长度小于目标长度，使用上采样增加长度
            x = tf.keras.layers.UpSampling1D(size=self.prediction_range // current_length)(x)
        
        # 确保最终长度精确匹配
        x = tf.keras.layers.Conv1D(filters=32, kernel_size=1)(x)  # 1x1 卷积调整
        x = tf.keras.layers.Reshape((-1, 32))(x)  # 重塑确保维度正确
        x = tf.keras.layers.Dense(32)(x)  # 保持特征维度
        
        return x
        
    def _build_lstm_gru_attention(self, x, params):
        """构建LSTM+GRU+注意力模型"""
        x = LSTM(params['lstm_units'], return_sequences=True)(x)
        x = GRU(params['gru_units'], return_sequences=True)(x)
        x = MultiHeadAttention(
            num_heads=4,
            key_dim=16,
            value_dim=16
        )(x, x)
        return x
        
    def _build_bilstm(self, x, params):
        """构建双向LSTM模型"""
        return Bidirectional(LSTM(params['lstm_units'], return_sequences=True))(x)
        
    def _build_cnn_lstm(self, x, params):
        """构建CNN+LSTM模型"""
        x = Conv1D(filters=16, kernel_size=3, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        return LSTM(params['lstm_units'], return_sequences=True)(x)
        
    def _build_transformer(self, x, params):
        """构建Transformer模型"""
        x = MultiHeadAttention(num_heads=params['num_heads'], 
                             key_dim=params['key_dim'])(x, x)
        return LayerNormalization()(x)
        
    def _build_gru_attention(self, x, params):
        """构建GRU+注意力模型"""
        x = GRU(params['gru_units'], return_sequences=True)(x)
        return MultiHeadAttention(num_heads=4, key_dim=16)(x, x)
        
    def _build_lstm_cnn(self, x, params):
        """构建LSTM+CNN模型"""
        x = LSTM(params['lstm_units'], return_sequences=True)(x)
        x = Conv1D(filters=16, kernel_size=3, padding='same')(x)
        return tf.keras.layers.BatchNormalization()(x)
        
    def update_weights(self, performance_metrics):
        """更新模型权重"""
        try:
            if not performance_metrics:
                return
                
            # 根据性能指标计算新权重
            scores = np.array([metrics['score'] for metrics in performance_metrics])
            new_weights = scores / np.sum(scores)
            
            # 平滑更新
            alpha = 0.3
            self.weights = alpha * new_weights + (1 - alpha) * self.weights
            
            logger.info(f"更新模型权重: {self.weights}")
            
        except Exception as e:
            logger.error(f"更新模型权重时出错: {str(e)}")
            
    def get_ensemble_prediction(self, X):
        """获取集成预测结果"""
        try:
            predictions = []
            for i, model in enumerate(self.models):
                pred = model.predict(X)  # shape: (batch_size, prediction_range, 5)
                predictions.append(pred * self.weights[i])
                
            # 加权平均
            ensemble_pred = np.sum(predictions, axis=0)  # shape: (batch_size, prediction_range, 5)
            return ensemble_pred
            
        except Exception as e:
            logger.error(f"获取集成预测时出错: {str(e)}")
            return None

    def _build_advanced_features(self, x):
        """构建高级特征分析"""
        features = []
        
        # 1. 冷热号分析
        hot_cold = self._analyze_hot_cold_numbers(x)
        features.append(hot_cold)
        
        # 2. 号码频率统计
        freq = self._analyze_frequency(x)
        features.append(freq)
        
        # 3. 和值分析
        sum_features = self._analyze_sum_value(x)
        features.append(sum_features)
        
        # 4. 数字特征分析
        digit_features = self._analyze_digit_patterns(x)
        features.append(digit_features)
        
        # 5. 形态分析
        pattern_features = self._analyze_number_patterns(x)
        features.append(pattern_features)
        
        # 6. 012路分析
        route_features = self._analyze_012_routes(x)
        features.append(route_features)
        
        return tf.keras.layers.Concatenate()(features)

    def _analyze_hot_cold_numbers(self, x, window_sizes=[100, 500, 1000]):
        """分析冷热号"""
        features = []
        
        for window in window_sizes:
            # 最近window期的频率
            recent = x[:, -window:]
            freq = tf.reduce_sum(tf.one_hot(tf.cast(recent, tf.int32), 10), axis=1)
            features.append(freq)
        
        return tf.keras.layers.Concatenate()(features)

    def _analyze_digit_patterns(self, x):
        """分析数字特征"""
        # 1. 奇偶比
        odd_even = tf.reduce_sum(tf.cast(x % 2 == 1, tf.float32), axis=-1, keepdims=True)
        
        # 2. 大小比 (5-9为大)
        big_small = tf.reduce_sum(tf.cast(x >= 5, tf.float32), axis=-1, keepdims=True)
        
        # 3. 质合比
        prime_numbers = tf.constant([2, 3, 5, 7])
        is_prime = tf.reduce_sum(tf.cast(
            tf.equal(x[..., None], prime_numbers), tf.float32
        ), axis=-1)
        prime_composite = tf.reduce_sum(is_prime, axis=-1, keepdims=True)
        
        # 4. 跨度
        span = tf.reduce_max(x, axis=-1) - tf.reduce_min(x, axis=-1)
        
        return tf.concat([odd_even, big_small, prime_composite, span[..., None]], axis=-1)

    def _analyze_number_patterns(self, x):
        """分析号码形态"""
        # 1. 连号分析
        consecutive = tf.reduce_sum(tf.cast(
            x[:, 1:] == x[:, :-1] + 1, tf.float32
        ), axis=-1, keepdims=True)
        
        # 2. 重复号分析
        unique_counts = tf.reduce_sum(tf.one_hot(tf.cast(x, tf.int32), 10), axis=-2)
        repeats = tf.reduce_sum(tf.cast(unique_counts > 1, tf.float32), axis=-1, keepdims=True)
        
        # 3. 形态识别
        patterns = self._identify_number_patterns(x)
        
        return tf.concat([consecutive, repeats, patterns], axis=-1)

    def _identify_number_patterns(self, x):
        """识别特定号码形态"""
        # 1. 豹子号(AAAAA)
        baozi = tf.reduce_all(x == x[..., :1], axis=-1, keepdims=True)
        
        # 2. 组5(AAAAB)
        zu5 = tf.logical_or(
            tf.reduce_sum(tf.cast(x == x[..., :1], tf.float32), axis=-1) == 4,
            tf.reduce_sum(tf.cast(x == x[..., 1:2], tf.float32), axis=-1) == 4
        )
        
        # 3. 组10(AAABB)
        sorted_x = tf.sort(x, axis=-1)
        zu10 = tf.logical_and(
            tf.reduce_sum(tf.cast(sorted_x[..., :2] == sorted_x[..., :1], tf.float32), axis=-1) == 2,
            tf.reduce_sum(tf.cast(sorted_x[..., 2:] == sorted_x[..., 2:3], tf.float32), axis=-1) == 2
        )
        
        # 4. 顺子(连续5个数)
        shunzi = tf.reduce_all(sorted_x[..., 1:] == sorted_x[..., :-1] + 1, axis=-1)
        
        return tf.cast(tf.stack([baozi, zu5, zu10, shunzi], axis=-1), tf.float32)

    def _analyze_sum_value(self, x):
        """分析和值特征"""
        # 1. 计算和值
        sum_value = tf.reduce_sum(x, axis=-1, keepdims=True)
        
        # 2. 和值分布区间
        sum_ranges = [
            (0, 10), (11, 20), (21, 30), (31, 40), (41, 45)
        ]
        sum_dist = []
        for low, high in sum_ranges:
            in_range = tf.logical_and(
                sum_value >= low,
                sum_value <= high
            )
            sum_dist.append(tf.cast(in_range, tf.float32))
        
        # 3. 和值特征
        sum_features = tf.concat([sum_value, tf.concat(sum_dist, axis=-1)], axis=-1)
        
        return sum_features

    def _analyze_012_routes(self, x):
        """分析012路特征"""
        # 1. 计算每个数字的路数
        routes = tf.math.floormod(x, 3)  # 对3取余
        
        # 2. 统计每个位置的路数分布
        route_distributions = []
        for i in range(5):  # 五个位置
            digit_routes = routes[..., i:i+1]
            # 统计0,1,2路的数量
            route_counts = []
            for r in range(3):
                count = tf.reduce_sum(
                    tf.cast(digit_routes == r, tf.float32),
                    axis=-1, keepdims=True
                )
                route_counts.append(count)
            route_distributions.append(tf.concat(route_counts, axis=-1))
        
        # 3. 计算整体012路比例
        total_route_dist = tf.reduce_mean(tf.concat(route_distributions, axis=-1), axis=-1, keepdims=True)
        
        # 4. 分析路数组合特征
        route_patterns = self._analyze_route_patterns(routes)
        
        # 5. 计算相邻位置的路数关系
        route_transitions = []
        for i in range(4):
            transition = tf.cast(
                routes[..., i:i+1] == routes[..., i+1:i+2],
                tf.float32
            )
            route_transitions.append(transition)
        
        # 6. 特征组合
        features = [
            *route_distributions,  # 每位路数分布
            total_route_dist,     # 整体路数比例
            route_patterns,       # 路数组合特征
            *route_transitions    # 相邻位置路数关系
        ]
        
        return tf.concat(features, axis=-1)

    def _analyze_route_patterns(self, routes):
        """分析路数组合模式"""
        # 1. 全0路
        all_zero = tf.reduce_all(routes == 0, axis=-1, keepdims=True)
        
        # 2. 全1路
        all_one = tf.reduce_all(routes == 1, axis=-1, keepdims=True)
        
        # 3. 全2路
        all_two = tf.reduce_all(routes == 2, axis=-1, keepdims=True)
        
        # 4. 012路是否均匀分布(各有至少一个)
        has_zero = tf.reduce_any(routes == 0, axis=-1, keepdims=True)
        has_one = tf.reduce_any(routes == 1, axis=-1, keepdims=True)
        has_two = tf.reduce_any(routes == 2, axis=-1, keepdims=True)
        balanced = tf.logical_and(
            tf.logical_and(has_zero, has_one),
            has_two
        )
        
        # 5. 主路特征(出现最多的路数)
        route_counts = []
        for r in range(3):
            count = tf.reduce_sum(
                tf.cast(routes == r, tf.float32),
                axis=-1, keepdims=True
            )
            route_counts.append(count)
        main_route = tf.argmax(tf.concat(route_counts, axis=-1), axis=-1)
        
        return tf.concat([
            tf.cast(all_zero, tf.float32),
            tf.cast(all_one, tf.float32),
            tf.cast(all_two, tf.float32),
            tf.cast(balanced, tf.float32),
            tf.cast(main_route, tf.float32)[..., tf.newaxis]
        ], axis=-1)

    def _build_advanced_digit_features(self, x):
        """构建高级数字特征"""
        features = []
        
        # 1. 每位数字的独立特征
        for i in range(5):
            digit = x[..., i:i+1]  # 提取第i位数字
            
            # 数字频率统计
            freq = tf.keras.layers.Lambda(
                lambda x: tf.cast(tf.histogram_fixed_width(x, [0, 9], nbins=10), tf.float32)
            )(digit)
            
            # 数字转换模式
            transitions = self._build_digit_transitions(digit)
            
            features.extend([freq, transitions])
        
        # 2. 数字组合特征
        for i in range(5):
            for j in range(i+1, 5):
                # 两位数字组合
                pair = tf.stack([x[..., i], x[..., j]], axis=-1)
                pair_features = self._build_pair_features(pair)
                features.append(pair_features)
        
        # 3. 完整号码特征
        full_number = tf.reshape(x, (-1, self.sequence_length))  # 将5位数字合并为一个完整号码
        number_features = self._build_number_features(full_number)
        features.append(number_features)
        
        return tf.keras.layers.Concatenate()(features)

    def _extract_periodic_pattern(self, x, period):
        """提取周期性模式"""
        # 重塑以匹配周期
        batch_size = tf.shape(x)[0]
        length = tf.shape(x)[1]
        n_periods = length // period
        
        # 重塑为(batch, n_periods, period, features)
        x_reshaped = tf.reshape(x[:, :n_periods*period], 
                               (batch_size, n_periods, period, -1))
        
        # 计算周期内模式
        pattern = tf.reduce_mean(x_reshaped, axis=1)  # 平均周期模式
        variance = tf.math.reduce_variance(x_reshaped, axis=1)  # 周期变异性
        
        return tf.concat([pattern, variance], axis=-1)

    def _build_statistical_features(self, x):
        """构建统计特征"""
        # 1. 移动统计
        windows = [60, 360, 720]  # 1小时、6小时、12小时
        stats = []
        
        for window in windows:
            # 移动平均
            ma = tf.keras.layers.AveragePooling1D(
                pool_size=window, strides=1, padding='same')(x)
            # 移动标准差
            std = tf.math.reduce_std(
                tf.stack([x, ma], axis=-1), axis=-1)
            # 移动极差
            pooled_max = tf.keras.layers.MaxPooling1D(
                pool_size=window, strides=1, padding='same')(x)
            pooled_min = -tf.keras.layers.MaxPooling1D(
                pool_size=window, strides=1, padding='same')(-x)
            range_stat = pooled_max - pooled_min
            
            stats.extend([ma, std, range_stat])
        
        # 2. 概率分布特征
        probs = self._build_probability_features(x)
        
        return tf.concat(stats + [probs], axis=-1)

    def _build_probability_features(self, x):
        """构建概率分布特征"""
        # 1. 条件概率矩阵
        cond_matrix = tf.zeros((10, 10, 5))  # 每位数字的转移概率
        
        # 2. 计算历史转移概率
        for i in range(5):
            current = tf.cast(x[..., i], tf.int32)
            next_digit = tf.roll(current, shift=-1, axis=0)
            
            # 更新转移矩阵
            for j in range(10):
                for k in range(10):
                    mask_current = tf.cast(current == j, tf.float32)
                    mask_next = tf.cast(next_digit == k, tf.float32)
                    prob = tf.reduce_mean(mask_current * mask_next)
                    cond_matrix = tf.tensor_scatter_nd_update(
                        cond_matrix,
                        [[j, k, i]],
                        [prob]
                    )
        
        return cond_matrix

    def _build_periodic_pattern_model(self, x, params):
        """周期性模式捕获模型"""
        # 1. 多尺度周期性分析
        periods = [12, 24, 60, 120, 360]
        period_features = []
        
        for period in periods:
            # 周期性卷积
            conv = Conv1D(32, kernel_size=period, strides=1, padding='same')(x)
            # 周期性池化
            pool = tf.keras.layers.MaxPooling1D(pool_size=period)(conv)
            period_features.append(pool)
        
        # 合并周期特征
        x = tf.concat(period_features, axis=-1)
        
        # 2. 时序注意力
        x = self._build_temporal_attention(x, params)
        
        return x

    def _build_probability_head(self, x):
        """概率输出头"""
        # 1. 每位数字的概率分布
        digit_probs = []
        for i in range(5):
            # 输出每位数字0-9的概率
            digit_prob = Dense(10, activation='softmax', name=f'digit_{i}')(x)
            digit_probs.append(digit_prob)
        
        # 2. 组合概率
        combination_prob = Dense(1, activation='sigmoid', name='combination')(x)
        
        # 3. 置信度
        confidence = Dense(1, activation='sigmoid', name='confidence')(x)
        
        return {
            'digits': digit_probs,
            'combination': combination_prob,
            'confidence': confidence
        }

    def compile_model(self, model):
        """编译模型"""
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=self.enhanced_match_loss,  # 使用增强型匹配损失函数
            metrics=['accuracy']
        )

    def _build_digit_correlation_model(self, x, params):
        """数字关联分析模型"""
        # 1. 相邻数字关系
        adjacent_patterns = Conv1D(64, kernel_size=2, strides=1, padding='same')(x)
        
        # 2. 数字组合模式
        combination_patterns = []
        for window in [3, 5, 7]:
            pattern = Conv1D(32, kernel_size=window, padding='same')(x)
            combination_patterns.append(pattern)
        
        # 3. 数字重复模式
        repeat_patterns = self._build_repeat_patterns(x)
        
        # 合并所有模式
        x = tf.keras.layers.Concatenate()([
            adjacent_patterns,
            *combination_patterns,
            repeat_patterns
        ])
        
        # LSTM层捕获长期关联
        x = LSTM(128, return_sequences=True)(x)
        return x

    def _build_trend_prediction_model(self, x, params):
        """趋势预测模型"""
        # 1. 移动平均特征
        ma_features = []
        for window in [12, 24, 48, 96]:
            ma = tf.keras.layers.AveragePooling1D(
                pool_size=window, 
                strides=1, 
                padding='same'
            )(x)
            ma_features.append(ma)
        
        # 2. 趋势特征
        trend = self._build_trend_features(x)
        
        # 3. 组合特征
        x = tf.keras.layers.Concatenate()([*ma_features, trend])
        
        # 4. 双向GRU捕获双向趋势
        x = Bidirectional(GRU(64, return_sequences=True))(x)
        return x

    def _build_combination_pattern_model(self, x, params):
        """组合模式识别模型"""
        # 1. 局部组合模式
        local_patterns = self._build_local_patterns(x)
        
        # 2. 全局组合模式
        global_patterns = self._build_global_patterns(x)
        
        # 3. 注意力机制关注重要组合
        x = MultiHeadAttention(
            num_heads=8,
            key_dim=32
        )(tf.concat([local_patterns, global_patterns], axis=-1))
        
        return x

    def _build_probability_model(self, x, params):
        """概率分布学习模型"""
        # 1. 历史概率分布
        hist_probs = self._build_historical_probabilities(x)
        
        # 2. 条件概率特征
        cond_probs = self._build_conditional_probabilities(x)
        
        # 3. 组合概率模型
        x = tf.keras.layers.Concatenate()([hist_probs, cond_probs])
        x = Dense(256, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        
        return x

    def _build_digit_transition_model(self, x, params):
        """数字转换预测模型"""
        # 1. 数字转换矩阵
        transition_matrix = self._build_transition_matrix(x)
        
        # 2. 状态转换LSTM
        x = LSTM(128, return_sequences=True)(x)
        
        # 3. 注意力加权
        x = MultiHeadAttention(
            num_heads=4,
            key_dim=32
        )(x, transition_matrix)
        
        return x

    def _build_temporal_attention(self, x, params):
        """构建时序注意力层"""
        # 多头自注意力
        attention_output = MultiHeadAttention(
            num_heads=8,
            key_dim=64,
            value_dim=64,
            dropout=0.1
        )(x, x)
        
        # 残差连接和层归一化
        x = Add()([x, attention_output])
        x = LayerNormalization()(x)
        
        # 前馈网络
        ffn = Dense(256, activation='relu')(x)
        ffn = Dense(x.shape[-1])(ffn)
        
        # 再次残差连接和层归一化
        x = Add()([x, ffn])
        x = LayerNormalization()(x)
        
        return x 

    def _build_combination_predictor(self, x, params):
        """组合模式预测器"""
        # 1. 提取数字组合特征
        combinations = []
        for i in range(5):
            for j in range(i+1, 5):
                # 两位数字组合的频率分析
                pair = tf.stack([x[..., i], x[..., j]], axis=-1)
                pair_conv = Conv1D(32, kernel_size=2, padding='same')(pair)
                combinations.append(pair_conv)
        
        # 2. 组合模式分析
        x = tf.keras.layers.Concatenate()(combinations)
        x = LSTM(128, return_sequences=True)(x)
        x = MultiHeadAttention(num_heads=8, key_dim=32)(x, x)
        return x

    def _build_probability_predictor(self, x, params):
        """概率分布预测器"""
        # 1. 计算历史概率分布
        digit_probs = []
        for i in range(5):
            # 每位数字的概率分布
            digit = tf.cast(x[..., i], tf.int32)
            probs = tf.keras.layers.Lambda(
                lambda x: tf.cast(tf.histogram_fixed_width(x, [0, 9], nbins=10), tf.float32)
            )(digit)
            digit_probs.append(probs)
        
        # 2. 条件概率计算
        cond_probs = self._build_conditional_probabilities(x)
        
        # 3. 概率模型
        x = tf.keras.layers.Concatenate()(digit_probs + [cond_probs])
        x = Dense(256, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        return x

    def _build_periodic_predictor(self, x, params):
        """周期模式预测器"""
        # 1. 多周期分析
        periods = [60, 120, 360, 720, 1440]  # 1小时到24小时的周期
        period_features = []
        
        for period in periods:
            # 提取周期特征
            pattern = self._extract_periodic_pattern(x, period)
            # 周期性注意力
            attention = MultiHeadAttention(
                num_heads=4, 
                key_dim=32
            )(pattern, pattern)
            period_features.append(attention)
        
        # 2. 周期特征融合
        x = tf.keras.layers.Concatenate()(period_features)
        x = Dense(256, activation='relu')(x)
        return x

    def _build_trend_predictor(self, x, params):
        """趋势预测器"""
        # 1. 多尺度趋势分析
        trends = []
        windows = [60, 360, 720, 1440]  # 不同时间尺度
        
        for window in windows:
            # 移动平均
            ma = tf.keras.layers.AveragePooling1D(
                pool_size=window, strides=1, padding='same')(x)
            # 趋势方向
            trend = tf.sign(x - ma)
            trends.append(trend)
        
        # 2. 趋势特征融合
        x = tf.keras.layers.Concatenate()(trends)
        x = Bidirectional(LSTM(64, return_sequences=True))(x)
        return x

    def _build_statistical_predictor(self, x, params):
        """统计模式预测器"""
        # 1. 统计特征提取
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
        
        # 2. 统计模型
        x = tf.keras.layers.Concatenate()(stats)
        x = Dense(128, activation='relu')(x)
        return x

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
        
        # 3. 注意力加权
        x = MultiHeadAttention(
            num_heads=8,
            key_dim=64,
            value_dim=64
        )(x, x)
        
        # 4. 最终预测层
        x = Dense(256, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        
        return x 

    def _build_number_features(self, numbers):
        """提取完整号码特征"""
        # 1. 号码间隔分析
        intervals = self._analyze_intervals(numbers)
        
        # 2. 号码重复模式
        repeats = self._analyze_repeats(numbers)
        
        # 3. 号码差值特征
        diffs = self._analyze_differences(numbers)
        
        # 4. 号码分布特征
        distributions = self._analyze_distributions(numbers)
        
        return tf.keras.layers.Concatenate()([
            intervals, repeats, diffs, distributions
        ])

    def _analyze_intervals(self, numbers):
        """分析号码出现间隔"""
        def get_intervals(sequence):
            # 计算每个号码的出现间隔
            intervals_dict = {}
            for i in range(len(sequence)):
                num = sequence[i]
                if num not in intervals_dict:
                    intervals_dict[num] = []
                else:
                    last_pos = max(idx for idx, val in enumerate(sequence[:i]) if val == num)
                    intervals_dict[num].append(i - last_pos)
            return intervals_dict
        
        intervals = tf.keras.layers.Lambda(get_intervals)(numbers)
        return Dense(64, activation='relu')(intervals) 

    def _build_prediction_head(self, x):
        """构建预测输出头"""
        # 1. 每位数字的概率分布
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

    def _build_pattern_features(self, x):
        """构建形态特征分析"""
        # 1. 当前形态识别
        current_patterns = self._identify_patterns(x)
        
        # 2. 形态遗漏值分析
        pattern_gaps = self._analyze_pattern_gaps(x)
        
        # 3. 形态转换规律
        pattern_transitions = self._analyze_pattern_transitions(x)
        
        return tf.keras.layers.Concatenate()([
            current_patterns,
            pattern_gaps,
            pattern_transitions
        ])

    def _identify_patterns(self, x):
        """识别号码形态"""
        patterns = []
        
        # 1. 豹子号(AAAAA)
        baozi = tf.reduce_all(x == x[..., :1], axis=-1, keepdims=True)
        patterns.append(tf.cast(baozi, tf.float32))
        
        # 2. 组5(AAAAB)
        sorted_x = tf.sort(x, axis=-1)
        zu5 = tf.logical_and(
            tf.reduce_sum(tf.cast(sorted_x[..., :4] == sorted_x[..., :1], tf.float32), axis=-1) == 4,
            sorted_x[..., 4] != sorted_x[..., 0]
        )
        patterns.append(tf.cast(zu5, tf.float32)[..., tf.newaxis])
        
        # 3. 组10(AAABB)
        zu10 = tf.logical_and(
            tf.reduce_sum(tf.cast(sorted_x[..., :3] == sorted_x[..., :1], tf.float32), axis=-1) == 3,
            tf.reduce_sum(tf.cast(sorted_x[..., 3:] == sorted_x[..., 3:4], tf.float32), axis=-1) == 2
        )
        patterns.append(tf.cast(zu10, tf.float32)[..., tf.newaxis])
        
        # 4. 组20(AAABC)
        zu20 = tf.logical_and(
            tf.reduce_sum(tf.cast(sorted_x[..., :3] == sorted_x[..., :1], tf.float32), axis=-1) == 3,
            sorted_x[..., 3] != sorted_x[..., 4]
        )
        patterns.append(tf.cast(zu20, tf.float32)[..., tf.newaxis])
        
        # 5. 组30(AABBC)
        zu30 = tf.logical_and(
            tf.reduce_sum(tf.cast(sorted_x[..., :2] == sorted_x[..., :1], tf.float32), axis=-1) == 2,
            tf.reduce_sum(tf.cast(sorted_x[..., 2:4] == sorted_x[..., 2:3], tf.float32), axis=-1) == 2
        )
        patterns.append(tf.cast(zu30, tf.float32)[..., tf.newaxis])
        
        # 6. 组60(AABCD)
        zu60 = tf.logical_and(
            tf.reduce_sum(tf.cast(sorted_x[..., :2] == sorted_x[..., :1], tf.float32), axis=-1) == 2,
            tf.reduce_all(sorted_x[..., 2:] != sorted_x[..., 1:4], axis=-1)
        )
        patterns.append(tf.cast(zu60, tf.float32)[..., tf.newaxis])
        
        # 7. 组120(ABCDE)
        zu120 = tf.reduce_all(sorted_x[..., 1:] > sorted_x[..., :-1], axis=-1, keepdims=True)
        patterns.append(tf.cast(zu120, tf.float32))
        
        return tf.concat(patterns, axis=-1)

    def _analyze_pattern_gaps(self, x):
        """分析形态遗漏值"""
        # 1. 初始化遗漏值计数器
        gap_counters = tf.zeros_like(x[..., :7])  # 7种形态
        
        # 2. 计算每种形态的遗漏值
        def update_gaps(sequence):
            gaps = []
            for i in range(7):  # 7种形态
                last_pos = -1
                current_gap = 0
                pattern_positions = tf.where(sequence[:, i])
                
                if tf.size(pattern_positions) > 0:
                    last_pos = tf.reduce_max(pattern_positions)
                    current_gap = tf.shape(sequence)[0] - 1 - last_pos
                
                gaps.append(current_gap)
            return tf.stack(gaps)
        
        # 3. 应用遗漏值计算
        patterns = self._identify_patterns(x)
        gaps = tf.keras.layers.Lambda(update_gaps)(patterns)
        
        # 4. 遗漏值特征
        gap_features = []
        
        # 当前遗漏值
        gap_features.append(gaps)
        
        # 历史最大遗漏值
        max_gaps = tf.reduce_max(gaps, axis=0, keepdims=True)
        gap_features.append(max_gaps)
        
        # 历史平均遗漏值
        mean_gaps = tf.reduce_mean(gaps, axis=0, keepdims=True)
        gap_features.append(mean_gaps)
        
        # 遗漏值分布
        gap_dist = tf.keras.layers.Lambda(
            lambda x: tf.cast(tf.histogram_fixed_width(x, [0, 1000], nbins=50), tf.float32)
        )(gaps)
        gap_features.append(gap_dist)
        
        return tf.concat(gap_features, axis=-1)

    def _analyze_pattern_transitions(self, x):
        """分析形态转换规律"""
        # 1. 识别所有形态
        patterns = self._identify_patterns(x)
        
        # 2. 计算形态转换矩阵
        def get_transition_matrix(sequence):
            matrix = tf.zeros((7, 7))  # 7x7转换矩阵
            for i in range(len(sequence)-1):
                current = tf.argmax(sequence[i])
                next_pattern = tf.argmax(sequence[i+1])
                matrix = tf.tensor_scatter_nd_update(
                    matrix,
                    [[current, next_pattern]],
                    [1.0]
                )
            return matrix
        
        transition_matrix = tf.keras.layers.Lambda(get_transition_matrix)(patterns)
        
        # 3. 提取转换特征
        transitions = []
        
        # 转换概率
        prob_matrix = transition_matrix / (tf.reduce_sum(transition_matrix, axis=-1, keepdims=True) + 1e-7)
        transitions.append(tf.reshape(prob_matrix, [-1]))
        
        # 最常见转换路径
        common_transitions = tf.reduce_max(prob_matrix, axis=-1)
        transitions.append(common_transitions)
        
        # 形态稳定性(自我转换概率)
        stability = tf.linalg.diag_part(prob_matrix)
        transitions.append(stability)
        
        return tf.concat(transitions, axis=-1) 

    def parallel_training(self, data):
        """补全并行训练逻辑"""
        try:
            # 根据当前线程数调整
            with ThreadPoolExecutor(max_workers=config_manager.SYSTEM_CONFIG['max_threads']) as executor:
                # 动态分配任务
                chunk_size = len(data) // self.threads
                futures = []
                for i in range(self.threads):
                    chunk = data[i*chunk_size : (i+1)*chunk_size]
                    futures.append(executor.submit(self._train_chunk, chunk))
            return True
        except Exception as e:
            logger.error(f"训练失败: {str(e)}")
            return False

    def _forward_pass(self, inputs):
        """实现多模型并行前向传播"""
        # 使用线程池并行执行
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(model, inputs) for model in self.models]
        return [f.result() for f in futures]

    def _calculate_loss(self, preds, targets):
        """计算集成损失"""
        # 加权平均各模型损失
        losses = [self.enhanced_match_loss(p, targets) for p in preds]
        return sum(w*l for w, l in zip(self.weights, losses))

    def _backward_pass(self, loss):
        """实现多模型并行反向传播"""
        # 使用线程池并行执行
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self._backward_single_model, loss) for _ in range(len(self.models))]
        for f in futures:
            f.result()

    def _update_parameters(self):
        """实现多模型并行参数更新"""
        # 使用线程池并行执行
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self._update_single_model) for _ in range(len(self.models))]
        for f in futures:
            f.result()

    def _init_training(self, data):
        """实现训练初始化逻辑"""
        # 实现训练初始化逻辑
        pass  # 临时实现，需要根据实际需求实现

    def _backward_single_model(self, loss):
        """实现单个模型反向传播逻辑"""
        # 实现单个模型反向传播逻辑
        pass  # 临时实现，需要根据实际需求实现

    def _update_single_model(self):
        """实现单个模型参数更新逻辑"""
        # 实现单个模型参数更新逻辑
        pass  # 临时实现，需要根据实际需求实现

    def _save_final_issue(self, final_issue):
        """保存最终期号"""
        # 由最后一个完成训练的模型执行
        if self.finished_models == self.total_models:
            with open('D:\JupyterWork\comparison\issue_number.txt', 'a') as f:
                f.write(f"\n{final_issue}")

    def save_model(self, model_idx):
        model = self.models[model_idx]
        model.save(os.path.join(config_manager.MODEL_DIR, f'model_{model_idx}')) 

    def save_ensemble_info(self):
        with open('ensemble_info.json', 'w') as f:
            json.dump({
                'weights': self.weights.tolist(),
                'performance': self.performance_history
            }, f) 

    def load_pretrained_models(self, model_paths):
        for path in model_paths:
            # 添加device参数适配CPU
            model = torch.load(path, map_location=torch.device('cpu'))  
            self.models.append(model) 