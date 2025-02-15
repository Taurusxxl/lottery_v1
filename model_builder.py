
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, GRU, Bidirectional, Conv1D
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Add
import logging

# 获取logger实例
logger = logging.getLogger(__name__)

class ModelBuilder:
    """模型构建类"""
    
    def __init__(self, sequence_length=14400, feature_dim=5):
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.prediction_range = 2880
    
    def build_model(self, model_num, params):
        """构建增强版单个模型"""
        try:
            inputs = tf.keras.Input(shape=(self.sequence_length, self.feature_dim))
            
            # 1. 基础特征提取
            x = self._build_basic_features(inputs)
            
            # 2. 高级特征分析
            advanced_features = self._build_advanced_features(inputs)
            
            # 3. 特征融合
            x = tf.keras.layers.Concatenate()([x, advanced_features])
            
            # 4. 根据模型编号选择不同的深度特征提取方法
            x = self._build_model_specific_features(x, model_num, params)
            
            # 5. 预测头
            outputs = self._build_prediction_head(x)
            
            return Model(inputs=inputs, outputs=outputs)
            
        except Exception as e:
            logger.error(f"构建模型 {model_num} 时出错: {str(e)}")
            raise

    def _build_model_specific_features(self, x, model_num, params):
        """根据模型编号构建特定特征"""
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
        else:
            return self._build_lstm_cnn(x, params)

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

    def _build_transformer(self, x, params):
        """构建Transformer模型"""
        x = MultiHeadAttention(
            num_heads=params['num_heads'],
            key_dim=params['key_dim']
        )(x, x)
        return LayerNormalization()(x)

    def _build_gru_attention_skip(self, x, params):
        """GRU + 自注意力 + 跳跃连接"""
        # GRU层
        gru_out = GRU(params['gru_units'], return_sequences=True)(x)
        
        # 自注意力
        att = MultiHeadAttention(num_heads=4, key_dim=16)(gru_out, gru_out)
        
        # 跳跃连接
        x = Add()([att, x])
        return LayerNormalization()(x)

    def _build_lstm_cnn(self, x, params):
        """LSTM + CNN模型"""
        x = LSTM(params['lstm_units'], return_sequences=True)(x)
        x = Conv1D(filters=16, kernel_size=3, padding='same')(x)
        return tf.keras.layers.BatchNormalization()(x)

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

    def _build_digit_correlation_model(self, x, params):
        """数字关联分析模型"""
        # 1. 相邻数字关系
        adjacent_patterns = Conv1D(64, kernel_size=2, strides=1, padding='same')(x)
        
        # 2. 数字组合模式
        combination_patterns = []
        for window in [3, 5, 7]:
            pattern = Conv1D(32, kernel_size=window, padding='same')(x)
            combination_patterns.append(pattern)
        
        # 3. 合并所有模式
        x = tf.keras.layers.Concatenate()([
            adjacent_patterns,
            *combination_patterns
        ])
        
        # LSTM层捕获长期关联
        x = LSTM(128, return_sequences=True)(x)
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

# 创建全局实例
model_builder = ModelBuilder()
