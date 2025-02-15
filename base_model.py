import tensorflow as tf
import numpy as np
import logging
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MultiHeadAttention, LayerNormalization

# 获取logger实例
logger = logging.getLogger(__name__)

class BaseModel:
    """模型基础类"""
    
    def __init__(self, sequence_length=14400, feature_dim=5):
        """
        初始化基础模型
        Args:
            sequence_length: 输入序列长度
            feature_dim: 特征维度
        """
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.prediction_range = 2880  # 预测范围
        
        # 初始化性能追踪
        self.performance_history = []
        logger.info("基础模型初始化完成")

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
            value_diff = tf.squeeze(best_targets - y_pred, axis=1)  # (batch_size, 5)
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
            direction_weight = tf.exp(-best_match_counts / 5.0) * 0.5
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

    def _build_basic_features(self, x):
        """构建基础特征"""
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
        """添加位置编码"""
        seq_len = tf.shape(x)[1]
        d_model = tf.shape(x)[-1]
        
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

    def _build_statistical_features(self, x):
        """构建统计特征"""
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
        return tf.keras.layers.Concatenate()(stats)

    def compile_model(self, model):
        """编译模型"""
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=self.enhanced_match_loss,
            metrics=['accuracy']
        )

    def _adjust_sequence_length(self, x):
        """调整序列长度到预测范围"""
        current_length = x.shape[1]
        
        if current_length > self.prediction_range:
            pool_size = current_length // self.prediction_range
            x = tf.keras.layers.AveragePooling1D(pool_size=pool_size)(x)
        elif current_length < self.prediction_range:
            x = tf.keras.layers.UpSampling1D(size=self.prediction_range // current_length)(x)
        
        x = tf.keras.layers.Conv1D(filters=32, kernel_size=1)(x)
        x = tf.keras.layers.Reshape((-1, 32))(x)
        x = tf.keras.layers.Dense(32)(x)
        
        return x

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

    def build_model(self, model_type='default'):
        """构建模型
        
        Args:
            model_type: 模型类型，可选值['default', 'probability', 'multi_task']
        """
        try:
            inputs = tf.keras.Input(shape=(self.sequence_length, self.feature_dim))
            
            # 1. 基础特征提取
            x = self._build_basic_features(inputs)
            
            # 2. 根据模型类型选择输出头
            if model_type == 'probability':
                outputs = self._build_probability_head(x)
            elif model_type == 'multi_task':
                outputs = self._build_multi_task_head(x)
            else:
                outputs = self._build_prediction_head(x)
            
            model = Model(inputs=inputs, outputs=outputs)
            self.compile_model(model)
            
            return model
            
        except Exception as e:
            logger.error(f"构建模型时出错: {str(e)}")
            raise

    def train_step(self, batch_data):
        """执行一个训练步骤"""
        try:
            with tf.GradientTape() as tape:
                predictions = self.model(batch_data['input'], training=True)
                loss = self.enhanced_match_loss(batch_data['target'], predictions)
            
            # 计算梯度
            gradients = tape.gradient(loss, self.model.trainable_variables)
            # 应用梯度
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            
            return loss
            
        except Exception as e:
            logger.error(f"训练步骤执行出错: {str(e)}")
            return None

    def validate_step(self, batch_data):
        """执行一个验证步骤"""
        try:
            predictions = self.model(batch_data['input'], training=False)
            loss = self.enhanced_match_loss(batch_data['target'], predictions)
            
            # 计算匹配准确率
            matches = self._calculate_matches(predictions, batch_data['target'])
            
            return {
                'loss': loss,
                'accuracy': tf.reduce_mean(matches)
            }
            
        except Exception as e:
            logger.error(f"验证步骤执行出错: {str(e)}")
            return None

    def _calculate_matches(self, predictions, targets):
        """计算预测匹配程度"""
        try:
            # 对预测值进行四舍五入
            rounded_preds = tf.round(predictions)
            
            # 计算每位数字的匹配情况
            matches = tf.cast(tf.equal(rounded_preds, targets), tf.float32)
            
            # 计算完全匹配的比例
            full_matches = tf.reduce_all(matches, axis=-1)
            
            return full_matches
            
        except Exception as e:
            logger.error(f"计算匹配程度时出错: {str(e)}")
            return tf.zeros_like(predictions[..., 0])

    def _build_temporal_features(self, x):
        """构建时序特征"""
        try:
            # 1. 时间编码
            time_encoding = self._add_positional_encoding(x)
            
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

    def _build_periodic_features(self, x):
        """构建周期性特征"""
        features = []
        
        for period in [60, 120, 360, 720, 1440]:  # 1小时到24小时的周期
            # 1. 提取周期模式
            pattern = self._extract_periodic_pattern(x, period)
            
            # 2. 周期性偏差
            deviation = x - pattern
            
            # 3. 周期强度
            strength = tf.reduce_mean(tf.abs(pattern), axis=-1, keepdims=True)
            
            features.extend([pattern, deviation, strength])
            
        return tf.keras.layers.Concatenate()(features)

    def _build_trend_features(self, x):
        """构建趋势特征"""
        # 1. 短期趋势
        short_ma = tf.keras.layers.AveragePooling1D(
            pool_size=12, strides=1, padding='same')(x)
        short_trend = tf.sign(x - short_ma)
        
        # 2. 中期趋势
        medium_ma = tf.keras.layers.AveragePooling1D(
            pool_size=60, strides=1, padding='same')(x)
        medium_trend = tf.sign(x - medium_ma)
        
        # 3. 长期趋势
        long_ma = tf.keras.layers.AveragePooling1D(
            pool_size=360, strides=1, padding='same')(x)
        long_trend = tf.sign(x - long_ma)
        
        # 4. 趋势一致性
        trend_consistency = tf.reduce_mean(
            tf.cast(short_trend == medium_trend, tf.float32) * 
            tf.cast(medium_trend == long_trend, tf.float32),
            axis=-1, keepdims=True
        )
        
        return tf.concat([short_trend, medium_trend, long_trend, trend_consistency], axis=-1)

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

    def save_model(self, path):
        """保存模型"""
        try:
            self.model.save(path)
            logger.info(f"模型已保存到: {path}")
        except Exception as e:
            logger.error(f"保存模型时出错: {str(e)}")

    def load_model(self, path):
        """加载模型"""
        try:
            self.model = tf.keras.models.load_model(
                path,
                custom_objects={'enhanced_match_loss': self.enhanced_match_loss}
            )
            logger.info(f"已加载模型: {path}")
        except Exception as e:
            logger.error(f"加载模型时出错: {str(e)}")

# 创建全局实例
base_model = BaseModel()
