
import tensorflow as tf
import numpy as np
import logging
from datetime import datetime
from cell1 import config_instance
from collections import deque

# 获取logger实例
logger = logging.getLogger(__name__)

class Predictor:
    """预测器类"""
    
    def __init__(self, model_ensemble):
        self.model_ensemble = model_ensemble
        self.prediction_range = 2880
        self.prediction_history = deque(maxlen=1000)
        self.confidence_threshold = 0.8
        
        # 预测结果缓存
        self.cache = {}
        self.cache_timeout = 300  # 5分钟缓存超时
        
        # 初始化各类预测器
        self._init_predictors()
        logger.info("预测器初始化完成")
        
    def _init_predictors(self):
        """初始化专门的预测器"""
        self.combination_predictor = self._build_combination_predictor()
        self.probability_predictor = self._build_probability_predictor()
        self.periodic_predictor = self._build_periodic_predictor()
        self.trend_predictor = self._build_trend_predictor()
        self.statistical_predictor = self._build_statistical_predictor()
    
    def get_ensemble_prediction(self, X):
        """获取集成预测结果"""
        try:
            # 1. 获取各模型预测
            predictions = []
            confidences = []
            
            for i, model in enumerate(self.model_ensemble.models):
                pred = model.predict(X)
                pred_value = pred['digits']
                confidence = pred['confidence']
                
                predictions.append(pred_value * self.model_ensemble.weights[i])
                confidences.append(confidence)
            
            # 2. 集成预测结果
            ensemble_pred = np.sum(predictions, axis=0)
            mean_confidence = np.mean(confidences)
            
            # 3. 记录预测历史
            self._record_prediction(ensemble_pred, mean_confidence)
            
            # 4. 返回结果和置信度
            return {
                'prediction': ensemble_pred,
                'confidence': mean_confidence
            }
            
        except Exception as e:
            logger.error(f"集成预测失败: {str(e)}")
            return None

    def _build_combination_predictor(self):
        """组合模式预测器"""
        try:
            inputs = tf.keras.Input(shape=(self.sequence_length, 5))
            
            # 1. 提取数字组合特征
            combinations = []
            for i in range(5):
                for j in range(i+1, 5):
                    # 两位数字组合的频率分析
                    pair = tf.stack([inputs[..., i], inputs[..., j]], axis=-1)
                    pair_conv = tf.keras.layers.Conv1D(32, kernel_size=2, padding='same')(pair)
                    combinations.append(pair_conv)
            
            # 2. 组合模式分析
            x = tf.keras.layers.Concatenate()(combinations)
            x = tf.keras.layers.LSTM(128, return_sequences=True)(x)
            x = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=32)(x, x)
            
            # 3. 预测头
            outputs = self._build_prediction_head(x)
            
            return tf.keras.Model(inputs=inputs, outputs=outputs)
            
        except Exception as e:
            logger.error(f"构建组合预测器失败: {str(e)}")
            return None

    def _build_probability_predictor(self):
        """概率分布预测器"""
        try:
            inputs = tf.keras.Input(shape=(self.sequence_length, 5))
            
            # 1. 计算历史概率分布
            digit_probs = []
            for i in range(5):
                # 每位数字的概率分布
                digit = tf.cast(inputs[..., i], tf.int32)
                probs = tf.keras.layers.Lambda(
                    lambda x: tf.cast(tf.histogram_fixed_width(
                        x, [0, 9], nbins=10
                    ), tf.float32)
                )(digit)
                digit_probs.append(probs)
            
            # 2. 条件概率计算
            cond_probs = self._build_conditional_probabilities(inputs)
            
            # 3. 概率模型
            x = tf.keras.layers.Concatenate()(digit_probs + [cond_probs])
            x = tf.keras.layers.Dense(256, activation='relu')(x)
            x = tf.keras.layers.Dense(128, activation='relu')(x)
            
            # 4. 预测头
            outputs = self._build_prediction_head(x)
            
            return tf.keras.Model(inputs=inputs, outputs=outputs)
            
        except Exception as e:
            logger.error(f"构建概率预测器失败: {str(e)}")
            return None

    def _build_periodic_predictor(self):
        """周期模式预测器"""
        try:
            inputs = tf.keras.Input(shape=(self.sequence_length, 5))
            
            # 1. 多周期分析
            periods = [60, 120, 360, 720, 1440]  # 1小时到24小时的周期
            period_features = []
            
            for period in periods:
                # 提取周期特征
                pattern = self._extract_periodic_pattern(inputs, period)
                # 周期性注意力
                attention = tf.keras.layers.MultiHeadAttention(
                    num_heads=4, 
                    key_dim=32
                )(pattern, pattern)
                period_features.append(attention)
            
            # 2. 周期特征融合
            x = tf.keras.layers.Concatenate()(period_features)
            x = tf.keras.layers.Dense(256, activation='relu')(x)
            
            # 3. 预测头
            outputs = self._build_prediction_head(x)
            
            return tf.keras.Model(inputs=inputs, outputs=outputs)
            
        except Exception as e:
            logger.error(f"构建周期预测器失败: {str(e)}")
            return None

    def _build_trend_predictor(self):
        """趋势预测器"""
        try:
            inputs = tf.keras.Input(shape=(self.sequence_length, 5))
            
            # 1. 多尺度趋势分析
            trends = []
            windows = [60, 360, 720, 1440]  # 不同时间尺度
            
            for window in windows:
                # 移动平均
                ma = tf.keras.layers.AveragePooling1D(
                    pool_size=window, strides=1, padding='same')(inputs)
                # 趋势方向
                trend = tf.sign(inputs - ma)
                trends.append(trend)
            
            # 2. 趋势特征融合
            x = tf.keras.layers.Concatenate()(trends)
            x = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(64, return_sequences=True)
            )(x)
            
            # 3. 预测头
            outputs = self._build_prediction_head(x)
            
            return tf.keras.Model(inputs=inputs, outputs=outputs)
            
        except Exception as e:
            logger.error(f"构建趋势预测器失败: {str(e)}")
            return None

    def _build_statistical_predictor(self):
        """统计模式预测器"""
        try:
            inputs = tf.keras.Input(shape=(self.sequence_length, 5))
            
            # 1. 统计特征提取
            stats = []
            
            # 均值特征
            mean = tf.reduce_mean(inputs, axis=1, keepdims=True)
            # 标准差特征
            std = tf.math.reduce_std(inputs, axis=1, keepdims=True)
            # 峰度
            kurtosis = tf.reduce_mean(
                tf.pow(inputs - mean, 4), axis=1, keepdims=True
            ) / tf.pow(std, 4)
            # 偏度
            skewness = tf.reduce_mean(
                tf.pow(inputs - mean, 3), axis=1, keepdims=True
            ) / tf.pow(std, 3)
            
            stats.extend([mean, std, kurtosis, skewness])
            
            # 2. 统计模型
            x = tf.keras.layers.Concatenate()(stats)
            x = tf.keras.layers.Dense(128, activation='relu')(x)
            
            # 3. 预测头
            outputs = self._build_prediction_head(x)
            
            return tf.keras.Model(inputs=inputs, outputs=outputs)
            
        except Exception as e:
            logger.error(f"构建统计预测器失败: {str(e)}")
            return None

    def _build_prediction_head(self, x):
        """构建预测输出头"""
        # 1. 每位数字的概率分布
        digit_predictions = []
        for i in range(5):
            digit_pred = tf.keras.layers.Dense(
                10, activation='softmax', name=f'digit_{i}'
            )(x)
            digit_predictions.append(digit_pred)
        
        # 2. 完整号码匹配概率
        match_prob = tf.keras.layers.Dense(
            self.prediction_range, activation='sigmoid', name='match_prob'
        )(x)
        
        # 3. 预测置信度
        confidence = tf.keras.layers.Dense(
            1, activation='sigmoid', name='confidence'
        )(x)
        
        return {
            'digits': digit_predictions,
            'match_prob': match_prob,
            'confidence': confidence
        }

    def _record_prediction(self, prediction, confidence):
        """记录预测结果"""
        self.prediction_history.append({
            'prediction': prediction,
            'confidence': confidence,
            'timestamp': datetime.now()
        })

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

    def _build_conditional_probabilities(self, x):
        """构建条件概率特征"""
        # 条件概率矩阵初始化
        cond_matrix = tf.zeros((10, 10, 5))
        
        # 计算每位数字的转移概率
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
            
        # 计算整体准确率
        correct = 0
        total = 0
        
        for pred in self.prediction_history:
            if pred.get('actual') is not None:
                correct += int(np.array_equal(
                    pred['prediction'], pred['actual']
                ))
                total += 1
        
        accuracy = correct / total if total > 0 else 0
        
        return {
            'accuracy': accuracy,
            'total_predictions': total,
            'correct_predictions': correct
        }

# 创建全局实例 
predictor = Predictor(model_ensemble=None)
