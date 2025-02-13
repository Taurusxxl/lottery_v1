#12 数据优化策略\data_optimizer.py
import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import adfuller
import logging

logger = logging.getLogger(__name__)

class DataOptimizer:
    def _evaluate_distribution(self, X):
        """评估数据分布情况"""
        try:
            # 1. 检查数据偏度
            skewness = np.abs(np.mean([np.abs(stats.skew(X[:, i])) for i in range(X.shape[1])]))
            skewness_score = 1 / (1 + skewness)  # 转换为0-1分数
            
            # 2. 检查数据峰度
            kurtosis = np.abs(np.mean([np.abs(stats.kurtosis(X[:, i])) for i in range(X.shape[1])]))
            kurtosis_score = 1 / (1 + kurtosis)  # 转换为0-1分数
            
            # 3. 检查异常值比例
            z_scores = np.abs(stats.zscore(X))
            outlier_ratio = np.mean(z_scores > 3)  # 3个标准差以外视为异常值
            outlier_score = 1 - outlier_ratio
            
            # 计算加权平均分数
            distribution_score = (
                0.4 * skewness_score +
                0.3 * kurtosis_score +
                0.3 * outlier_score
            )
            
            return distribution_score
            
        except Exception as e:
            logger.error(f"评估数据分布时出错: {str(e)}")
            return 0.0

    def _evaluate_correlation(self, X):
        """评估特征相关性"""
        try:
            # 1. 计算特征间相关系数矩阵
            corr_matrix = np.corrcoef(X.T)
            
            # 2. 计算特征间的平均相关性
            # 去除对角线上的1
            mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
            avg_correlation = np.mean(np.abs(corr_matrix[mask]))
            
            # 3. 计算相关性得分
            correlation_score = 1 - avg_correlation  # 越小越好
            
            return correlation_score
            
        except Exception as e:
            logger.error(f"评估特征相关性时出错: {str(e)}")
            return 0.0

    def _evaluate_time_series(self, X):
        """评估时间序列特性"""
        try:
            # 1. 检查平稳性
            stationarity_scores = []
            for i in range(X.shape[1]):
                # 使用ADF测试检查平稳性
                adf_result = adfuller(X[:, i])[1]  # 获取p值
                stationarity_scores.append(1 - min(adf_result, 1))  # 转换为0-1分数
            
            stationarity_score = np.mean(stationarity_scores)
            
            # 2. 检查自相关性
            autocorr_scores = []
            for i in range(X.shape[1]):
                # 计算滞后1期的自相关系数
                autocorr = np.corrcoef(X[1:, i], X[:-1, i])[0, 1]
                autocorr_scores.append(abs(autocorr))
            
            autocorr_score = np.mean(autocorr_scores)
            
            # 3. 检查趋势性
            trend_scores = []
            for i in range(X.shape[1]):
                # 使用简单线性回归检测趋势
                slope = np.polyfit(np.arange(len(X)), X[:, i], 1)[0]
                trend_scores.append(abs(slope))
            
            trend_score = 1 / (1 + np.mean(trend_scores))  # 转换为0-1分数
            
            # 计算加权平均分数
            time_series_score = (
                0.4 * stationarity_score +
                0.3 * autocorr_score +
                0.3 * trend_score
            )
            
            return time_series_score
            
        except Exception as e:
            logger.error(f"评估时间序列特性时出错: {str(e)}")
            return 0.0