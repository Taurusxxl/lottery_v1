#7 Data Optimization Module / 数据优化模块
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

    def _calculate_trend(self, values):
        """计算趋势"""
        if len(values) < 2:
            return "INSUFFICIENT_DATA"
            
        # 使用简单线性回归
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope < -0.01:
            return "IMPROVING"
        elif slope > 0.01:
            return "DEGRADING"
        else:
            return "STABLE"

    def _compute_correlation(self, x1, x2):
        """计算相关系数"""
        try:
            # 标准化
            x1_norm = (x1 - np.mean(x1)) / np.std(x1)
            x2_norm = (x2 - np.mean(x2)) / np.std(x2)
            
            # 计算相关系数
            corr = np.mean(x1_norm * x2_norm)
            return corr
            
        except Exception as e:
            logger.error(f"计算相关系数时出错: {str(e)}")
            return 0.0

    def optimize_feature_selection(self, X, y, n_features=10):
        """优化特征选择"""
        try:
            # 1. 计算特征重要性
            importance = self._calculate_feature_importance(X, y)
            
            # 2. 计算特征冗余度
            redundancy = self._calculate_feature_redundancy(X)
            
            # 3. 综合评分
            final_score = importance * (1 - redundancy)
            
            # 4. 选择最优特征
            selected = np.argsort(final_score)[-n_features:]
            
            return selected, final_score[selected]
            
        except Exception as e:
            logger.error(f"优化特征选择时出错: {str(e)}")
            return None, None
    
    def _calculate_feature_importance(self, X, y):
        """计算特征重要性分数"""
        try:
            correlations = []
            for i in range(X.shape[1]):
                corr = np.abs(np.corrcoef(X[:, i], y)[0, 1])
                correlations.append(corr)
            return np.array(correlations)
        except Exception as e:
            logger.error(f"计算特征重要性时出错: {str(e)}")
            return None

    def _calculate_feature_redundancy(self, X):
        """计算特征冗余度"""
        try:
            n_features = X.shape[1]
            redundancy = np.zeros(n_features)
            
            for i in range(n_features):
                correlations = []
                for j in range(n_features):
                    if i != j:
                        corr = np.abs(np.corrcoef(X[:, i], X[:, j])[0, 1])
                        correlations.append(corr)
                redundancy[i] = np.mean(correlations)
            
            return redundancy
            
        except Exception as e:
            logger.error(f"计算特征冗余度时出错: {str(e)}")
            return None

    def evaluate_feature_correlation(self, features):
        """评估特征相关性矩阵"""
        try:
            corr_matrix = np.corrcoef(features.T)
            return corr_matrix
        except Exception as e:
            logger.error(f"评估特征相关性时出错: {str(e)}")
            return None

    def optimize_feature_combination(self, features, target):
        """优化特征组合"""
        try:
            # 1. 计算相关性
            correlations = self.evaluate_feature_correlation(features)
            
            # 2. 计算稳定性
            stability = self._analyze_feature_stability(features)
            
            # 3. 计算目标相关性
            target_corr = np.array([
                abs(np.corrcoef(features[:, i], target)[0, 1])
                for i in range(features.shape[1])
            ])
            
            # 4. 优化组合
            scores = target_corr * stability
            return scores
            
        except Exception as e:
            logger.error(f"优化特征组合时出错: {str(e)}")
            return None

    def _analyze_feature_stability(self, features):
        """分析特征稳定性"""
        try:
            stability_scores = []
            for i in range(features.shape[1]):
                # 计算特征的变异系数
                cv = np.std(features[:, i]) / np.mean(np.abs(features[:, i]))
                # 转换为稳定性分数
                stability = 1 / (1 + cv)
                stability_scores.append(stability)
            return np.array(stability_scores)
        except Exception as e:
            logger.error(f"分析特征稳定性时出错: {str(e)}")
            return None