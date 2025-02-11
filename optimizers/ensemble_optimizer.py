# 集成优化控制\ensemble_optimizer.py
import numpy as np
from bayes_opt import BayesianOptimization
from bayes_opt.util import Events
import tensorflow as tf
import logging
import os
import json
from sklearn.metrics import mutual_info_score
from scipy.stats import entropy

# 获取logger实例
logger = logging.getLogger(__name__)

class EnsembleOptimizer:
    """集成学习超参数优化器"""
    def __init__(self, model_ensemble, data_processor):
        self.model_ensemble = model_ensemble
        self.data_processor = data_processor
        self.best_params = None
        self.optimization_history = []
        
        # 定义参数范围
        self.param_ranges = {
            'weight_params': {
                'initial_weights': (0.1, 0.3),
                'min_weight': (0.05, 0.2),
                'max_weight': (0.8, 0.95),
                'weight_smoothing': (0.1, 0.5)
            },
            'diversity_params': {
                'correlation_threshold': (0.5, 0.9),
                'diversity_weight': (0.1, 0.5),
                'agreement_threshold': (0.7, 0.9)
            },
            'adaptation_params': {
                'adaptation_rate': (0.1, 0.5),
                'performance_window': (100, 1000),
                'min_improvement': (0.001, 0.01)
            }
        }
    
    def _flatten_param_ranges(self):
        """将嵌套的参数范围展平为一维字典"""
        flat_ranges = {}
        for category, params in self.param_ranges.items():
            for param_name, param_range in params.items():
                flat_name = f"{category}__{param_name}"
                flat_ranges[flat_name] = param_range
        return flat_ranges
    
    def _process_params(self, flat_params):
        """将一维参数重构为嵌套结构"""
        nested_params = {}
        for flat_name, value in flat_params.items():
            category, param_name = flat_name.split('__')
            if category not in nested_params:
                nested_params[category] = {}
            nested_params[category][param_name] = value
        return nested_params

    def objective_function(self, **params):
        """优化目标函数"""
        try:
            # 重构参数
            nested_params = self._process_params(params)
            
            # 更新集成参数
            self._update_ensemble_params(nested_params)
            
            # 获取验证数据
            X_val, y_val = self.data_processor.get_validation_data()
            if X_val is None or y_val is None:
                return float('-inf')
            
            # 评估集成性能
            ensemble_score = self._evaluate_ensemble(X_val, y_val)
            diversity_score = self._calculate_diversity()
            
            # 综合得分 (考虑性能和多样性)
            final_score = (0.7 * ensemble_score + 
                         0.3 * diversity_score)
            
            # 记录历史
            self.optimization_history.append({
                'params': nested_params,
                'ensemble_score': ensemble_score,
                'diversity_score': diversity_score,
                'final_score': final_score
            })
            
            return final_score
            
        except Exception as e:
            logger.error(f"优化目标函数执行出错: {str(e)}")
            return float('-inf')
    
    def _evaluate_ensemble(self, X, y):
        """评估集成模型性能"""
        try:
            predictions = self.model_ensemble.predict(X)
            matches = np.any(np.round(predictions) == y, axis=1)
            return np.mean(matches)
        except Exception as e:
            logger.error(f"评估集成性能时出错: {str(e)}")
            return 0.0
    
    def _calculate_diversity(self):
        """计算模型多样性得分"""
        try:
            # 获取所有模型的预测
            predictions = []
            X_val, _ = self.data_processor.get_validation_data()
            
            for model in self.model_ensemble.models:
                pred = model.predict(X_val)
                predictions.append(pred)
            
            # 计算模型间的互信息
            diversity_scores = []
            n_models = len(predictions)
            
            for i in range(n_models):
                for j in range(i + 1, n_models):
                    mi_score = mutual_info_score(
                        predictions[i].ravel(),
                        predictions[j].ravel()
                    )
                    diversity_scores.append(1 - mi_score)  # 转换为多样性分数
            
            return np.mean(diversity_scores)
            
        except Exception as e:
            logger.error(f"计算多样性得分时出错: {str(e)}")
            return 0.0
    
    def optimize(self, n_iter=50):
        """运行优化过程"""
        try:
            optimizer = BayesianOptimization(
                f=self.objective_function,
                pbounds=self._flatten_param_ranges(),
                random_state=42
            )
            
            # 定义优化回调
            def optimization_callback(res):
                progress = self._check_optimization_progress()
                if progress and not progress['is_improving'] and progress['is_stable']:
                    logger.info("优化进入稳定阶段，但仍将继续探索")
            
            optimizer.subscribe(Events.OPTIMIZATION_STEP, optimization_callback)
            optimizer.maximize(
                init_points=5,
                n_iter=n_iter
            )
            
            self.best_params = self._process_params(optimizer.max['params'])
            self._save_best_params()
            
            # 最终优化结果总结
            final_progress = self._check_optimization_progress()
            if final_progress:
                logger.info("优化完成总结:")
                logger.info(f"最终得分: {final_progress['current_score']:.4f}")
                logger.info(f"最佳得分: {final_progress['best_score']:.4f}")
                logger.info(f"优化稳定性: {final_progress['std_score']:.4f}")
            
            return self.best_params
            
        except Exception as e:
            logger.error(f"运行优化过程时出错: {str(e)}")
            return None
    
    def _check_optimization_progress(self):
        """检查优化进度和效果"""
        try:
            if len(self.optimization_history) < 5:
                return
            
            # 获取最近5次和历史最佳得分
            recent_scores = [h['final_score'] for h in self.optimization_history[-5:]]
            best_score = max(h['final_score'] for h in self.optimization_history)
            current_score = recent_scores[-1]
            
            # 计算最近5次的统计信息
            mean_score = np.mean(recent_scores)
            std_score = np.std(recent_scores)
            
            # 判断优化趋势
            is_improving = current_score > mean_score
            is_stable = std_score < 0.1  # 根据实际情况调整阈值
            
            # 记录详细的优化进展
            logger.info("-" * 50)
            logger.info("优化进度检查:")
            logger.info(f"当前得分: {current_score:.4f}")
            logger.info(f"历史最佳: {best_score:.4f}")
            logger.info(f"最近5次得分: {[f'{s:.4f}' for s in recent_scores]}")
            logger.info(f"最近5次平均: {mean_score:.4f} (标准差: {std_score:.4f})")
            logger.info(f"优化趋势: {'改善中' if is_improving else '停滞'}")
            logger.info(f"稳定性: {'稳定' if is_stable else '波动'}")
            logger.info("-" * 50)
            
            return {
                'is_improving': is_improving,
                'is_stable': is_stable,
                'current_score': current_score,
                'best_score': best_score,
                'mean_score': mean_score,
                'std_score': std_score
            }
            
        except Exception as e:
            logger.error(f"检查优化进度时出错: {str(e)}")
            return None
    
    def _save_best_params(self):
        """保存最佳参数"""
        try:
            save_path = os.path.join(BASE_SAVE_DIR, 'best_ensemble_params.json')
            with open(save_path, 'w') as f:
                json.dump(self.best_params, f, indent=4)
            logger.info(f"最佳集成参数已保存到: {save_path}")
        except Exception as e:
            logger.error(f"保存最佳参数时出错: {str(e)}")

    def _update_ensemble_params(self, params):
        """更新集成模型参数"""
        try:
            # 更新权重参数
            weight_params = params.get('weight_params', {})
            self.model_ensemble.update_weights(weight_params)
            
            # 更新多样性参数
            diversity_params = params.get('diversity_params', {})
            self.model_ensemble.update_diversity_settings(diversity_params)
            
            # 更新自适应参数
            adaptation_params = params.get('adaptation_params', {})
            self.model_ensemble.update_adaptation_settings(adaptation_params)
            
        except Exception as e:
            logger.error(f"更新集成参数时出错: {str(e)}")

    def adjust_ensemble_strategy(self, match_distribution):
        """调整集成策略"""
        try:
            total_samples = sum(match_distribution.values())
            
            # 1. 分析集成效果
            high_match_ratio = (match_distribution[4] + match_distribution[5]) / total_samples
            low_match_ratio = (match_distribution[0] + match_distribution[1]) / total_samples
            
            # 2. 根据分布调整集成策略
            if high_match_ratio < 0.1:  # 高匹配率太低
                # 增加模型多样性
                self.increase_model_diversity()
                # 调整模型权重
                self.adjust_model_weights()
                
            elif low_match_ratio > 0.5:  # 低匹配率太高
                # 强化表现好的模型
                self.strengthen_best_models()
                # 重新训练表现差的模型
                self.retrain_weak_models()
                
            return True
            
        except Exception as e:
            logger.error(f"调整集成策略时出错: {str(e)}")
            return False

# 创建全局实例
ensemble_optimizer = EnsembleOptimizer(model_ensemble=None, data_processor=None)
