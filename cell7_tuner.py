# Parameter Tuning System / 参数调优系统
import numpy as np
import tensorflow as tf
import logging
import os
import json
from datetime import datetime
from bayes_opt import BayesianOptimization
from bayes_opt.logger import Events
import optuna
from sklearn.metrics import mutual_info_score
from typing import Dict, Any, Optional

# 获取logger实例
logger = logging.getLogger(__name__)

class OptimizerManager:
    """优化器管理类 - 整合模型、动态和集成优化"""
    
    def __init__(self, model_ensemble, data_processor, performance_monitor):
        """
        初始化优化器管理器
        Args:
            model_ensemble: 模型集成实例
            data_processor: 数据处理器实例
            performance_monitor: 性能监控器实例
        """
        self.model_ensemble = model_ensemble
        self.data_processor = data_processor
        self.performance_monitor = performance_monitor
        
        # 优化历史记录
        self.optimization_history = []
        
        # 初始化参数范围
        self._init_param_ranges()
        
        # 初始化调整阈值
        self._init_thresholds()
        
        logger.info("优化器管理器初始化完成")

    def _init_param_ranges(self):
        """初始化所有参数范围"""
        self.param_ranges = {
            # 模型架构参数
            'model_params': {
                'lstm_units': (64, 512),
                'lstm_layers': (1, 4),
                'cnn_filters': (32, 256),
                'transformer_heads': (4, 16)
            },
            
            # 动态调整参数
            'dynamic_params': {
                'learning_rate': (1e-5, 1e-2),
                'batch_size': (16, 128),
                'dropout_rate': (0.1, 0.5)
            },
            
            # 集成参数
            'ensemble_params': {
                'initial_weights': (0.1, 0.3),
                'diversity_weight': (0.1, 0.5),
                'adaptation_rate': (0.1, 0.5)
            },
            
            # 添加训练优化参数范围
            'training_params': {
                'optimizer_params': {
                    'learning_rate': (1e-5, 1e-2),
                    'beta_1': (0.8, 0.999),
                    'beta_2': (0.8, 0.999),
                    'epsilon': (1e-8, 1e-6)
                },
                'lr_schedule_params': {
                    'decay_rate': (0.9, 0.99),
                    'decay_steps': (100, 1000),
                    'warmup_steps': (0, 100),
                    'min_lr': (1e-6, 1e-4)
                },
                'training_control': {
                    'batch_size': (16, 128),
                    'epochs_per_iteration': (1, 10),
                    'validation_frequency': (1, 10),
                    'early_stopping_patience': (10, 50)
                }
            }
        }
        
        # 添加离散参数选项
        self.discrete_params = {
            'optimizer_type': ['adam', 'adamw', 'radam'],
            'scheduler_type': ['exponential', 'cosine', 'step']
        }

    def _init_thresholds(self):
        """初始化调整阈值"""
        self.thresholds = {
            'performance_drop': 0.1,    # 性能下降阈值
            'loss_spike': 0.5,         # 损失突增阈值
            'diversity_min': 0.3,      # 最小多样性要求
            'weight_change': 0.2       # 权重调整阈值
        }

    def optimize_all(self, n_iter=50):
        """执行全面优化"""
        try:
            # 1. 模型架构优化
            model_params = self._optimize_model_architecture(n_iter)
            
            # 2. 动态参数优化
            dynamic_params = self._optimize_dynamic_params(n_iter)
            
            # 3. 集成策略优化
            ensemble_params = self._optimize_ensemble_strategy(n_iter)
            
            # 整合优化结果
            optimized_params = {
                'model_params': model_params,
                'dynamic_params': dynamic_params,
                'ensemble_params': ensemble_params
            }
            
            # 保存优化结果
            self._save_optimization_results(optimized_params)
            
            return optimized_params
            
        except Exception as e:
            logger.error(f"执行全面优化时出错: {str(e)}")
            return None

    def _optimize_model_architecture(self, n_iter):
        """模型架构优化"""
        try:
            optimizer = BayesianOptimization(
                f=self._model_objective,
                pbounds=self.param_ranges['model_params'],
                random_state=42
            )
            
            optimizer.maximize(init_points=5, n_iter=n_iter)
            return optimizer.max['params']
            
        except Exception as e:
            logger.error(f"模型架构优化失败: {str(e)}")
            return None

    def _optimize_dynamic_params(self, n_iter):
        """动态参数优化"""
        try:
            optimizer = BayesianOptimization(
                f=self._dynamic_objective,
                pbounds=self.param_ranges['dynamic_params'],
                random_state=42
            )
            
            optimizer.maximize(init_points=5, n_iter=n_iter)
            return optimizer.max['params']
            
        except Exception as e:
            logger.error(f"动态参数优化失败: {str(e)}")
            return None

    def _optimize_ensemble_strategy(self, n_iter):
        """集成策略优化"""
        try:
            optimizer = BayesianOptimization(
                f=self._ensemble_objective,
                pbounds=self.param_ranges['ensemble_params'],
                random_state=42
            )
            
            optimizer.maximize(init_points=5, n_iter=n_iter)
            return optimizer.max['params']
            
        except Exception as e:
            logger.error(f"集成策略优化失败: {str(e)}")
            return None

    def _model_objective(self, **params):
        """模型优化目标函数"""
        try:
            self.model_ensemble.update_architecture(params)
            return self._evaluate_performance()
        except Exception as e:
            logger.error(f"模型目标函数评估失败: {str(e)}")
            return float('-inf')

    def _dynamic_objective(self, **params):
        """动态优化目标函数"""
        try:
            self.model_ensemble.update_dynamic_params(params)
            return self._evaluate_performance()
        except Exception as e:
            logger.error(f"动态目标函数评估失败: {str(e)}")
            return float('-inf')

    def _ensemble_objective(self, **params):
        """集成优化目标函数"""
        try:
            self.model_ensemble.update_ensemble_params(params)
            performance = self._evaluate_performance()
            diversity = self._calculate_diversity()
            return 0.7 * performance + 0.3 * diversity
        except Exception as e:
            logger.error(f"集成目标函数评估失败: {str(e)}")
            return float('-inf')

    def _evaluate_performance(self):
        """评估性能"""
        try:
            X_val, y_val = self.data_processor.get_validation_data()
            predictions = self.model_ensemble.predict(X_val)
            matches = np.any(np.round(predictions) == y_val, axis=1)
            return np.mean(matches)
        except Exception as e:
            logger.error(f"性能评估失败: {str(e)}")
            return 0.0

    def _calculate_diversity(self):
        """计算模型多样性"""
        try:
            predictions = []
            X_val, _ = self.data_processor.get_validation_data()
            
            for model in self.model_ensemble.models:
                pred = model.predict(X_val)
                predictions.append(pred)
            
            diversity_scores = []
            n_models = len(predictions)
            
            for i in range(n_models):
                for j in range(i + 1, n_models):
                    mi_score = mutual_info_score(
                        predictions[i].ravel(),
                        predictions[j].ravel()
                    )
                    diversity_scores.append(1 - mi_score)
            
            return np.mean(diversity_scores)
            
        except Exception as e:
            logger.error(f"多样性计算失败: {str(e)}")
            return 0.0

    def _save_optimization_results(self, results):
        """保存优化结果"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"optimization_results_{timestamp}.json"
            
            save_path = os.path.join(os.getcwd(), 'optimization_results', filename)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            with open(save_path, 'w') as f:
                json.dump(results, f, indent=4)
                
            logger.info(f"优化结果已保存到: {save_path}")
            
        except Exception as e:
            logger.error(f"保存优化结果失败: {str(e)}")

    def dynamic_adjust(self, metrics):
        """动态参数调整"""
        try:
            if self._needs_adjustment(metrics):
                suggestions = self._get_adjustment_suggestions(metrics)
                return self._apply_adjustments(suggestions)
            return None
        except Exception as e:
            logger.error(f"动态调整失败: {str(e)}")
            return None

    def _needs_adjustment(self, metrics):
        """检查是否需要调整"""
        return any([
            metrics['performance_change'] < -self.thresholds['performance_drop'],
            metrics['loss_change'] > self.thresholds['loss_spike'],
            metrics['diversity'] < self.thresholds['diversity_min']
        ])

    def _get_adjustment_suggestions(self, metrics):
        """获取参数调整建议"""
        try:
            suggestions = {}
            
            # 基于性能变化的学习率调整
            if metrics['performance_change'] < 0:
                current_lr = metrics['current_lr']
                suggestions['learning_rate'] = self._adjust_learning_rate(
                    current_lr, 
                    metrics['performance_change']
                )
            
            # 基于内存使用的批次大小调整    
            if metrics['memory_usage'] > 0.9:
                current_batch = metrics['current_batch_size']
                suggestions['batch_size'] = self._adjust_batch_size(current_batch)
            
            # 基于过拟合风险的正则化调整
            if metrics['validation_loss'] > metrics['training_loss'] * 1.2:
                suggestions['dropout_rate'] = self._adjust_dropout_rate(
                    metrics['current_dropout']
                )
                
            return suggestions
            
        except Exception as e:
            logger.error(f"生成调整建议失败: {str(e)}")
            return {}

    def _adjust_learning_rate(self, current_lr, performance_change):
        """调整学习率"""
        try:
            if performance_change < -0.2:  # 性能显著下降
                return current_lr * 0.5
            elif performance_change < -0.1:  # 性能轻微下降
                return current_lr * 0.8
            return current_lr
        except Exception as e:
            logger.error(f"调整学习率失败: {str(e)}")
            return current_lr

    def _adjust_batch_size(self, current_batch_size):
        """调整批次大小"""
        try:
            return max(16, current_batch_size // 2)
        except Exception as e:
            logger.error(f"调整批次大小失败: {str(e)}")
            return current_batch_size

    def _adjust_dropout_rate(self, current_dropout):
        """调整dropout率"""
        try:
            return min(0.5, current_dropout + 0.1)
        except Exception as e:
            logger.error(f"调整dropout率失败: {str(e)}")
            return current_dropout

    def _apply_adjustments(self, suggestions):
        """应用参数调整"""
        try:
            new_params = {}
            
            for param_name, new_value in suggestions.items():
                # 验证参数范围
                if param_name in self.param_ranges['dynamic_params']:
                    min_val, max_val = self.param_ranges['dynamic_params'][param_name]
                    new_value = np.clip(new_value, min_val, max_val)
                    new_params[param_name] = new_value
                    
            # 更新模型参数
            if new_params:
                self.model_ensemble.update_dynamic_params(new_params)
                logger.info(f"应用参数调整: {new_params}")
                
            return new_params
            
        except Exception as e:
            logger.error(f"应用参数调整失败: {str(e)}")
            return {}

    def adjust_ensemble_weights(self, performance_metrics):
        """调整集成权重"""
        try:
            weights = []
            for model_idx, metrics in enumerate(performance_metrics):
                # 基于性能计算新权重
                performance_score = 1.0 - metrics['loss']
                diversity_score = self._calculate_model_diversity(model_idx)
                weight = 0.7 * performance_score + 0.3 * diversity_score
                weights.append(weight)
            
            # 归一化权重
            weights = np.array(weights)
            weights = weights / np.sum(weights)
            
            # 更新模型集成权重
            self.model_ensemble.update_weights(weights)
            logger.info(f"更新集成权重: {weights}")
            
            return weights
            
        except Exception as e:
            logger.error(f"调整集成权重失败: {str(e)}")
            return None

    def _calculate_model_diversity(self, model_idx):
        """计算单个模型的多样性得分"""
        try:
            predictions = []
            X_val, _ = self.data_processor.get_validation_data()
            
            # 获取当前模型和其他模型的预测
            current_pred = self.model_ensemble.models[model_idx].predict(X_val)
            other_preds = []
            for i, model in enumerate(self.model_ensemble.models):
                if i != model_idx:
                    other_preds.append(model.predict(X_val))
            
            # 计算与其他模型的平均互信息分数
            diversity_scores = []
            for other_pred in other_preds:
                mi_score = mutual_info_score(
                    current_pred.ravel(),
                    other_pred.ravel()
                )
                diversity_scores.append(1 - mi_score)
            
            return np.mean(diversity_scores)
            
        except Exception as e:
            logger.error(f"计算模型多样性失败: {str(e)}")
            return 0.0

    def get_optimization_summary(self):
        """获取优化过程摘要"""
        try:
            if not self.optimization_history:
                return None
                
            latest_results = self.optimization_history[-1]
            best_results = max(
                self.optimization_history,
                key=lambda x: x.get('final_score', 0)
            )
            
            return {
                'latest': {
                    'params': latest_results['params'],
                    'performance': latest_results.get('final_score', 0)
                },
                'best': {
                    'params': best_results['params'],
                    'performance': best_results.get('final_score', 0)
                },
                'total_iterations': len(self.optimization_history)
            }
            
        except Exception as e:
            logger.error(f"获取优化摘要失败: {str(e)}")
            return None

    def reset_optimization(self):
        """重置优化状态"""
        try:
            self.optimization_history.clear()
            self._init_param_ranges()
            self._init_thresholds()
            logger.info("优化状态已重置")
            return True
        except Exception as e:
            logger.error(f"重置优化状态失败: {str(e)}")
            return False

    def optimize_model_params(self, training_direction):
        """根据训练方向优化模型参数 (from cell13)"""
        try:
            if isinstance(training_direction, dict):
                # 1. 学习率调整
                if training_direction['learning_rate'] == 'INCREASE':
                    self.current_lr *= 1.5
                elif training_direction['learning_rate'] == 'DECREASE':
                    self.current_lr *= 0.7
                
                # 2. 批次大小调整
                if training_direction['batch_size'] == 'DECREASE':
                    self.batch_size = max(16, self.batch_size // 2)
                
                # 3. 模型复杂度调整
                if training_direction['model_complexity'] == 'INCREASE':
                    self._increase_model_complexity()
                
                # 4. 正则化调整
                if training_direction.get('regularization') == 'INCREASE':
                    self._increase_regularization()
                
            return True
            
        except Exception as e:
            logger.error(f"优化模型参数时出错: {str(e)}")
            return False

    def _increase_model_complexity(self):
        """增加模型复杂度 (from cell13)"""
        try:
            current_params = self.model_ensemble.get_current_params()
            new_params = {
                'lstm_units': int(current_params['lstm_units'] * 1.5),
                'transformer_heads': current_params['transformer_heads'] + 2,
                'cnn_filters': int(current_params['cnn_filters'] * 1.3)
            }
            self.model_ensemble.update_architecture(new_params)
        except Exception as e:
            logger.error(f"增加模型复杂度失败: {str(e)}")

    def _increase_regularization(self):
        """增加正则化强度 (from cell13)"""
        try:
            current_params = self.model_ensemble.get_current_params()
            new_params = {
                'dropout_rate': min(0.5, current_params['dropout_rate'] + 0.1),
                'weight_decay': current_params['weight_decay'] * 2
            }
            self.model_ensemble.update_dynamic_params(new_params)
        except Exception as e:
            logger.error(f"增加正则化强度失败: {str(e)}")

    def adjust_ensemble_strategy(self, match_distribution):
        """调整集成策略 (from cell15)"""
        try:
            total_samples = sum(match_distribution.values())
            
            # 1. 分析集成效果
            high_match_ratio = (match_distribution[4] + match_distribution[5]) / total_samples
            low_match_ratio = (match_distribution[0] + match_distribution[1]) / total_samples
            
            # 2. 根据分布调整集成策略
            if high_match_ratio < 0.1:  # 高匹配率太低
                # 增加模型多样性
                self._increase_model_diversity()
                # 调整模型权重
                self._adjust_model_weights()
                
            elif low_match_ratio > 0.5:  # 低匹配率太高
                # 强化表现好的模型
                self._strengthen_best_models()
                # 重新训练表现差的模型
                self._retrain_weak_models()
                
            return True
            
        except Exception as e:
            logger.error(f"调整集成策略时出错: {str(e)}")
            return False

    def _increase_model_diversity(self):
        """增加模型多样性 (from cell15)"""
        try:
            # 1. 计算当前多样性矩阵
            diversity_matrix = self._calculate_diversity_matrix()
            
            # 2. 找出相似度最高的模型对
            similar_pairs = self._find_similar_model_pairs(diversity_matrix)
            
            # 3. 对相似模型进行差异化训练
            for model_i, model_j in similar_pairs:
                self._differentiate_models(model_i, model_j)
                
        except Exception as e:
            logger.error(f"增加模型多样性失败: {str(e)}")

    def _calculate_diversity_matrix(self):
        """计算模型间多样性矩阵 (from cell15)"""
        try:
            n_models = len(self.model_ensemble.models)
            diversity_matrix = np.zeros((n_models, n_models))
            
            X_val, _ = self.data_processor.get_validation_data()
            predictions = [model.predict(X_val) for model in self.model_ensemble.models]
            
            for i in range(n_models):
                for j in range(i+1, n_models):
                    mi_score = mutual_info_score(
                        predictions[i].ravel(),
                        predictions[j].ravel()
                    )
                    diversity_matrix[i, j] = mi_score
                    diversity_matrix[j, i] = mi_score
                    
            return diversity_matrix
            
        except Exception as e:
            logger.error(f"计算多样性矩阵失败: {str(e)}")
            return None

    def _find_similar_model_pairs(self, diversity_matrix):
        """找出相似度高的模型对 (from cell15)"""
        try:
            n_models = len(self.model_ensemble.models)
            similar_pairs = []
            
            for i in range(n_models):
                for j in range(i+1, n_models):
                    if diversity_matrix[i, j] > 0.8:  # 相似度阈值
                        similar_pairs.append((i, j))
                        
            return similar_pairs
            
        except Exception as e:
            logger.error(f"寻找相似模型对失败: {str(e)}")
            return []

    def _differentiate_models(self, model_i, model_j):
        """对相似模型进行差异化训练 (from cell15)"""
        try:
            # 1. 调整模型架构
            self.model_ensemble.update_model_architecture({
                model_i: {'dropout_rate': 0.3},
                model_j: {'dropout_rate': 0.5}
            })
            
            # 2. 使用不同的优化器
            self.model_ensemble.update_optimizer_settings({
                model_i: {'learning_rate': 0.001},
                model_j: {'learning_rate': 0.0005}
            })
            
        except Exception as e:
            logger.error(f"模型差异化失败: {str(e)}")

    def _strengthen_best_models(self):
        """强化表现好的模型 (from cell15)"""
        try:
            performance_metrics = self.performance_monitor.get_model_metrics()
            best_models = self._identify_best_models(performance_metrics)
            
            for model_idx in best_models:
                # 增加模型权重
                self.model_ensemble.increase_model_weight(model_idx)
                # 微调学习率
                self.model_ensemble.fine_tune_model(model_idx)
                
        except Exception as e:
            logger.error(f"强化最佳模型失败: {str(e)}")

    def _retrain_weak_models(self):
        """重新训练表现差的模型 (from cell15)"""
        try:
            performance_metrics = self.performance_monitor.get_model_metrics()
            weak_models = self._identify_weak_models(performance_metrics)
            
            for model_idx in weak_models:
                # 重置模型参数
                self.model_ensemble.reset_model(model_idx)
                # 使用新的训练策略
                self.model_ensemble.retrain_model(
                    model_idx, 
                    strategy='adaptive'
                )
                
        except Exception as e:
            logger.error(f"重训弱模型失败: {str(e)}")

    def _identify_best_models(self, metrics):
        """识别最佳模型 (from cell15)"""
        try:
            scores = [m['performance'] for m in metrics]
            threshold = np.percentile(scores, 75)  # 上四分位数
            return [i for i, score in enumerate(scores) if score >= threshold]
        except Exception as e:
            logger.error(f"识别最佳模型失败: {str(e)}")
            return []

    def _identify_weak_models(self, metrics):
        """识别弱模型 (from cell15)"""
        try:
            scores = [m['performance'] for m in metrics]
            threshold = np.percentile(scores, 25)  # 下四分位数
            return [i for i, score in enumerate(scores) if score <= threshold]
        except Exception as e:
            logger.error(f"识别弱模型失败: {str(e)}")
            return []

    def adjust_after_sample(self, model, sample, current_params):
        """基于样本梯度调整参数 (from cell13)"""
        try:
            with tf.GradientTape() as tape:
                predictions = model(sample['input'])
                loss = tf.keras.losses.MSE(sample['target'], predictions)
            grads = tape.gradient(loss, model.trainable_variables)
            
            # 生成参数调整建议
            adjusted_params = {
                'learning_rate': self._adjust_lr_from_grads(grads, current_params),
                'batch_size': current_params['batch_size']
            }
            return adjusted_params
            
        except Exception as e:
            logger.error(f"样本级参数调整失败: {str(e)}")
            return current_params

    def _adjust_lr_from_grads(self, grads, current_params):
        """根据梯度调整学习率"""
        try:
            grad_norm = tf.linalg.global_norm(grads)
            if grad_norm > 10.0:  # 梯度爆炸
                return current_params['learning_rate'] * 0.5
            elif grad_norm < 0.1:  # 梯度消失
                return current_params['learning_rate'] * 1.5
            return current_params['learning_rate']
        except Exception as e:
            logger.error(f"学习率梯度调整失败: {str(e)}")
            return current_params['learning_rate']

    def on_train_end(self):
        """训练结束时的优化操作 (from cell13)"""
        try:
            # 获取新的参数建议
            new_params = self.suggest_next_params()
            # 更新集成模型参数
            self.model_ensemble.update_params(new_params)
            # 保存优化记录
            self._save_optimization_record()
            return True
        except Exception as e:
            logger.error(f"训练结束优化操作失败: {str(e)}")
            return False

    def suggest_next_params(self):
        """使用Optuna生成下一组参数"""
        try:
            study = optuna.create_study(
                study_name="model_optim_v1",
                storage="sqlite:///optuna.db",
                load_if_exists=True
            )
            
            trial = study.ask()
            params = {
                'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-2),
                'batch_size': trial.suggest_int('batch_size', 16, 128),
                'dropout_rate': trial.suggest_uniform('dropout_rate', 0.1, 0.5)
            }
            return params
        except Exception as e:
            logger.error(f"生成参数建议失败: {str(e)}")
            return None

    def optimize_parameters(self):
        """执行参数优化 (from cell14)"""
        try:
            # 贝叶斯优化初始化基础参数
            initial_params = self.bayesian_optimization()
            # 使用Optuna进行细粒度优化
            final_params = self._optuna_optimization(initial_params)
            
            # 更新并保存最佳参数
            self.best_params = final_params
            self.save_best_params()
            
            logger.info(f"参数优化完成: {final_params}")
            return final_params
        except Exception as e:
            logger.error(f"参数优化失败: {str(e)}")
            return None

    def save_best_params(self):
        """保存最佳参数配置 (from cell14)"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"best_params_{timestamp}.json"
            save_path = os.path.join(os.getcwd(), 'optimization_params', filename)
            
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                json.dump(self.best_params, f, indent=4)
            
            logger.info(f"最佳参数已保存到: {save_path}")
            return True
        except Exception as e:
            logger.error(f"保存最佳参数失败: {str(e)}")
            return False

    def _save_optimization_record(self):
        """保存优化记录"""
        try:
            record = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'history': self.optimization_history,
                'best_params': self.best_params,
                'performance_summary': self.get_optimization_summary()
            }
            
            filename = f"optimization_record_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            save_path = os.path.join(os.getcwd(), 'optimization_records', filename)
            
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                json.dump(record, f, indent=4)
                
            logger.info(f"优化记录已保存到: {save_path}")
            return True
        except Exception as e:
            logger.error(f"保存优化记录失败: {str(e)}")
            return False

    def analyze_training_direction(self, match_counts, current_params):
        """分析训练方向"""
        try:
            # 1. 初始化/更新匹配分布
            if not hasattr(self, 'match_distribution'):
                self.match_distribution = {i: 0 for i in range(6)}
                
            # 2. 更新分布
            for count in match_counts:
                self.match_distribution[count] += 1
            
            # 3. 判断当前状态
            if self.match_distribution[5] > 0:
                return "OPTIMAL"
                
            avg_match = sum(k * v for k, v in self.match_distribution.items()) / sum(self.match_distribution.values())
            
            # 4. 根据匹配分布给出调整建议
            if avg_match < 2:
                return {
                    'learning_rate': 'INCREASE',
                    'batch_size': 'DECREASE',
                    'model_complexity': 'INCREASE'
                }
            elif avg_match > 3:
                return {
                    'learning_rate': 'DECREASE',
                    'regularization': 'INCREASE',
                    'ensemble_diversity': 'INCREASE'
                }
            
            return "CONTINUE"
            
        except Exception as e:
            logger.error(f"分析训练方向时出错: {str(e)}")
            return None

    def optimize_training_flow(self):
        """优化训练流程"""
        try:
            self._dynamic_resource_adjust()
            self._dynamic_batch_adjust()
            self._enable_mixed_precision()
            return True
        except Exception as e:
            logger.error(f"优化训练流程时出错: {str(e)}")
            return False

    def _dynamic_resource_adjust(self):
        """根据硬件资源动态调整参数"""
        try:
            # 获取资源信息
            mem_info = memory_manager.get_memory_info()
            cpu_usage = psutil.cpu_percent()
            
            # 内存调整策略
            if mem_info['percent'] > 75:
                new_batch = max(4, self.batch_size // 2)
                logger.info(f"内存使用{mem_info['percent']}% → 批次从{self.batch_size}调整为{new_batch}")
                self.batch_size = new_batch
            
            # CPU线程调整策略
            if hasattr(self, 'threads'):
                if cpu_usage < 60:
                    self.threads = min(12, self.threads + 2)
                else:
                    self.threads = max(4, self.threads - 2)
            
            # GPU显存优化
            if tf.config.list_physical_devices('GPU'):
                gpu_mem = tf.config.experimental.get_memory_info('GPU:0')
                used_percent = gpu_mem['current'] / gpu_mem['total']
                if used_percent > 0.8:
                    tf.config.experimental.set_memory_growth(True)
                    
            return True
        except Exception as e:
            logger.error(f"资源调整失败: {str(e)}")
            return False

    def _dynamic_batch_adjust(self):
        """动态调整批次大小"""
        try:
            if hasattr(self, 'batch_size'):
                mem_usage = memory_manager.get_memory_info()
                if mem_usage['percent'] > 80:
                    new_size = max(8, self.batch_size // 2)
                    logger.info(f"批次大小从{self.batch_size}调整为{new_size}")
                    self.batch_size = new_size
            return True
        except Exception as e:
            logger.error(f"批次调整失败: {str(e)}")
            return False

    def _enable_mixed_precision(self):
        """启用混合精度训练"""
        try:
            if tf.config.list_physical_devices('GPU'):
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                logger.info("已启用混合精度训练")
            return True
        except Exception as e:
            logger.error(f"启用混合精度失败: {str(e)}")
            return False

    def setup_mixed_precision(self):
        """配置混合精度训练"""
        try:
            if tf.config.list_physical_devices('GPU'):
                # 启用mixed precision policy
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                logger.info("已启用混合精度训练")
                
                # 配置优化器
                self.model_ensemble.update_optimizer_settings({
                    'mixed_precision': True,
                    'loss_scale': 'dynamic'
                })
                return True
        except Exception as e:
            logger.error(f"配置混合精度训练失败: {str(e)}")
            return False

    def setup_checkpoints(self):
        """配置检查点"""
        try:
            checkpoint_config = {
                'save_freq': 100,  # 每100步保存一次
                'max_to_keep': 5,  # 保留最新的5个检查点
                'include_optimizer': True,
                'save_best_only': True,
                'monitor': 'val_accuracy'
            }
            self.model_ensemble.setup_model_checkpoint(checkpoint_config)
            logger.info("检查点配置完成")
            return True
        except Exception as e:
            logger.error(f"配置检查点失败: {str(e)}")
            return False

    def adjust_learning_rate(self, metrics):
        """智能调整学习率"""
        try:
            accuracy = metrics.get('accuracy', 0)
            loss_change = metrics.get('loss_change', 0)
            
            # 基于性能调整学习率
            if accuracy < 0.5 and loss_change > 0:
                # 性能差且损失在增加，大幅降低学习率
                return self.current_lr * 0.5
            elif accuracy < 0.7 and loss_change > 0:
                # 性能一般且损失增加，小幅降低学习率
                return self.current_lr * 0.8
            elif accuracy > 0.9 and loss_change < 0:
                # 性能好且损失在下降，小幅提高学习率
                return self.current_lr * 1.1
            
            return self.current_lr
        except Exception as e:
            logger.error(f"调整学习率失败: {str(e)}")
            return self.current_lr

    def _update_training_params(self, params):
        """更新训练参数"""
        try:
            nested_params = self._process_params(params)
            
            # 1. 更新优化器参数
            optimizer_params = nested_params['optimizer_params']
            for model in self.model_ensemble.models:
                model.optimizer.learning_rate = optimizer_params['learning_rate']
                if hasattr(model.optimizer, 'beta_1'):
                    model.optimizer.beta_1 = optimizer_params['beta_1']
                if hasattr(model.optimizer, 'beta_2'):
                    model.optimizer.beta_2 = optimizer_params['beta_2']
            
            # 2. 更新学习率调度
            lr_params = nested_params['lr_schedule_params']
            self.model_ensemble.update_learning_rate_schedule(
                decay_rate=lr_params['decay_rate'],
                decay_steps=int(lr_params['decay_steps']),
                warmup_steps=int(lr_params['warmup_steps']),
                min_lr=lr_params['min_lr']
            )
            
            # 3. 更新训练控制参数
            training_params = nested_params['training_control']
            self.model_ensemble.batch_size = int(training_params['batch_size'])
            self.model_ensemble.epochs_per_iteration = int(training_params['epochs_per_iteration'])
            self.model_ensemble.validation_frequency = int(training_params['validation_frequency'])
            
            logger.info("训练参数已更新")
            
        except Exception as e:
            logger.error(f"更新训练参数时出错: {str(e)}")
            raise

# 创建全局实例
optimizer_manager = OptimizerManager(
    model_ensemble=None,
    data_processor=None,
    performance_monitor=None
)
