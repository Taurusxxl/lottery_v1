# 训练策略优化\training_optimizer.py
import numpy as np
from bayes_opt import BayesianOptimization
from bayes_opt.util import Events
import tensorflow as tf
import logging
import os
import json
from sklearn.model_selection import cross_val_score
import psutil

# 获取logger实例
logger = logging.getLogger(__name__)

class TrainingOptimizer:
    """训练策略优化器类"""
    
    def __init__(self, model_ensemble, data_processor):
        """
        初始化训练优化器
        Args:
            model_ensemble: 模型集成实例
            data_processor: 数据处理器实例
        """
        self.model_ensemble = model_ensemble
        self.data_processor = data_processor
        self.best_params = None
        self.optimization_history = []
        self.best_match_history = []  # 记录最佳匹配数历史
        self.match_distribution = {i: 0 for i in range(6)}  # 记录各匹配数的分布
        
        # 定义参数范围
        self.param_ranges = {
            # 1. 优化器参数
            'optimizer_params': {
                'learning_rate': (1e-5, 1e-2),
                'beta_1': (0.8, 0.999),
                'beta_2': (0.8, 0.999),
                'epsilon': (1e-8, 1e-6)
            },
            
            # 2. 学习率调度参数
            'lr_schedule_params': {
                'decay_rate': (0.9, 0.99),
                'decay_steps': (100, 1000),
                'warmup_steps': (0, 100),
                'min_lr': (1e-6, 1e-4)
            },
            
            # 3. 训练控制参数
            'training_control': {
                'batch_size': (16, 128),
                'epochs_per_iteration': (1, 10),
                'validation_frequency': (1, 10),
                'early_stopping_patience': (10, 50)
            }
        }
        
        # 定义离散参数
        self.discrete_params = {
            'optimizer_type': ['adam', 'adamw', 'radam'],
            'scheduler_type': ['exponential', 'cosine', 'step']
        }
        
        logger.info("训练优化器初始化完成")

    def optimize(self, n_iter=50):
        """
        运行优化过程
        Args:
            n_iter: 优化迭代次数
        Returns:
            dict: 最佳参数
        """
        try:
            optimizer = BayesianOptimization(
                f=self._objective_function,
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

    def _objective_function(self, **params):
        """优化目标函数"""
        try:
            # 1. 验证参数
            is_valid, message = self._validate_params(params)
            if not is_valid:
                logger.warning(f"参数验证失败: {message}")
                return float('-inf')
            
            # 2. 更新训练参数
            self._update_training_params(params)
            
            # 3. 获取评估数据
            X, y = self.data_processor.get_validation_data()
            if X is None or y is None:
                return float('-inf')
            
            # 4. 评估性能
            score = self._evaluate_performance(X, y)
            
            # 5. 记录历史
            self.optimization_history.append({
                'params': params,
                'score': score
            })
            
            return score
            
        except Exception as e:
            logger.error(f"目标函数执行出错: {str(e)}")
            return float('-inf')

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

    def _evaluate_performance(self, X, y):
        """评估性能"""
        try:
            # 1. 获取集成预测
            predictions = self.model_ensemble.predict(X)
            
            # 2. 计算匹配率
            matches = np.any(np.round(predictions) == y, axis=1)
            accuracy = np.mean(matches)
            
            # 3. 计算其他指标
            mse = np.mean((predictions - y) ** 2)
            mae = np.mean(np.abs(predictions - y))
            
            # 4. 计算综合得分
            score = (0.6 * accuracy + 
                    0.2 * (1 / (1 + mse)) + 
                    0.2 * (1 / (1 + mae)))
            
            logger.debug(f"性能评估 - 准确率: {accuracy:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}")
            return score
            
        except Exception as e:
            logger.error(f"评估性能时出错: {str(e)}")
            return float('-inf')

    def _save_best_params(self):
        """保存最佳参数"""
        try:
            save_path = os.path.join('config', 'best_training_params.json')
            os.makedirs('config', exist_ok=True)
            
            with open(save_path, 'w') as f:
                json.dump(self.best_params, f, indent=4)
            logger.info(f"最佳训练参数已保存到: {save_path}")
            
        except Exception as e:
            logger.error(f"保存最佳参数时出错: {str(e)}")

    def _validate_params(self, params):
        """
        验证参数有效性
        Args:
            params: 待验证的参数
        Returns:
            tuple: (是否有效, 错误信息)
        """
        try:
            nested_params = self._process_params(params)
            
            # 验证优化器参数
            optimizer_params = nested_params['optimizer_params']
            assert 0 < optimizer_params['learning_rate'] < 1
            assert 0 < optimizer_params['beta_1'] < 1
            assert 0 < optimizer_params['beta_2'] < 1
            
            # 验证学习率调度参数
            lr_params = nested_params['lr_schedule_params']
            assert 0 < lr_params['decay_rate'] < 1
            assert lr_params['decay_steps'] > 0
            
            # 验证训练控制参数
            training_params = nested_params['training_control']
            assert training_params['batch_size'] > 0
            assert training_params['epochs_per_iteration'] > 0
            
            return True, "参数验证通过"
            
        except AssertionError as e:
            return False, f"参数验证失败: {str(e)}"
        except Exception as e:
            return False, f"参数验证出错: {str(e)}"

    def _flatten_param_ranges(self):
        """
        将嵌套的参数范围展平为一维字典
        Returns:
            dict: 展平后的参数范围
        """
        flat_ranges = {}
        for category, params in self.param_ranges.items():
            for param_name, param_range in params.items():
                flat_ranges[f"{category}__{param_name}"] = param_range
        return flat_ranges

    def _process_params(self, flat_params):
        """
        将一维参数字典重构为嵌套结构
        Args:
            flat_params: 展平的参数字典
        Returns:
            dict: 嵌套的参数字典
        """
        nested_params = {
            'optimizer_params': {},
            'lr_schedule_params': {},
            'training_control': {}
        }
        
        for flat_name, value in flat_params.items():
            category, param_name = flat_name.split('__')
            nested_params[category][param_name] = value
            
        return nested_params

    def _check_optimization_progress(self):
        """检查优化进度和效果"""
        try:
            if len(self.optimization_history) < 5:
                return
            
            # 获取最近5次和历史最佳得分
            recent_scores = [h['score'] for h in self.optimization_history[-5:]]
            best_score = max(h['score'] for h in self.optimization_history)
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

    def analyze_training_direction(self, match_counts, current_params):
        """分析训练方向"""
        try:
            # 1. 分析匹配分布
            for count in match_counts:
                self.match_distribution[count] += 1
            
            # 2. 判断当前状态
            if self.match_distribution[5] > 0:
                return "OPTIMAL"  # 已达到最优
                
            avg_match = sum(k * v for k, v in self.match_distribution.items()) / sum(self.match_distribution.values())
            
            # 3. 根据匹配分布给出调整建议
            if avg_match < 2:  # 大部分预测匹配数低于2
                return {
                    'learning_rate': 'INCREASE',  # 增大学习率
                    'batch_size': 'DECREASE',     # 减小批次大小
                    'model_complexity': 'INCREASE' # 增加模型复杂度
                }
            elif avg_match > 3:  # 大部分预测匹配数高于3
                return {
                    'learning_rate': 'DECREASE',   # 减小学习率
                    'regularization': 'INCREASE',  # 增加正则化
                    'ensemble_diversity': 'INCREASE' # 增加集成多样性
                }
            
            return "CONTINUE"  # 保持当前方向
            
        except Exception as e:
            logger.error(f"分析训练方向时出错: {str(e)}")
            return None

    def adjust_training_strategy(self):
        # 根据当前批次表现调整
        if self.performance_monitor.get_recent_accuracy() < 0.5:
            self._adjust_learning_rate()

    def optimize_training_flow(self):
        """补全训练流程优化"""
        # 新增功能
        self._dynamic_batch_adjust()  # 动态批次调整
        self._enable_mixed_precision()  # 混合精度训练
        self._setup_checkpoints()  # 检查点配置

    def _dynamic_batch_adjust(self):
        """根据内存使用动态调整批次大小"""
        mem_usage = memory_manager.get_memory_info()
        if mem_usage['percent'] > 80:
            new_size = max(8, self.batch_size // 2)
            logger.info(f"批次大小从{self.batch_size}调整为{new_size}")
            self.batch_size = new_size

    def _enable_mixed_precision(self):
        # 当前仅设置标志位
        self.mixed_precision = True  # 需添加具体实现

    def _dynamic_resource_adjust(self):
        """根据硬件资源动态调整参数"""
        # 获取实时资源数据
        mem_info = memory_manager.get_memory_info()
        cpu_usage = psutil.cpu_percent()
        
        # 内存调整策略
        if mem_info['percent'] > 75:
            new_batch = max(4, self.batch_size // 2)
            logger.info(f"内存使用{mem_info['percent']}% → 批次从{self.batch_size}调整为{new_batch}")
            self.batch_size = new_batch
        
        # CPU线程调整策略
        if cpu_usage < 60:
            self.threads = min(12, self.threads + 2)  # 最大12线程
        else:
            self.threads = max(4, self.threads - 2)
        
        # GPU显存优化
        if tf.config.list_physical_devices('GPU'):
            gpu_mem = tf.config.experimental.get_memory_info('GPU:0')
            used_percent = gpu_mem['current'] / gpu_mem['total']
            if used_percent > 0.8:
                tf.config.experimental.set_memory_growth(True)

# 创建全局实例
training_optimizer = TrainingOptimizer(model_ensemble=None, data_processor=None)
