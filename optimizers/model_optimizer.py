# 模型参数优化\model_optimizer.py
import numpy as np
from bayes_opt import BayesianOptimization  # 修正包名导入方式
from bayes_opt.logger import JSONLogger
from bayes_opt.util import load_logs
import tensorflow as tf
import logging
import os
import json
from datetime import datetime
from packaging import version
import bayes_opt
import optuna
import importlib

# 获取logger实例
logger = logging.getLogger(__name__)

# 修改版本检查方式
try:
    from bayes_opt import __version__ as bo_version
except ImportError:
    bo_version = "1.2.0"  # 默认兼容旧版本

if version.parse(bo_version) >= version.parse("1.3"):
    from bayes_opt import Events
else:
    from bayes_opt import Events  # 统一导入方式

# 检查是否有非常规导入方式
from bayesian_optimization import ...  # 正常情况
# 或
from bayes_opt import ...  # 需要对应不同版本

# 动态导入BayesianOptimization
def load_bayesian_optimization():
    try:
        from bayesian_optimization import BayesianOptimization
        return BayesianOptimization
    except ImportError:
        try:
            from bayes_opt import BayesianOptimization
            return BayesianOptimization
        except ImportError as e:
            raise ImportError(
                "无法导入BayesianOptimization，请执行：\n"
                "!conda install -c conda-forge bayesian-optimization=1.2.0\n"
                "或\n"
                "!pip install --force-reinstall bayesian-optimization==1.2.0"
            ) from e

BayesianOptimization = load_bayesian_optimization()

# 性能优化配置
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'  # 启用OneDNN优化
os.environ['OMP_NUM_THREADS'] = '4'        # 设置OpenMP线程数

class ModelOptimizer:
    def __init__(self, model_ensemble, data_processor):
        self.model_ensemble = model_ensemble
        self.data_processor = data_processor
        self.best_params = None
        self.optimization_history = []
        
        # 定义参数范围
        self.param_ranges = {
            # LSTM相关参数
            'lstm_params': {
                'units': (64, 512),
                'layers': (1, 4),
                'dropout': (0.1, 0.5),
                'recurrent_dropout': (0.0, 0.3)
            },
            
            # CNN相关参数
            'cnn_params': {
                'filters': (32, 256),
                'kernel_size': (2, 5),
                'pool_size': (2, 4),
                'conv_layers': (1, 3)
            },
            
            # Transformer相关参数
            'transformer_params': {
                'heads': (4, 16),
                'd_model': (128, 512),
                'dff': (256, 1024),
                'num_layers': (2, 6)
            }
        }
        
        # 定义验证规则
        self.validation_rules = {
            'units': lambda x: isinstance(x, int) and x > 0,
            'layers': lambda x: isinstance(x, int) and x > 0,
            'dropout': lambda x: 0 <= x <= 1,
            'heads': lambda x: isinstance(x, int) and x > 0
        }
    
    def optimize(self, n_iter=50):
        """运行优化过程"""
        try:
            optimizer = BayesianOptimization(
                f=self.objective_function,
                pbounds=self._flatten_param_ranges(),
                random_state=42
            )
            
            # 注册回调函数
            optimizer.subscribe(
                event=Events.OPTIMIZATION_STEP,
                subscriber=self._optimization_callback
            )
            
            optimizer.maximize(
                init_points=5,
                n_iter=n_iter
            )
            
            self.best_params = self._process_params(optimizer.max['params'])
            self._save_best_params()
            
            return self.best_params
            
        except Exception as e:
            logger.error(f"运行优化过程时出错: {str(e)}")
            return None
    
    def objective_function(self, **params):
        """优化目标函数"""
        try:
            # 验证参数
            is_valid, message = self.validate_params(params)
            if not is_valid:
                logger.warning(message)
                return float('-inf')
            
            # 更新模型架构
            self._update_model_architecture(params)
            
            # 评估性能
            score = self._evaluate_model_performance()
            
            # 记录历史
            self.optimization_history.append({
                'params': params,
                'score': score
            })
            
            return score
            
        except Exception as e:
            logger.error(f"优化目标函数执行出错: {str(e)}")
            return float('-inf')
    
    def validate_params(self, params):
        """验证参数有效性"""
        try:
            for param_name, value in params.items():
                for rule_name, rule_func in self.validation_rules.items():
                    if rule_name in param_name and not rule_func(value):
                        return False, f"{param_name}参数验证失败"
            return True, "参数验证通过"
            
        except Exception as e:
            return False, f"参数验证出错: {str(e)}"
    
    def _update_model_architecture(self, params):
        """更新模型架构"""
        try:
            nested_params = self._process_params(params)
            self.model_ensemble.update_model_architecture(nested_params)
            
        except Exception as e:
            logger.error(f"更新模型架构时出错: {str(e)}")
            raise
    
    def _evaluate_model_performance(self):
        """评估模型性能"""
        try:
            X_val, y_val = self.data_processor.get_validation_data()
            predictions = self.model_ensemble.get_ensemble_prediction(X_val)
            
            # 使用匹配率作为评估指标
            matches = np.any(np.round(predictions) == y_val, axis=1)
            accuracy = np.mean(matches)
            
            return accuracy
            
        except Exception as e:
            logger.error(f"评估模型性能时出错: {str(e)}")
            return float('-inf')
    
    def _save_best_params(self):
        """保存最佳参数"""
        try:
            save_path = os.path.join(BASE_SAVE_DIR, 'best_model_params.json')
            with open(save_path, 'w') as f:
                json.dump(self.best_params, f, indent=4)
            logger.info(f"最佳模型参数已保存到: {save_path}")
            
        except Exception as e:
            logger.error(f"保存最佳参数时出错: {str(e)}")

    def _flatten_param_ranges(self):
        """将嵌套参数范围展平"""
        flat_ranges = {}
        for model_type, params in self.param_ranges.items():
            for param_name, param_range in params.items():
                flat_name = f"{model_type}__{param_name}"
                flat_ranges[flat_name] = param_range
        return flat_ranges
    
    def _process_params(self, flat_params):
        """将展平的参数重构为嵌套结构"""
        nested_params = {}
        for flat_name, value in flat_params.items():
            model_type, param_name = flat_name.split('__')
            if model_type not in nested_params:
                nested_params[model_type] = {}
            nested_params[model_type][param_name] = value
        return nested_params

    def _optimization_callback(self, event, instance):
        """优化进度回调"""
        try:
            iteration = len(instance.res)
            current_best = instance.max['target']
            current_params = instance.max['params']
            
            # 记录优化进度
            logger.info(
                f"优化进度: 第{iteration}次迭代\n"
                f"当前最佳分数: {current_best:.4f}\n"
                f"参数: {json.dumps(current_params, indent=2)}"
            )
            
            # 保存阶段性结果
            if iteration % 10 == 0:  # 每10次迭代保存一次
                self._save_checkpoint(iteration, current_best, current_params)
            
        except Exception as e:
            logger.error(f"回调函数执行出错: {str(e)}")

    def _save_checkpoint(self, iteration, score, params):
        """保存优化检查点"""
        try:
            checkpoint = {
                'iteration': iteration,
                'best_score': score,
                'best_params': params,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            save_path = os.path.join(
                BASE_SAVE_DIR, 
                f'model_opt_checkpoint_{iteration}.json'
            )
            
            with open(save_path, 'w') as f:
                json.dump(checkpoint, f, indent=4)
            
            logger.info(f"保存优化检查点: {save_path}")
            
        except Exception as e:
            logger.error(f"保存检查点时出错: {str(e)}")

    def optimize_model_params(self, training_direction):
        """根据训练方向优化模型参数"""
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
                    self.increase_model_complexity()
                
                # 4. 正则化调整
                if training_direction.get('regularization') == 'INCREASE':
                    self.increase_regularization()
                
            return True
            
        except Exception as e:
            logger.error(f"优化模型参数时出错: {str(e)}")
            return False

    def on_train_end(self):
        new_params = self.bayesian_optimizer.suggest()
        self.model_ensemble.update_params(new_params)  # 更新集成模型参数
        self._save_optimization_record()  # 保存优化轨迹

    def adjust_after_sample(self, model, sample, current_params):
        # 使用sample['input']，但data_manager未生成该字段
        # 需要与data_manager的数据格式对齐
        # 基于样本梯度调整
        with tf.GradientTape() as tape:
            predictions = model(sample['input'])
            loss = tf.keras.losses.MSE(sample['target'], predictions)
        grads = tape.gradient(loss, model.trainable_variables)
        
        # 生成参数调整建议
        adjusted_params = {
            'learning_rate': self._adjust_learning_rate(grads, current_params),
            'batch_size': current_params['batch_size']  # 保持原批次大小
        }
        return adjusted_params

# 创建全局实例
model_optimizer = ModelOptimizer(model_ensemble=None, data_processor=None)

# 使用Optuna替代
study = optuna.create_study()

# 统一导入方式
try:
    from bayes_opt import BayesianOptimization  # 实际包名
except ImportError:
    raise ImportError("请执行：conda install -c conda-forge bayesian-optimization=1.2.0") 