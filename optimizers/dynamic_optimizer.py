# 动态调整策略\dynamic_optimizer.py
import numpy as np
import tensorflow as tf
import logging
from bayes_opt import BayesianOptimization
import json
import os
from datetime import datetime

# 获取logger实例
logger = logging.getLogger(__name__)

class DynamicOptimizer:
    """动态参数优化器类"""
    
    def __init__(self, model_ensemble, performance_monitor):
        """
        初始化动态优化器
        Args:
            model_ensemble: 模型集成实例
            performance_monitor: 性能监控器实例
        """
        self.model_ensemble = model_ensemble
        self.performance_monitor = performance_monitor
        self.optimization_history = []
        
        # 定义参数范围
        self.param_ranges = {
            'learning_rate': (1e-5, 1e-2),
            'batch_size': (16, 128),
            'dropout_rate': (0.1, 0.5),
            'weight_decay': (1e-6, 1e-3)
        }
        
        # 定义调整阈值
        self.adjustment_thresholds = {
            'performance_drop': 0.1,  # 性能下降阈值
            'loss_spike': 0.5,       # 损失突增阈值
            'gradient_norm': 10.0     # 梯度范数阈值
        }
        
        logger.info("动态优化器初始化完成")

    def optimize(self, current_metrics):
        """
        根据当前指标优化参数
        Args:
            current_metrics: 当前性能指标
        Returns:
            dict: 优化后的参数
        """
        try:
            # 检查是否需要调整
            if not self._needs_adjustment(current_metrics):
                return None
                
            # 获取优化建议
            suggestions = self._get_optimization_suggestions(current_metrics)
            
            # 应用参数调整
            new_params = self._apply_adjustments(suggestions)
            
            # 记录优化历史
            self.optimization_history.append({
                'metrics': current_metrics,
                'suggestions': suggestions,
                'new_params': new_params
            })
            
            return new_params
            
        except Exception as e:
            logger.error(f"参数优化过程出错: {str(e)}")
            return None

    def _needs_adjustment(self, metrics):
        """
        判断是否需要调整参数
        Args:
            metrics: 当前性能指标
        Returns:
            bool: 是否需要调整
        """
        try:
            # 检查性能下降
            if metrics['performance_change'] < -self.adjustment_thresholds['performance_drop']:
                return True
                
            # 检查损失突增
            if metrics['loss_change'] > self.adjustment_thresholds['loss_spike']:
                return True
                
            # 检查梯度不稳定
            if metrics['gradient_norm'] > self.adjustment_thresholds['gradient_norm']:
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"检查参数调整需求时出错: {str(e)}")
            return False

    def _get_optimization_suggestions(self, metrics):
        """
        获取参数优化建议
        Args:
            metrics: 当前性能指标
        Returns:
            dict: 参数调整建议
        """
        try:
            suggestions = {}
            
            # 根据性能变化调整学习率
            if metrics['performance_change'] < 0:
                suggestions['learning_rate'] = self._adjust_learning_rate(
                    metrics['current_lr'],
                    metrics['performance_change']
                )
            
            # 根据内存使用调整批次大小
            if metrics['memory_usage'] > 0.9:  # 内存使用超过90%
                suggestions['batch_size'] = self._adjust_batch_size(
                    metrics['current_batch_size']
                )
            
            # 根据过拟合风险调整dropout
            if metrics['validation_loss'] > metrics['training_loss'] * 1.2:
                suggestions['dropout_rate'] = self._adjust_dropout_rate(
                    metrics['current_dropout']
                )
            
            return suggestions
            
        except Exception as e:
            logger.error(f"生成优化建议时出错: {str(e)}")
            return {}

    def _apply_adjustments(self, suggestions):
        """
        应用参数调整
        Args:
            suggestions: 参数调整建议
        Returns:
            dict: 调整后的参数
        """
        try:
            new_params = {}
            
            for param_name, new_value in suggestions.items():
                # 验证参数范围
                if param_name in self.param_ranges:
                    min_val, max_val = self.param_ranges[param_name]
                    new_value = np.clip(new_value, min_val, max_val)
                    new_params[param_name] = new_value
                    
            return new_params
            
        except Exception as e:
            logger.error(f"应用参数调整时出错: {str(e)}")
            return {}

    def _adjust_learning_rate(self, current_lr, performance_change):
        """调整学习率"""
        if performance_change < -0.2:  # 性能显著下降
            return current_lr * 0.5
        elif performance_change < -0.1:  # 性能轻微下降
            return current_lr * 0.8
        return current_lr

    def _adjust_batch_size(self, current_batch_size):
        """调整批次大小"""
        return max(16, current_batch_size // 2)  # 减小批次大小，但不小于16

    def _adjust_dropout_rate(self, current_dropout):
        """调整dropout率"""
        return min(0.5, current_dropout + 0.1)  # 增加dropout，但不超过0.5

    def optimize_parameters(self):
        self.bayesian_optimization()  # 贝叶斯优化初始化基础参数
        logger.info("初始参数设置完成：%s", self.best_params)

    def save_best_params(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"dynamic_params_{timestamp}.json"
        with open(os.path.join(config_manager.MODEL_DIR, filename), 'w') as f:
            json.dump(self.best_params, f)
