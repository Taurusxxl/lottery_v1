
import tensorflow as tf
import numpy as np
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from cell1 import config_instance
from cell7 import resource_monitor
from cell8 import performance_monitor
from datetime import datetime
from collections import deque

# 获取logger实例
logger = logging.getLogger(__name__)

class TrainingManager:
    """训练管理器类"""
    
    def __init__(self, model_ensemble):
        self.model_ensemble = model_ensemble
        self.batch_size = 32
        self.epochs = 100
        
        # 训练状态
        self.is_training = False
        self.pause_training = False
        self.training_thread = None
        
        # 训练进度
        self.current_epoch = 0
        self.current_batch = 0
        self.total_batches = 0
        
        # 训练性能监控
        self.performance_history = deque(maxlen=1000)
        self.resource_monitor = resource_monitor
        self.performance_monitor = performance_monitor
        
        # 训练同步机制
        self.training_lock = threading.Lock()
        self.finished_models = 0
        self.total_models = 6
        
        # 训练配置
        self.config = config_instance.SYSTEM_CONFIG['TRAINING_CONFIG']
        
        logger.info("训练管理器初始化完成")
    
    def start_training(self, training_data):
        """启动训练"""
        try:
            if self.is_training:
                logger.warning("训练已在进行中")
                return False
            
            self.is_training = True
            self.pause_training = False
            
            # 初始化训练状态
            self._init_training(training_data)
            
            # 启动训练线程
            self.training_thread = threading.Thread(
                target=self._training_loop,
                args=(training_data,)
            )
            self.training_thread.start()
            
            logger.info("训练已启动")
            return True
            
        except Exception as e:
            logger.error(f"启动训练失败: {str(e)}")
            self.is_training = False
            return False
    
    def stop_training(self):
        """停止训练"""
        try:
            self.is_training = False
            if self.training_thread and self.training_thread.is_alive():
                self.training_thread.join()
            logger.info("训练已停止")
            return True
        except Exception as e:
            logger.error(f"停止训练失败: {str(e)}")
            return False
    
    def pause_resume_training(self):
        """暂停/恢复训练"""
        try:
            self.pause_training = not self.pause_training
            status = "暂停" if self.pause_training else "恢复"
            logger.info(f"训练已{status}")
            return True
        except Exception as e:
            logger.error(f"训练暂停/恢复失败: {str(e)}")
            return False
    
    def _init_training(self, training_data):
        """初始化训练"""
        self.current_epoch = 0
        self.current_batch = 0
        self.total_batches = len(training_data) // self.batch_size
        self.performance_history.clear()
        
        # 初始化资源监控
        self.resource_monitor.start()
        
        # 初始化性能监控
        self.performance_monitor.reset()
    
    def _training_loop(self, training_data):
        """训练主循环"""
        try:
            while self.is_training and self.current_epoch < self.epochs:
                
                # 检查暂停状态
                if self.pause_training:
                    time.sleep(1)
                    continue
                
                # 检查系统资源
                if not self._check_resources():
                    time.sleep(5)
                    continue
                
                # 获取训练批次
                batch_data = self._get_next_batch(training_data)
                if batch_data is None:
                    continue
                
                # 并行训练模型
                self._parallel_train_models(batch_data)
                
                # 更新训练状态
                self._update_training_status()
                
                # 记录训练性能
                self._record_performance()
                
            logger.info("训练完成")
            
        except Exception as e:
            logger.error(f"训练循环出错: {str(e)}")
            self.is_training = False
    
    def _check_resources(self):
        """检查系统资源"""
        try:
            # 检查CPU使用率
            if self.resource_monitor.cpu_usage > 90:
                logger.warning("CPU使用率过高,暂停训练")
                return False
            
            # 检查内存使用率    
            if self.resource_monitor.memory_usage > 90:
                logger.warning("内存使用率过高,暂停训练")
                return False
            
            # 检查GPU使用率
            if self.resource_monitor.gpu_usage > 90:
                logger.warning("GPU使用率过高,暂停训练")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"资源检查失败: {str(e)}")
            return False
    
    def _get_next_batch(self, training_data):
        """获取下一个训练批次"""
        try:
            start_idx = self.current_batch * self.batch_size
            end_idx = start_idx + self.batch_size
            
            if end_idx > len(training_data):
                self.current_epoch += 1
                self.current_batch = 0
                return None
                
            batch = training_data[start_idx:end_idx]
            self.current_batch += 1
            
            return self._preprocess_batch(batch)
            
        except Exception as e:
            logger.error(f"获取训练批次失败: {str(e)}")
            return None
    
    def _parallel_train_models(self, batch_data):
        """并行训练模型"""
        try:
            with ThreadPoolExecutor(max_workers=self.total_models) as executor:
                futures = []
                for i, model in enumerate(self.model_ensemble.models):
                    future = executor.submit(
                        self._train_single_model,
                        model,
                        batch_data,
                        i
                    )
                    futures.append(future)
                
                # 等待所有模型训练完成
                for future in futures:
                    future.result()
                    
        except Exception as e:
            logger.error(f"并行训练失败: {str(e)}")
    
    def _train_single_model(self, model, batch_data, model_idx):
        """训练单个模型"""
        try:
            # 1. 前向传播
            with tf.GradientTape() as tape:
                predictions = model(batch_data['input'])
                loss = self._calculate_loss(predictions, batch_data['target'])
            
            # 2. 反向传播
            gradients = tape.gradient(loss, model.trainable_variables)
            self.model_ensemble.optimizer.apply_gradients(
                zip(gradients, model.trainable_variables)
            )
            
            # 3. 更新模型权重
            self._update_model_weights(model_idx, loss.numpy())
            
            # 4. 记录训练进度
            with self.training_lock:
                self.finished_models += 1
                if self.finished_models == self.total_models:
                    self.finished_models = 0
                    self._on_batch_complete()
                    
        except Exception as e:
            logger.error(f"训练模型 {model_idx} 失败: {str(e)}")
    
    def _calculate_loss(self, predictions, targets):
        """计算训练损失"""
        try:
            # 使用增强型匹配损失函数
            return self.model_ensemble.enhanced_match_loss(targets, predictions)
        except Exception as e:
            logger.error(f"计算损失失败: {str(e)}")
            return tf.constant(0.0)
    
    def _update_model_weights(self, model_idx, loss):
        """更新模型权重"""
        try:
            # 记录性能
            self.performance_history.append({
                'model_idx': model_idx,
                'loss': loss,
                'timestamp': datetime.now()
            })
            
            # 更新权重
            performance = np.exp(-loss)  # 损失越小,性能越好
            self.model_ensemble.weights[model_idx] = performance
            
            # 归一化权重
            total = np.sum(self.model_ensemble.weights)
            self.model_ensemble.weights /= total
            
        except Exception as e:
            logger.error(f"更新模型权重失败: {str(e)}")
    
    def _update_training_status(self):
        """更新训练状态"""
        try:
            # 计算训练进度
            total_steps = self.epochs * self.total_batches
            current_steps = self.current_epoch * self.total_batches + self.current_batch
            progress = current_steps / total_steps
            
            # 更新监控指标
            self.performance_monitor.update_metrics({
                'progress': progress,
                'current_epoch': self.current_epoch,
                'current_batch': self.current_batch,
                'loss': np.mean([p['loss'] for p in self.performance_history])
            })
            
        except Exception as e:
            logger.error(f"更新训练状态失败: {str(e)}")
    
    def _record_performance(self):
        """记录训练性能"""
        try:
            metrics = {
                'timestamp': datetime.now(),
                'epoch': self.current_epoch,
                'batch': self.current_batch,
                'loss': np.mean([p['loss'] for p in self.performance_history]),
                'resource_usage': {
                    'cpu': self.resource_monitor.cpu_usage,
                    'memory': self.resource_monitor.memory_usage,
                    'gpu': self.resource_monitor.gpu_usage
                }
            }
            
            self.performance_monitor.update_metrics(metrics)
            
        except Exception as e:
            logger.error(f"记录性能失败: {str(e)}")
    
    def _preprocess_batch(self, batch):
        """预处理训练批次"""
        try:
            return {
                'input': tf.convert_to_tensor(batch['input']),
                'target': tf.convert_to_tensor(batch['target'])
            }
        except Exception as e:
            logger.error(f"预处理批次失败: {str(e)}")
            return None
    
    def _on_batch_complete(self):
        """批次训练完成回调"""
        try:
            # 保存检查点
            if self.current_batch % self.config['save_frequency'] == 0:
                self._save_checkpoint()
            
            # 评估性能
            if self.current_batch % self.config['eval_frequency'] == 0:
                self._evaluate_performance()
                
            # 调整学习率
            if self.current_batch % self.config['lr_update_frequency'] == 0:
                self._adjust_learning_rate()
                
        except Exception as e:
            logger.error(f"批次完成处理失败: {str(e)}")
    
    def _save_checkpoint(self):
        """保存训练检查点"""
        try:
            checkpoint = {
                'epoch': self.current_epoch,
                'batch': self.current_batch,
                'model_states': [model.get_weights() for model in self.model_ensemble.models],
                'optimizer_state': self.model_ensemble.optimizer.get_weights(),
                'performance_history': list(self.performance_history)
            }
            
            save_path = f"checkpoints/checkpoint_e{self.current_epoch}_b{self.current_batch}.h5"
            tf.keras.models.save_model(checkpoint, save_path)
            logger.info(f"保存检查点: {save_path}")
            
        except Exception as e:
            logger.error(f"保存检查点失败: {str(e)}")
    
    def _evaluate_performance(self):
        """评估训练性能"""
        try:
            # 计算平均损失
            avg_loss = np.mean([p['loss'] for p in self.performance_history])
            
            # 计算性能改进
            if len(self.performance_history) > 1:
                prev_loss = self.performance_history[-2]['loss']
                improvement = (prev_loss - avg_loss) / prev_loss
                
                if improvement < self.config['min_improvement']:
                    logger.warning("性能改进不足")
                    
            logger.info(f"当前平均损失: {avg_loss:.4f}")
            
        except Exception as e:
            logger.error(f"评估性能失败: {str(e)}")
    
    def _adjust_learning_rate(self):
        """调整学习率"""
        try:
            if len(self.performance_history) < 2:
                return
                
            # 计算最近的性能变化
            recent_loss = np.mean([p['loss'] for p in list(self.performance_history)[-10:]])
            previous_loss = np.mean([p['loss'] for p in list(self.performance_history)[-20:-10]])
            
            # 根据性能变化调整学习率
            if recent_loss > previous_loss:
                new_lr = self.model_ensemble.optimizer.learning_rate * 0.8
                self.model_ensemble.optimizer.learning_rate.assign(new_lr)
                logger.info(f"降低学习率至: {new_lr:.6f}")
            
        except Exception as e:
            logger.error(f"调整学习率失败: {str(e)}")

    def get_training_status(self):
        """获取训练状态"""
        return {
            'is_training': self.is_training,
            'is_paused': self.pause_training,
            'current_epoch': self.current_epoch,
            'current_batch': self.current_batch,
            'total_epochs': self.epochs,
            'total_batches': self.total_batches,
            'progress': (self.current_epoch * self.total_batches + self.current_batch) / 
                       (self.epochs * self.total_batches)
        }

    def get_performance_metrics(self):
        """获取性能指标"""
        if not self.performance_history:
            return None
            
        recent_records = list(self.performance_history)[-100:]
        return {
            'average_loss': np.mean([r['loss'] for r in recent_records]),
            'min_loss': np.min([r['loss'] for r in recent_records]),
            'max_loss': np.max([r['loss'] for r in recent_records]),
            'loss_trend': self._calculate_trend([r['loss'] for r in recent_records])
        }

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

# 创建全局实例
training_manager = TrainingManager(model_ensemble=None)
