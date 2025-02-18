#10 Training Management System / 训练管理系统
import traceback
import os
import tensorflow as tf
import numpy as np
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from cell1_core import core_manager
from cell3_monitor import resource_monitor, performance_monitor
from datetime import datetime, timedelta
from collections import deque
from cell6_model import model_core
import gc
from cell4_data import data_manager

# 获取logger实例
logger = logging.getLogger(__name__)

class TrainingManager:
    """训练管理器类"""
    
    def __init__(self, model_ensemble):
        self.model_ensemble = model_ensemble
        
        # 获取训练配置
        self.config = core_manager.SYSTEM_CONFIG['TRAINING_CONFIG']
        self.batch_size = self.config['batch_size']
        self.epochs = self.config['max_epochs']
        
        # 添加数据管理器
        self.data_manager = data_manager
        
        # 训练状态
        self.is_training = False
        self.pause_training = False
        self._training_loop_running = False
        self._training_thread = None
        
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
        
        logger.info("训练管理器初始化完成")
        
        self.issue_file = "D:\\JupyterWork\\notebooks\\period_number\\issue.txt"
        self.current_target_period = None
    
    def start_training(self, training_data):
        """启动训练（增加数据量检查）"""
        try:
            # 修改后的数据量检查
            if training_data.shape[0] != 14400 + 2880:  # 检查数组第一维长度
                logger.error(f"数据量不匹配，预期{14400+2880}条，实际{training_data.shape[0]}条")
                return False
            
            # 添加数据校验
            if training_data is None:
                logger.error("训练数据为空")
                return False
            if len(training_data) == 0:
                logger.error("训练数据长度为零")
                return False
            
            if self.is_training:
                logger.warning("训练已在进行中")
                return False
            
            self.is_training = True
            self.pause_training = False
            
            # 初始化训练状态
            self._init_training(training_data)
            
            # 启动训练线程
            self._training_thread = threading.Thread(
                target=self._training_loop,
                args=(training_data,)
            )
            self._training_thread.start()
            
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
            if self._training_thread and self._training_thread.is_alive():
                self._training_thread.join()
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
            strategy = tf.distribute.MirroredStrategy()
            with strategy.scope():
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

    def _calculate_next_period(self, current_period):
        """计算下一期号"""
        try:
            date_part, num_part = current_period.split('-')
            current_date = datetime.strptime(date_part, "%Y%m%d")
            current_num = int(num_part)
            
            if current_num < 1440:
                return f"{date_part}-{current_num+1:04d}"
            else:
                next_date = current_date + timedelta(days=1)
                return f"{next_date.strftime('%Y%m%d')}-0001"
        except Exception as e:
            logger.error(f"计算下一期号失败: {str(e)}")
            return None

    def _get_last_processed_period(self):
        """获取最后处理的期号"""
        try:
            if os.path.exists(self.issue_file):
                with open(self.issue_file, 'r') as f:
                    return f.read().strip()
            return None
        except Exception as e:
            logger.error(f"读取期号文件失败: {str(e)}")
            return None

    def _save_processed_period(self, period):
        """保存处理完成的期号"""
        try:
            os.makedirs(os.path.dirname(self.issue_file), exist_ok=True)
            with open(self.issue_file, 'w') as f:
                f.write(period)
        except Exception as e:
            logger.error(f"保存期号文件失败: {str(e)}")

    def _generate_sequence_periods(self, start_period, length):
        """生成连续的期号序列"""
        periods = [start_period]
        current = start_period
        for _ in range(length-1):
            current = self._calculate_next_period(current)
            if not current:
                return None
            periods.append(current)
        return periods

    def _fetch_training_data(self, start_period):
        """获取训练数据（增加数据验证）"""
        try:
            # 生成需要获取的期号范围
            total_periods = self._generate_sequence_periods(start_period, 14400+2880)
            if not total_periods or len(total_periods) != 14400+2880:
                logger.error("期号生成不完整")
                return None
                
            # 获取数据
            query = """
                SELECT number, date_period FROM admin_tab
                WHERE date_period IN %s
                ORDER BY date_period ASC
            """
            data = data_manager.execute_query(query, (tuple(total_periods),))
            
            # 验证数据完整性
            if len(data) != 14400+2880:
                logger.error(f"数据不完整，预期{14400+2880}条，实际获取{len(data)}条")
                return None
                
            # 新增数据量打印
            logger.info(f"获取到训练数据 {len(data)} 条，时间范围: {data[0]['date_period']} 至 {data[-1]['date_period']}")
            
            return data
        except Exception as e:
            logger.error(f"获取训练数据失败: {str(e)}")
            return None

    def _validate_sequence(self, sequence):
        """验证训练序列有效性"""
        try:
            if sequence is None:
                return False
                
            # 检查是否为numpy数组
            if not isinstance(sequence, np.ndarray):
                logger.warning(f"序列类型错误: {type(sequence)}")
                return False
                
            # 检查数据形状
            if sequence.shape != (14400+2880, 5):
                logger.warning(f"无效序列形状: {sequence.shape}")
                return False
                
            # 检查数值范围
            if np.min(sequence) < -1 or np.max(sequence) > 1:
                logger.warning(f"数值范围异常: [{np.min(sequence):.2f}, {np.max(sequence):.2f}]")
                return False
                
            return True
        except Exception as e:
            logger.error(f"序列验证失败: {str(e)}")
            return False

    def training_loop(self):
        """训练主循环"""
        logger.info("开始训练循环")
        issue_file = "D:/JupyterWork/notebooks/period_number/issue.txt"
        
        while True:
            try:
                # 1. 读取上一次训练的最后期号
                try:
                    with open(issue_file, 'r') as f:
                        last_issue = f.read().strip()
                    logger.info(f"读取到上次期号: {last_issue}")
                except Exception as e:
                    logger.error(f"读取期号文件失败: {str(e)}")
                    time.sleep(60)
                    continue
                
                # 2. 计算下一个目标期号
                next_target_issue = self._calculate_next_issue(last_issue)
                logger.info(f"下一目标期号: {next_target_issue}")
                
                # 3. 获取并处理训练序列
                sequence_data = self._get_training_sequence(next_target_issue)
                if sequence_data is None:
                    logger.info(f"等待期号 {next_target_issue} 的数据...")
                    time.sleep(60)
                    continue
                    
                input_data, target_data = sequence_data
                logger.info(f"获取到序列数据，输入形状: {input_data.shape}, 目标形状: {target_data.shape}")
                
                # 4. 训练所有模型
                with self.model_ensemble.session.as_default():
                    with self.model_ensemble.graph.as_default():
                        for i, model in enumerate(self.model_ensemble.models):
                            try:
                                # 训练前评估
                                initial_loss = model.evaluate(input_data, target_data, verbose=0)
                                
                                # 训练
                                loss = model.train_on_batch(input_data, target_data)
                                
                                # 训练后评估
                                final_loss = model.evaluate(input_data, target_data, verbose=0)
                                
                                logger.info(f"模型 {i+1} 训练: 初始损失={initial_loss:.4f}, "
                                          f"最终损失={final_loss:.4f}, 变化={initial_loss-final_loss:.4f}")
                                
                            except Exception as e:
                                logger.error(f"模型 {i+1} 训练失败: {str(e)}")
                                continue
                
                # 5. 保存新的期号
                try:
                    with open(issue_file, 'w') as f:
                        f.write(next_target_issue)
                    logger.info(f"已更新期号为: {next_target_issue}")
                except Exception as e:
                    logger.error(f"更新期号文件失败: {str(e)}")
                
                # 6. 等待一段时间再继续
                time.sleep(10)
                
            except Exception as e:
                logger.error(f"训练循环出错: {str(e)}")
                time.sleep(60)
                continue
            
    def _get_start_issue(self, end_issue):
        """计算起始期号（往前推12天）"""
        date_str, _ = end_issue.split('-')
        end_date = datetime.strptime(date_str, "%Y%m%d")
        start_date = end_date - timedelta(days=12)
        return f"{start_date.strftime('%Y%m%d')}-0001"

    def _calculate_next_issue(self, current_issue):
        """计算下一个期号"""
        # 解析当前期号 (格式: YYYYMMDD-XXXX)
        date_str, period = current_issue.split('-')
        year = int(date_str[:4])
        month = int(date_str[4:6])
        day = int(date_str[6:8])
        period_num = int(period)
        
        # 计算下一期
        if period_num < 1440:
            # 同一天的下一期
            next_period = f"{period_num + 1:04d}"
            next_date = date_str
        else:
            # 下一天的第一期
            next_period = "0001"
            next_date = datetime(year, month, day) + timedelta(days=1)
            next_date = next_date.strftime("%Y%m%d")
            
        return f"{next_date}-{next_period}"
        
    def _get_training_sequence(self, target_issue):
        """获取训练序列"""
        try:
            # 计算起始期号（往前推12天）
            start_issue = self._get_start_issue(target_issue)
            
            # 从数据库获取序列
            sequence = self.data_manager.get_sequence(start_issue, target_issue)
            
            if sequence is None:
                return None
            
            # 确保序列长度正确
            if sequence.shape[0] > 14400 + 2880:
                # 如果序列太长，截取最后的部分
                sequence = sequence[-(14400 + 2880):]
            elif sequence.shape[0] < 14400 + 2880:
                logger.error(f"序列长度不足: {sequence.shape[0]}, 需要 {14400 + 2880}")
                return None
            
            # 分割输入和目标序列
            input_sequence = sequence[:14400]
            target_sequence = sequence[14400:14400+2880]
            
            # 添加批次维度
            input_sequence = np.expand_dims(input_sequence, axis=0)
            target_sequence = np.expand_dims(target_sequence, axis=0)
            
            return (input_sequence, target_sequence)
            
        except Exception as e:
            logger.error(f"获取训练序列失败: {str(e)}")
            return None

    def ensure_training_thread(self):
        """确保训练线程运行"""
        if not self._training_loop_running:
            self._training_loop_running = True
            self._training_thread = threading.Thread(
                target=self.training_loop,
                name="TrainingThread",
                daemon=True
            )
            self._training_thread.start()
            logger.info("训练线程已启动")

    def _validate_training_effect(self, model, input_data, target_data, model_index):
        """验证训练效果"""
        try:
            # 添加batch维度
            input_data = np.expand_dims(input_data, axis=0)  # 转换为 (1, sequence_length, features)
            target_data = np.expand_dims(target_data, axis=0)  # 转换为 (1, target_length, features)
            
            # 训练前预测
            pred_before = model.predict(input_data)
            
            # 训练
            loss = model.train_on_batch(input_data, target_data)
            
            # 训练后预测
            pred_after = model.predict(input_data)
            
            # 计算预测变化
            pred_change = np.mean(np.abs(pred_after - pred_before))
            
            # 计算准确度变化
            acc_before = np.mean(np.abs(pred_before - target_data))
            acc_after = np.mean(np.abs(pred_after - target_data))
            
            logger.info(f"模型 {model_index} 训练效果:"
                       f"\n - 预测变化: {pred_change:.4f}"
                       f"\n - 准确度提升: {acc_before-acc_after:.4f}")
            
        except Exception as e:
            logger.error(f"验证训练效果失败: {str(e)}")

    def check_training_status(self):
        """检查训练状态"""
        try:
            # 初始化状态字典
            status = {
                'is_training': self._training_loop_running,
                'thread_alive': False,
                'models_compiled': False,
                'last_losses': [],
                'weights_initialized': []
            }
            
            # 检查训练线程状态
            if self._training_thread:
                status['thread_alive'] = self._training_thread.is_alive()
            
            # 检查模型编译状态
            if self.model_ensemble and hasattr(self.model_ensemble, 'models'):
                with self.model_ensemble.session.as_default():
                    with self.model_ensemble.graph.as_default():
                        models_compiled = []
                        weights_initialized = []
                        last_losses = []
                        
                        for model in self.model_ensemble.models:
                            # 检查模型是否已编译
                            has_optimizer = hasattr(model, 'optimizer')
                            models_compiled.append(has_optimizer)
                            
                            # 检查权重是否已初始化
                            if hasattr(model, 'get_weights'):
                                weights = model.get_weights()
                                weights_initialized.append(
                                    len(weights) > 0 and any(np.any(w != 0) for w in weights)
                                )
                            else:
                                weights_initialized.append(False)
                            
                            # 尝试获取最后的损失值
                            if hasattr(model, 'history') and model.history:
                                if model.history.history and 'loss' in model.history.history:
                                    last_losses.append(model.history.history['loss'][-1])
                                else:
                                    last_losses.append(None)
                            else:
                                last_losses.append(None)
                        
                        status['models_compiled'] = all(models_compiled)
                        status['weights_initialized'] = weights_initialized
                        status['last_losses'] = last_losses
            
            logger.info(f"训练状态检查完成: {status}")
            return status
            
        except Exception as e:
            logger.error(f"检查训练状态失败: {str(e)}")
            # 返回基本状态信息
            return {
                'is_training': False,
                'thread_alive': False,
                'models_compiled': False,
                'last_losses': [],
                'weights_initialized': [],
                'error': str(e)
            }

# 然后创建训练管理器
training_manager = TrainingManager(model_ensemble=model_core)

# 最后启动训练线程
if not hasattr(training_manager, '_training_loop_running'):
    training_manager.ensure_training_thread()
