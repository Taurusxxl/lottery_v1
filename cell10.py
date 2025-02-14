#10 模型配置管理\model_config.py
import os
import json
import logging
from typing import Dict, Any, Optional
import tensorflow as tf
from cell1 import config_instance  # 修改后的导入

# 获取logger实例
logger = logging.getLogger(__name__)

class ModelConfig:
    """模型配置管理类"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化模型配置
        
        Args:
            config_path: 配置文件路径,如果为None则使用默认配置
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.sequence_length = config_instance.SYSTEM_CONFIG['SAMPLE_CONFIG']['input_length']
        
    def _load_config(self) -> Dict[str, Any]:
        """加载配置"""
        try:
            if self.config_path and os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                logger.info(f"从{self.config_path}加载配置")
                return config
            
            # 使用默认配置
            return self._get_default_config()
            
        except Exception as e:
            logger.error(f"加载配置失败: {str(e)}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        sample_cfg = config_instance.SYSTEM_CONFIG['SAMPLE_CONFIG']
        return {
            # 1. 基础配置
            'base_config': {
                'sequence_length': sample_cfg['input_length'],
                'feature_dim': 5,
                'prediction_range': sample_cfg['target_length'],
                'batch_size': 32,
                'epochs': 100
            },
            
            # 2. 优化器配置
            'optimizer_config': {
                'optimizer_type': 'adam',
                'learning_rate': 0.001,
                'beta_1': 0.9,
                'beta_2': 0.999,
                'epsilon': 1e-7,
                'weight_decay': 0.0001
            },
            
            # 3. 损失函数配置
            'loss_config': {
                'loss_type': 'enhanced_match_loss',
                'huber_delta': 1.0,
                'focal_gamma': 2.0,
                'label_smoothing': 0.1
            },
            
            # 4. 模型架构配置
            'architecture_config': {
                # LSTM + GRU + 注意力模型
                'model_1': {
                    'lstm_units': 128,
                    'lstm_dropout': 0.3,
                    'gru_units': 64,
                    'attention_heads': 4
                },
                
                # 双向LSTM + Transformer模型
                'model_2': {
                    'bilstm_units': 128,
                    'transformer_blocks': 2,
                    'transformer_heads': 4
                },
                
                # TCN + LSTM模型
                'model_3': {
                    'tcn_filters': 64,
                    'tcn_kernel_size': 3,
                    'lstm_units': 128
                },
                
                # Transformer + 残差模型
                'model_4': {
                    'num_layers': 3,
                    'num_heads': 4,
                    'd_model': 128
                },
                
                # 双向GRU + 注意力模型
                'model_5': {
                    'gru_units': 128,
                    'attention_dim': 64
                },
                
                # LSTM + CNN + 注意力模型
                'model_6': {
                    'lstm_units': 128,
                    'cnn_filters': [64, 128],
                    'attention_units': 64
                }
            },
            
            # 5. 训练配置
            'training_config': {
                'early_stopping_patience': 20,
                'reduce_lr_patience': 10,
                'min_delta': 1e-4,
                'validation_split': 0.2,
                'shuffle': True
            },
            
            # 6. 集成配置
            'ensemble_config': {
                'voting_method': 'weighted',
                'min_weight': 0.1,
                'max_weight': 0.9,
                'weight_update_freq': 100
            },
            
            # 7. 监控配置
            'monitor_config': {
                'performance_window': 500,
                'alert_threshold': 0.2,
                'recovery_patience': 10
            },
            
            # 8. 保存配置
            'save_config': {
                'save_freq': 100,
                'max_to_keep': 5,
                'save_format': 'tf'
            }
        }
    
    def get_model_config(self, model_index: int) -> Dict[str, Any]:
        """获取指定模型的配置"""
        try:
            return self.config['architecture_config'][f'model_{model_index}']
        except KeyError:
            logger.error(f"未找到模型{model_index}的配置")
            return {}
    
    def get_optimizer_config(self) -> Dict[str, Any]:
        """获取优化器配置"""
        return self.config['optimizer_config']
    
    def get_loss_config(self) -> Dict[str, Any]:
        """获取损失函数配置"""
        return self.config['loss_config']
    
    def get_training_config(self) -> Dict[str, Any]:
        """获取训练配置"""
        return self.config['training_config']
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """更新配置"""
        try:
            self.config.update(new_config)
            self._save_config()
            logger.info("配置更新成功")
        except Exception as e:
            logger.error(f"更新配置失败: {str(e)}")
    
    def _save_config(self) -> None:
        """保存配置到文件"""
        if self.config_path:
            try:
                with open(self.config_path, 'w', encoding='utf-8') as f:
                    json.dump(self.config, f, indent=4)
                logger.info(f"配置已保存到{self.config_path}")
            except Exception as e:
                logger.error(f"保存配置失败: {str(e)}")
    
    def validate_config(self) -> bool:
        """验证配置有效性"""
        try:
            # 验证基础配置
            base_config = self.config['base_config']
            assert base_config['sequence_length'] > 0
            assert base_config['feature_dim'] > 0
            assert base_config['batch_size'] > 0
            
            # 验证优化器配置
            optimizer_config = self.config['optimizer_config']
            assert optimizer_config['learning_rate'] > 0
            assert 0 < optimizer_config['beta_1'] < 1
            assert 0 < optimizer_config['beta_2'] < 1
            
            # 验证模型架构配置
            for model_name, model_config in self.config['architecture_config'].items():
                assert all(value > 0 for value in model_config.values() if isinstance(value, (int, float)))
            
            return True
            
        except AssertionError as e:
            logger.error(f"配置验证失败: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"配置验证出错: {str(e)}")
            return False

    def save_best_params(self):
        # 保存当前最佳参数组合
        with open('best_params.json', 'w') as f:
            json.dump(self.best_params, f)

# 创建全局实例
model_config = ModelConfig()
