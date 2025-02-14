#18 配置系统测试用例\test_config.py
from cell1 import config_instance  # 从cell1导入配置实例
import numpy as np
import logging

# 添加配置测试用例
def test_config_system():
    """测试配置系统完整性"""
    required_keys = ['host', 'port', 'user', 'password']
    assert all(key in config_instance.DB_CONFIG for key in required_keys), "数据库配置缺失必要参数"
    logger.info("配置系统测试通过")

# 原有代码保持不变
class ConfigValidator:
    def __init__(self):
        self.validation_rules = {
            'learning_rate': lambda x: 0 < x < 1,
            'batch_size': lambda x: x > 0 and x & (x-1) == 0  # 验证是否为2的幂
        }
    
    def validate(self, config: dict) -> bool:
        valid = True
        for key, rule in self.validation_rules.items():
            if key in config:
                if not rule(config[key]):
                    logger.warning(f"参数 {key} 的值 {config[key]} 无效")
                    valid = False
        return valid

# 在模块加载时自动运行测试
if __name__ == "__main__":
    test_config_system()