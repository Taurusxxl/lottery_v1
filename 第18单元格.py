#18 配置系统测试用例\test_config.py
def test_config_instance():
    from core.config_manager import config_instance
    assert isinstance(config_instance.DB_CONFIG, dict)
    assert 'host' in config_instance.DB_CONFIG
    assert 'port' in config_instance.DB_CONFIG
    assert 'user' in config_instance.DB_CONFIG
    print("数据库配置测试通过")

    assert isinstance(config_instance.TRAINING_CONFIG, dict)
    assert 'max_epochs' in config_instance.TRAINING_CONFIG
    assert 'batch_size' in config_instance.TRAINING_CONFIG
    print("训练配置测试通过")

    assert isinstance(config_instance.SYSTEM_CONFIG, dict)
    assert 'memory_limit' in config_instance.SYSTEM_CONFIG
    assert 'gpu_memory_limit' in config_instance.SYSTEM_CONFIG
    print("系统配置测试通过")

# 运行测试
test_config_instance()