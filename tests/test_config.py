def test_config_instance():
    from core.config_manager import config_instance
    assert isinstance(config_instance.DB_CONFIG, dict)
    assert 'host' in config_instance.DB_CONFIG 