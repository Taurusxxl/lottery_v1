# 主程序入口
import logging
from datetime import datetime
import os
import sys
import json

# 导入自定义模块
from cell1 import ConfigManager
from cell5 import DatabaseManager
from cell6 import DataManager
from cell7 import ResourceMonitor
from cell8 import PerformanceMonitor
from cell11 import ModelEnsemble
from cell13 import ModelOptimizer
from cell14 import DynamicOptimizer
from cell16 import StateManager

def setup_monitoring():
    """初始化监控系统"""
    # 从配置获取保存目录
    config = ConfigManager()
    save_dir = os.path.join(config.BASE_DIR, 'performance_records')
    os.makedirs(save_dir, exist_ok=True)

    resource_monitor = ResourceMonitor()
    performance_monitor = PerformanceMonitor(save_dir=save_dir)
    
    # 启动监控
    resource_monitor.start()
    performance_monitor.start()
    
    return resource_monitor, performance_monitor

def setup_logging():
    """配置日志系统"""
    log_dir = os.path.join('JupyterWork', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f'system_{datetime.now().strftime("%Y%m%d")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def initialize_system():
    """初始化系统组件"""
    config = ConfigManager()
    db_manager = DatabaseManager()
    data_manager = DataManager()
    state_manager = StateManager()
    
    return config, db_manager, data_manager, state_manager

def main():
    # 设置日志
    logger = setup_logging()
    
    try:
        # 1. 初始化系统
        logger.info("正在初始化系统...")
        config, db_manager, data_manager, state_manager = initialize_system()
        
        # 2. 启动监控
        logger.info("启动系统监控...")
        resource_monitor, perf_monitor = setup_monitoring()
        
        # 3. 加载数据
        logger.info("正在加载数据...")
        data_manager._init_data_loader()
        train_data = data_manager.load_training_data()
        
        # 4. 创建模型和优化器
        logger.info("正在初始化模型...")
        model = ModelEnsemble()
        optimizer = ModelOptimizer(model)
        dynamic_optimizer = DynamicOptimizer()
        
        # 5. 开始训练循环
        logger.info("开始训练...")
        best_loss = float('inf')
        
        def optimization_callback(study, trial):
            nonlocal best_loss
            current_loss = trial.value
            if current_loss < best_loss:
                best_loss = current_loss
                logger.info(f"发现更好的模型，loss: {best_loss}")
                model.save_checkpoint()
        
        optimizer.optimize(
            train_data,
            callbacks=[optimization_callback]
        )
        
        # 6. 保存结果
        logger.info("保存结果...")
        model.save_weights()
        
        # 7. 记录性能指标
        perf_data = {
            'timestamp': datetime.now().isoformat(),
            'best_loss': float(best_loss),
            'training_time': perf_monitor.get_total_time(),
            'resource_usage': resource_monitor.get_summary()
        }
        
        performance_file = os.path.join('logs', 'performance', 
                                      f'performance_{datetime.now().strftime("%Y%m%d")}.json')
        os.makedirs(os.path.dirname(performance_file), exist_ok=True)
        
        with open(performance_file, 'w') as f:
            json.dump(perf_data, f, indent=4)
        
        logger.info("训练完成！")
        
    except Exception as e:
        logger.error(f"运行出错: {str(e)}")
        raise
    finally:
        # 清理资源
        logger.info("清理资源...")
        if 'resource_monitor' in locals():
            resource_monitor.stop()
        if 'perf_monitor' in locals():
            perf_monitor.stop()
        db_manager.close_all()

if __name__ == "__main__":
    main() 