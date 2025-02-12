"""
彩票预测系统主入口
Usage: 
  python main.py --mode train      # 训练模式
  python main.py --mode predict    # 预测模式
"""

import argparse
from core.system_manager import SystemManager
from core.config_manager import ConfigLoader
from core.logging_manager import init_logging

def main():
    # 初始化基础组件
    init_logging()
    config = ConfigLoader().config
    
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='彩票预测系统')
    parser.add_argument('--mode', choices=['train', 'predict'], required=True)
    parser.add_argument('--model', default='ensemble', help='选择模型类型')
    args = parser.parse_args()

    # 初始化系统管理器
    system = SystemManager(config)
    
    try:
        if args.mode == 'train':
            # 训练流程
            system.initialize_data_pipeline()
            system.build_model_ensemble(args.model)
            system.start_training()
        elif args.mode == 'predict':
            # 预测流程
            predictions = system.run_prediction()
            print(f"预测结果: {predictions}")
            
    except Exception as e:
        system.logger.error(f"系统运行异常: {str(e)}")
        raise

if __name__ == "__main__":
    main() 