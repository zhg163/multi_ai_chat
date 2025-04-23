#!/usr/bin/env python
"""
入口脚本 - 启动应用程序并确保正确的导入路径
"""

import os
import sys

# 将项目根目录添加到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# 导入主应用并运行
if __name__ == "__main__":
    try:
        import uvicorn
        from app.common.logger_config import setup_logger
        
        # 设置日志
        logger = setup_logger(name="app")
        logger.info("应用启动中...")
        
        # 运行应用
        uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
    except ImportError as e:
        print(f"启动失败，导入错误: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"启动失败，未知错误: {e}")
        sys.exit(1) 