"""
多AI聊天系统 - 主应用模块

包含FastAPI主应用实例和路由配置
"""

import os
import logging
from dotenv import load_dotenv

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 加载环境变量
# 获取项目根目录
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
env_path = os.path.join(project_root, ".env")

# 加载.env文件
if os.path.exists(env_path):
    load_dotenv(env_path)
    logging.info(f"已加载环境变量: {env_path}")
else:
    logging.warning(f"未找到.env文件: {env_path}")

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse

# 创建FastAPI实例
app = FastAPI(
    title="多AI聊天系统",
    description="支持多角色AI聊天的平台",
    version="0.1.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 实际生产环境中应限制为特定域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 导入嵌入服务
try:
    from app.services.embedding_service import embedding_service
    logging.info("嵌入服务已导入")
except ImportError as e:
    logging.error(f"无法导入嵌入服务: {str(e)}")

# 导入简单测试路由
try:
    from app.api.simple_routes import router as simple_router
    app.include_router(simple_router)
    logging.info("简单测试路由已加载")
except ImportError as e:
    logging.error(f"无法导入简单测试路由: {str(e)}")

# 导入路由模块
try:
    from app.api.message_processing_routes import router as message_processing_router
    app.include_router(message_processing_router)
    logging.info("消息处理路由已加载")
except ImportError as e:
    logging.error(f"无法导入消息处理路由: {str(e)}")

try:
    from app.api.role_routes import router as role_router
    app.include_router(role_router)
    logging.info("角色管理路由已加载")
except ImportError as e:
    logging.error(f"无法导入角色管理路由: {str(e)}")

# 导入LLM路由模块
try:
    from app.api.llm_routes import router as llm_router
    app.include_router(llm_router)
    logging.info("LLM服务路由已加载")
except ImportError as e:
    logging.error(f"无法导入LLM服务路由: {str(e)}")
    
    # 尝试修复并重新导入
    try:
        # 尝试清除模块缓存
        import sys
        import importlib
        
        # 删除可能有问题的模块缓存
        for module_name in list(sys.modules.keys()):
            if module_name.startswith('app.api.llm_routes') or module_name.startswith('app.memory.memory_manager'):
                del sys.modules[module_name]
        
        # 重新导入memory_manager模块
        importlib.import_module('app.memory.memory_manager')
        
        # 然后尝试重新导入llm_routes
        llm_routes_module = importlib.import_module('app.api.llm_routes')
        llm_router = llm_routes_module.router
        app.include_router(llm_router)
        logging.info("LLM服务路由已成功重新加载")
    except Exception as reimpot_error:
        logging.error(f"尝试修复LLM服务路由失败: {str(reimpot_error)}")

# 导入用户路由
try:
    from app.api.user_routes import router as user_router
    app.include_router(user_router)
    logging.info("用户管理路由已加载")
except ImportError as e:
    logging.error(f"无法导入用户管理路由: {str(e)}")
    
    # 添加辅助函数获取数据库连接
    async def get_database():
        try:
            from app.database.connection import Database
            if Database.db is None:
                await Database.connect()
            return Database.db
        except Exception as db_error:
            logging.error(f"获取数据库连接失败: {str(db_error)}")
            return None
    
    # 添加简易用户API路由
    @app.get("/api/users/", tags=["users"])
    async def get_users(
        limit: int = 100,
        offset: int = 0
    ):
        """
        获取用户列表，如果数据库中没有用户，则返回示例用户
        """
        try:
            # 获取MongoDB连接
            db = await get_database()
            if db is None:
                logging.warning("数据库连接失败，返回示例用户")
                return [
                    {
                        "_id": "677f834ddaaba35dd9149b0b",
                        "username": "zhangsan",
                        "name": "轻舞飞扬",
                        "email": "zhang@example.com",
                        "avatar": "https://example.com/avatars/user1.png",
                        "description": "普通用户",
                        "tags": ["电影", "篮球"],
                        "is_active": True
                    }
                ]
            
            # 记录日志
            logging.info(f"正在获取用户列表，limit: {limit}, offset: {offset}")
            
            # 从数据库查询用户
            users_cursor = db.users.find({}).skip(offset).limit(limit)
            users = await users_cursor.to_list(length=limit)
            
            # 记录查询结果
            logging.info(f"从数据库获取到 {len(users)} 个用户")
            
            # 如果数据库中有用户，则返回实际用户
            if users:
                # 转换ObjectId为字符串
                for user in users:
                    user["_id"] = str(user["_id"])
                return users
            
            # 如果数据库中没有用户，则返回示例用户
            logging.warning("数据库中没有找到用户，返回示例用户")
            return [
                {
                    "_id": "677f834ddaaba35dd9149b0b",
                    "username": "zhangsan",
                    "name": "轻舞飞扬",
                    "email": "zhang@example.com",
                    "avatar": "https://example.com/avatars/user1.png",
                    "description": "普通用户",
                    "tags": ["电影", "篮球"],
                    "is_active": True
                }
            ]
        
        except Exception as e:
            # 记录错误
            logging.error(f"获取用户列表时发生错误: {str(e)}")
            return [
                {
                    "_id": "677f834ddaaba35dd9149b0b",
                    "username": "zhangsan",
                    "name": "轻舞飞扬",
                    "email": "zhang@example.com",
                    "avatar": "https://example.com/avatars/user1.png",
                    "description": "普通用户",
                    "tags": ["电影", "篮球"],
                    "is_active": True
                }
            ]

# 添加无需认证的结束会话端点
@app.post("/api/sessions/{session_id}/end-and-archive", tags=["sessions"])
async def end_and_archive_session(
    session_id: str,
    request: Request
):
    """结束会话并强制归档所有消息 (无需认证)"""
    try:
        # 解析请求体
        data = await request.json()
        user_id = data.get("user_id", "anonymous_user")
        
        logging.info(f"结束并归档会话: session_id={session_id}, user_id={user_id}")
        
        # 获取记忆管理器
        from app.memory.memory_manager import get_memory_manager
        memory_manager = await get_memory_manager()
        
        # 获取会话中所有消息
        messages = memory_manager.short_term.get_session_messages(session_id, user_id)
        message_count = len(messages)
        
        if not messages:
            logging.warning(f"会话 {session_id} 没有消息可归档")
            return {
                "success": False,
                "archived_messages_count": 0,
                "total_messages": 0,
                "message": "会话没有消息可归档"
            }
        
        # 归档消息计数
        archived_count = 0
        
        # 逐条归档消息到MongoDB
        for message in messages:
            success = await memory_manager.archive_message(session_id, user_id, message)
            if success:
                archived_count += 1
        
        # 调用会话结束函数，生成摘要
        result = await memory_manager.end_session(session_id, user_id)
        
        return {
            "success": True,
            "archived_messages_count": archived_count,
            "total_messages": message_count,
            "summary": result.get("summary", ""),
            "session_id": session_id
        }
    except Exception as e:
        logging.error(f"结束并归档会话失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"结束并归档会话失败: {str(e)}")

# 导入会话路由
try:
    from app.api.session_routes import router as session_router
    # 将前缀设置为/api/sessions以符合前端调用
    app.include_router(session_router, prefix="/api")
    logging.info("会话管理路由已加载")
except ImportError as e:
    logging.error(f"无法导入会话管理路由: {str(e)}")

# 导入记忆模块路由
try:
    from app.api.endpoints.memory import router as memory_router
    app.include_router(memory_router, prefix="/api")
    logging.info("记忆管理路由已加载")
except ImportError as e:
    logging.error(f"无法导入记忆管理路由: {str(e)}")

# 配置静态文件服务
try:
    # 获取项目根目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    docs_dir = os.path.join(project_root, "docs")
    
    if os.path.exists(docs_dir):
        # 挂载静态文件目录
        app.mount("/docs", StaticFiles(directory=docs_dir), name="docs")
        logging.info(f"静态文件目录已挂载: {docs_dir}")
    else:
        logging.error(f"静态文件目录不存在: {docs_dir}")
except Exception as e:
    logging.error(f"无法挂载静态文件目录: {str(e)}")

# 应用启动事件
@app.on_event("startup")
async def startup_event():
    """应用启动时执行的操作"""
    try:
        # 创建必要的目录
        os.makedirs("app/data", exist_ok=True)
        
        # 初始化嵌入服务
        try:
            await embedding_service.initialize()
            logging.info("嵌入服务初始化完成")
        except Exception as e:
            logging.error(f"嵌入服务初始化失败: {str(e)}")
            
        # 初始化记忆模块
        try:
            from app.memory.memory_manager import get_memory_manager
            # 获取并初始化记忆管理器
            memory_manager = await get_memory_manager()
            logging.info("记忆管理器初始化成功")
        except Exception as e:
            logging.error(f"记忆管理器初始化失败: {str(e)}")
            
    except Exception as e:
        logging.error(f"应用启动事件执行失败: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def root():
    """重定向到演示页面"""
    return RedirectResponse(url="/docs/stream_example.html")

@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "ok"} 