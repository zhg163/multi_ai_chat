"""
多AI聊天系统 - 主应用模块

包含FastAPI主应用实例和路由配置
"""

import os
import logging
from dotenv import load_dotenv
from datetime import datetime
import time
from typing import Dict
from fastapi import FastAPI, HTTPException, Request, Response, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    # ,
    # filename='app.log',
    # filemode='a'
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

from fastapi import FastAPI, HTTPException, Request, Response
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

# 导入角色路由
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
    
    # 添加LLM修复路由
    try:
        from app.api import llm_fix
        app.include_router(llm_fix.router, tags=["llm_fix"])
        logging.info("LLM修复路由已加载")
    except ImportError as fix_error:
        logging.error(f"无法导入LLM修复路由: {str(fix_error)}")
    
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

# 导入会话路由
try:
    from app.api.session_routes import router as session_router
    app.include_router(session_router)
    logging.info("会话管理路由已加载")
except ImportError as e:
    logging.error(f"无法导入会话管理路由: {str(e)}")

# 导入自定义会话路由
try:
    from app.api.custom_session_routes import router as custom_session_router
    app.include_router(custom_session_router)
    logging.info("自定义会话路由已加载")
except ImportError as e:
    logging.error(f"无法导入自定义会话路由: {str(e)}")
    
# 导入标准数据库模块
from app.database.mongodb import get_database, get_collection

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
        # 使用标准方法获取MongoDB连接
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
        logging.error(f"获取用户列表失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取用户列表失败: {str(e)}")

# 添加用户选择端点
@app.post("/api/users/select-user", tags=["users"])
async def select_user_login(
    request: Request,
    user_data: Dict = Body(...)
):
    """选择用户并登录"""
    try:
        user_id = user_data.get("user_id")
        session_id = user_data.get("session_id")
        
        logging.info(f"用户选择请求，user_id={user_id}, session_id={session_id}")
        
        if not user_id:
            raise HTTPException(status_code=400, detail="缺少用户ID")
        
        # 验证用户ID格式    
        if user_id == 'undefined' or user_id == 'null':
            raise HTTPException(status_code=400, detail="无效的用户ID格式：不能为'undefined'或'null'")
            
        # 验证用户是否存在
        from app.database.connection import get_database
        from bson.objectid import ObjectId, InvalidId
        
        db = await get_database()
        if db is None:
            raise HTTPException(status_code=500, detail="数据库连接失败")
            
        try:
            # 尝试转换为ObjectId
            obj_id = ObjectId(user_id)
            logging.info(f"尝试查找用户，ObjectId={obj_id}")
            
            # 查询用户
            user = await db.users.find_one({"_id": obj_id})
            logging.info(f"用户查询结果: {user is not None}")
        except InvalidId as e:
            logging.error(f"无效的ObjectId格式: {user_id}, 错误: {str(e)}")
            raise HTTPException(status_code=400, detail=f"无效的用户ID格式: {str(e)}")
        except Exception as e:
            logging.error(f"查询用户时出错: {str(e)}", exc_info=True)
            raise HTTPException(status_code=400, detail=f"无效的用户ID或查询失败: {str(e)}")
            
        if not user:
            raise HTTPException(status_code=404, detail="用户不存在")
            
        # 生成访问令牌
        from app.auth.auth_handler import create_access_token
        from datetime import timedelta
        
        expires_delta = timedelta(days=7)
        access_token = create_access_token(
            data={"sub": str(user["_id"])},
            expires_delta=expires_delta
        )
        
        # 如果提供了会话ID，更新会话所有者
        new_session_id = None
        if session_id:
            try:
                # 获取原始匿名会话的消息
                from app.memory.memory_manager import get_memory_manager
                
                memory_manager = await get_memory_manager()
                messages = memory_manager.short_term.get_session_messages(session_id, "anonymous_user")
                
                if messages:
                    # 创建一个新会话，归属于选中的用户
                    new_session_id = await memory_manager.start_new_session(str(user["_id"]))
                    
                    # 将消息迁移到新会话
                    for msg in reversed(messages):  # 从旧到新迁移消息
                        await memory_manager.add_message(
                            new_session_id, 
                            str(user["_id"]), 
                            msg.get("role", "user"), 
                            msg.get("content", ""),
                            msg.get("roleid"),
                            msg.get("message_id")
                        )
            except Exception as e:
                logging.error(f"会话迁移失败: {str(e)}", exc_info=True)
                # 即使迁移失败也继续处理，允许用户选择
        
        # 返回用户信息和令牌
        response_data = {
            "success": True, 
            "access_token": access_token, 
            "token_type": "bearer",
            "user_id": str(user["_id"]),
            "name": user.get("name", "")
        }
        
        # 如果有新会话ID，添加到响应中
        if new_session_id:
            response_data["session_id"] = new_session_id
        
        logging.info(f"用户选择成功: {response_data['user_id']}")
        return response_data
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"用户选择失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"用户选择失败: {str(e)}")

# 导入记忆模块路由
try:
    from app.api.endpoints.memory import router as memory_router
    app.include_router(memory_router)
    logging.info("记忆管理路由已加载")
except ImportError as e:
    logging.error(f"无法导入记忆管理路由: {str(e)}")

# 配置静态文件服务
try:
    # 获取项目根目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    static_dir = os.path.join(current_dir, "static")
    
    if os.path.exists(static_dir):
        # 挂载静态文件目录
        app.mount("/static", StaticFiles(directory=static_dir), name="static")
        logging.info(f"静态文件目录已挂载: {static_dir}")
    else:
        logging.error(f"静态文件目录不存在: {static_dir}")
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
        
        # 加载LLM修复模块
        try:
            from app.api import llm_fix
            app.include_router(llm_fix.router)
            logging.info("LLM修复路由已加载(启动时)")
            
            # 尝试自动修复LLM服务的generate方法
            try:
                import httpx
                async with httpx.AsyncClient() as client:
                    fix_url = "http://localhost:8000/api/llm/fix_generate_method"
                    response = await client.get(fix_url)
                    if response.status_code == 200:
                        logging.info(f"自动修复LLM generate方法成功: {response.json()}")
                    else:
                        logging.warning(f"自动修复LLM generate方法失败: {response.status_code}, {response.text}")
            except Exception as auto_fix_error:
                logging.error(f"尝试自动修复LLM generate方法时出错: {str(auto_fix_error)}")
        except Exception as e:
            logging.error(f"加载LLM修复模块失败: {str(e)}")
            
    except Exception as e:
        logging.error(f"应用启动事件执行失败: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def root():
    """重定向到演示页面"""
    return RedirectResponse(url="/static/stream_example.html")

@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "ok"}

@app.get("/api/routes/test")
async def test_routes():
    """测试所有API路由的可用性"""
    # 检查已注册的路由
    routes = []
    for route in app.routes:
        if hasattr(route, "path") and "/api/" in route.path:
            method = "GET"
            if hasattr(route, "methods"):
                method = list(route.methods)[0] if route.methods else "GET"
            routes.append({"path": route.path, "method": method})
    
    return {
        "status": "ok",
        "routes": routes,
        "total_routes": len(routes),
        "timestamp": datetime.now().isoformat()
    }

# 添加一个HTTP请求日志记录中间件，帮助调试路由问题
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """记录每个HTTP请求的详细信息，帮助调试404错误"""
    start_time = time.time()
    path = request.url.path
    method = request.method
    
    # 记录请求详情
    logging.info(f"请求开始: {method} {path}")
    
    # 如果是API请求，记录更多详情
    if path.startswith("/api/"):
        # 记录注册路由信息
        routes_info = []
        for route in app.routes:
            if hasattr(route, "path"):
                route_methods = list(getattr(route, "methods", ["GET"]))
                if route_methods and method in route_methods:
                    routes_info.append(f"{route.path} [{','.join(route_methods)}]")
        
        if routes_info:
            logging.info(f"可能匹配的路由: {routes_info}")
    
    # 执行请求
    response = await call_next(request)
    
    # 计算处理时间并记录响应状态
    process_time = time.time() - start_time
    logging.info(f"请求完成: {method} {path} - 状态: {response.status_code} - 处理时间: {process_time:.4f}秒")
    
    # 如果是404错误，记录更多信息帮助调试
    if response.status_code == 404 and path.startswith("/api/"):
        logging.warning(f"404 NOT FOUND: {method} {path}")
        # 记录所有API路由，帮助判断是否有拼写错误
        api_routes = []
        for route in app.routes:
            if hasattr(route, "path") and "/api/" in route.path:
                api_routes.append(f"{route.path} [{','.join(getattr(route, 'methods', ['GET']))}]")
        logging.warning(f"全部API路由: {api_routes}")
    
    response.headers["X-Process-Time"] = str(process_time)
    return response 