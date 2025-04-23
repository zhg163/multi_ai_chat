"""
API路由模块 - 导出所有路由模块

包含API路由的导出声明，方便在主应用中导入
"""

from fastapi import APIRouter

# 导出各个路由模块
try:
    from app.api.simple_routes import router as simple_router
except ImportError:
    simple_router = None

try:
    from app.api.llm_routes import router as llm_router
except ImportError:
    llm_router = None

try:
    from app.api.rag_chat_routes import router as rag_chat_router
except ImportError:
    rag_chat_router = None

try:
    from app.api.message_routes import router as message_router
except ImportError:
    message_router = None

try:
    from app.api.message_processing_routes import router as message_processing_router
except ImportError:
    message_processing_router = None

try:
    from app.api.session_routes import router as session_router
except ImportError:
    session_router = None

try:
    from app.api.role_routes import router as role_router
except ImportError:
    role_router = None

# 不再导入内存路由模块，功能已整合至其他路由

router = APIRouter()

# 注册所有路由
if simple_router:
    router.include_router(simple_router)

if llm_router:
    router.include_router(llm_router)

if rag_chat_router:
    router.include_router(rag_chat_router)

if message_router:
    router.include_router(message_router)

if message_processing_router:
    router.include_router(message_processing_router)

if session_router:
    router.include_router(session_router)

if role_router:
    router.include_router(role_router)
    # 确保角色匹配API可用
    print("角色管理路由已注册，包含角色匹配API")

# 不再注册内存路由，功能已整合至其他路由

# 导出所有路由
__all__ = ["router"] 