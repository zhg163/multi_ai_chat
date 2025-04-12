from fastapi import APIRouter
from app.api.role_routes import router as role_router
from app.api.message_routes import router as message_router
from app.api.session_routes import router as session_router
from app.api.session_role_routes import router as session_role_router
from app.api.message_processing_routes import router as message_processing_router

router = APIRouter()

# 注册所有路由
router.include_router(role_router)
router.include_router(message_router)
router.include_router(session_router)
router.include_router(session_role_router)
router.include_router(message_processing_router)

# 导出所有路由
__all__ = ["router"] 