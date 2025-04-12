"""
JWT Bearer认证

处理JWT令牌认证
"""

from fastapi import Request, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import logging

from .auth_handler import decode_token

logger = logging.getLogger(__name__)

class JWTBearer(HTTPBearer):
    """JWT Bearer认证处理器"""
    
    def __init__(self, auto_error: bool = True):
        super(JWTBearer, self).__init__(auto_error=auto_error)
    
    async def __call__(self, request: Request):
        """验证认证信息"""
        # 演示模式：允许使用demo_token直接通过
        if request.headers.get("Authorization") == "Bearer demo_token":
            logger.warning("Using demo token for authentication")
            return {"id": "demo_user_id", "username": "demo_user"}
            
        credentials: HTTPAuthorizationCredentials = await super(JWTBearer, self).__call__(request)
        
        if credentials:
            if not credentials.scheme == "Bearer":
                raise HTTPException(status_code=403, detail="Invalid authentication scheme.")
            
            try:
                payload = decode_token(credentials.credentials)
                return payload
            except Exception as e:
                logger.error(f"Token validation error: {str(e)}")
                raise HTTPException(status_code=403, detail=f"Invalid token or expired token: {str(e)}")
        else:
            raise HTTPException(status_code=403, detail="Invalid authorization code.") 