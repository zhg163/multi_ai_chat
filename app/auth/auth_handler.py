"""
JWT认证处理

提供JWT令牌编码和解码功能
"""

import os
import time
import jwt
from typing import Dict, Optional
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.security.utils import get_authorization_scheme_param

# JWT配置
JWT_SECRET = os.getenv("JWT_SECRET", "demo_secret_key")  # 实际生产中应使用环境变量
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXPIRES_SECONDS = int(os.getenv("JWT_EXPIRES_SECONDS", "3600"))  # 1小时

# 认证处理器
jwt_bearer = HTTPBearer()

# 可选认证处理器
class OptionalHTTPBearer(HTTPBearer):
    def __init__(self, auto_error: bool = False):
        super(OptionalHTTPBearer, self).__init__(auto_error=auto_error)
        
    async def __call__(self, request: Request) -> Optional[HTTPAuthorizationCredentials]:
        authorization = request.headers.get("Authorization")
        scheme, credentials = get_authorization_scheme_param(authorization)
        
        if not authorization or scheme.lower() != "bearer":
            if self.auto_error:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Not authenticated",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            else:
                return None
                
        return HTTPAuthorizationCredentials(scheme=scheme, credentials=credentials)

# 创建可选认证处理器实例
optional_jwt_bearer = OptionalHTTPBearer(auto_error=False)

def sign_token(user_id: str, username: str) -> str:
    """
    生成JWT令牌
    
    Args:
        user_id: 用户ID
        username: 用户名
        
    Returns:
        JWT令牌
    """
    payload = {
        "user_id": user_id,
        "username": username,
        "exp": time.time() + JWT_EXPIRES_SECONDS,
        "iat": time.time()
    }
    
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def decode_token(token: str) -> Dict:
    """
    解码JWT令牌
    
    Args:
        token: JWT令牌
        
    Returns:
        解码后的数据
    """
    try:
        decoded_token = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        if decoded_token["exp"] < time.time():
            raise HTTPException(status_code=401, detail="Token expired")
        return {
            "id": decoded_token["user_id"], 
            "username": decoded_token["username"]
        }
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

async def get_current_user(token=Depends(jwt_bearer)):
    """
    获取当前用户
    
    Args:
        token: JWT令牌（通过依赖注入获取）
        
    Returns:
        当前用户信息
    """
    # 演示模式
    if isinstance(token, dict) and token.get("id") == "demo_user_id":
        return token
        
    # 正常模式
    try:
        payload = decode_token(token.credentials)
        return payload
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Could not validate credentials: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )

async def get_current_user_optional(token=Depends(optional_jwt_bearer)):
    """
    获取当前用户（可选认证）
    
    与get_current_user不同，认证失败时不会抛出异常，而是返回默认匿名用户或None
    
    Args:
        token: JWT令牌（通过依赖注入获取，可选）
        
    Returns:
        当前用户信息，认证失败时返回默认用户或None
    """
    if token is None:
        # 返回默认匿名用户
        return {
            "id": "anonymous_user",
            "username": "anonymous",
            "is_anonymous": True
        }
        
    # 尝试解析令牌
    try:
        payload = decode_token(token.credentials)
        return payload
    except Exception as e:
        # 出错时，记录日志但返回匿名用户
        print(f"认证失败，但继续处理: {str(e)}")
        return {
            "id": "anonymous_user",
            "username": "anonymous",
            "is_anonymous": True
        } 