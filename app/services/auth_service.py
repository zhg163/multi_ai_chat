"""
认证服务模块

提供用户认证、授权相关功能
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# 配置日志
logger = logging.getLogger(__name__)

# 从环境变量获取密钥，如果不存在则使用默认值
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-for-development")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24小时

# 安全依赖
security = HTTPBearer()

class AuthService:
    """认证服务类"""
    
    @staticmethod
    def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """
        创建访问令牌
        
        Args:
            data: 要编码到令牌中的数据
            expires_delta: 令牌有效期
            
        Returns:
            JWT令牌字符串
        """
        to_encode = data.copy()
        
        # 设置过期时间
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
            
        to_encode.update({"exp": expire})
        
        # 创建JWT令牌
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    
    @staticmethod
    def verify_token(token: str) -> Dict[str, Any]:
        """
        验证令牌
        
        Args:
            token: JWT令牌
            
        Returns:
            解码后的令牌数据
            
        Raises:
            HTTPException: 如果令牌无效或已过期
        """
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning(f"令牌已过期: {token[:10]}...")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="令牌已过期",
                headers={"WWW-Authenticate": "Bearer"},
            )
        except jwt.InvalidTokenError:
            logger.warning(f"无效的令牌: {token[:10]}...")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="无效的令牌",
                headers={"WWW-Authenticate": "Bearer"},
            )

# 依赖函数：获取当前用户
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """
    获取当前用户
    
    Args:
        credentials: HTTP授权凭证
        
    Returns:
        当前用户数据
        
    Raises:
        HTTPException: 如果授权失败
    """
    try:
        token = credentials.credentials
        user_data = AuthService.verify_token(token)
        return user_data
    except Exception as e:
        logger.error(f"获取当前用户失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的认证凭据",
            headers={"WWW-Authenticate": "Bearer"},
        )

# 认证依赖
async def get_current_active_user(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """
    获取当前激活用户
    
    Args:
        current_user: 当前用户信息
        
    Returns:
        当前激活用户信息
        
    Raises:
        HTTPException: 如果用户未激活
    """
    if current_user.get("disabled"):
        raise HTTPException(status_code=400, detail="用户未激活")
    return current_user

# 快速模拟JWT令牌的功能，用于测试
def get_demo_token() -> str:
    """
    获取演示用JWT令牌
    
    Returns:
        JWT令牌字符串
    """
    data = {"sub": "demo", "name": "演示用户", "admin": True}
    return AuthService.create_access_token(data) 