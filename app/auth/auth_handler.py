import os, jwt, time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# 创建日志记录器
logger = logging.getLogger(__name__)

# JWT配置
JWT_SECRET = os.getenv("JWT_SECRET", "demo_secret_key")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXPIRES_SECONDS = int(os.getenv("JWT_EXPIRES_SECONDS", "3600"))

# 创建安全依赖
security = HTTPBearer(auto_error=False)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(seconds=JWT_EXPIRES_SECONDS)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return encoded_jwt

def decode_token(token: str) -> Dict:
    """
    解码并验证JWT令牌
    
    Args:
        token: JWT令牌字符串
        
    Returns:
        解码后的令牌数据字典
        
    Raises:
        jwt.PyJWTError: 当令牌无效或过期时
    """
    try:
        # 打印收到的原始token
        logger.info(f"auth_handler收到的token: {token[:10]}...{token[-10:] if len(token) > 20 else token}")
        
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        
        # 打印完整的解码后用户信息
        logger.info(f"============用户信息开始============")
        logger.info(f"auth_handler解码后的完整用户信息: {json.dumps(payload, ensure_ascii=False, indent=2)}")
        
        # 打印常见关键字段
        if "sub" in payload:
            logger.info(f"用户ID (sub): {payload['sub']}")
        if "username" in payload:
            logger.info(f"用户名 (username): {payload['username']}")
        if "name" in payload:
            logger.info(f"显示名称 (name): {payload['name']}")
        if "role" in payload:
            logger.info(f"角色 (role): {payload['role']}")
        if "exp" in payload:
            exp_time = datetime.fromtimestamp(payload["exp"])
            logger.info(f"过期时间 (exp): {exp_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
        # 打印其他字段
        for key, value in payload.items():
            if key not in ["sub", "username", "name", "role", "exp"]:
                logger.info(f"其他字段 - {key}: {value}")
        
        logger.info(f"============用户信息结束============")
        
        return payload
    except jwt.ExpiredSignatureError:
        # 令牌已过期
        logger.error("令牌已过期")
        raise jwt.PyJWTError("令牌已过期")
    except jwt.InvalidTokenError as e:
        # 令牌无效
        logger.error(f"无效的令牌: {str(e)}")
        raise jwt.PyJWTError("无效的令牌")
    except Exception as e:
        # 其他异常
        logger.error(f"解码令牌时发生未知错误: {str(e)}")
        raise jwt.PyJWTError(f"令牌处理错误: {str(e)}")

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    从HTTP请求中提取并验证JWT令牌，返回当前用户信息
    
    Args:
        credentials: 从HTTP请求头中提取的Authorization凭证
        
    Returns:
        当前用户信息字典
        
    Raises:
        HTTPException: 当身份验证失败时
    """
    if credentials is None:
        # 为开发环境提供一个默认用户，生产环境应该删除这部分
        return {
            "user_id": "default_user_id",
            "username": "default_user",
            "role": "user",
            "is_temporary": True
        }
    
    try:
        token = credentials.credentials
        payload = decode_token(token)
        
        # 检查令牌是否包含必要信息
        if "sub" not in payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="无效的身份认证凭据",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return {
            "user_id": payload.get("sub"),
            "username": payload.get("username", ""),
            "role": payload.get("role", "user"),
            "exp": payload.get("exp", 0)
        }
        
    except jwt.PyJWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"身份认证失败: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )

async def get_current_user_optional(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    从HTTP请求中提取并验证JWT令牌，返回当前用户信息，如果验证失败则返回None
    
    Args:
        credentials: 从HTTP请求头中提取的Authorization凭证
        
    Returns:
        当前用户信息字典，或者在验证失败时返回None
    """
    if credentials is None:
        # 没有提供凭证，返回开发环境的默认用户
        return {
            "user_id": "default_user_id",
            "username": "default_user",
            "role": "user",
            "is_temporary": True
        }
    
    try:
        token = credentials.credentials
        payload = decode_token(token)
        
        # 检查令牌是否包含必要信息
        if "sub" not in payload:
            return None
        
        return {
            "user_id": payload.get("sub"),
            "username": payload.get("username", ""),
            "role": payload.get("role", "user"),
            "exp": payload.get("exp", 0)
        }
        
    except jwt.PyJWTError:
        # 令牌验证失败，返回None
        return None

# 为兼容性提供别名
get_current_user_or_none = get_current_user_optional
