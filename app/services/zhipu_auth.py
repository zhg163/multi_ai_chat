"""
智谱AI认证辅助模块

为智谱AI API提供JWT令牌生成功能
"""

import json
import time
import jwt
import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

def generate_zhipu_token(api_key: str, exp_seconds: int = 3600) -> str:
    """
    生成智谱AI API的JWT令牌
    
    Args:
        api_key: 智谱AI API密钥，格式为"id.secret"
        exp_seconds: 令牌有效期（秒），默认3600秒（1小时）
        
    Returns:
        生成的JWT令牌
    """
    try:
        # 智谱AI的API密钥格式为"id.secret"
        key_parts = api_key.split('.')
        if len(key_parts) != 2:
            raise ValueError("API密钥格式不正确，应为'id.secret'")
        
        api_id, api_secret = key_parts
        
        # 准备JWT负载
        payload = {
            "api_key": api_id,
            "exp": int(time.time()) + exp_seconds,  # 过期时间
            "timestamp": int(time.time()),          # 当前时间戳
        }
        
        # 生成JWT令牌
        token = jwt.encode(
            payload=payload,
            key=api_secret,
            algorithm="HS256"
        )
        
        return token
    
    except Exception as e:
        logger.error(f"生成智谱AI令牌时出错: {str(e)}")
        raise

def parse_zhipu_api_key(api_key: str) -> Tuple[str, str]:
    """
    解析智谱AI API密钥
    
    Args:
        api_key: 智谱AI API密钥，格式为"id.secret"
        
    Returns:
        (api_id, api_secret)元组
    """
    key_parts = api_key.split('.')
    if len(key_parts) != 2:
        raise ValueError("API密钥格式不正确，应为'id.secret'")
    
    return key_parts[0], key_parts[1]

def get_zhipu_auth_headers(api_key: str) -> Dict[str, str]:
    """
    获取智谱AI API的认证头
    
    Args:
        api_key: 智谱AI API密钥，格式为"id.secret"
        
    Returns:
        包含Authorization头的字典
    """
    token = generate_zhipu_token(api_key)
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    } 