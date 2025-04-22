"""
角色工具模块 - 提供处理会话角色数据的工具函数
"""

import json
import logging
import uuid
from typing import Dict, Any, Optional, List, Union

# 配置日志记录器
logger = logging.getLogger(__name__)

def parse_role_from_session(session_data: Dict[str, Any], role_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    从会话数据中解析角色信息
    
    Args:
        session_data: 会话数据字典
        role_id: 角色ID（可选）
        
    Returns:
        Optional[Dict[str, Any]]: 角色信息字典，如果找不到则返回None
    """
    # 获取角色数据
    roles_data = session_data.get("roles", [])
    roles = []
    
    # 解析角色数据（可能是字符串或列表）
    if isinstance(roles_data, str):
        try:
            roles = json.loads(roles_data)
        except json.JSONDecodeError:
            logger.error(f"角色数据解析失败: {roles_data[:100]}...")
            return None
    elif isinstance(roles_data, list):
        roles = roles_data
    else:
        logger.error(f"无效的角色数据类型: {type(roles_data)}")
        return None
    
    # 如果未指定角色ID，返回第一个角色
    if not role_id and roles:
        return roles[0]
    
    # 查找指定ID的角色
    for role in roles:
        if role.get("role_id") == role_id:
            return role
    
    # 找不到指定角色，返回第一个角色（如果有）
    if not role_id and roles:
        return roles[0]
    
    # 没有找到任何角色
    return None

def get_role_prompt(role_data: Dict[str, Any]) -> str:
    """
    获取角色的提示信息
    
    Args:
        role_data: 角色数据字典
        
    Returns:
        str: 角色提示信息
    """
    # 尝试获取系统提示
    system_prompt = role_data.get("system_prompt", "")
    if system_prompt:
        return system_prompt
    
    # 尝试获取普通提示
    prompt = role_data.get("prompt", "")
    if prompt:
        return prompt
    
    # 如果两者都没有，返回角色名称
    return get_role_name(role_data)

def get_role_name(role_data: Dict[str, Any]) -> str:
    """
    获取角色名称
    
    Args:
        role_data: 角色数据字典
        
    Returns:
        str: 角色名称
    """
    return role_data.get("name", "未知角色")

def is_role_expired(role_data: Dict[str, Any], max_usage: int = 100) -> bool:
    """
    检查角色是否已过期（超过使用次数限制）
    
    Args:
        role_data: 角色数据字典
        max_usage: 最大使用次数，默认为100
        
    Returns:
        bool: 是否已过期
    """
    # 获取角色设置的最大使用次数（如果有）
    role_max_usage = role_data.get("max_usage", max_usage)
    
    # 获取当前使用次数
    usage_count = role_data.get("usage_count", 0)
    
    # 检查是否超过限制
    # 如果role_max_usage为0或负数，表示不限制使用次数
    if role_max_usage <= 0:
        return False
    
    return usage_count >= role_max_usage

def normalize_role_data(role_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    标准化角色数据，确保所有必要字段都存在
    
    Args:
        role_data: 原始角色数据字典
        
    Returns:
        Dict[str, Any]: 标准化后的角色数据字典
    """
    # 创建新的字典，避免修改原始数据
    normalized = role_data.copy()
    
    # 确保角色ID存在
    if "role_id" not in normalized or not normalized["role_id"]:
        normalized["role_id"] = str(uuid.uuid4())
    
    # 确保其他必要字段存在
    if "name" not in normalized:
        normalized["name"] = "未命名角色"
    
    if "usage_count" not in normalized:
        normalized["usage_count"] = 0
    
    if "max_usage" not in normalized:
        normalized["max_usage"] = 0  # 0表示不限制
    
    # 确保prompt字段存在
    if "prompt" not in normalized and "system_prompt" not in normalized:
        normalized["prompt"] = ""
    
    return normalized

async def normalize_role_data_async(role_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    标准化角色数据的异步版本，功能与同步版本相同
    
    Args:
        role_data: 原始角色数据字典
        
    Returns:
        Dict[str, Any]: 标准化后的角色数据字典
    """
    return normalize_role_data(role_data)