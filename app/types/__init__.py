"""
类型定义模块 - 包含应用中使用的公共类型定义
"""

from typing import Dict, List, Optional, Any, Union, TypeVar, Generic, TypedDict

# 用户相关类型
UserId = str
UserDict = Dict[str, Any]
UserList = List[UserDict]

# 会话相关类型
SessionId = str
SessionDict = Dict[str, Any]
SessionList = List[SessionDict]

# 消息相关类型
MessageId = str
MessageDict = Dict[str, Any]
MessageList = List[MessageDict]

# 角色相关类型
RoleId = str
RoleDict = Dict[str, Any]
RoleList = List[RoleDict]

# 添加JsonObject类型
class JsonObject(TypedDict, total=False):
    """
    JSON对象类型，表示可序列化为JSON的对象
    用于描述通过API传递的数据结构
    """
    pass

# 通用响应类型
class ApiResponse:
    """API响应基类"""
    success: bool
    message: str
    data: Optional[Any] = None
    
    def __init__(self, success: bool, message: str, data: Any = None):
        self.success = success
        self.message = message
        self.data = data
        
    def dict(self) -> Dict[str, Any]:
        """将响应转换为字典"""
        return {
            "success": self.success,
            "message": self.message,
            "data": self.data
        }

# 类型变量
T = TypeVar('T')

# 分页结果
class PagedResult(Generic[T]):
    """分页结果"""
    items: List[T]
    total: int
    page: int
    size: int
    
    def __init__(self, items: List[T], total: int, page: int, size: int):
        self.items = items
        self.total = total
        self.page = page
        self.size = size
        
    def dict(self) -> Dict[str, Any]:
        """将分页结果转换为字典"""
        return {
            "items": self.items,
            "total": self.total,
            "page": self.page,
            "size": self.size
        } 