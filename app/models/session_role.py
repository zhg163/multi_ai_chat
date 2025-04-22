"""
会话角色模型 - 定义SessionRole类用于角色管理
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from datetime import datetime
from uuid import uuid4


class SessionRole(BaseModel):
    """会话角色模型类"""
    
    role_id: str = Field(default_factory=lambda: str(uuid4()))
    role_name: str
    description: Optional[str] = None
    system_prompt: str
    role_type: str = "custom"
    personality: Optional[str] = None
    speech_style: Optional[str] = None
    temperature: float = 0.7
    usage_count: int = 0
    max_usage: Optional[int] = None
    created_at: datetime = Field(default_factory=datetime.now)
    modified_at: datetime = Field(default_factory=datetime.now)
    is_active: bool = True
    is_default: bool = False
    keywords: List[str] = []
    metadata: Dict[str, Any] = {}
    
    class Config:
        from_attributes = True
        
    def to_dict(self) -> Dict[str, Any]:
        """将角色转换为字典"""
        return self.model_dump()
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionRole":
        """从字典创建角色实例"""
        if "created_at" in data and isinstance(data["created_at"], str):
            try:
                data["created_at"] = datetime.fromisoformat(data["created_at"])
            except ValueError:
                data["created_at"] = datetime.now()
                
        if "modified_at" in data and isinstance(data["modified_at"], str):
            try:
                data["modified_at"] = datetime.fromisoformat(data["modified_at"])
            except ValueError:
                data["modified_at"] = datetime.now()
        
        return cls(**data) 