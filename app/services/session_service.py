from datetime import datetime
from typing import List, Dict, Optional, Any, Union
from bson import ObjectId
import logging
from pymongo.errors import PyMongoError

from app.models.session import Session, SessionStatus
from app.models.role import Role
# Remove top-level import to break circular dependency
# from app.services.role_service import RoleService
from ..database.mongodb import get_db

logger = logging.getLogger(__name__)

class SessionService:
    """会话管理服务，提供会话创建、配置和管理功能"""
    
    @staticmethod
    async def create_session(
        user_id: str,
        title: Optional[str] = None,
        description: Optional[str] = None,
        role_ids: Optional[List[str]] = None,
        settings: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        创建新会话
        
        参数:
            user_id: 用户ID
            title: 会话标题
            description: 会话描述
            role_ids: 角色ID列表
            settings: 会话设置
            
        返回:
            新创建的会话信息
        """
        try:
            # 验证角色是否存在且处于活跃状态
            if role_ids:
                # Import RoleService here to avoid circular dependency
                from app.services.role_service import RoleService
                
                valid_role_ids = []
                for role_id in role_ids:
                    role = await RoleService.get_role_by_id(role_id)
                    if role and role.get("active", False):
                        valid_role_ids.append(role_id)
                    
                role_ids = valid_role_ids
            
            # 创建会话
            session = await Session.create(
                user_id=user_id,
                title=title,
                description=description,
                role_ids=role_ids,
                settings=settings
            )
            
            return session
        except Exception as e:
            logger.error(f"创建会话失败: {e}")
            raise
    
    @staticmethod
    async def get_session(
        session_id: Union[str, ObjectId],
        user_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        获取会话信息
        
        参数:
            session_id: 会话ID
            user_id: 用户ID（用于权限验证）
            
        返回:
            会话信息，如果未找到则返回None
        """
        try:
            # 获取会话
            session = await Session.get_by_id(session_id, user_id)
            
            # 检查会话是否存在
            if not session:
                return None
                
            # 检查是否是会话所有者
            if str(session["user_id"]) != user_id:
                return None
                
            # 返回会话数据
            return session.to_dict()
        except Exception as e:
            logger.error(f"获取会话失败: {e}")
            raise
    
    @staticmethod
    async def list_sessions(
        user_id: str,
        limit: int = 20,
        offset: int = 0,
        status: str = "active"
    ) -> List[Dict[str, Any]]:
        """
        获取用户的会话列表
        
        参数:
            user_id: 用户ID
            limit: 最大返回数量
            offset: 跳过的记录数
            status: 会话状态过滤
            
        返回:
            会话列表
        """
        try:
            # 获取会话列表
            sessions = await Session.list_by_user(
                user_id=user_id,
                limit=limit,
                offset=offset,
                status=status
            )
            
            # 转换为字典列表
            return [session.to_dict() for session in sessions]
        except Exception as e:
            logger.error(f"获取会话列表失败: {e}")
            raise
    
    @staticmethod
    async def update_session(
        session_id: Union[str, ObjectId],
        user_id: str,
        update_data: Dict[str, Any]
    ) -> bool:
        """
        更新会话信息
        
        参数:
            session_id: 会话ID
            user_id: 用户ID
            update_data: 需要更新的字段
            
        返回:
            更新是否成功
        """
        try:
            # 检查会话是否存在
            session = await Session.get_by_id(session_id, user_id)
            if not session:
                raise ValueError(f"会话ID '{session_id}' 不存在或无权访问")
            
            # 移除不可直接更新的字段
            for field in ["_id", "user_id", "created_at", "updated_at", "status"]:
                if field in update_data:
                    del update_data[field]
            
            # 更新会话
            success = await session.update(update_data)
            
            return success
        except Exception as e:
            logger.error(f"更新会话失败: {e}")
            raise
    
    @staticmethod
    async def configure_session_roles(
        session_id: Union[str, ObjectId],
        user_id: str,
        role_ids: List[str]
    ) -> bool:
        """
        配置会话的角色列表
        
        参数:
            session_id: 会话ID
            user_id: 用户ID
            role_ids: 角色ID列表
            
        返回:
            配置是否成功
        """
        try:
            # 检查会话是否存在
            session = await Session.get_by_id(session_id, user_id)
            if not session:
                raise ValueError(f"会话ID '{session_id}' 不存在或无权访问")
            
            # 验证角色是否存在且处于活跃状态
            # Import RoleService here to avoid circular dependency
            from app.services.role_service import RoleService
            
            valid_role_ids = []
            for role_id in role_ids:
                role = await RoleService.get_role_by_id(role_id)
                if role and role.get("active", False):
                    valid_role_ids.append(role_id)
            
            if not valid_role_ids and role_ids:
                raise ValueError("提供的角色ID列表中没有有效的角色ID")
            
            # 更新会话角色
            success = await session.update_roles(valid_role_ids)
            
            return success
        except Exception as e:
            logger.error(f"配置会话角色失败: {e}")
            raise
    
    @staticmethod
    async def get_session_roles(
        session_id: Union[str, ObjectId],
        user_id: str
    ) -> List[Dict[str, Any]]:
        """
        获取会话分配的所有角色
        
        参数:
            session_id: 会话ID
            user_id: 用户ID
            
        返回:
            角色对象列表
        """
        try:
            # 检查会话是否存在
            session = await Session.get_by_id(session_id, user_id)
            if not session:
                raise ValueError(f"会话ID '{session_id}' 不存在或无权访问")
            
            # 获取会话中的角色ID
            role_ids = session.get("role_ids", [])
            if not role_ids:
                return []
            
            # Import RoleService here to avoid circular dependency
            from app.services.role_service import RoleService
            
            # 获取角色详细信息
            roles = []
            for role_id in role_ids:
                role = await RoleService.get_role_by_id(role_id)
                if role and role.get("active", False):
                    roles.append(role)
            
            return roles
        except Exception as e:
            logger.error(f"获取会话角色失败: {e}")
            raise
    
    @staticmethod
    async def add_session_role(
        session_id: Union[str, ObjectId],
        user_id: str,
        role_id: str
    ) -> bool:
        """
        向会话添加单个角色
        
        参数:
            session_id: 会话ID
            user_id: 用户ID
            role_id: 角色ID
            
        返回:
            操作是否成功
        """
        try:
            # 检查会话是否存在
            session = await Session.get_by_id(session_id, user_id)
            if not session:
                raise ValueError(f"会话ID '{session_id}' 不存在或无权访问")
            
            # Import RoleService here to avoid circular dependency
            from app.services.role_service import RoleService
            
            # 验证角色是否存在且处于活跃状态
            role = await RoleService.get_role_by_id(role_id)
            if not role or not role.get("active", False):
                raise ValueError(f"角色ID '{role_id}' 不存在或未激活")
            
            # 检查角色是否已经分配给会话
            session_role_ids = session.get("role_ids", [])
            if role_id in session_role_ids:
                # 角色已存在，视为成功
                return True
            
            # 添加角色
            success = await session.add_roles([role_id])
            
            return success
        except Exception as e:
            logger.error(f"添加角色到会话失败: {e}")
            raise
    
    @staticmethod
    async def remove_session_role(
        session_id: Union[str, ObjectId],
        user_id: str,
        role_id: str
    ) -> bool:
        """
        从会话中移除单个角色
        
        参数:
            session_id: 会话ID
            user_id: 用户ID
            role_id: 要移除的角色ID
            
        返回:
            操作是否成功
        """
        try:
            # 检查会话是否存在
            session = await Session.get_by_id(session_id, user_id)
            if not session:
                raise ValueError(f"会话ID '{session_id}' 不存在或无权访问")
            
            # Import RoleService here to avoid circular dependency
            from app.services.role_service import RoleService
            
            # 验证角色是否存在
            role = await RoleService.get_role_by_id(role_id)
            if not role:
                raise ValueError(f"角色ID '{role_id}' 不存在")
            
            # 检查角色是否已经分配给会话
            session_role_ids = session.get("role_ids", [])
            if role_id not in session_role_ids:
                # 角色不存在，视为成功
                return True
            
            # 移除角色
            success = await session.remove_roles([role_id])
            
            return success
        except Exception as e:
            logger.error(f"从会话移除角色失败: {e}")
            raise
    
    @staticmethod
    async def has_session_role(
        session_id: Union[str, ObjectId],
        user_id: str,
        role_id: str
    ) -> bool:
        """
        检查会话是否包含特定角色
        
        参数:
            session_id: 会话ID
            user_id: 用户ID
            role_id: 角色ID
            
        返回:
            如果会话包含该角色则返回True，否则返回False
        """
        try:
            # 检查会话是否存在
            session = await Session.get_by_id(session_id, user_id)
            if not session:
                raise ValueError(f"会话ID '{session_id}' 不存在或无权访问")
            
            # 检查角色是否已分配给会话
            session_role_ids = session.get("role_ids", [])
            role_id_strs = [str(rid) for rid in session_role_ids]
            
            return str(role_id) in role_id_strs
        except Exception as e:
            logger.error(f"检查会话角色失败: {e}")
            raise
    
    @staticmethod
    async def configure_session_settings(
        session_id: Union[str, ObjectId],
        user_id: str,
        settings: Dict[str, Any]
    ) -> bool:
        """
        配置会话设置
        
        参数:
            session_id: 会话ID
            user_id: 用户ID
            settings: 会话设置
            
        返回:
            配置是否成功
        """
        try:
            # 检查会话是否存在
            session = await Session.get_by_id(session_id, user_id)
            if not session:
                raise ValueError(f"会话ID '{session_id}' 不存在或无权访问")
            
            # 获取当前设置并更新
            current_settings = session.get("settings", {})
            
            # 合并设置，只更新提供的字段
            for key, value in settings.items():
                current_settings[key] = value
            
            # 更新会话设置
            success = await session.update_settings(current_settings)
            
            return success
        except Exception as e:
            logger.error(f"配置会话设置失败: {e}")
            raise
    
    @staticmethod
    async def archive_session(
        session_id: Union[str, ObjectId],
        user_id: str
    ) -> bool:
        """
        归档会话
        
        参数:
            session_id: 会话ID
            user_id: 用户ID
            
        返回:
            操作是否成功
        """
        try:
            # 检查会话是否存在
            session = await Session.get_by_id(session_id, user_id)
            if not session:
                raise ValueError(f"会话ID '{session_id}' 不存在或无权访问")
            
            # 检查会话是否已经是archived状态
            if session.get("status") == "archived":
                return True
            
            # 更新会话状态
            update_data = {"status": "archived"}
            success = await session.update(update_data)
            
            return success
        except Exception as e:
            logger.error(f"归档会话失败: {e}")
            raise
    
    @staticmethod
    async def restore_session(
        session_id: Union[str, ObjectId],
        user_id: str
    ) -> bool:
        """
        恢复已归档的会话
        
        参数:
            session_id: 会话ID
            user_id: 用户ID
            
        返回:
            操作是否成功
        """
        try:
            # 检查会话是否存在
            session = await Session.get_by_id(session_id, user_id)
            if not session:
                raise ValueError(f"会话ID '{session_id}' 不存在或无权访问")
            
            if session.get("status") != "archived":
                raise ValueError(f"会话ID '{session_id}' 不是已归档状态，无法恢复")
            
            # 更新会话状态
            update_data = {"status": "active"}
            success = await session.update(update_data)
            
            return success
        except Exception as e:
            logger.error(f"恢复会话失败: {e}")
            raise
    
    @staticmethod
    async def delete_session(
        session_id: Union[str, ObjectId],
        user_id: str
    ) -> bool:
        """
        删除会话（标记为已删除）
        
        参数:
            session_id: 会话ID
            user_id: 用户ID
            
        返回:
            操作是否成功
        """
        try:
            # 检查会话是否存在
            session = await Session.get_by_id(session_id, user_id)
            if not session:
                raise ValueError(f"会话ID '{session_id}' 不存在或无权访问")
            
            # 删除会话
            success = await session.delete()
            
            return success
        except Exception as e:
            logger.error(f"删除会话失败: {e}")
            raise
    
    @staticmethod
    async def permanently_delete_session(
        session_id: Union[str, ObjectId],
        user_id: str
    ) -> bool:
        """
        永久删除会话（从数据库彻底删除，不可恢复）
        
        参数:
            session_id: 会话ID
            user_id: 用户ID
            
        返回:
            操作是否成功
        """
        try:
            # 检查会话是否存在且状态为已删除
            session = await Session.get_by_id(session_id, user_id)
            if not session:
                raise ValueError(f"会话ID '{session_id}' 不存在或无权访问")
            
            if session.get("status") != "deleted":
                raise ValueError(f"只能永久删除已标记为删除的会话")
            
            # 永久删除会话
            success = await Session.permanently_delete(session_id, user_id)
            
            return success
        except Exception as e:
            logger.error(f"永久删除会话失败: {e}")
            raise
    
    @staticmethod
    async def clean_deleted_sessions(
        user_id: str,
        older_than_days: Optional[int] = None
    ) -> int:
        """
        清理用户的已删除会话
        
        参数:
            user_id: 用户ID
            older_than_days: 如果提供，只删除超过指定天数的已删除会话
            
        返回:
            删除的会话数量
        """
        try:
            # 永久删除所有标记为删除的会话
            count = await Session.permanently_delete_all_user_deleted(user_id, older_than_days)
            
            return count
        except Exception as e:
            logger.error(f"清理已删除会话失败: {e}")
            raise
    
    @staticmethod
    async def change_session_status(
        session_id: Union[str, ObjectId],
        user_id: str,
        new_status: str
    ) -> bool:
        """
        修改会话状态
        
        参数:
            session_id: 会话ID
            user_id: 用户ID
            new_status: 新的会话状态
            
        返回:
            操作是否成功
        """
        try:
            # 检查会话是否存在
            session = await Session.get_by_id(session_id, user_id)
            if not session:
                raise ValueError(f"会话ID '{session_id}' 不存在或无权访问")
            
            # 检查状态是否有效
            valid_statuses = [SessionStatus.ACTIVE, SessionStatus.ARCHIVED, SessionStatus.DELETED]
            if new_status not in valid_statuses:
                raise ValueError(f"无效的会话状态: {new_status}")
                
            # 检查当前状态
            current_status = session.get("status")
            
            # 如果状态相同，不需要更新
            if current_status == new_status:
                return True
                
            # 验证状态转换是否合法
            if current_status == SessionStatus.DELETED and new_status == SessionStatus.ACTIVE:
                raise ValueError("已删除的会话不能直接恢复为活跃状态，请先恢复到归档状态")
            
            # 更新会话状态
            update_data = {"status": new_status}
            
            # 如果是归档操作，记录归档时间
            if new_status == SessionStatus.ARCHIVED:
                update_data["archived_at"] = datetime.utcnow()
            
            success = await session.update(update_data)
            
            return success
        except Exception as e:
            logger.error(f"修改会话状态失败: {e}")
            raise
    
    @staticmethod
    async def search_sessions(
        user_id: str,
        query: Optional[str] = None,
        status: Optional[str] = None,
        role_id: Optional[str] = None,
        created_after: Optional[datetime] = None,
        created_before: Optional[datetime] = None,
        updated_after: Optional[datetime] = None,
        updated_before: Optional[datetime] = None,
        sort_by: str = "updated_at",
        sort_direction: int = -1,
        limit: int = 20,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        高级会话搜索
        
        参数:
            user_id: 用户ID
            query: 搜索关键词
            status: 会话状态
            role_id: 角色ID
            created_after: 创建时间下限
            created_before: 创建时间上限
            updated_after: 更新时间下限
            updated_before: 更新时间上限
            sort_by: 排序字段
            sort_direction: 排序方向
            limit: 最大返回数量
            offset: 跳过的记录数
            
        返回:
            包含会话列表和总数的字典
        """
        try:
            # 构建过滤条件
            filters = {}
            if status:
                filters["status"] = status
            if role_id:
                filters["role_ids"] = role_id
            if created_after:
                filters["created_after"] = created_after
            if created_before:
                filters["created_before"] = created_before
            if updated_after:
                filters["updated_after"] = updated_after
            if updated_before:
                filters["updated_before"] = updated_before
            
            # 验证排序字段
            valid_sort_fields = ["created_at", "updated_at", "last_message_at", "title"]
            if sort_by not in valid_sort_fields:
                sort_by = "updated_at"
            
            # 验证排序方向
            if sort_direction not in [1, -1]:
                sort_direction = -1
            
            # 执行搜索
            sessions, total = await Session.search_sessions(
                user_id=user_id,
                query=query,
                filters=filters,
                sort_by=sort_by,
                sort_direction=sort_direction,
                limit=limit,
                offset=offset
            )
            
            # 格式化会话数据
            formatted_sessions = []
            for session in sessions:
                session_dict = {
                    "id": str(session["_id"]),
                    "title": session["title"],
                    "description": session["description"],
                    "status": session["status"],
                    "role_ids": [str(role_id) for role_id in session["role_ids"]],
                    "settings": session["settings"],
                    "created_at": session["created_at"].isoformat(),
                    "updated_at": session["updated_at"].isoformat(),
                    "last_message_at": session["last_message_at"].isoformat()
                }
                formatted_sessions.append(session_dict)
            
            return {
                "sessions": formatted_sessions,
                "total": total,
                "limit": limit,
                "offset": offset
            }
        except Exception as e:
            logger.error(f"搜索会话失败: {e}")
            raise
    
    @staticmethod
    async def batch_remove_session_roles(
        session_id: Union[str, ObjectId],
        user_id: str,
        role_ids: List[str]
    ) -> bool:
        """
        批量从会话中移除角色
        
        参数:
            session_id: 会话ID
            user_id: 用户ID
            role_ids: 要移除的角色ID列表
            
        返回:
            操作是否成功
        """
        if not role_ids:
            return True
            
        try:
            # 检查会话是否存在
            session = await Session.get_by_id(session_id, user_id)
            if not session:
                raise ValueError(f"会话ID '{session_id}' 不存在或无权访问")
            
            # Import RoleService here to avoid circular dependency
            from app.services.role_service import RoleService
            
            # 验证所有角色是否存在
            for role_id in role_ids:
                role = await RoleService.get_role_by_id(role_id)
                if not role:
                    raise ValueError(f"角色ID '{role_id}' 不存在")
            
            # 移除角色
            success = await session.remove_roles(role_ids)
            
            return success
        except Exception as e:
            logger.error(f"批量从会话移除角色失败: {e}")
            raise
    
    @staticmethod
    async def list_session_roles(
        session_id: Union[str, ObjectId],
        user_id: str
    ) -> List[dict]:
        """
        获取会话关联的所有角色
        
        参数:
            session_id: 会话ID
            user_id: 用户ID
            
        返回:
            角色列表，每个角色包含角色的详细信息
        """
        try:
            # 检查会话是否存在
            session = await Session.get_by_id(session_id, user_id)
            if not session:
                raise ValueError(f"会话ID '{session_id}' 不存在或无权访问")
            
            # Import RoleService here to avoid circular dependency
            from app.services.role_service import RoleService
            
            # 获取会话中的角色ID
            role_ids = session.role_ids if hasattr(session, 'role_ids') else []
            
            # 如果没有角色，返回空列表
            if not role_ids:
                return []
            
            # 获取角色详细信息
            roles = []
            for role_id in role_ids:
                role = await RoleService.get_role_by_id(role_id)
                if role:
                    roles.append(role.dict())
            
            return roles
        except Exception as e:
            logger.error(f"获取会话角色失败: {e}")
            raise 