from datetime import datetime
from typing import List, Dict, Optional, Any, Union, Tuple
from bson import ObjectId
import re
import jieba
import time
import uuid
import asyncio
import logging

from app.models.role import Role

logger = logging.getLogger(__name__)

class RoleService:
    """角色管理服务，提供角色CRUD操作"""
    
    @staticmethod
    async def create_role(
        name: str,
        description: str,
        personality: str,
        speech_style: str,
        keywords: List[str] = None,
        temperature: float = 0.7,
        prompt_template: str = None,
        system_prompt: str = None
    ) -> Dict[str, Any]:
        """
        创建新角色
        
        参数:
            name: 角色名称
            description: 角色描述
            personality: 角色性格
            speech_style: 角色说话风格
            keywords: 关键词列表，用于相关性匹配
            temperature: 生成参数，控制输出随机性
            prompt_template: 角色提示模板
            system_prompt: 系统提示
            
        返回:
            新创建的角色信息
        """
        # 检查角色名称是否已存在
        existing_role = await Role.get_by_name(name)
        if existing_role:
            raise ValueError(f"角色名 '{name}' 已存在")
            
        # 创建角色
        role = await Role.create(
            name=name,
            description=description,
            personality=personality,
            speech_style=speech_style,
            keywords=keywords,
            temperature=temperature,
            prompt_template=prompt_template,
            system_prompt=system_prompt
        )
        
        return role
    
    @staticmethod
    async def get_role_by_id(role_id: Union[str, ObjectId]) -> Optional[Dict[str, Any]]:
        """
        通过ID获取角色
        
        参数:
            role_id: 角色ID
            
        返回:
            角色信息，如果未找到则返回None
        """
        return await Role.get_by_id(role_id)
    
    @staticmethod
    async def get_role_by_name(name: str) -> Optional[Dict[str, Any]]:
        """
        通过名称获取角色
        
        参数:
            name: 角色名称
            
        返回:
            角色信息，如果未找到则返回None
        """
        return await Role.get_by_name(name)
    
    @staticmethod
    async def list_roles(active_only: bool = True, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """
        获取角色列表
        
        参数:
            active_only: 是否只返回活跃角色
            limit: 最大返回数量
            offset: 跳过的记录数
            
        返回:
            角色列表
        """
        if active_only:
            return await Role.list_active(limit=limit, offset=offset)
        else:
            # 需要在Role模型中添加list_all方法
            # 暂时使用list_active替代
            return await Role.list_active(limit=limit, offset=offset)
    
    @staticmethod
    async def update_role(
        role_id: Union[str, ObjectId],
        update_data: Dict[str, Any]
    ) -> bool:
        """
        更新角色信息
        
        参数:
            role_id: 角色ID
            update_data: 需要更新的字段
            
        返回:
            更新是否成功
        """
        # 检查角色是否存在
        role = await Role.get_by_id(role_id)
        if not role:
            raise ValueError(f"角色ID '{role_id}' 不存在")
            
        # 如果更新名称，检查新名称是否与其他角色冲突
        if "name" in update_data and update_data["name"] != role["name"]:
            existing_role = await Role.get_by_name(update_data["name"])
            if existing_role and str(existing_role["_id"]) != str(role_id):
                raise ValueError(f"角色名 '{update_data['name']}' 已被其他角色使用")
        
        # 更新角色
        return await Role.update(role_id, update_data)
    
    @staticmethod
    async def update_role_keywords(
        role_id: Union[str, ObjectId],
        keywords: List[str]
    ) -> bool:
        """
        更新角色关键词
        
        参数:
            role_id: 角色ID
            keywords: 新的关键词列表
            
        返回:
            更新是否成功
        """
        # 检查角色是否存在
        role = await Role.get_by_id(role_id)
        if not role:
            raise ValueError(f"角色ID '{role_id}' 不存在")
            
        return await Role.update_keywords(role_id, keywords)
    
    @staticmethod
    async def deactivate_role(role_id: Union[str, ObjectId]) -> bool:
        """
        停用角色
        
        参数:
            role_id: 角色ID
            
        返回:
            操作是否成功
        """
        # 检查角色是否存在
        role = await Role.get_by_id(role_id)
        if not role:
            raise ValueError(f"角色ID '{role_id}' 不存在")
            
        return await Role.deactivate(role_id)
    
    @staticmethod
    async def activate_role(role_id: Union[str, ObjectId]) -> bool:
        """
        启用角色
        
        参数:
            role_id: 角色ID
            
        返回:
            操作是否成功
        """
        # 检查角色是否存在
        role = await Role.get_by_id(role_id)
        if not role:
            raise ValueError(f"角色ID '{role_id}' 不存在")
            
        return await Role.activate(role_id)
    
    @staticmethod
    async def match_roles_by_keywords(
        message: str, 
        limit: int = 3, 
        min_score: float = 0.1,
        active_only: bool = True
    ) -> List[Dict[str, Any]]:
        """
        根据用户消息中的关键词匹配最合适的角色
        
        参数:
            message: 用户消息
            limit: 最多返回的角色数量
            min_score: 最小匹配分数，低于此分数的匹配将被忽略
            active_only: 是否只匹配活跃角色
            
        返回:
            按匹配度排序的角色列表，每个角色包含匹配分数
        """
        # 获取所有活跃角色
        roles = await RoleService.list_roles(active_only=active_only)
        if not roles:
            return []
            
        # 对消息进行分词
        words = RoleService._tokenize_message(message)
        
        # 计算每个角色的匹配分数
        scored_roles = []
        for role in roles:
            score = RoleService._calculate_match_score(words, role.get("keywords", []), message)
            if score >= min_score:
                role_with_score = role.copy()
                role_with_score["match_score"] = score
                scored_roles.append(role_with_score)
                
        # 按匹配分数排序并限制返回数量
        return sorted(scored_roles, key=lambda x: x["match_score"], reverse=True)[:limit]
    
    @staticmethod
    def _tokenize_message(message: str) -> List[str]:
        """
        对消息进行分词
        
        参数:
            message: 原始消息文本
            
        返回:
            分词列表
        """
        # 使用jieba分词
        seg_list = jieba.cut(message, cut_all=False)
        return [word for word in seg_list if len(word.strip()) > 0]
    
    @staticmethod
    def _calculate_match_score(
        message_words: List[str], 
        role_keywords: List[str],
        original_message: str
    ) -> float:
        """
        计算消息与角色关键词的匹配分数
        
        参数:
            message_words: 分词后的消息
            role_keywords: 角色关键词列表
            original_message: 原始消息，用于完整匹配
            
        返回:
            匹配分数 (0.0-1.0)
        """
        if not role_keywords:
            return 0.0
            
        total_score = 0.0
        matched_keywords = set()
        
        # 精确匹配算法
        for keyword in role_keywords:
            # 完整消息中包含关键词（权重较高）
            if keyword in original_message:
                total_score += 1.0
                matched_keywords.add(keyword)
                continue
                
            # 分词结果中包含关键词（权重中等）
            if keyword in message_words:
                total_score += 0.8
                matched_keywords.add(keyword)
                continue
                
            # 分词结果中部分匹配关键词（权重较低）
            for word in message_words:
                if keyword in word or word in keyword:
                    total_score += 0.5
                    matched_keywords.add(keyword)
                    break
        
        # 归一化分数
        match_ratio = len(matched_keywords) / len(role_keywords)
        keyword_score = total_score / (len(role_keywords) * 1.0)  # 最高可能分数
        
        # 综合分数，同时考虑匹配关键词比例和匹配强度
        return (match_ratio * 0.5) + (keyword_score * 0.5)
    
    @staticmethod
    async def get_role_for_message(message: str) -> Optional[Dict[str, Any]]:
        """
        为给定消息获取最匹配的角色
        
        参数:
            message: 用户消息
            
        返回:
            匹配度最高的角色，如果没有合适匹配则返回None
        """
        matched_roles = await RoleService.match_roles_by_keywords(
            message, 
            limit=1, 
            min_score=0.2  # 设置一个最小阈值，避免不相关匹配
        )
        
        if matched_roles:
            return matched_roles[0]
        return None
        
    @staticmethod
    async def match_role_for_message(message: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        匹配最适合的角色并返回结果

        Args:
            message: 用户消息
            session_id: 可选的会话ID

        Returns:
            Dict: 匹配结果，包含成功状态、角色信息、匹配原因和错误信息
        """
        start_time = time.time()
        request_id = str(uuid.uuid4())
        logger.info(f"[{request_id}] 开始角色匹配, 消息长度: {len(message)}, 会话ID: {session_id}")
        
        try:
            # 验证消息
            if not message or not isinstance(message, str):
                logger.error(f"[{request_id}] 无效的消息格式")
                return {"success": False, "error": "无效的消息格式"}
                
            # 执行角色匹配
            kw_extract_start = time.time()
            logger.debug(f"[{request_id}] 开始从消息中提取关键词")
            try:
                keywords = await asyncio.wait_for(
                    RoleService.extract_keywords_from_text(message, top_k=5),
                    timeout=5.0
                )
                kw_extract_time = time.time() - kw_extract_start
                logger.debug(f"[{request_id}] 关键词提取完成，耗时: {kw_extract_time:.4f}秒, 关键词: {keywords}")
            except asyncio.TimeoutError:
                logger.error(f"[{request_id}] 关键词提取超时")
                return {"success": False, "error": "关键词提取超时"}
            except Exception as e:
                logger.error(f"[{request_id}] 关键词提取失败: {str(e)}")
                return {"success": False, "error": f"关键词提取失败: {str(e)}"}
                
            # 根据关键词匹配角色
            match_start = time.time()
            logger.debug(f"[{request_id}] 开始根据关键词匹配角色")
            matched_roles = await RoleService.match_roles_by_keywords(keywords, top_k=3)
            match_time = time.time() - match_start
            logger.debug(f"[{request_id}] 角色匹配完成，耗时: {match_time:.4f}秒, 匹配到 {len(matched_roles)} 个角色")
            
            # 处理匹配结果
            if not matched_roles:
                logger.warning(f"[{request_id}] 未找到匹配的角色")
                return {"success": False, "error": "未找到匹配的角色"}
                
            # 获取最佳匹配角色
            best_match = matched_roles[0]
            
            # 构建匹配原因
            match_reason = "根据以下关键词匹配: "
            for kw in keywords[:3]:
                if kw in best_match.get("matched_keywords", []):
                    match_reason += f"「{kw}」"
                    
            logger.info(f"[{request_id}] 匹配成功，最佳角色: {best_match.get('name')}, 匹配原因: {match_reason}")
            
            # 构建完整返回结果
            result = {
                "success": True,
                "role": best_match,
                "match_reason": match_reason,
                "matched_keywords": best_match.get("matched_keywords", [])
            }
            
            total_time = time.time() - start_time
            logger.info(f"[{request_id}] 角色匹配过程完成，总耗时: {total_time:.4f}秒")
            return result
            
        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"[{request_id}] 角色匹配过程中出错，耗时: {total_time:.4f}秒, 错误: {str(e)}", exc_info=True)
            return {"success": False, "error": str(e)}
        
    @staticmethod
    async def extract_keywords_from_text(text: str, top_k: int = 10) -> List[str]:
        """
        从文本中提取关键词，可用于辅助设置角色关键词
        
        参数:
            text: 要分析的文本
            top_k: 返回的关键词数量
            
        返回:
            关键词列表
        """
        import jieba.analyse
        
        # 使用jieba的TextRank算法提取关键词
        keywords = jieba.analyse.textrank(text, topK=top_k, withWeight=False)
        return keywords

    @staticmethod
    async def add_prompt_template(
        role_id: Union[str, ObjectId], 
        template_name: str, 
        template_content: str,
        is_default: bool = False,
        description: str = ""
    ) -> Tuple[bool, Optional[str]]:
        """
        为角色添加提示模板
        
        参数:
            role_id: 角色ID
            template_name: 模板名称
            template_content: 模板内容
            is_default: 是否设为默认模板
            description: 模板描述
            
        返回:
            (成功状态, 模板ID)
        """
        # 检查角色是否存在
        role = await Role.get_by_id(role_id)
        if not role:
            raise ValueError(f"角色ID '{role_id}' 不存在")
        
        # 检查模板名称是否重复
        templates = await Role.get_prompt_templates(role_id)
        if any(t["name"] == template_name for t in templates):
            raise ValueError(f"模板名称 '{template_name}' 已存在")
        
        success, template_id = await Role.add_prompt_template(
            role_id, 
            template_name, 
            template_content,
            is_default,
            description
        )
        
        return success, template_id

    @staticmethod
    async def get_prompt_templates(
        role_id: Union[str, ObjectId]
    ) -> List[Dict[str, Any]]:
        """
        获取角色的所有提示模板
        
        参数:
            role_id: 角色ID
            
        返回:
            提示模板列表
        """
        # 检查角色是否存在
        role = await Role.get_by_id(role_id)
        if not role:
            raise ValueError(f"角色ID '{role_id}' 不存在")
        
        return await Role.get_prompt_templates(role_id)

    @staticmethod
    async def update_prompt_template(
        role_id: Union[str, ObjectId],
        template_id: str,
        name: Optional[str] = None,
        content: Optional[str] = None,
        description: Optional[str] = None
    ) -> bool:
        """
        更新角色提示模板
        
        参数:
            role_id: 角色ID
            template_id: 模板ID
            name: 新的模板名称
            content: 新的模板内容
            description: 新的模板描述
            
        返回:
            更新是否成功
        """
        # 检查角色是否存在
        role = await Role.get_by_id(role_id)
        if not role:
            raise ValueError(f"角色ID '{role_id}' 不存在")
        
        # 检查模板是否存在
        templates = await Role.get_prompt_templates(role_id)
        template = next((t for t in templates if t["id"] == template_id), None)
        if not template:
            raise ValueError(f"模板ID '{template_id}' 不存在")
        
        # 检查新名称是否与其他模板重复
        if name and name != template["name"]:
            if any(t["name"] == name and t["id"] != template_id for t in templates):
                raise ValueError(f"模板名称 '{name}' 已存在")
        
        # 构建更新数据
        update_data = {}
        if name is not None:
            update_data["name"] = name
        if content is not None:
            update_data["content"] = content
        if description is not None:
            update_data["description"] = description
        
        if not update_data:
            return True  # 没有更新，视为成功
        
        return await Role.update_prompt_template(role_id, template_id, update_data)

    @staticmethod
    async def delete_prompt_template(
        role_id: Union[str, ObjectId],
        template_id: str
    ) -> bool:
        """
        删除角色提示模板
        
        参数:
            role_id: 角色ID
            template_id: 模板ID
            
        返回:
            删除是否成功
        """
        # 检查角色是否存在
        role = await Role.get_by_id(role_id)
        if not role:
            raise ValueError(f"角色ID '{role_id}' 不存在")
        
        # 检查模板是否存在
        templates = await Role.get_prompt_templates(role_id)
        if not any(t["id"] == template_id for t in templates):
            raise ValueError(f"模板ID '{template_id}' 不存在")
        
        return await Role.delete_prompt_template(role_id, template_id)

    @staticmethod
    async def set_default_template(
        role_id: Union[str, ObjectId],
        template_id: str
    ) -> bool:
        """
        设置角色默认提示模板
        
        参数:
            role_id: 角色ID
            template_id: 模板ID
            
        返回:
            设置是否成功
        """
        # 检查角色是否存在
        role = await Role.get_by_id(role_id)
        if not role:
            raise ValueError(f"角色ID '{role_id}' 不存在")
        
        # 检查模板是否存在
        templates = await Role.get_prompt_templates(role_id)
        if not any(t["id"] == template_id for t in templates):
            raise ValueError(f"模板ID '{template_id}' 不存在")
        
        return await Role.set_default_template(role_id, template_id)

    @staticmethod
    async def get_default_template(
        role_id: Union[str, ObjectId]
    ) -> Optional[Dict[str, Any]]:
        """
        获取角色的默认提示模板
        
        参数:
            role_id: 角色ID
            
        返回:
            默认提示模板，如果没有则返回None
        """
        # 检查角色是否存在
        role = await Role.get_by_id(role_id)
        if not role:
            raise ValueError(f"角色ID '{role_id}' 不存在")
        
        default_id = role.get("default_template_id")
        if not default_id:
            return None
        
        templates = await Role.get_prompt_templates(role_id)
        return next((t for t in templates if t["id"] == default_id), None) 