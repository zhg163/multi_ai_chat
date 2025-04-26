"""
数据库降级和模拟模块

当数据库连接不可用时提供模拟功能，用于开发和降级
"""

import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from bson import ObjectId

logger = logging.getLogger(__name__)

class MockCollection:
    """模拟MongoDB集合的类，提供基本的CRUD操作"""
    
    def __init__(self, name: str):
        """初始化模拟集合"""
        self.name = name
        self.data = []
        #logger.debug(f"创建模拟集合: {name}")
    
    async def insert_one(self, document: Dict[str, Any]):
        """插入一个文档"""
        if "_id" not in document:
            document["_id"] = ObjectId()
        self.data.append(document)
        logger.debug(f"向集合 {self.name} 插入文档: {document.get('_id')}")
        
        class Result:
            @property
            def inserted_id(self):
                return document["_id"]
        
        return Result()
    
    async def insert_many(self, documents: List[Dict[str, Any]]):
        """插入多个文档"""
        inserted_ids = []
        for doc in documents:
            if "_id" not in doc:
                doc["_id"] = ObjectId()
            self.data.append(doc)
            inserted_ids.append(doc["_id"])
        
        logger.debug(f"向集合 {self.name} 插入 {len(documents)} 个文档")
        
        class Result:
            @property
            def inserted_ids(self):
                return inserted_ids
        
        return Result()
    
    async def find_one(self, query: Dict[str, Any]):
        """查找单个文档"""
        #logger.debug(f"在集合 {self.name} 中查找: {query}")
        for doc in self.data:
            match = True
            for key, value in query.items():
                if key not in doc or doc[key] != value:
                    match = False
                    break
            if match:
                #logger.debug(f"找到匹配文档: {doc.get('_id')}")
                return doc
        #logger.debug(f"未找到匹配文档")
        return None
    
    async def update_one(self, query: Dict[str, Any], update: Dict[str, Any]):
        """更新单个文档"""
        logger.debug(f"在集合 {self.name} 中更新: {query}")
        modified_count = 0
        
        for i, doc in enumerate(self.data):
            match = True
            for key, value in query.items():
                if key not in doc or doc[key] != value:
                    match = False
                    break
            
            if match:
                # 处理 $set 操作
                if "$set" in update:
                    for k, v in update["$set"].items():
                        self.data[i][k] = v
                
                # 处理 $push 操作
                if "$push" in update:
                    for k, v in update["$push"].items():
                        if k not in self.data[i]:
                            self.data[i][k] = []
                        self.data[i][k].append(v)
                
                # 处理 $pull 操作
                if "$pull" in update:
                    for k, v in update["$pull"].items():
                        if k in self.data[i] and isinstance(self.data[i][k], list):
                            # 简单实现，仅支持直接值比较
                            self.data[i][k] = [item for item in self.data[i][k] if item != v]
                
                modified_count += 1
                logger.debug(f"已更新文档: {doc.get('_id')}")
                break
        
        class Result:
            @property
            def modified_count(self):
                return modified_count
        
        return Result()
    
    async def delete_one(self, query: Dict[str, Any]):
        """删除单个文档"""
        logger.debug(f"在集合 {self.name} 中删除: {query}")
        deleted_count = 0
        
        for i, doc in enumerate(self.data):
            match = True
            for key, value in query.items():
                if key not in doc or doc[key] != value:
                    match = False
                    break
            
            if match:
                self.data.pop(i)
                deleted_count = 1
                logger.debug(f"已删除文档: {doc.get('_id')}")
                break
        
        class Result:
            @property
            def deleted_count(self):
                return deleted_count
        
        return Result()
    
    async def delete_many(self, query: Dict[str, Any]):
        """删除多个文档"""
        logger.debug(f"在集合 {self.name} 中批量删除: {query}")
        to_delete = []
        
        for i, doc in enumerate(self.data):
            match = True
            for key, value in query.items():
                if key not in doc or doc[key] != value:
                    match = False
                    break
            
            if match:
                to_delete.append(i)
        
        # 从后向前删除，以避免索引变化问题
        deleted_count = len(to_delete)
        for i in sorted(to_delete, reverse=True):
            self.data.pop(i)
        
        logger.debug(f"已删除 {deleted_count} 个文档")
        
        class Result:
            @property
            def deleted_count(self):
                return deleted_count
        
        return Result()
    
    def find(self, query: Dict[str, Any] = None):
        """查找多个文档，返回游标"""
        if query is None:
            query = {}
        
        logger.debug(f"在集合 {self.name} 中查找多个文档: {query}")
        matched_docs = []
        
        for doc in self.data:
            match = True
            for key, value in query.items():
                if key not in doc or doc[key] != value:
                    match = False
                    break
            
            if match:
                matched_docs.append(doc)
        
        logger.debug(f"找到 {len(matched_docs)} 个匹配文档")
        return MockCursor(matched_docs)

class MockCursor:
    """模拟MongoDB游标的类"""
    
    def __init__(self, data: List[Dict[str, Any]]):
        """初始化游标"""
        self.data = data
        self.sort_field = None
        self.sort_direction = 1
        self.skip_count = 0
        self.limit_count = None
    
    def sort(self, field: Union[str, tuple], direction: int = 1):
        """设置排序字段和方向"""
        self.sort_field = field
        self.sort_direction = direction
        return self
    
    def skip(self, count: int):
        """设置跳过的文档数"""
        self.skip_count = count
        return self
    
    def limit(self, count: int):
        """设置返回的最大文档数"""
        self.limit_count = count
        return self
    
    async def to_list(self, length: int = None):
        """转换为列表"""
        # 应用排序
        if self.sort_field:
            field_name = self.sort_field
            if isinstance(self.sort_field, tuple):
                field_name = self.sort_field[0]
            
            def sort_key(doc):
                return doc.get(field_name, "")
            
            self.data.sort(key=sort_key, reverse=(self.sort_direction == -1))
        
        # 应用skip
        result = self.data[self.skip_count:]
        
        # 应用limit
        limit = length if length is not None else self.limit_count
        if limit is not None:
            result = result[:limit]
        
        return result 