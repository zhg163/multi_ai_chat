import requests
import json
import sys
from pprint import pprint

class RetrievalApiTest:
    def __init__(self, 
                 base_url="http://localhost:9222", 
                 api_key="ragflow-Q3Njg2ODNjMTNjMDExZjBhYTE4MzU1Yz",
                 auth_method="bearer"):
        """
        初始化检索API测试类
        
        参数:
            base_url: API基础URL
            api_key: API密钥
            auth_method: 授权方法，可选值: "bearer", "header", "param"
        """
        self.base_url = base_url
        self.api_key = api_key
        self.auth_method = auth_method
        self.retrieval_url = f"{base_url}/api/v1/retrieval"
        self.datasets_url = f"{base_url}/api/v1/datasets"
        self.results = []
        
        # 根据授权方法设置不同的请求头
        self.headers = {
            "Content-Type": "application/json"
        }
        
        if auth_method == "bearer":
            self.headers["Authorization"] = f"Bearer {api_key}"
        elif auth_method == "header":
            self.headers["api-key"] = api_key
    
    def get_datasets(self):
        """
        获取所有数据集的列表
        
        返回:
            包含API响应的字典对象
        """
        url = self.datasets_url
        
        # 准备参数
        params = {}
        if self.auth_method == "param":
            params["api_key"] = self.api_key
        
        print(f"发送GET请求到 {url}")
        print(f"请求头: {json.dumps(self.headers, indent=2, ensure_ascii=False)}")
        
        try:
            # 发送GET请求
            response = requests.get(
                url, 
                headers=self.headers, 
                params=params
            )
            print(f"\n响应状态码: {response.status_code}")
            
            # 尝试解析响应为JSON
            try:
                resp_data = response.json()
                #print("响应内容:", json.dumps(resp_data, indent=2, ensure_ascii=False))
                
                # 如果返回成功并包含数据集信息，打印数据集列表
                if response.status_code == 200 and resp_data.get("code") == 0 and "data" in resp_data:
                    print("\n数据集列表:")
                    print("-" * 50)
                    
                    # 根据响应结构来处理
                    if isinstance(resp_data["data"], list):
                        datasets = resp_data["data"]
                    else:
                        datasets = resp_data["data"].get("datasets", [])
                    
                    for i, dataset in enumerate(datasets):
                        dataset_id = dataset.get("id")
                        dataset_name = dataset.get("name")
                        print(f"{i+1}. ID: {dataset_id} | 名称: {dataset_name}")
                    
                    print("-" * 50)
                    
                return resp_data
                
            except ValueError:
                print("响应不是有效的JSON格式")
                print(response.text)
                return {"success": False, "error": "Invalid JSON response", "text": response.text}
            
        except requests.RequestException as e:
            print(f"请求过程中出错: {e}")
            
            error_data = {"success": False, "error": str(e)}
            
            if hasattr(e, 'response') and e.response:
                print(f"响应状态码: {e.response.status_code}")
                
                try:
                    error_data["response"] = e.response.json()
                except:
                    error_data["response"] = e.response.text
                    
                print(f"响应内容: {error_data['response']}")
            
            return error_data
    
    def get_dataset_documents(self, 
                             dataset_id, 
                             page=1, 
                             page_size=10, 
                             orderby=None, 
                             desc=False, 
                             keywords=None, 
                             document_id=None, 
                             document_name=None):
        """
        获取指定数据集下的文档
        
        参数:
            dataset_id: 数据集ID
            page: 页码
            page_size: 每页大小
            orderby: 排序字段
            desc: 是否降序
            keywords: 搜索关键词
            document_id: 文档ID筛选
            document_name: 文档名称筛选
            
        返回:
            包含API响应的字典对象
        """
        url = f"{self.datasets_url}/{dataset_id}/documents"
        
        # 准备查询参数
        params = {
            "page": page,
            "page_size": page_size
        }
        
        if orderby:
            params["orderby"] = orderby
        
        if desc:
            params["desc"] = desc
            
        if keywords:
            params["keywords"] = keywords
            
        if document_id:
            params["id"] = document_id
            
        if document_name:
            params["name"] = document_name
            
        if self.auth_method == "param":
            params["api_key"] = self.api_key
        
        print(f"发送GET请求到 {url}")
        print(f"请求头: {json.dumps(self.headers, indent=2, ensure_ascii=False)}")
        print(f"查询参数: {json.dumps(params, indent=2, ensure_ascii=False)}")
        
        try:
            # 发送GET请求
            response = requests.get(
                url, 
                headers=self.headers, 
                params=params
            )
            print(f"\n响应状态码: {response.status_code}")
            
            # 尝试解析响应为JSON
            try:
                resp_data = response.json()
                print("响应内容:", json.dumps(resp_data, indent=2, ensure_ascii=False))
                
                # 如果返回成功并包含文档信息，打印文档列表
                if response.status_code == 200 and resp_data.get("code") == 0 and "data" in resp_data:
                    print("\n文档列表:")
                    print("-" * 50)
                    
                    # 根据响应结构来处理
                    if isinstance(resp_data["data"], list):
                        documents = resp_data["data"]
                    else:
                        documents = resp_data["data"].get("documents", [])
                    
                    for i, doc in enumerate(documents):
                        doc_id = doc.get("id")
                        doc_name = doc.get("name")
                        print(f"{i+1}. ID: {doc_id} | 名称: {doc_name}")
                    
                    print("-" * 50)
                    
                return resp_data
                
            except ValueError:
                print("响应不是有效的JSON格式")
                print(response.text)
                return {"success": False, "error": "Invalid JSON response", "text": response.text}
            
        except requests.RequestException as e:
            print(f"请求过程中出错: {e}")
            
            error_data = {"success": False, "error": str(e)}
            
            if hasattr(e, 'response') and e.response:
                print(f"响应状态码: {e.response.status_code}")
                
                try:
                    error_data["response"] = e.response.json()
                except:
                    error_data["response"] = e.response.text
                    
                print(f"响应内容: {error_data['response']}")
            
            return error_data
    
    def test_retrieval(self, 
                       question="What is advantage of ragflow?",
                       dataset_ids=None, 
                       document_ids=None,
                       page=1,
                       page_size=30,
                       similarity_threshold=0.5,
                       vector_similarity_weight=0.8,
                       top_k=1024,
                       rerank_id=None,
                       keyword=False,
                       highlight=False):
        """
        测试检索API
        
        参数:
            question: 用户查询或关键词
            dataset_ids: 要搜索的数据集ID列表
            document_ids: 要搜索的文档ID列表
            page: 显示块的页码，默认为1
            page_size: 每页的最大块数，默认为30
            similarity_threshold: 最小相似度分数，默认为0.2
            vector_similarity_weight: 向量余弦相似度的权重，默认为0.3
            top_k: 参与向量余弦计算的块数，默认为1024
            rerank_id: 重排模型的ID
            keyword: 是否启用基于关键词的匹配，默认为False
            highlight: 是否在结果中突出显示匹配的术语，默认为False
        """
        # 构建请求体
        payload = {
            "question": question,
            "page": page,
            "page_size": page_size,
            "similarity_threshold": similarity_threshold,
            "vector_similarity_weight": vector_similarity_weight,
            "top_k": top_k,
            "keyword": keyword,
            "highlight": highlight
        }
        
        # 添加可选参数
        if dataset_ids:
            payload["dataset_ids"] = dataset_ids
            
        if document_ids:
            payload["document_ids"] = document_ids
            
        if rerank_id:
            payload["rerank_id"] = rerank_id
        
        # 准备参数
        params = {}
        if self.auth_method == "param":
            params["api_key"] = self.api_key
        
        print(f"发送请求到 {self.retrieval_url}")
        print(f"请求头: {json.dumps(self.headers, indent=2, ensure_ascii=False)}")
        print(f"请求体: {json.dumps(payload, indent=2, ensure_ascii=False)}")
        
        # 执行请求
        result = self._make_request(self.retrieval_url, self.headers, payload, params)
        self.results.append(result)
        return result
    
    def _make_request(self, url, headers, payload, params=None):
        """
        内部方法，用于执行请求并处理响应
        """
        result = {
            "url": url,
            "headers": headers,
            "payload": payload,
            "params": params,
            "status_code": None,
            "response": None,
            "success": False,
            "chunks": []
        }
        
        try:
            # 发送POST请求
            with requests.post(
                url, 
                headers=headers, 
                json=payload,
                params=params
            ) as response:
                result["status_code"] = response.status_code
                
                # 尝试解析响应为JSON
                try:
                    resp_data = response.json()
                    result["response"] = resp_data
                    
                    # 检查响应是否成功
                    if response.status_code == 200 and resp_data.get("code") == 0:
                        result["success"] = True
                        
                        # 提取并存储返回的chunks
                        if "data" in resp_data and "chunks" in resp_data["data"]:
                            result["chunks"] = resp_data["data"]["chunks"]
                            
                            # 模拟流式输出
                            print("\n模拟流式输出检索结果:")
                            print("-" * 50)
                            
                            for i, chunk in enumerate(resp_data["data"]["chunks"]):
                                # 显示进度
                                progress = f"[{i+1}/{len(resp_data['data']['chunks'])}]"
                                
                                # 获取内容 (优先使用highlight，如果有的话)
                                content = chunk.get("highlight", chunk.get("content", "无内容"))
                                
                                # 显示来源
                                doc_name = chunk.get("document_keyword", "未知文档")
                                
                                # 显示相似度分数
                                similarity = chunk.get("similarity", 0)
                                
                                # 打印结果块
                                print(f"\n{progress} 文档: {doc_name} (相似度: {similarity:.4f})")
                                print(f"内容: {content}")
                                
                                # 模拟流式延迟
                                # (在实际测试中可以考虑添加time.sleep)
                            
                            print("\n检索完成，共返回 {} 个结果块。".format(len(resp_data["data"]["chunks"])))
                            print("-" * 50)
                            
                except ValueError:
                    print("响应不是有效的JSON格式")
                    result["response"] = response.text
                
                # 打印响应结果详情
                print("\n响应状态码:", response.status_code)
                if result["success"]:
                    print("检索成功! 返回了 {} 个结果块。".format(len(result["chunks"])))
                else:
                    print("检索失败!")
                    print("响应内容:", json.dumps(result["response"], indent=2, ensure_ascii=False))
                
        except requests.RequestException as e:
            print(f"请求过程中出错: {e}")
            result["error"] = str(e)
            
            if hasattr(e, 'response') and e.response:
                result["status_code"] = e.response.status_code
                print(f"响应状态码: {e.response.status_code}")
                
                try:
                    result["response"] = e.response.json()
                except:
                    result["response"] = e.response.text
                    
                print(f"响应内容: {result['response']}")
        
        return result

def run_test():
    """运行测试示例"""
    # 创建测试对象，使用Bearer认证方式
    api_test = RetrievalApiTest(auth_method="bearer")
    
    # 获取可用的数据集列表
    print("\n=== 获取数据集列表 ===")
    datasets_result = api_test.get_datasets()
    
    # 获取第一个数据集的ID作为测试使用（如果存在）
    dataset_id = None
    if datasets_result.get("code") == 0 and "data" in datasets_result:
        # 尝试从不同的数据结构获取数据集
        if isinstance(datasets_result["data"], list):
            datasets = datasets_result["data"]
        else:
            datasets = datasets_result["data"].get("datasets", [])
            
        if datasets and len(datasets) > 0:
            dataset_id = datasets[1].get("id")
            print(f"\n找到数据集ID: {dataset_id}")
    
    # # 如果获取到数据集ID，查询该数据集下的文档
    # if dataset_id:
    #     print(f"\n=== 获取数据集 {dataset_id} 的文档 ===")
    #     documents_result = api_test.get_dataset_documents(dataset_id)
        
    #     # 获取文档ID
    #     document_id = None
    #     if documents_result.get("code") == 0 and "data" in documents_result:
    #         # 尝试从不同的数据结构获取文档
    #         if isinstance(documents_result["data"], list):
    #             documents = documents_result["data"]
    #         else:
    #             documents = documents_result["data"].get("documents", [])
                
    #         if documents and len(documents) > 0:
    #             document_id = documents[0].get("id")
    #             print(f"\n找到文档ID: {document_id}")
        
        # 测试场景1: 使用获取到的数据集ID
        print(f"\n=== 测试场景1: 使用获取到的数据集ID: {dataset_id} ===")
        result1 = api_test.test_retrieval(
            question="记忆晶体",
            dataset_ids=[dataset_id],
            highlight=True
        )
        
    #     # 测试场景2: 同时使用获取到的数据集ID和文档ID
    #     if document_id:
    #         print(f"\n=== 测试场景2: 同时使用数据集ID和文档ID ===")
    #         result2 = api_test.test_retrieval(
    #             question="What is ragflow?",
    #             dataset_ids=[dataset_id],
    #             document_ids=[document_id],
    #             highlight=True
    #         )
    # else:
    #     print("\n无法获取数据集ID，跳过测试")
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    print("检索API测试工具已创建，可以直接运行脚本获取数据集ID和测试检索功能。")
    print("正在运行测试...")
    run_test() 