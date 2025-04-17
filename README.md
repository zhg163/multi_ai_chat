# 多AI聊天系统

本项目提供了一个支持多角色的AI聊天系统，能够根据用户消息自动选择最合适的角色进行回复。

## 主要功能

- 多角色支持：系统支持多个AI角色，每个角色有自己的性格、说话风格和专业领域
- 角色匹配：根据用户消息自动选择最合适的角色
- 实时对话：支持流式输出，提供更好的用户体验
- 角色管理：可以添加、编辑、激活或禁用角色

## 安装与配置

1. 克隆项目仓库
```bash
git clone <项目地址>
cd multi_ai_chat
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 配置环境变量
创建`.env`文件，填入必要的配置：
```
MONGODB_URL=mongodb://localhost:27017
DB_NAME=multi_ai_chat
```

4. 启动服务
```bash
uvicorn app.main:app --reload
```

## 关于嵌入服务

本系统提供了两种嵌入服务实现：

1. **标准实现（使用sentence-transformers）**：
   - 需要Python环境支持`_lzma`模块
   - 提供高质量的语义向量表示
   - 使用多语言模型支持中英文

2. **简化实现（不依赖外部模型）**：
   - 不依赖`_lzma`模块，适用于所有Python环境
   - 使用基于哈希的方法生成伪向量表示
   - 性能较轻量，但语义理解能力有限

系统会自动选择合适的实现。如果您希望使用标准实现，请确保您的Python安装支持`lzma`模块，或者安装更新版本的`sentence-transformers`。

## API文档

启动服务后，访问以下地址查看API文档：
- Swagger UI：http://localhost:8000/docs
- ReDoc：http://localhost:8000/redoc

## 示例页面

访问 http://localhost:8000/ 查看聊天演示页面。 



python /Users/zhg/.cursor/extensions/ms-python.debugpy-2024.6.0-darwin-arm64/bundled/libs/debugpy/adapter/../../debugpy/launcher 55819 -- -m uvicorn main:app --reload 



docker-compose up -d

# 查看服务状态
docker-compose ps

# 停止服务
docker-compose stop

# 重启服务
docker-compose restart

# 删除服务(数据会保留在volume中)
docker-compose down

# 完全删除(包括数据)
docker-compose down -v


# 进入Redis容器
docker exec -it ai_chat_redis redis-cli -a your_redis_password

# 进入MongoDB容器
docker exec -it ai_chat_mongodb mongosh -u root -p example


uvicorn app.main:app --reload 


docker-compose stop app
docker-compose build app
docker-compose up -d app


# 登录私有仓库
docker login --username=<用户名> crpi-vevfocpds07ux96j.cn-beijing.personal.cr.aliyuncs.com/dockhub-zhg0914/dockhub-zhg


docker pull redis:6.2.8
docker pull mongo:5.0.22


git config --global http.proxy http://127.0.0.1:15236
git config --global https.proxy https://127.0.0.1:15236

git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/zhg163/multi_ai_chat.git
git push -u origin main


git clone https://github.com/zhg163/multi_ai_chat.git
cd multi_ai_chat
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

nohup python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 > output.log 2>&1 &