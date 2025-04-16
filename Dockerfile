# 使用Debian Slim基础镜像
FROM python:3.10.17-slim

# 清除任何代理设置
ENV http_proxy=""
ENV https_proxy=""
ENV HTTP_PROXY=""
ENV HTTPS_PROXY=""

# 设置工作目录
WORKDIR /app

# 安装必要的系统包
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 配置pip使用国内镜像源
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ \
    && pip config set global.trusted-host mirrors.aliyun.com

# 更新pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# 先复制requirements文件单独安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout=100 -r requirements.txt

# 复制项目文件
COPY . .

# 暴露端口
EXPOSE 8000

# 运行命令
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]