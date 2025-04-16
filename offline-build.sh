#!/bin/bash
set -e

# 配置变量
IMAGE_NAME="crpi-vevfocpds07ux96j.cn-beijing.personal.cr.aliyuncs.com/zhgpub/dokcerhubzhg"
IMAGE_TAG="v1.0.0"
BASE_IMAGE="python:3.10.17-slim"

# 检查基础镜像是否存在
if ! docker images | grep -q "${BASE_IMAGE}"; then
  echo "错误: 基础镜像 ${BASE_IMAGE} 未找到"
  exit 1
fi

# 构建镜像
echo "使用本地 ${BASE_IMAGE} 构建应用..."
docker build --network=none \
  -t ${IMAGE_NAME}:${IMAGE_TAG} \
  multi_ai_chat

echo "构建完成: ${IMAGE_NAME}:${IMAGE_TAG}"