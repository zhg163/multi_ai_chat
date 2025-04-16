#!/bin/bash
set -e

# 配置变量
IMAGE_NAME="crpi-vevfocpds07ux96j.cn-beijing.personal.cr.aliyuncs.com/zhgpub/dokcerhubzhg"
IMAGE_TAG="v1.0.0"
OUTPUT_DIR="./docker-images"

# 创建输出目录
mkdir -p $OUTPUT_DIR

echo "=== 准备构建器 ==="
# 确保构建器存在
docker buildx inspect multiarch-builder >/dev/null 2>&1 || docker buildx create --name multiarch-builder --driver docker-container --use
docker buildx use multiarch-builder
docker buildx inspect --bootstrap

echo "=== 开始构建多架构镜像 ==="
# 构建多架构镜像并导出到本地目录
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  --tag $IMAGE_NAME:$IMAGE_TAG \
  --output "type=local,dest=$OUTPUT_DIR" \
  .

echo "=== 构建完成 ==="
echo "多架构镜像已保存到: $OUTPUT_DIR"
echo "这些镜像可以使用 'docker load' 命令加载"

# 可选：构建当前平台镜像并加载到本地Docker
echo "=== 构建当前平台镜像并加载到本地 ==="
CURRENT_PLATFORM=$(docker version -f '{{.Server.Os}}/{{.Server.Arch}}')
docker buildx build \
  --platform $CURRENT_PLATFORM \
  --tag $IMAGE_NAME:$IMAGE_TAG-local \
  --load \
  .

echo "=== 本地镜像构建完成 ==="
echo "本地镜像标签: $IMAGE_NAME:$IMAGE_TAG-local"