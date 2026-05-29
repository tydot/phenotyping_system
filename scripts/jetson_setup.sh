#!/bin/bash
# =============================================================
# ARM 功能表型系统 - Jetson Orin Nano Super 8GB 一键部署脚本
# 用法：chmod +x scripts/jetson_setup.sh && sudo ./scripts/jetson_setup.sh
# =============================================================

set -e

echo "=========================================="
echo " ARM 功能表型系统 - Jetson 部署脚本"
echo "=========================================="

# ---- 1. 环境检查 ----
echo ""
echo "[1/8] 检查 Jetson 环境..."

if ! command -v nvcc &> /dev/null; then
    echo "错误：未检测到 CUDA，请确认已刷好 JetPack 6.x"
    exit 1
fi

CUDA_VER=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9.]+' || echo "unknown")
echo "  CUDA 版本：$CUDA_VER"
echo "  架构：$(uname -m)"
echo "  内存：$(free -h | awk '/Mem:/ {print $2}')"

# JetPack 版本检测
if [ -f /etc/nv_tegra_release ]; then
    L4T_VER=$(head -1 /etc/nv_tegra_release | grep -oP 'R\d+' || echo "unknown")
    echo "  L4T 版本：$L4T_VER"
elif command -v nv_tegra_release &> /dev/null; then
    echo "  L4T 版本：$(nv_tegra_release 2>/dev/null | head -1)"
else
    echo "  L4T 版本：未能检测"
fi

# 检查 NVMe SSD
echo ""
echo "  存储设备："
if lsblk -d -o NAME,TYPE,SIZE,MOUNTPOINT 2>/dev/null | grep -q nvme; then
    echo "  [OK] 检测到 NVMe SSD："
    lsblk -d -o NAME,SIZE,MOUNTPOINT 2>/dev/null | grep nvme
else
    echo "  [WARN] 未检测到 NVMe SSD，建议使用 NVMe 存放数据"
    echo "         当前存储："
    lsblk -d -o NAME,TYPE,SIZE 2>/dev/null | head -10
fi

# ---- 2. 设置最大性能模式 ----
echo ""
echo "[2/8] 设置 Jetson 最大性能模式..."

if command -v nvpmodel &> /dev/null; then
    sudo nvpmodel -m 0
    echo "  已设置为 MAXN 模式"
else
    echo "  跳过：nvpmodel 不可用（可能非 Jetson 设备）"
fi

if command -v jetson_clocks &> /dev/null; then
    sudo jetson_clocks
    echo "  已锁定最大频率"
fi

# ---- 3. 增加 swap 空间 ----
echo ""
echo "[3/8] 配置 swap 空间（4GB）..."

SWAPFILE="/swapfile"
if [ ! -f "$SWAPFILE" ]; then
    sudo fallocate -l 4G "$SWAPFILE"
    sudo chmod 600 "$SWAPFILE"
    sudo mkswap "$SWAPFILE"
    sudo swapon "$SWAPFILE"
    echo "  swap 已启用：4GB"

    # 持久化
    if ! grep -q "$SWAPFILE" /etc/fstab; then
        echo "$SWAPFILE none swap sw 0 0" | sudo tee -a /etc/fstab > /dev/null
        echo "  已添加到 /etc/fstab 实现开机自动挂载"
    fi
else
    echo "  swap 文件已存在，跳过"
fi

# ---- 4. 安装 Docker ----
echo ""
echo "[4/8] 检查 Docker..."

if ! command -v docker &> /dev/null; then
    echo "  安装 Docker..."
    sudo apt-get update
    sudo apt-get install -y docker.io docker-compose-v2
else
    echo "  Docker 已安装：$(docker --version)"
fi

# 确保 Docker 服务运行
sudo systemctl enable docker
sudo systemctl start docker

# 将当前用户添加到 docker 组
if ! groups "$USER" | grep -q docker; then
    sudo usermod -aG docker "$USER"
    echo "  已将 $USER 添加到 docker 组（需要重新登录生效）"
fi

# ---- 5. 准备环境配置 ----
echo ""
echo "[5/8] 准备环境配置..."

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

if [ ! -f .env ]; then
    cp .env.jetson .env
    echo "  已从 .env.jetson 复制 .env 文件"
else
    echo "  .env 文件已存在，跳过"
fi

# ---- 6. 检查推理产物 ----
echo ""
echo "[6/8] 检查推理产物..."

ARTIFACT_DIR="$PROJECT_DIR/outputs/inference_artifacts"
REQUIRED_FILES=(
    "scaler_mean.npy"
    "scaler_scale.npy"
    "pca_components.npy"
    "pca_mean.npy"
    "cluster_prototypes_pca.npy"
    "reference_patient_embeddings_pca.csv"
)

ARTIFACTS_OK=true
for f in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$ARTIFACT_DIR/$f" ]; then
        echo "  [MISSING] $f"
        ARTIFACTS_OK=false
    fi
done

if [ "$ARTIFACTS_OK" = true ]; then
    echo "  [OK] 全部推理产物已就绪"
else
    echo ""
    echo "  [ERROR] 缺少推理产物！请先在开发机上运行："
    echo "    python scripts/generate_inference_artifacts.py"
    echo "  然后将 outputs/inference_artifacts/ 目录拷贝到 Jetson"
    exit 1
fi

# ---- 7. 预缓存 DINOv2 模型 ----
echo ""
echo "[7/8] 检查 DINOv2 模型缓存..."

TORCH_HUB_DIR="$HOME/.cache/torch/hub"
DINOV2_DIR="$TORCH_HUB_DIR/facebookresearch_dinov2_main"

if [ -d "$DINOV2_DIR" ]; then
    echo "  DINOv2 模型缓存已存在"
else
    echo "  DINOv2 首次运行时会自动下载模型权重"
    echo "  如需预缓存，可从开发机拷贝 ~/.cache/torch/hub/ 目录"
    echo "  或在 Docker 构建阶段自动下载（Dockerfile.jetson 已配置）"
fi

# ---- 8. 构建并启动 ----
echo ""
echo "[8/8] 构建 Docker 镜像并启动服务..."
echo "  这可能需要 10-30 分钟，取决于网络速度"
echo ""

docker compose -f docker-compose.jetson.yml build

echo ""
echo "启动服务..."
docker compose -f docker-compose.jetson.yml up -d

echo ""
echo "=========================================="
echo " 部署完成！"
echo "=========================================="
echo ""
echo " 访问地址："
echo "   前端界面：http://$(hostname -I | awk '{print $1}'):8501"
echo "   Neo4j：  http://$(hostname -I | awk '{print $1}'):7474"
echo ""
echo " 默认账号：admin / admin123"
echo ""
echo " 内存分配（8GB 总预算）："
echo "   Neo4j:  1.5GB (heap 768m + pagecache 256m)"
echo "   App:    5.0GB (含 DINOv2 FP16 推理)"
echo "   系统:   ~1.5GB"
echo ""
echo " 常用命令："
echo "   查看日志：docker compose -f docker-compose.jetson.yml logs -f"
echo "   停止服务：docker compose -f docker-compose.jetson.yml down"
echo "   重启服务：docker compose -f docker-compose.jetson.yml restart"
echo "   监控资源：sudo tegrastats"
echo "   GPU 状态：sudo jtop"
echo ""
