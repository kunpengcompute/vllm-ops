#!/bin/bash
# OpenVLA 前处理优化 — 部署脚本
# 将 NEON 加速 kernel 注入 vLLM 源码树
#
# 用法:
#   bash deploy.sh /path/to/vllm
#
# 设计原则: 只"追加", 不替换上游文件
#   - 新增:  openvla_image_preprocess.cpp (kernel 实现)
#   - 替换:  openvla.py (processor + torch.library 算子注册)
#   - sed:   cpu_extension.cmake (追加 kernel 源文件)
#
# 注意: C++ TORCH_LIBRARY 静态初始化在 torch 2.13.0 ARM 平台不可靠,
#       改为在 Python 端用 torch.library.Library 注册算子.

set -e

VLLM_DIR="${1:?请指定 vLLM 源码目录, 例如: bash deploy.sh /home/j50058823/openvla/vllm}"
DEPLOY_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== OpenVLA 前处理 NEON 加速 — 部署 ==="
echo "vLLM 目录: $VLLM_DIR"
echo "deploy 目录: $DEPLOY_DIR"

# ── 1. 新增 kernel 源文件 ──────────────────────────────────────
echo ""
echo "[1/4] 新增 OpenVLA kernel 源文件"

cp -v "$DEPLOY_DIR/vllm/csrc/cpu/openvla_image_preprocess.cpp" \
      "$VLLM_DIR/csrc/cpu/openvla_image_preprocess.cpp"

# ── 2. cpu_extension.cmake: 追加 kernel 源文件 ──────────────────
echo ""
echo "[2/4] 追加源文件到 cpu_extension.cmake"

CMAKE_FILE="$VLLM_DIR/cmake/cpu_extension.cmake"

if ! grep -q "openvla_image_preprocess.cpp" "$CMAKE_FILE"; then
    sed -i '/"csrc\/cpu\/shm.cpp"/a\        "csrc\/cpu\/openvla_image_preprocess.cpp"' \
        "$CMAKE_FILE"
    echo "  [OK] 已追加 openvla_image_preprocess.cpp"
else
    echo "  [SKIP] 已存在, 不重复追加"
fi

# ── 3. 替换 processor (含 torch.library 算子注册) ──────────────
echo ""
echo "[3/4] 替换 OpenVLA processor (含 torch.library 注册)"

cp -v "$DEPLOY_DIR/vllm/vllm/transformers_utils/processors/openvla.py" \
      "$VLLM_DIR/vllm/transformers_utils/processors/openvla.py"

# ── 4. 验证 ──────────────────────────────────────────────────
echo ""
echo "[4/4] 验证文件存在"

for f in \
    csrc/cpu/openvla_image_preprocess.cpp \
    vllm/transformers_utils/processors/openvla.py
do
    if [ -f "$VLLM_DIR/$f" ]; then
        echo "  [OK] $f"
    else
        echo "  [FAIL] $f 不存在!"
        exit 1
    fi
done

echo ""
echo "=== 部署完成 ==="
echo ""
echo "下一步构建:"
echo "  cd $VLLM_DIR"
echo "  VLLM_TARGET_DEVICE=cpu pip3 install -e . --no-build-isolation"
echo ""

