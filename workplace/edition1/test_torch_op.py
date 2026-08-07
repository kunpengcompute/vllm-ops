#!/usr/bin/env python3
"""vLLM torch op 集成验证 — 编译 + 正确性 + 性能"""
import torch, numpy as np, time, sys
from torch.utils.cpp_extension import load_inline
from PIL import Image

IMAGENET_MEAN = np.array([0.484375, 0.455078125, 0.40625], dtype=np.float32)
IMAGENET_STD  = np.array([0.228515625, 0.2236328125, 0.224609375], dtype=np.float32)
SIGLIP_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
SIGLIP_STD  = np.array([0.5, 0.5, 0.5], dtype=np.float32)

# Step 1: 编译 — 自动查找 kernel 源文件
print("=== Step 1: torch.utils.cpp_extension.load_inline 编译 ===")
import os, pathlib

# 按优先级查找: 命令行参数 > 相对路径 > 绝对路径
script_dir = pathlib.Path(__file__).resolve().parent
search_paths = [
    script_dir / "../../vllm/csrc/cpu/openvla_image_preprocess.cpp",
    script_dir / "../../../vllm/csrc/cpu/openvla_image_preprocess.cpp",
    pathlib.Path("/home/j50058823/openvla/vllm/csrc/cpu/openvla_image_preprocess.cpp"),
]
cpp_path = None
for p in search_paths:
    if p.exists():
        cpp_path = str(p)
        break
if cpp_path is None:
    print("ERROR: 找不到 openvla_image_preprocess.cpp, 尝试的路径:")
    for p in search_paths:
        print(f"  {p}")
    sys.exit(1)
print(f"  源文件: {cpp_path}")
cpp_source = open(cpp_path).read()

mod = load_inline(
    name='openvla_preprocess',
    cpp_sources=[cpp_source],
    functions=['openvla_fused_preprocess'],
    extra_cflags=['-std=c++17', '-march=armv8.2-a+fp16+dotprod', '-fopenmp', '-O3', '-D__aarch64__'],
    extra_ldflags=['-fopenmp'],
)
print("[OK] 编译成功")
print(f"   函数签名: {mod.openvla_fused_preprocess}")

# Step 2: 正确性验证
print("\n=== Step 2: 正确性验证 (vs PIL+NumPy reference) ===")
rng = np.random.RandomState(42)
all_ok = True

for w, h in [(224, 224), (481, 321), (640, 480), (1920, 1080)]:
    pixels = rng.randint(0, 256, (h, w, 3), dtype=np.uint8)
    t = torch.from_numpy(pixels.copy()).contiguous()

    # 加速路径
    out_acc = mod.openvla_fused_preprocess(t, 4)

    # PIL 参考
    img = Image.fromarray(pixels).resize((224, 224), Image.Resampling.BICUBIC)
    raw = np.asarray(img, dtype=np.float32) / 255.0
    dino = ((raw - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)
    siglip = ((raw - SIGLIP_MEAN) / SIGLIP_STD).transpose(2, 0, 1)
    ref = np.concatenate([dino, siglip], axis=0).astype(np.float32)

    max_err = np.abs(out_acc.numpy() - ref).max()
    ok = "OK" if max_err < 0.02 else "FAIL"
    if max_err >= 0.02:
        all_ok = False
    print(f"  {w}x{h}: max_err={max_err:.2e}  {ok}")

# Step 3: 性能
print("\n=== Step 3: 性能基准 (10 次均值) ===")
for w, h in [(224, 224), (481, 321), (640, 480), (1920, 1080)]:
    pixels = rng.randint(0, 256, (h, w, 3), dtype=np.uint8)
    t = torch.from_numpy(pixels.copy()).contiguous()

    # 预热
    for _ in range(5):
        _ = mod.openvla_fused_preprocess(t, 4)

    t0 = time.perf_counter()
    for _ in range(50):
        _ = mod.openvla_fused_preprocess(t, 4)
    ms = (time.perf_counter() - t0) / 50 * 1000
    print(f"  {w}x{h}: {ms:.3f} ms")

print()
print(f"[{'OK' if all_ok else 'FAIL'}] vLLM torch op 集成验证{'全部通过' if all_ok else '存在差异'}")
sys.exit(0 if all_ok else 1)
