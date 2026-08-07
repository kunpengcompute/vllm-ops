#!/usr/bin/env python3
"""
OpenVLA 图片前处理独立流水线 — 从 vLLM 中提取, 供优化验证使用。

完整流程:
  输入图片 (PIL/NumPy/Torch, 任意尺寸, uint8 或 float)
    │
    ▼ step1: to_rgb_image()
  PIL RGB Image
    │
    ▼ step2: resize (224x224, BICUBIC)  [仅当尺寸不匹配时]
  PIL RGB (224x224)
    │
    ▼ step3: raw = np.asarray(...) / 255.0
  float32 HWC (224,224,3)
    │
    ▼ step4: 双路归一化 + HWC→CHW + concat
    │   ├── dinov2_pixels = (raw - IMAGENET_MEAN) / IMAGENET_STD  → (3,224,224)
    │   └── siglip_pixels = (raw - SIGLIP_MEAN) / SIGLIP_STD      → (3,224,224)
    │
    ▼ step5: np.concatenate([dinov2, siglip], axis=0)
  float32 (6, 224, 224)
    │
    ▼ step6: torch.from_numpy + torch.stack (batching)
  float32 (B, 6, 224, 224)

用法:
  python openvla_preprocess_pipeline.py              # 基准测试
  python openvla_preprocess_pipeline.py --validate   # 正确性验证
  python openvla_preprocess_pipeline.py --profile    # 逐步骤性能分析
  python openvla_preprocess_pipeline.py --batch 8    # 批量测试
"""

from __future__ import annotations

import argparse
import time
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

# ==============================================================================
# 常量: 与 vLLM openvla.py 完全一致
# ==============================================================================

IMAGENET_MEAN = np.array([0.484375, 0.455078125, 0.40625], dtype=np.float32)
IMAGENET_STD = np.array([0.228515625, 0.2236328125, 0.224609375], dtype=np.float32)
SIGLIP_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
SIGLIP_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)

IMAGE_SIZE = 224
PATCH_SIZE = 14
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2  # 256

# 预计算常量, 用于 fused kernel 优化
INV_255 = np.float32(1.0 / 255.0)
IMAGENET_MEAN_DIV_STD = IMAGENET_MEAN / IMAGENET_STD  # (3,)
IMAGENET_STD_INV = np.float32(1.0) / IMAGENET_STD      # (3,)
SIGLIP_MEAN_DIV_STD = SIGLIP_MEAN / SIGLIP_STD          # (3,)
SIGLIP_STD_INV = np.float32(1.0) / SIGLIP_STD            # (3,)


# ==============================================================================
# Step 1: 输入归一化 → PIL RGB
# ==============================================================================

def to_rgb_image(image: Any) -> Image.Image:
    """将任意格式输入转化为 PIL RGB Image。

    对应: vllm/transformers_utils/processors/openvla.py::to_rgb_image()
    """
    if isinstance(image, Image.Image):
        return image if image.mode == "RGB" else image.convert("RGB")

    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if not isinstance(image, np.ndarray):
        raise TypeError(
            f"OpenVLA image input must be a PIL image, numpy array, or torch tensor; "
            f"got {type(image)}"
        )

    if image.ndim != 3:
        raise ValueError(
            f"OpenVLA image input must have 3 dimensions, got shape {image.shape}"
        )

    if image.shape[0] in (1, 3):
        image = np.moveaxis(image, 0, -1)

    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    elif image.shape[-1] != 3:
        raise ValueError(
            f"OpenVLA image input must have 1 or 3 channels, got shape {image.shape}"
        )

    if image.dtype != np.uint8:
        image = image.astype(np.float32)
        if image.max(initial=0.0) <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)

    return Image.fromarray(image).convert("RGB")


# ==============================================================================
# Step 2: 图像缩放 (可选, 仅在尺寸不匹配时)
# ==============================================================================

def resize_if_needed(rgb_image: Image.Image, target_size: int) -> Image.Image:
    """仅在尺寸不匹配时执行 BICUBIC resize。

    对应: openvla.py::preprocess_openvla_image() 的 resize 部分
    """
    if rgb_image.size == (target_size, target_size):
        return rgb_image
    return rgb_image.resize((target_size, target_size), Image.Resampling.BICUBIC)


# ==============================================================================
# Step 3-5: 数值处理 — NumPy Reference 路径
# ==============================================================================

def preprocess_numpy_reference(rgb_image: Image.Image) -> np.ndarray:
    """NumPy 参考实现: resize + 双路归一化 + 6CHW 输出。

    对应: openvla.py::preprocess_openvla_image() — 原始实现, 逐值基准。

    Returns:
        np.ndarray, shape (6, 224, 224), dtype float32
    """
    # Step 2: resize
    image = resize_if_needed(rgb_image, IMAGE_SIZE)

    # Step 3: uint8 → float32 [0,1]
    raw = np.asarray(image, dtype=np.float32) / 255.0  # (224, 224, 3)

    # Step 4: 双路归一化 + HWC → CHW
    dinov2_pixels = ((raw - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)
    siglip_pixels = ((raw - SIGLIP_MEAN) / SIGLIP_STD).transpose(2, 0, 1)

    # Step 5: concat → 6CHW
    return np.concatenate([dinov2_pixels, siglip_pixels], axis=0)


# ==============================================================================
# Step 3-5: 数值处理 — NumPy Fused 路径 (一次遍历, 作为 NEON 的前身)
# ==============================================================================

def preprocess_numpy_fused(rgb_image: Image.Image) -> np.ndarray:
    """NumPy fused 实现: 单次遍历完成所有数值处理。

    此版本用于验证 fused kernel 的数学等价性, 以及作为非 ARM 平台的优化路径。
    避免多次临时数组分配和 3 次完整遍历。
    """
    image = resize_if_needed(rgb_image, IMAGE_SIZE)

    # uint8 HWC → float32 CHW 的 3 个通道 + 两组归一化 → 直接写入 6CHW
    raw = np.asarray(image, dtype=np.float32)  # (224, 224, 3) float32

    out = np.empty((6, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

    # 通道 0-2: DINOv2 归一化
    inv255 = INV_255
    for c in range(3):
        out[c] = (raw[:, :, c] * inv255 - IMAGENET_MEAN[c]) / IMAGENET_STD[c]

    # 通道 3-5: SigLIP 归一化
    for c in range(3):
        out[c + 3] = (raw[:, :, c] * inv255 - SIGLIP_MEAN[c]) / SIGLIP_STD[c]

    return out


# ==============================================================================
# Step 3-5: Native 路径 (ARM NEON — 待实现)
# ==============================================================================

_HAS_NATIVE = False

try:
    if hasattr(torch.ops, "_C") and hasattr(
        getattr(torch.ops, "_C"), "openvla_fused_preprocess"
    ):
        _HAS_NATIVE = True
except Exception:
    pass


def preprocess_native(rgb_image: Image.Image) -> np.ndarray:
    """Native fused kernel 路径 (ARM NEON + OpenMP)。

    Operator 契约:
      - 输入: CPU uint8 tensor [224, 224, 3]
      - 输出: CPU float32 tensor [1, 6, 224, 224]
      - 通道 0-2: DINOv2 归一化
      - 通道 3-5: SigLIP 归一化
    """
    if not _HAS_NATIVE:
        raise RuntimeError(
            "Native kernel 未编译。在 ARM 平台编译 vLLM CPU 扩展后可用。"
        )

    image = resize_if_needed(rgb_image, IMAGE_SIZE)
    raw = np.asarray(image, dtype=np.uint8)
    input_tensor = torch.from_numpy(raw).contiguous()
    output = torch.ops._C.openvla_fused_preprocess(input_tensor)
    return output.numpy().squeeze(0)  # [1, 6, 224, 224] → [6, 224, 224]


# ==============================================================================
# Step 6: 批处理
# ==============================================================================

def batch_process(
    images: Sequence[Any],
    method: str = "reference",
    fast_path: bool = True,
) -> torch.Tensor:
    """完整前处理流水线: 多图 → 批处理 tensor。

    Args:
        images: 输入图片列表 (PIL/NumPy/Torch)
        method: "reference" | "fused" | "native"
        fast_path: 是否启用 PIL RGB 复用和跳过 resize

    Returns:
        torch.Tensor, shape (B, 6, 224, 224), dtype float32
    """
    processors = {
        "reference": preprocess_numpy_reference,
        "fused": preprocess_numpy_fused,
        "native": preprocess_native,
    }
    if method not in processors:
        raise ValueError(f"Unknown method: {method}, choose from {list(processors)}")

    preprocess = processors[method]

    pixel_values = []
    for image in images:
        if fast_path:
            rgb = to_rgb_image(image)
        else:
            # 模拟原始慢路径: 强制 convert + 强制 resize
            rgb = to_rgb_image(image)
            if rgb.mode != "RGB":
                rgb = rgb.convert("RGB")

        result = preprocess(rgb)
        pixel_values.append(torch.from_numpy(result))

    return torch.stack(pixel_values, dim=0)


# ==============================================================================
# 测试图片生成
# ==============================================================================

def make_test_image(size: tuple[int, int] = (224, 224)) -> Image.Image:
    """生成随机测试图片。"""
    return Image.fromarray(
        np.random.RandomState(42).randint(0, 256, (*size, 3), dtype=np.uint8),
        mode="RGB",
    )


def make_test_images(
    num: int = 1,
    sizes: Sequence[tuple[int, int]] | None = None,
) -> list[Image.Image]:
    """生成多张不同尺寸的测试图片。"""
    if sizes is None:
        # 覆盖常见尺寸: 已对齐、更大、更小、非正方形
        sizes = [(224, 224), (640, 480), (1920, 1080), (480, 640), (100, 200)]
    images = []
    for i in range(num):
        h, w = sizes[i % len(sizes)]
        images.append(make_test_image((h, w)))
    return images


# ==============================================================================
# 正确性验证: reference vs fused vs native (逐值对比)
# ==============================================================================

def validate(output_ref: np.ndarray, output_test: np.ndarray,
             name: str, atol: float = 1e-5) -> bool:
    """逐值对比两个输出。"""
    if output_ref.shape != output_test.shape:
        print(f"  [FAIL] {name}: shape 不匹配 "
              f"ref={output_ref.shape} vs test={output_test.shape}")
        return False
    if output_ref.dtype != output_test.dtype:
        print(f"  [FAIL] {name}: dtype 不匹配 "
              f"ref={output_ref.dtype} vs test={output_test.dtype}")
        return False

    max_err = np.abs(output_ref - output_test).max()
    if max_err > atol:
        print(f"  [FAIL] {name}: max_err={max_err:.2e} > atol={atol:.2e}")
        return False
    print(f"  [OK] {name}: shape={output_ref.shape}, dtype={output_ref.dtype}, "
          f"max_err={max_err:.2e}")
    return True


def run_validation() -> bool:
    """完整正确性验证: 所有输入格式 × 所有处理方法。"""
    print("=" * 60)
    print("正确性验证: reference vs fused")
    print("=" * 60)

    test_sizes = [(224, 224), (640, 480), (100, 200)]
    all_pass = True

    for h, w in test_sizes:
        print(f"\n--- 图片尺寸: {h}x{w} ---")

        # PIL Image
        pil_img = make_test_image((h, w))
        ref = preprocess_numpy_reference(pil_img)
        fused = preprocess_numpy_fused(pil_img)
        all_pass &= validate(ref, fused, f"PIL ({h}x{w})")

        # NumPy uint8 HWC
        np_img = np.random.RandomState(123).randint(0, 256, (h, w, 3), dtype=np.uint8)
        rgb_img = to_rgb_image(np_img)
        ref = preprocess_numpy_reference(rgb_img)
        fused = preprocess_numpy_fused(rgb_img)
        all_pass &= validate(ref, fused, f"NumPy uint8 HWC ({h}x{w})")

        # Torch uint8 CHW
        t_img = torch.randint(0, 256, (3, h, w), dtype=torch.uint8)
        rgb_img = to_rgb_image(t_img)
        ref = preprocess_numpy_reference(rgb_img)
        fused = preprocess_numpy_fused(rgb_img)
        all_pass &= validate(ref, fused, f"Torch uint8 CHW ({h}x{w})")

    # 灰度图
    print("\n--- 灰度图 ---")
    gray = np.full((100, 100, 1), 128, dtype=np.uint8)
    rgb_img = to_rgb_image(gray)
    ref = preprocess_numpy_reference(rgb_img)
    fused = preprocess_numpy_fused(rgb_img)
    all_pass &= validate(ref, fused, "灰度图")

    # 浮点 [0,1] 输入
    print("\n--- 浮点 [0,1] 输入 ---")
    fp_img = np.random.RandomState(99).rand(300, 200, 3).astype(np.float32)
    rgb_img = to_rgb_image(fp_img)
    ref = preprocess_numpy_reference(rgb_img)
    fused = preprocess_numpy_fused(rgb_img)
    all_pass &= validate(ref, fused, "float32 [0,1]")

    print("\n" + "=" * 60)
    if all_pass:
        print("全部验证通过!")
    else:
        print("存在验证失败!")
    return all_pass


# ==============================================================================
# 性能分析
# ==============================================================================

class Timer:
    """简单的墙钟计时器。"""

    def __init__(self, name: str = "") -> None:
        self.name = name
        self.elapsed = 0.0

    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args: object) -> None:
        self.elapsed = time.perf_counter() - self.start


def run_benchmark(
    num_images: int = 100,
    batch_size: int = 1,
    image_size: tuple[int, int] = (224, 224),
    method: str = "reference",
) -> dict[str, float]:
    """单次基准测试。

    Returns:
        { "rgb": sec, "resize": sec, "numerical": sec, "batch": sec }
    """
    # 生成图片
    images = []
    for _ in range(num_images):
        images.append(make_test_image(image_size))

    # ---- 基准: 不做任何处理的遍历 ----
    t_total = Timer("total")
    t_total.__enter__()

    # Step 1: to_rgb_image (可能无操作)
    t_rgb = Timer("rgb")
    t_rgb.__enter__()
    rgb_images = [to_rgb_image(img) for img in images]
    t_rgb.__exit__()

    # Step 2: resize
    t_resize = Timer("resize")
    t_resize.__enter__()
    resized = [resize_if_needed(img, IMAGE_SIZE) for img in rgb_images]
    t_resize.__exit__()

    # Step 3-5: 数值处理
    t_num = Timer("numerical")
    t_num.__enter__()
    preprocess_fn = {
        "reference": preprocess_numpy_reference,
        "fused": preprocess_numpy_fused,
        "native": preprocess_native,
    }[method]
    arrays = []
    for img in resized:
        arrays.append(preprocess_fn(img))
    t_num.__exit__()

    # Step 6: batch
    t_batch = Timer("batch")
    t_batch.__enter__()
    tensors = [torch.from_numpy(a) for a in arrays]
    batch = torch.stack(tensors, dim=0)
    t_batch.__exit__()

    t_total.__exit__()

    return {
        "rgb_ms": t_rgb.elapsed * 1000 / num_images,
        "resize_ms": t_resize.elapsed * 1000 / num_images,
        "numerical_ms": t_num.elapsed * 1000 / num_images,
        "batch_ms": t_batch.elapsed * 1000 / num_images,
        "total_ms": t_total.elapsed * 1000 / num_images,
        "throughput_ips": num_images / t_total.elapsed,
        "batch_shape": str(list(batch.shape)),
    }


def run_benchmarks(
    num_images: int = 100,
    warmup: int = 10,
) -> None:
    """完整性能基准: 多种图片尺寸 × 两种方法。"""
    print("=" * 70)
    print("性能基准测试")
    print("=" * 70)

    methods = ["reference", "fused"]
    sizes = [(224, 224), (640, 480), (1920, 1080)]

    # Warmup
    print(f"\n预热 {warmup} 轮...")
    warmup_img = make_test_image((224, 224))
    for _ in range(warmup):
        preprocess_numpy_reference(warmup_img)
        preprocess_numpy_fused(warmup_img)

    for h, w in sizes:
        print(f"\n{'─' * 70}")
        print(f"图片尺寸: {h}x{w}  |  样本数: {num_images}")
        print(f"{'─' * 70}")
        print(f"{'方法':<12} {'RGB转换':>8}  {'Resize':>8}  "
              f"{'数值处理':>8}  {'Batch':>8}  {'总计':>8}  {'吞吐量':>10}")
        print(f"{'':─<12} {'':─>8}  {'':─>8}  {'':─>8}  {'':─>8}  {'':─>8}  {'':─>10}")

        for method in methods:
            stats = run_benchmark(
                num_images=num_images,
                image_size=(h, w),
                method=method,
            )
            print(
                f"{method:<12} "
                f"{stats['rgb_ms']:7.3f}ms "
                f"{stats['resize_ms']:7.3f}ms "
                f"{stats['numerical_ms']:7.3f}ms "
                f"{stats['batch_ms']:7.3f}ms "
                f"{stats['total_ms']:7.3f}ms "
                f"{stats['throughput_ips']:8.1f} img/s"
            )

    # 快速路径效果
    print(f"\n{'─' * 70}")
    print("快速路径效果 (224x224 已对齐图片, 跳过 RGB 转换和 resize)")
    print(f"{'─' * 70}")

    images_224 = make_test_images(num_images, sizes=[(224, 224)] * num_images)
    t_fast = Timer("fast_path")
    t_fast.__enter__()
    for img in images_224:
        rgb = to_rgb_image(img)
        _ = preprocess_numpy_reference(rgb)
    t_fast.__exit__()

    images_var = make_test_images(
        num_images,
        sizes=[(640, 480), (1920, 1080), (480, 640), (100, 200)],
    )
    t_slow = Timer("slow_path")
    t_slow.__enter__()
    for img in images_var:
        rgb = to_rgb_image(img)
        rgb = rgb.convert("RGB") if rgb.mode != "RGB" else rgb
        _ = preprocess_numpy_reference(rgb)
    t_slow.__exit__()

    print(f"  快速路径 (224x224): {t_fast.elapsed * 1000 / num_images:.3f} ms/img")
    print(f"  通用路径 (混合尺寸): {t_slow.elapsed * 1000 / num_images:.3f} ms/img")
    print(f"  加速比: {t_slow.elapsed / t_fast.elapsed:.2f}x")


# ==============================================================================
# 逐步骤性能分析 (profile)
# ==============================================================================

def run_profile() -> None:
    """单张图片逐步骤微基准。"""
    img = make_test_image((1920, 1080))  # 典型高清输入

    print("=" * 60)
    print("单张图片逐步骤性能分析 (1920x1080 → 224x224)")
    print("=" * 60)

    # Step 1
    for _ in range(10):
        to_rgb_image(img)
    t1 = Timer("to_rgb")
    t1.__enter__()
    for _ in range(100):
        rgb = to_rgb_image(img)
    t1.__exit__()
    print(f"  to_rgb_image:       {t1.elapsed / 100 * 1000:.4f} ms")

    # Step 2
    rgb = to_rgb_image(img)
    for _ in range(10):
        resize_if_needed(rgb, IMAGE_SIZE)
    t2 = Timer("resize")
    t2.__enter__()
    for _ in range(100):
        resized = resize_if_needed(rgb, IMAGE_SIZE)
    t2.__exit__()
    print(f"  resize (BICUBIC):   {t2.elapsed / 100 * 1000:.4f} ms")

    # Step 3: np.asarray + div 255
    resized = resize_if_needed(rgb, IMAGE_SIZE)
    for _ in range(10):
        np.asarray(resized, dtype=np.float32) / 255.0
    t3 = Timer("asarray+div")
    t3.__enter__()
    for _ in range(500):
        raw = np.asarray(resized, dtype=np.float32) / 255.0
    t3.__exit__()
    print(f"  asarray + /255:      {t3.elapsed / 500 * 1000:.4f} ms")

    # Step 4: 双路归一化
    raw = np.asarray(resized, dtype=np.float32) / 255.0
    for _ in range(10):
        ((raw - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)
    t4 = Timer("normalize")
    t4.__enter__()
    for _ in range(500):
        d = ((raw - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)
        s = ((raw - SIGLIP_MEAN) / SIGLIP_STD).transpose(2, 0, 1)
    t4.__exit__()
    print(f"  双路归一化 (ref):   {t4.elapsed / 500 * 1000:.4f} ms")

    # Step 4-fused
    for _ in range(10):
        preprocess_numpy_fused(resized)
    t5 = Timer("fused")
    t5.__enter__()
    for _ in range(500):
        preprocess_numpy_fused(resized)
    t5.__exit__()
    print(f"  双路归一化 (fused):  {t5.elapsed / 500 * 1000:.4f} ms")

    # Step 5: concat
    for _ in range(10):
        d = ((raw - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)
        s = ((raw - SIGLIP_MEAN) / SIGLIP_STD).transpose(2, 0, 1)
        np.concatenate([d, s], axis=0)
    t6 = Timer("concat")
    t6.__enter__()
    for _ in range(500):
        d = ((raw - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)
        s = ((raw - SIGLIP_MEAN) / SIGLIP_STD).transpose(2, 0, 1)
        np.concatenate([d, s], axis=0)
    t6.__exit__()
    print(f"  concat (ref 含 norm): {t6.elapsed / 500 * 1000:.4f} ms")

    # Step 6: torch
    arr = preprocess_numpy_reference(resized)
    for _ in range(10):
        torch.from_numpy(arr)
    t7 = Timer("torch")
    t7.__enter__()
    for _ in range(1000):
        torch.from_numpy(arr)
    t7.__exit__()
    print(f"  torch.from_numpy:    {t7.elapsed / 1000 * 1000:.4f} ms")

    # 总览对比
    print(f"\n  ── 总览 (参考路径 vs fused) ──")
    for _ in range(10):
        preprocess_numpy_reference(resized)
    t_ref = Timer("ref_total")
    t_ref.__enter__()
    for _ in range(500):
        preprocess_numpy_reference(resized)
    t_ref.__exit__()

    for _ in range(10):
        preprocess_numpy_fused(resized)
    t_fused = Timer("fused_total")
    t_fused.__enter__()
    for _ in range(500):
        preprocess_numpy_fused(resized)
    t_fused.__exit__()

    print(f"  reference (数值部分): {t_ref.elapsed / 500 * 1000:.4f} ms")
    print(f"  fused (数值部分):     {t_fused.elapsed / 500 * 1000:.4f} ms")
    print(f"  加速比:               {t_ref.elapsed / t_fused.elapsed:.2f}x")


# ==============================================================================
# Main CLI
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="OpenVLA 图片前处理独立流水线 — 优化验证工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python openvla_preprocess_pipeline.py                  # 快速基准 (默认)
  python openvla_preprocess_pipeline.py --validate       # 正确性验证
  python openvla_preprocess_pipeline.py --profile        # 逐步骤性能
  python openvla_preprocess_pipeline.py --bench --images 500  # 大规模基准
        """,
    )
    parser.add_argument("--validate", action="store_true",
                        help="运行正确性验证 (reference vs fused)")
    parser.add_argument("--profile", action="store_true",
                        help="单张图片逐步骤性能分析")
    parser.add_argument("--bench", action="store_true",
                        help="完整基准测试 (多种尺寸对比)")
    parser.add_argument("--images", type=int, default=100,
                        help="基准测试的图片数量 (默认: 100)")
    parser.add_argument("--method", choices=["reference", "fused", "native"],
                        default="reference",
                        help="处理方法 (默认: reference)")

    args = parser.parse_args()

    # 如果没有指定任何选项, 默认运行快速基准
    if not (args.validate or args.profile or args.bench):
        args.bench = True

    if args.validate:
        ok = run_validation()
        if not ok:
            sys.exit(1)

    if args.profile:
        run_profile()

    if args.bench:
        run_benchmarks(num_images=args.images)


if __name__ == "__main__":
    main()
