#!/usr/bin/env python3
"""
多尺寸 + 多线程完整基准: 原版 vs NEON 加速版
输出 Markdown 表格, 直接可嵌入技术报告

用法: python3 bench_all.py
"""

import ctypes, sys, time
import numpy as np
from PIL import Image

IMAGENET_MEAN = np.array([0.484375, 0.455078125, 0.40625], dtype=np.float32)
IMAGENET_STD  = np.array([0.228515625, 0.2236328125, 0.224609375], dtype=np.float32)
SIGLIP_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
SIGLIP_STD  = np.array([0.5, 0.5, 0.5], dtype=np.float32)

SO_PATH = "../libpreprocess.so"

SIZES = [
    (224, 224, "224×224 (等比例)"),
    (100, 200, "100×200 (上采样)"),
    (481, 321, "481×321 (数据集)"),
    (640, 480, "640×480"),
    (1920, 1080, "1920×1080 (大图)"),
]

THREADS = [1, 2, 4, 8, 16, 32]


def original_preprocess(pixels, iters):
    img = Image.fromarray(pixels, mode="RGB")
    t0 = time.perf_counter()
    for _ in range(iters):
        r = img.resize((224, 224), Image.Resampling.BICUBIC)
        raw = np.asarray(r, dtype=np.float32) / 255.0
        dino = ((raw - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)
        siglip = ((raw - SIGLIP_MEAN) / SIGLIP_STD).transpose(2, 0, 1)
        _ = np.concatenate([dino, siglip], axis=0).astype(np.float32)
    return (time.perf_counter() - t0) / iters * 1000


def accelerated_preprocess(pixels, iters, num_threads):
    h, w = pixels.shape[:2]
    out = np.empty((6, 224, 224), dtype=np.float32)

    lib = ctypes.CDLL(SO_PATH)
    lib.combined_preprocess.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.combined_preprocess.restype = None

    ptr_in = pixels.ctypes.data_as(ctypes.c_void_p)
    ptr_out = out.ctypes.data_as(ctypes.c_void_p)

    for _ in range(5):
        lib.combined_preprocess(ptr_in, ptr_out, h, w, num_threads)

    t0 = time.perf_counter()
    for _ in range(iters):
        lib.combined_preprocess(ptr_in, ptr_out, h, w, num_threads)
    return (time.perf_counter() - t0) / iters * 1000, out


def main():
    rng = np.random.RandomState(42)

    # 表头
    print("| 尺寸 | 原版(ms) | t=1(ms) | t=4(ms) | t=8(ms) | t=16(ms) | t=32(ms) | "
          "加速比(16t) | max_err |")
    print("|------|----------|---------|---------|---------|----------|----------|"
          "------------|---------|")

    for w, h, label in SIZES:
        pixels = rng.randint(0, 256, (h, w, 3), dtype=np.uint8)
        iters = 50 if w * h > 1000000 else (100 if w * h > 200000 else 200)

        orig_ms = original_preprocess(pixels, iters)
        neon_ms = {}
        max_err = 0

        for t in THREADS:
            ms, out = accelerated_preprocess(pixels, iters, t)
            neon_ms[t] = ms

            # 正确性 (只做一次)
            if t == THREADS[0]:
                img = Image.fromarray(pixels, mode="RGB")
                r = img.resize((224, 224), Image.Resampling.BICUBIC)
                raw = np.asarray(r, dtype=np.float32) / 255.0
                dino = ((raw - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)
                siglip = ((raw - SIGLIP_MEAN) / SIGLIP_STD).transpose(2, 0, 1)
                ref = np.concatenate([dino, siglip], axis=0).astype(np.float32)
                max_err = np.abs(ref - out).max()

        best_t = 16
        speedup = orig_ms / neon_ms[best_t]
        print(f"| {label:<12s} | {orig_ms:>8.3f} | {neon_ms[1]:>7.3f} | "
              f"{neon_ms[4]:>7.3f} | {neon_ms[8]:>7.3f} | "
              f"{neon_ms[16]:>8.3f} | {neon_ms[32]:>8.3f} | "
              f"{speedup:>10.1f}× | {max_err:>7.2e} |")

    # 分解表
    print()
    print("### 1920×1080 逐步骤拆解")
    print()
    print("| 步骤 | 原版(ms) | 占比 | 优化方式 |")
    print("|------|----------|------|----------|")
    pixels = rng.randint(0, 256, (1080, 1920, 3), dtype=np.uint8)

    img = Image.fromarray(pixels, mode="RGB")
    iters = 50

    t0 = time.perf_counter()
    for _ in range(iters): _ = img.resize((224, 224), Image.Resampling.BICUBIC)
    t_resize = (time.perf_counter() - t0) / iters * 1000

    r = img.resize((224, 224), Image.Resampling.BICUBIC)
    t0 = time.perf_counter()
    for _ in range(iters): raw = np.asarray(r, dtype=np.float32) / 255.0
    t_div = (time.perf_counter() - t0) / iters * 1000

    raw = np.asarray(r, dtype=np.float32) / 255.0
    t0 = time.perf_counter()
    for _ in range(iters): d = ((raw - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)
    t_dino = (time.perf_counter() - t0) / iters * 1000

    t0 = time.perf_counter()
    for _ in range(iters): s = ((raw - SIGLIP_MEAN) / SIGLIP_STD).transpose(2, 0, 1)
    t_sig = (time.perf_counter() - t0) / iters * 1000

    d = ((raw - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)
    s = ((raw - SIGLIP_MEAN) / SIGLIP_STD).transpose(2, 0, 1)
    t0 = time.perf_counter()
    for _ in range(iters): _ = np.concatenate([d, s], axis=0)
    t_cat = (time.perf_counter() - t0) / iters * 1000

    total = t_resize + t_div + t_dino + t_sig + t_cat

    steps = [
        ("PIL BICUBIC resize", t_resize, "OMP 多线程 + RGB stride=3 + float32"),
        ("asarray + /255", t_div, "合并进 normalize, 零开销"),
        ("DINOv2 normalize", t_dino, "NEON vld3q+FMA, 预计算 scale/offset"),
        ("SigLIP normalize", t_sig, "同 DINOv2, 6通道一次遍历"),
        ("concatenate", t_cat, "C++ 直接写 6 平面, 零开销"),
    ]
    for name, ms, opt in steps:
        print(f"| {name:<25s} | {ms:>8.3f} | {ms/total*100:>5.1f}% | {opt:<45s} |")
    print(f"| {'**总计**':<25s} | **{total:>7.3f}** | | |")


if __name__ == "__main__":
    main()
