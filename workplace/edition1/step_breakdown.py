#!/usr/bin/env python3
"""
逐过程拆解: 原版 (PIL+NumPy) vs 加速版 (NEON C++) 每一步的准确耗时

用法: python3 step_breakdown.py [--size W H] [--threads T] [--iters N]
输出: 每个子步骤的延迟 (ms) 和占比
"""

import argparse, subprocess, sys, time
import numpy as np
from PIL import Image

# ============================================================================
# 常量
# ============================================================================
IMAGENET_MEAN = np.array([0.484375, 0.455078125, 0.40625], dtype=np.float32)
IMAGENET_STD  = np.array([0.228515625, 0.2236328125, 0.224609375], dtype=np.float32)
SIGLIP_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
SIGLIP_STD  = np.array([0.5, 0.5, 0.5], dtype=np.float32)

C_PROG = "../libpreprocess.so"   # ctypes 直调, 零 subprocess 开销

# ============================================================================
# 原版逐步骤计时
# ============================================================================
def benchmark_original(pixels, iters):
    """逐步骤计时原版流程, 返回每步 ms"""
    results = {}

    # Step 1: to_rgb_image (PIL fromarray + convert)
    t0 = time.perf_counter()
    for _ in range(iters):
        img = Image.fromarray(pixels, mode="RGB")
    results["to_rgb"] = (time.perf_counter() - t0) / iters * 1000

    # 预先 to_rgb (只计一次, 后续复用)
    img = Image.fromarray(pixels, mode="RGB")

    # Step 2: PIL BICUBIC resize
    t0 = time.perf_counter()
    for _ in range(iters):
        resized = img.resize((224, 224), Image.Resampling.BICUBIC)
    results["resize"] = (time.perf_counter() - t0) / iters * 1000

    resized = img.resize((224, 224), Image.Resampling.BICUBIC)

    # Step 3: np.asarray + /255.0
    t0 = time.perf_counter()
    for _ in range(iters):
        raw = np.asarray(resized, dtype=np.float32) / 255.0
    results["asarray_div"] = (time.perf_counter() - t0) / iters * 1000

    raw = np.asarray(resized, dtype=np.float32) / 255.0

    # Step 4: DINOv2 normalize + transpose
    t0 = time.perf_counter()
    for _ in range(iters):
        dino = ((raw - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)
    results["dino_norm"] = (time.perf_counter() - t0) / iters * 1000

    # Step 5: SigLIP normalize + transpose
    t0 = time.perf_counter()
    for _ in range(iters):
        siglip = ((raw - SIGLIP_MEAN) / SIGLIP_STD).transpose(2, 0, 1)
    results["siglip_norm"] = (time.perf_counter() - t0) / iters * 1000

    dino = ((raw - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)
    siglip = ((raw - SIGLIP_MEAN) / SIGLIP_STD).transpose(2, 0, 1)

    # Step 6: concatenate
    t0 = time.perf_counter()
    for _ in range(iters):
        out = np.concatenate([dino, siglip], axis=0)
    results["concat"] = (time.perf_counter() - t0) / iters * 1000

    # Step 7: torch.from_numpy
    out = np.concatenate([dino, siglip], axis=0)
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = __import__("torch").from_numpy(out)
    results["to_torch"] = (time.perf_counter() - t0) / iters * 1000

    return results


# ============================================================================
# 加速版 (ctypes 直调 C++ .so)
# ============================================================================
def benchmark_accelerated(pixels, iters, num_threads):
    """ctypes 直调 C++ .so"""
    import ctypes
    h, w = pixels.shape[:2]
    out = np.empty((6, 224, 224), dtype=np.float32)

    lib = ctypes.CDLL(C_PROG)
    lib.combined_preprocess.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]
    lib.combined_preprocess.restype = None

    # 预热
    for _ in range(min(5, iters)):
        lib.combined_preprocess(
            pixels.ctypes.data_as(ctypes.c_void_p),
            out.ctypes.data_as(ctypes.c_void_p),
            h, w, num_threads,
        )

    t0 = time.perf_counter()
    for _ in range(iters):
        lib.combined_preprocess(
            pixels.ctypes.data_as(ctypes.c_void_p),
            out.ctypes.data_as(ctypes.c_void_p),
            h, w, num_threads,
        )
    ms = (time.perf_counter() - t0) / iters * 1000
    return ms, out


# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", nargs=2, type=int, default=[1920, 1080])
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--iters", type=int, default=200)
    args = parser.parse_args()

    w, h = args.size
    iters = min(args.iters, 200 if w * h > 500000 else 500)

    rng = np.random.RandomState(42)
    pixels = rng.randint(0, 256, (h, w, 3), dtype=np.uint8)

    print(f"=== {w}×{h}, {args.threads} threads, {iters} iters ===\n")

    # 原版
    steps = benchmark_original(pixels, iters)
    total_orig = sum(steps.values())

    # 加速
    acc_ms, acc_out = benchmark_accelerated(pixels, iters, args.threads)

    # 正确性
    img = Image.fromarray(pixels, mode="RGB")
    ref = np.asarray(img.resize((224, 224), Image.Resampling.BICUBIC), dtype=np.float32) / 255.0
    dino = ((ref - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)
    siglip = ((ref - SIGLIP_MEAN) / SIGLIP_STD).transpose(2, 0, 1)
    ref_out = np.concatenate([dino, siglip], axis=0).astype(np.float32)
    max_err = np.abs(ref_out - acc_out).max()

    # 输出
    print(f"{'步骤':<25s} {'原版(ms)':>10s} {'占比':>7s}")
    print("-" * 45)
    names = [
        ("to_rgb",      "① to_rgb_image"),
        ("resize",      "② PIL BICUBIC resize"),
        ("asarray_div", "③ np.asarray + /255.0"),
        ("dino_norm",   "④ DINOv2 normalize+transpose"),
        ("siglip_norm", "⑤ SigLIP normalize+transpose"),
        ("concat",      "⑥ np.concatenate"),
        ("to_torch",    "⑦ torch.from_numpy"),
    ]
    for key, name in names:
        ms = steps[key]
        pct = ms / total_orig * 100
        print(f"{name:<25s} {ms:>10.3f} {pct:>6.1f}%")

    print("-" * 45)
    print(f"{'原版总耗时':<25s} {total_orig:>10.3f}")
    print()
    print(f"{'加速版总耗时':<25s} {acc_ms:>10.3f} ms")
    print(f"{'加速比':<25s} {total_orig/acc_ms:>10.2f}×")
    print(f"{'max_err (float32)':<25s} {max_err:>10.2e}")
    status = "OK" if max_err <= 0.02 else "FAIL"
    print(f"{'正确性':<25s} {status:>10s}")


if __name__ == "__main__":
    main()
