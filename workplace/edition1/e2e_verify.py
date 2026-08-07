#!/usr/bin/env python3
"""
端到端正确性验证: 加速版 vs 原版, 多种尺寸, 确定性种子

用法: python3 e2e_verify.py
"""

import ctypes, sys
import numpy as np
from PIL import Image

IMAGENET_MEAN = np.array([0.484375, 0.455078125, 0.40625], dtype=np.float32)
IMAGENET_STD  = np.array([0.228515625, 0.2236328125, 0.224609375], dtype=np.float32)
SIGLIP_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
SIGLIP_STD  = np.array([0.5, 0.5, 0.5], dtype=np.float32)

SO_PATH = "../libpreprocess.so"

SIZES = [
    (224, 224, "等比例"),
    (100, 200, "上采样"),
    (150, 300, "上采样2"),
    (400, 300, "小图"),
    (481, 321, "数据集"),
    (640, 480, "中等"),
    (800, 600, "中等2"),
    (1280, 720, "720p"),
    (1920, 1080, "1080p"),
    (2560, 1440, "1440p"),
]


def reference(pixels):
    img = Image.fromarray(pixels, mode="RGB")
    r = img.resize((224, 224), Image.Resampling.BICUBIC)
    raw = np.asarray(r, dtype=np.float32) / 255.0
    dino = ((raw - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)
    siglip = ((raw - SIGLIP_MEAN) / SIGLIP_STD).transpose(2, 0, 1)
    return np.concatenate([dino, siglip], axis=0).astype(np.float32)


def accelerated(pixels, nthreads):
    h, w = pixels.shape[:2]
    out = np.empty((6, 224, 224), dtype=np.float32)
    lib = ctypes.CDLL(SO_PATH)
    lib.combined_preprocess.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.combined_preprocess.restype = None
    lib.combined_preprocess(
        pixels.ctypes.data_as(ctypes.c_void_p),
        out.ctypes.data_as(ctypes.c_void_p),
        h, w, nthreads)
    return out


def main():
    rng = np.random.RandomState(42)
    all_ok = True

    print(f"{'尺寸':>14s}  {'max_err':>10s}  {'err>0.01':>9s}  "
          f"{'DINOv2 max':>11s}  {'SigLIP max':>11s}  {'结果':>5s}")
    print("-" * 72)

    for w, h, label in SIZES:
        pixels = rng.randint(0, 256, (h, w, 3), dtype=np.uint8)
        ref = reference(pixels)
        acc = accelerated(pixels, 8)

        diff = np.abs(ref - acc)
        max_err = diff.max()
        n_big = int((diff > 0.01).sum())

        # 逐通道最大误差
        dino_max = diff[:3].max()
        siglip_max = diff[3:].max()

        ok = max_err <= 0.02
        if not ok:
            all_ok = False

        print(f"{label:>8s} {w}×{h:<6d}  {max_err:>10.2e}  {n_big:>9d}  "
              f"{dino_max:>11.2e}  {siglip_max:>11.2e}  {'OK':>5s}" if ok
              else f"{'FAIL':>5s}")

    print("-" * 72)
    print(f"\n{'[OK] 全部通过' if all_ok else '[FAIL] 存在差异'}")
    print(f"容忍度: max_err ≤ 0.02 (≈ 1-pixel uint8 × scale 因子)")
    return all_ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
