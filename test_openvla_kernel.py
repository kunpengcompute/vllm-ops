#!/usr/bin/env python3
"""
OpenVLA NEON Kernel — 端到端测试脚本

用法:
    python3 test_openvla_kernel.py          # 基本测试
    python3 test_openvla_kernel.py --bench  # 性能对比

在 vLLM 安装后运行此脚本验证 NEON kernel 是否正确部署。
"""

import sys
import time
from typing import Any

import numpy as np

# ═══════════════════════════════════════════════════════════════════════════
# 测试辅助
# ═══════════════════════════════════════════════════════════════════════════

PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        print(f"  [PASS] {name} {detail}")
        PASS += 1
    else:
        print(f"  [FAIL] {name} {detail}")
        FAIL += 1


def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ═══════════════════════════════════════════════════════════════════════════
# 测试 1: 算子注册
# ═══════════════════════════════════════════════════════════════════════════

def test_op_registration() -> bool:
    """检查 torch.ops._C.openvla_fused_preprocess 是否已注册."""
    section("Test 1/4: Op Registration")

    try:
        import torch
        import vllm  # noqa: F401
        from vllm.transformers_utils.processors import openvla  # noqa: F401
    except ImportError as e:
        check("Import vllm", False, str(e))
        return False

    check("Import vllm", True)

    has_op = hasattr(torch.ops._C, "openvla_fused_preprocess")
    check("torch.ops._C.openvla_fused_preprocess exists", has_op)

    if has_op:
        op = torch.ops._C.openvla_fused_preprocess
        check(f"Op callable: {op}", callable(op))

        # 检查 openvla 模块中的标志
        neon_ok = getattr(openvla, "_neon_available", False)
        check("_neon_available flag", neon_ok)
        return neon_ok

    return False


# ═══════════════════════════════════════════════════════════════════════════
# 测试 2: 端到端功能
# ═══════════════════════════════════════════════════════════════════════════

def test_end_to_end() -> None:
    """端到端测试: 输入图片 → 输出 tensor."""
    section("Test 2/4: End-to-End")

    from PIL import Image
    from vllm.transformers_utils.processors.openvla import (
        preprocess_openvla_image,
    )

    # 测试不同尺寸的图片
    test_sizes = [
        (224, 224),   # 无需 resize
        (100, 150),   # 需要 resize
        (480, 640),   # 大图 resize
        (50, 200),    # 窄图
    ]

    for h, w in test_sizes:
        img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        pil_img = Image.fromarray(img)
        try:
            result = preprocess_openvla_image(pil_img, 224)
            ok = (
                isinstance(result, type(__import__("torch").empty(0)))
                and tuple(result.shape) == (6, 224, 224)
                and str(result.dtype) == "torch.float32"
            )
            check(f"Input [{h},{w},3] → [{result.shape[0]},{result.shape[1]},{result.shape[2]}]", ok)
        except Exception as e:
            check(f"Input [{h},{w},3]", False, str(e))


# ═══════════════════════════════════════════════════════════════════════════
# 测试 3: 与 NumPy 参考实现精度对比
# ═══════════════════════════════════════════════════════════════════════════

def test_numpy_cross_check() -> None:
    """对比 NEON kernel 输出与 NumPy 参考实现的精度."""
    section("Test 3/4: NumPy Cross-Check")

    import torch
    from PIL import Image

    IMAGENET_MEAN = np.array([0.484375, 0.455078125, 0.40625], dtype=np.float32)
    IMAGENET_STD = np.array([0.228515625, 0.2236328125, 0.224609375], dtype=np.float32)
    SIGLIP_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    SIGLIP_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)

    def numpy_reference(pil_image: Any) -> np.ndarray:
        """NumPy 参考实现, 与 openvla_preprocess_pipeline.py 一致."""
        img = pil_image.resize((224, 224), Image.Resampling.BICUBIC)
        raw = np.asarray(img, dtype=np.float32) / 255.0
        dinov2 = ((raw - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)
        siglip = ((raw - SIGLIP_MEAN) / SIGLIP_STD).transpose(2, 0, 1)
        return np.concatenate([dinov2, siglip], axis=0)

    test_cases = [
        np.random.RandomState(42).randint(0, 256, (100, 150, 3), dtype=np.uint8),
        np.random.RandomState(42).randint(0, 256, (224, 224, 3), dtype=np.uint8),
        np.random.RandomState(42).randint(0, 256, (300, 400, 3), dtype=np.uint8),
        np.zeros((50, 80, 3), dtype=np.uint8),
        np.full((200, 100, 3), 255, dtype=np.uint8),
    ]

    max_diff = 0.0
    all_pass = True

    for i, img in enumerate(test_cases):
        pil_img = Image.fromarray(img)

        # NumPy 参考
        ref = numpy_reference(pil_img)

        # NEON kernel (通过 torch op)
        input_t = torch.from_numpy(img.copy()).contiguous()
        try:
            neon = torch.ops._C.openvla_fused_preprocess(input_t).numpy()
        except AttributeError:
            check(f"Case {i}: NEON op not available", False)
            all_pass = False
            continue

        diff = np.abs(neon - ref)
        case_max = float(diff.max())
        max_diff = max(max_diff, case_max)

        # resize 精度容限: BICUBIC 涉及浮点累加, 允许 ~0.02 差异
        ok = case_max < 0.05
        if not ok:
            all_pass = False
        check(
            f"Case {i} ({img.shape[0]}×{img.shape[1]})",
            ok,
            f"max_diff={case_max:.6f}",
        )

    check(
        "Overall cross-check",
        all_pass,
        f"worst max_diff={max_diff:.6f} " + ("OK" if max_diff < 0.05 else "FAIL"),
    )


# ═══════════════════════════════════════════════════════════════════════════
# 测试 4: Python fallback 路径
# ═══════════════════════════════════════════════════════════════════════════

def test_fallback() -> None:
    """验证 NEON kernel 不可用时 Python fallback 正常工作."""
    section("Test 4/4: Python Fallback")

    from PIL import Image
    from vllm.transformers_utils.processors.openvla import (
        _openvla_preprocess_python,
    )

    img = np.random.RandomState(123).randint(0, 256, (100, 150, 3), dtype=np.uint8)
    pil_img = Image.fromarray(img)

    result = _openvla_preprocess_python(pil_img, 224)
    ok = (
        isinstance(result, type(__import__("torch").empty(0)))
        and tuple(result.shape) == (6, 224, 224)
        and str(result.dtype) == "torch.float32"
    )
    check("Python fallback produces correct shape", ok)

    # 检查输出范围 (归一化后应在 ~[-2, 2] 范围内)
    arr = result.numpy()
    check(
        "Output range reasonable",
        float(arr.min()) > -3.0 and float(arr.max()) < 3.0,
        f"range=[{arr.min():.3f}, {arr.max():.3f}]",
    )


# ═══════════════════════════════════════════════════════════════════════════
# 性能基准 (可选)
# ═══════════════════════════════════════════════════════════════════════════

def bench() -> None:
    """NEON vs NumPy 性能对比."""
    section("Benchmark: NEON vs NumPy")

    import torch
    from PIL import Image
    from vllm.transformers_utils.processors.openvla import (
        _openvla_preprocess_python,
    )

    img = np.random.RandomState(99).randint(0, 256, (480, 640, 3), dtype=np.uint8)

    # NumPy warmup
    for _ in range(5):
        _openvla_preprocess_python(Image.fromarray(img), 224)
    t0 = time.perf_counter()
    for _ in range(50):
        _openvla_preprocess_python(Image.fromarray(img), 224)
    numpy_time = (time.perf_counter() - t0) / 50

    # NEON warmup
    input_t = torch.from_numpy(img.copy()).contiguous()
    for _ in range(5):
        torch.ops._C.openvla_fused_preprocess(input_t)
    t0 = time.perf_counter()
    for _ in range(50):
        torch.ops._C.openvla_fused_preprocess(input_t)
    neon_time = (time.perf_counter() - t0) / 50

    speedup = numpy_time / neon_time if neon_time > 0 else float("inf")
    print(f"  NumPy: {numpy_time*1000:.2f} ms/image")
    print(f"  NEON:  {neon_time*1000:.2f} ms/image")
    print(f"  Speedup: {speedup:.1f}×")


# ═══════════════════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 60)
    print("  OpenVLA NEON Kernel — Test Suite")
    print("=" * 60)

    # 测试 1
    neon_ok = test_op_registration()

    # 测试 2
    test_end_to_end()

    # 测试 3 (需要 NEON)
    if neon_ok:
        test_numpy_cross_check()
    else:
        section("Test 3/4: NumPy Cross-Check")
        print("  [SKIP] NEON kernel not available, running in Python mode")

    # 测试 4
    test_fallback()

    # 基准 (可选)
    if "--bench" in sys.argv and neon_ok:
        bench()
    elif "--bench" in sys.argv:
        section("Benchmark")
        print("  [SKIP] NEON kernel not available")

    # 结果
    section("Results")
    total = PASS + FAIL
    print(f"  Passed: {PASS}/{total}")
    if FAIL > 0:
        print(f"  Failed: {FAIL}/{total}")
        sys.exit(1)
    else:
        print("  All tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
