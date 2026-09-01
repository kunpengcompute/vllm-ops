# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ctypes
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import ImageProcessingMixin, ProcessorMixin

IMAGENET_MEAN = np.array([0.484375, 0.455078125, 0.40625], dtype=np.float32)
IMAGENET_STD = np.array([0.228515625, 0.2236328125, 0.224609375], dtype=np.float32)
SIGLIP_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
SIGLIP_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)

# ═══════════════════════════════════════════════════════════════════════════
# NEON kernel 加载 & torch.library 注册
# 在模块导入时完成 (import vllm._C 确保 _C 命名空间已初始化)
# ═══════════════════════════════════════════════════════════════════════════

_neon_available = False
_c_fn = None

try:
    import vllm._C  # noqa: F401  触发 _C.abi3.so 加载
    import torch.library

    # 加载 extern "C" 内核函数
    _so_path = vllm._C.__file__
    _c_lib = ctypes.CDLL(_so_path)
    _c_lib.openvla_fused_preprocess_c.argtypes = [
        ctypes.c_void_p,           # input: uint8_t*
        ctypes.c_int,              # H
        ctypes.c_int,              # W
        ctypes.c_void_p,           # output: float*
        ctypes.c_int64,            # num_threads
    ]
    _c_lib.openvla_fused_preprocess_c.restype = None
    _c_fn = _c_lib.openvla_fused_preprocess_c

    # 注册 torch 算子
    _lib = torch.library.Library("_C", "FRAGMENT")
    _lib.define(
        "openvla_fused_preprocess(Tensor input, int num_threads=4) -> Tensor"
    )

    @torch.library.impl(_lib, "openvla_fused_preprocess", "CPU")
    def _neon_kernel(input: torch.Tensor, num_threads: int = 4) -> torch.Tensor:
        """NEON fused kernel: uint8 [H,W,3] → resize + normalize → float32 [6,224,224]"""
        H, W = int(input.shape[0]), int(input.shape[1])
        input_contig = input.contiguous()
        output = torch.empty(6, 224, 224, dtype=torch.float32)
        _c_fn(input_contig.data_ptr(), H, W, output.data_ptr(), num_threads)
        return output

    _neon_available = True
except Exception:
    pass  # ctypes/torch.library 失败时走 Python fallback


# ═══════════════════════════════════════════════════════════════════════════
# Python fallback (全平台通用)
# ═══════════════════════════════════════════════════════════════════════════

def _openvla_preprocess_python(image: Any, image_size: int) -> torch.Tensor:
    """纯 NumPy 参考实现."""
    rgb_image = image.resize(
        (image_size, image_size), Image.Resampling.BICUBIC,
    )
    raw = np.asarray(rgb_image, dtype=np.float32) / 255.0
    dinov2_pixels = ((raw - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)
    siglip_pixels = ((raw - SIGLIP_MEAN) / SIGLIP_STD).transpose(2, 0, 1)
    pixel_values = np.concatenate([dinov2_pixels, siglip_pixels], axis=0)
    return torch.from_numpy(pixel_values.astype(np.float32))


# ═══════════════════════════════════════════════════════════════════════════
# OpenVLA image processor
# ═══════════════════════════════════════════════════════════════════════════

def to_rgb_image(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if not isinstance(image, np.ndarray):
        raise TypeError(
            "OpenVLA image input must be a PIL image, numpy array, or torch tensor; "
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


def preprocess_openvla_image(image: Any, image_size: int) -> torch.Tensor:
    rgb_image = to_rgb_image(image)

    # ARM NEON fused path: resize + normalize → [6, 224, 224]
    raw = np.asarray(rgb_image, dtype=np.uint8)
    if _neon_available and raw.ndim == 3 and raw.shape[2] == 3:
        input_tensor = torch.from_numpy(raw.copy()).contiguous()
        try:
            return torch.ops._C.openvla_fused_preprocess(input_tensor)
        except Exception:
            pass

    # Python fallback
    return _openvla_preprocess_python(rgb_image, image_size)


class OpenVLAImageProcessor(ImageProcessingMixin):
    def __init__(self, *, image_size: int) -> None:
        self.image_size = image_size

    def __call__(
        self,
        images: Any | None = None,
        **kwargs: object,
    ) -> dict[str, object]:
        if images is None:
            return {}
        if not isinstance(images, Sequence) or isinstance(images, (str, bytes)):
            images = [images]
        if len(images) == 0:
            return {}

        pixel_values = torch.stack(
            [
                preprocess_openvla_image(image, image_size=self.image_size)
                for image in images
            ],
            dim=0,
        )
        return {"pixel_values": pixel_values}


class OpenVLAProcessor(ProcessorMixin):
    def __init__(
        self,
        *,
        image_processor: OpenVLAImageProcessor,
        tokenizer: Any,
    ) -> None:
        self.image_processor = image_processor
        self.tokenizer = tokenizer
