"""共享内存模块，用于管理vLLM实例之间的共享内存通信。"""

from vllm.core.shared_memory.manager import SharedMemoryManager

__all__ = [
    "SharedMemoryManager"
]
