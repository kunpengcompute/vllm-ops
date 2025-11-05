"""共享内存管理器模块，用于在GPU和CPU vLLM实例之间共享KV缓存信息。"""

import contextlib
import json
import time
from multiprocessing import shared_memory
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from vllm.logger import init_logger

logger = init_logger(__name__)


class SharedMemoryManager:
    """
    管理vLLM实例之间的共享内存通信

    此类负责处理GPU和CPU两个独立vLLM实例之间的通信和数据共享，
    主要用于传递KV缓存的逻辑块与物理块映射信息，实现跨实例接力式推理。

    支持四种不同运行环境：
    1. GPU容器：负责创建请求映射表，作为服务端
    2. CPU容器：连接请求映射表，作为客户端
    3. GPU worker：使用GPU容器创建的映射表，export缓存
    4. CPU worker：使用CPU容器连接的映射表，import缓存
    """

    # 共享内存区块名称前缀
    SHM_PREFIX = "vllm_shared"

    # 物理块和逻辑块类型前缀
    LOGICAL_BLOCK_PREFIX = "logic"
    PHYSICAL_BLOCK_PREFIX = "phys"
    EXECUTE_MODEL_REQ_PREFIX = "emr"

    # 保存请求ID到共享内存区名称的映射
    REQUEST_MAP_NAME = f"{SHM_PREFIX}_request_map"

    def __init__(self, is_gpu_instance: Optional[bool] = None) -> None:
        """
        初始化共享内存管理器

        Args:
            is_gpu_instance: 是否为GPU主实例
                - True: 作为GPU主实例（服务端），创建映射表
                - False: 作为CPU主实例（客户端），连接映射表
                - None: 作为worker/测试/其他，连接映射表

        """
        self.is_gpu_instance = is_gpu_instance
        self.request_map_shm = None

        self._init_request_map()
        if is_gpu_instance is True:
            log_env = "GPU主实例（创建映射表）"
        elif is_gpu_instance is False:
            log_env = "CPU主实例（连接映射表）"
        else:
            log_env = "Worker/测试/其他（连接映射表）"
        logger.info(f"共享内存管理器初始化完成，当前模式: {log_env}")

    # ===== 初始化方法 =====
    def _init_request_map(self) -> None:
        """初始化或连接到请求映射表共享内存，使用辅助方法进行创建或重置"""
        try:
            self._create_request_map_shm()
        except FileExistsError:
            try:
                self._connect_or_reset_request_map_shm()
            except OSError as e:
                logger.warning(f"连接请求映射表失败: {type(e).__name__}({e})，跳过共享内存映射表")
                self.request_map_shm = None
        except OSError:
            logger.exception("创建或连接请求映射表失败")
            self.request_map_shm = None

    def _create_request_map_shm(self) -> None:
        """创建新的请求映射表共享内存并保存启动时间"""
        self.request_map_shm = shared_memory.SharedMemory(
            name=self.REQUEST_MAP_NAME,
            create=True,
            size=10 * 1024 * 1024,
        )
        init_time = time.time()
        initial_map = {"__init_time__": init_time}
        self._save_to_shm(self.request_map_shm, initial_map)
        logger.info(f"创建请求映射表: {self.REQUEST_MAP_NAME}，启动时间: {init_time}")

    def _is_request_map_stale(self, data: Dict[str, Any]) -> bool:
        """判断请求映射表是否超过10分钟"""
        init_time = data.get("__init_time__")
        return init_time is None or time.time() - init_time > 2 * 60

    def _connect_or_reset_request_map_shm(self) -> None:
        """连接到现有请求映射表，若过期则清理并重置"""
        existing_shm = shared_memory.SharedMemory(
            name=self.REQUEST_MAP_NAME, create=False)
        data = self._load_from_shm(existing_shm)
        if self._is_request_map_stale(data):
            # 清理旧请求的共享内存
            for req_id in data.keys():
                if req_id == "__init_time__":
                    continue
                self.cleanup_request(req_id)
            existing_shm.close()
            existing_shm.unlink()
            self._create_request_map_shm()
        else:
            self.request_map_shm = existing_shm
            init_time = data.get("__init_time__")
            logger.info(
                f"成功连接到现有请求映射表: {self.REQUEST_MAP_NAME}，启动时间: {init_time}")

    # ===== 通用工具方法 =====
    def _save_to_shm(self, shm: shared_memory.SharedMemory, data: Any) -> None:
        """将数据保存到共享内存中"""
        # 处理numpy数据类型，确保JSON可以序列化
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(convert_numpy_types(item) for item in obj)
            elif hasattr(obj, 'item'):  # numpy数值类型
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        converted_data = convert_numpy_types(data)
        serialized = json.dumps(converted_data).encode()
        assert len(
            serialized) <= shm.size, f"数据大小 ({len(serialized)} 字节) 超过共享内存大小 ({shm.size} 字节)"

        shm.buf[:] = b"\0" * shm.size
        shm.buf[: len(serialized)] = serialized

    def _load_from_shm(self, shm: shared_memory.SharedMemory) -> Any:
        """从共享内存加载数据"""
        i = 0
        while i < shm.size and shm.buf[i] != 0:
            i += 1
        data = shm.buf[:i].tobytes().decode()
        return json.loads(data)

    def get_logical_block_name(self, request_id: str) -> str:
        """生成逻辑块的共享内存名称"""
        return f"{self.SHM_PREFIX}_{request_id}_{self.LOGICAL_BLOCK_PREFIX}"

    def get_physical_block_name(self, request_id: str, block_id: int, tp_rank: int = 0) -> str:
        """生成物理块的共享内存名称"""
        return f"{self.SHM_PREFIX}_{request_id}_{self.PHYSICAL_BLOCK_PREFIX}_{block_id}_rank_{tp_rank}"

    # ===== 上传方法 =====
    def store_logical_block(self, request_id: str, kv_cache_data: Dict[str, Any]) -> bool:
        """
        将逻辑块数据存储到共享内存中

        Args:
            request_id: 请求ID，用作共享内存的唯一标识
            kv_cache_data: KV缓存数据，包含序列的KV缓存和相关元数据

        Returns:
            bool: 是否成功存储

        """
        logger.debug(f"正在存储id: {request_id} 的逻辑块数据")

        physical_mapping = kv_cache_data.get("physical_block_id_mapping", {})
        kv_cache_data["physical_block_id_mapping"] = physical_mapping

        # 生成逻辑块的共享内存名称并删除可能存在的旧共享内存
        kv_shm_name = self.get_logical_block_name(request_id)
        self._remove_existing_shm(kv_shm_name)

        # 序列化KV缓存数据
        serialized = json.dumps(kv_cache_data).encode()
        # 预留额外空间存放"physical_blocks_meta"
        size = len(serialized) + 5 * 1024 * 1024

        shm = None
        try:
            shm = shared_memory.SharedMemory(
                name=kv_shm_name, create=True, size=size)
            self._save_to_shm(shm, kv_cache_data)
        except OSError:
            logger.exception("创建共享内存时出错")
            return False
        finally:
            if shm is not None:
                shm.close()

        # 更新请求映射表
        request_map = self._load_from_shm(self.request_map_shm)
        if request_id not in request_map:
            request_map[request_id] = {}

        request_map[request_id].update(
            {
                "kv_shm_name": kv_shm_name,
                "timestamp": time.time(),
                "status": "kv_ready",
                "kv_cache_size": len(serialized),
            },
        )
        self._save_to_shm(self.request_map_shm, request_map)
        return True

    def set_physical_blocks_meta(self, request_id: str, meta_list: List[Dict[str, Any]], tp_world_size: int = 1) -> bool:
        """
        追加写入物理块元数据到已有 KV 缓存共享内存

        Args:
            request_id: 请求ID
            meta_list: 物理块元数据列表

        Returns:
            bool: 是否成功写入

        """
        shm = None
        try:
            # 从请求映射表获取共享内存名称
            request_map = self._load_from_shm(self.request_map_shm)
            if request_id not in request_map or "kv_shm_name" not in request_map[request_id]:
                logger.warning(f"无法找到请求 {request_id} 的 KV 缓存共享内存名称，跳过写入物理块元数据")
                return False

            kv_shm_name = request_map[request_id]["kv_shm_name"]

            shm = shared_memory.SharedMemory(name=kv_shm_name, create=False)
            kv_cache_data = self._load_from_shm(shm)
            kv_cache_data["physical_blocks_meta"] = meta_list
            kv_cache_data["tp_world_size"] = tp_world_size
            self._save_to_shm(shm, kv_cache_data)
            logger.debug(f"已将物理块元数据写入请求 {request_id} 的 KV 缓存共享内存，共 {len(meta_list)} 个块")
        except (OSError, json.JSONDecodeError):
            logger.exception("写入物理块元数据时出错")
            return False
        else:
            return True
        finally:
            if shm is not None:
                shm.close()

    def set_output_token_ids(self, request_id: str, output_token_ids: Dict[int, List[int]]) -> bool:
        """
        追加写入物理块元数据到已有 KV 缓存共享内存

        Args:
            request_id: 请求ID
            output_token_ids: 输出 token ID 列表

        Returns:
            bool: 是否成功写入

        """
        shm = None
        try:
            # 从请求映射表获取共享内存名称
            request_map = self._load_from_shm(self.request_map_shm)
            if request_id not in request_map or "kv_shm_name" not in request_map[request_id]:
                logger.warning(f"无法找到请求 {request_id} 的 KV 缓存共享内存名称，跳过写入物理块元数据")
                return False

            kv_shm_name = request_map[request_id]["kv_shm_name"]

            shm = shared_memory.SharedMemory(name=kv_shm_name, create=False)
            kv_cache_data = self._load_from_shm(shm)
            kv_cache_data["output_token_ids"] = output_token_ids
            self._save_to_shm(shm, kv_cache_data)
            logger.debug(f"已将输出 token ID 写入请求 {request_id}")
        except (OSError, json.JSONDecodeError):
            logger.exception("写入输出 token ID 时出错")
            return False
        else:
            return True
        finally:
            if shm is not None:
                shm.close()

    def upload_physical_blocks_gpu(
        self,
        request_id: str,
        block_id: int,
        tp_rank: int,
        worker_data: Dict
    ) -> bool:
        """
        上传GPU物理块到共享内存

        每个worker调用此方法上传自己的rank数据。直接上传到共享内存，不进行合并。

        Args:
            request_id: 请求ID
            block_id: 块ID
            tp_rank: tensor并行的rank
            worker_data: worker数据字典

        Returns:
            bool: 是否成功上传
        """
        shm = None
        try:
            # 通过get_physical_block_name获取唯一的共享内存名称
            shm_name = self.get_physical_block_name(
                request_id, block_id, tp_rank)
            data_array = worker_data["data"]

            # 检查是否已存在
            try:
                existing_shm = shared_memory.SharedMemory(
                    name=shm_name, create=False)
                existing_shm.close()
                logger.debug(f"Rank {tp_rank} 块已存在: {shm_name}")
                return True
            except FileNotFoundError:
                pass

            # 创建共享内存并上传数据
            shm = shared_memory.SharedMemory(
                name=shm_name,
                create=True,
                size=data_array.nbytes
            )
            shm_array = np.ndarray(
                data_array.shape, dtype=data_array.dtype, buffer=shm.buf)
            shm_array[:] = data_array[:]

            logger.debug(
                f"Rank {tp_rank} 块已上传: {shm_name}, 大小 {data_array.nbytes / (1024*1024):.2f}MB")
            return True

        except Exception as e:
            logger.error(f"上传GPU块失败: {e}")
            return False
        finally:
            if shm is not None:
                shm.close()

    def upload_physical_blocks_cpu(self, request_id: str, block_id: int, block: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        上传物理块到共享内存

        Args:
            request_id: 请求ID
            block_id: 物理块ID
            block: 物理块数据

        Returns:
            Optional[Dict[str, Any]]: 物理块元数据

        """
        shm = None
        try:
            # 生成物理块的共享内存名称并删除可能存在的旧共享内存
            shm_name = self.get_physical_block_name(request_id, block_id)
            self._remove_existing_shm(shm_name)

            logger.debug(
                f"物理块 {block_id} 的形状: {block.shape}, "
                f"数据类型: {block.dtype}, "
                f"大小: {block.nbytes / (1024 * 1024):.2f}MB",
            )
            shm = shared_memory.SharedMemory(
                name=shm_name, create=True, size=block.nbytes)
            shm_array = np.ndarray(
                block.shape, dtype=block.dtype, buffer=shm.buf)
            shm_array[:] = block[:]
            meta = {
                "block_id": block_id,
                "shm_name": shm_name,
                "shape": shm_array.shape,
                "dtype": str(shm_array.dtype),
                "nbytes": shm_array.nbytes,
            }
            logger.debug(
                f"已导出物理块 {block_id} 到共享内存 {shm_name}, 大小: {block.nbytes / (1024 * 1024):.2f}MB")
        except OSError:
            logger.exception("导出物理块 %s 到共享内存时出错", block_id)
            return None
        else:
            return meta
        finally:
            if shm is not None:
                shm.close()

    # ===== 下载相关方法 =====
    def load_logical_block(self, request_id: str) -> Optional[Dict[str, Any]]:
        """
        从共享内存加载逻辑块数据

        Args:
            request_id: 请求ID

        Returns:
            Optional[Dict[str, Any]]: KV缓存数据，如果不存在则返回None

        """
        logger.debug(f"尝试从共享内存加载请求 {request_id} 的逻辑块数据")

        # 从请求映射表查找
        request_map = self._load_from_shm(self.request_map_shm)
        if request_id not in request_map or "kv_shm_name" not in request_map[request_id]:
            logger.warning(f"请求 {request_id} 没有逻辑块共享内存")
            return None

        shm_info: Dict[str, Any] = request_map[request_id]
        logger.debug(f"请求 {request_id} 的共享内存信息: {shm_info}")

        shm = None
        try:
            kv_shm_name = shm_info["kv_shm_name"]
            shm = shared_memory.SharedMemory(name=kv_shm_name, create=False)
            logger.debug(f"成功连接到共享内存 {kv_shm_name}, 分配大小: {shm.size / (1024 * 1024):.2f} MB")

            kv_cache_data = self._load_from_shm(shm)
            logger.debug(f"成功从共享内存加载请求 {request_id} 的逻辑块, 包含的键: {list(kv_cache_data.keys())}")
        except (OSError, json.JSONDecodeError):
            logger.exception("加载逻辑块时出错")
            return None
        else:
            return kv_cache_data
        finally:
            if shm is not None:
                shm.close()

    def merge_physical_blocks_meta(self, request_id: str, meta: Dict[str, Any], tp_world_size: int) -> np.ndarray:
        """
        合并多个rank的物理块碎片

        Args:
            request_id: 请求ID
            meta: 物理块元数据（来自rank_0）
            tp_world_size: tensor并行的世界大小

        Returns:
            np.ndarray: 拼接后的完整数据
        """
        block_id = meta["block_id"]
        fragment_shape = meta["shape"]  # rank_0的形状
        fragment_dtype = np.dtype(meta["dtype"])

        logger.debug(f"开始合并块 {block_id} 的 {tp_world_size} 个碎片")

        # 读取所有rank的碎片
        fragments = []
        fragment_shms = []

        try:
            for tp_rank in range(tp_world_size):
                fragment_name = self.get_physical_block_name(
                    request_id, block_id, tp_rank)
                fragment_shm = shared_memory.SharedMemory(
                    name=fragment_name, create=False)
                fragment_shms.append(fragment_shm)

                # 读取碎片数据
                fragment_data = np.ndarray(
                    fragment_shape, dtype=fragment_dtype, buffer=fragment_shm.buf)
                fragments.append(fragment_data.copy())

                logger.debug(
                    f"读取rank {tp_rank} 碎片: {fragment_name}, shape={fragment_data.shape}")

            # 拼接碎片：在head维度上拼接
            # fragment_shape: [num_layers, 2, block_size, current_heads, head_size]
            # 需要在dim=3（head维度）上拼接
            merged_data = np.concatenate(fragments, axis=3)

            logger.debug(
                f"成功拼接块 {block_id}: 从 {len(fragments)} 个碎片 -> {merged_data.shape}, 大小 {merged_data.nbytes / (1024*1024):.2f}MB")

            return merged_data

        finally:
            # 清理碎片共享内存
            for shm in fragment_shms:
                try:
                    shm.close()
                except:
                    pass

    # ===== 清除工具方法 =====
    def _remove_existing_shm(self, shm_name: str) -> None:
        """删除指定共享内存，先关闭并unlink。"""
        try:
            existing_shm = shared_memory.SharedMemory(
                name=shm_name, create=False)
            existing_shm.close()
            existing_shm.unlink()
        except FileNotFoundError:
            return
        except OSError:
            with contextlib.suppress(FileNotFoundError):
                Path(f"/dev/shm/{shm_name}").unlink()

    def cleanup_request(self, request_id: str) -> None:
        """
        清理指定请求的逻辑块和物理块共享内存，以及从请求映射表中移除记录。

        Args:
            request_id: 请求ID

        """
        try:
            # 获取逻辑块名称
            logic_name = self.get_logical_block_name(request_id)
            # 读取逻辑块数据以获取物理块元数据
            shm = None
            try:
                shm = shared_memory.SharedMemory(name=logic_name, create=False)
                kv_cache_data = self._load_from_shm(shm)
            except (OSError, json.JSONDecodeError):
                kv_cache_data = {}
            finally:
                if shm is not None:
                    shm.close()

            # 获取tensor parallel world size
            tp_world_size = kv_cache_data.get("tp_world_size", 1)

            # 释放所有物理块共享内存（包括碎片）
            for meta in kv_cache_data.get("physical_blocks_meta") or []:
                block_id = meta.get("block_id")
                # 清理所有tensor parallel rank的碎片
                for tp_rank in range(tp_world_size):
                    fragment_name = self.get_physical_block_name(
                        request_id, block_id, tp_rank)
                    self._remove_existing_shm(fragment_name)

            # 释放逻辑块
            self._remove_existing_shm(logic_name)
            # 从请求映射表中移除条目
            if self.request_map_shm is not None:
                request_map = self._load_from_shm(self.request_map_shm)
                if request_id in request_map:
                    del request_map[request_id]
                    self._save_to_shm(self.request_map_shm, request_map)
            logger.debug(f"已清理请求 {request_id} 的所有共享内存资源")
        except (OSError, json.JSONDecodeError):
            logger.exception("清理请求 %s 时出错", request_id)
