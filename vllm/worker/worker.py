# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A GPU worker class."""
import gc
import os
from typing import Dict, List, Optional, Set, Tuple, Type, Union

import torch
import torch.distributed

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.device_allocator.cumem import CuMemAllocator
from vllm.distributed import (ensure_model_parallel_initialized,
                              init_distributed_environment,
                              set_custom_all_reduce,
                              broadcast_tensor_dict,
                              get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size)
from vllm.distributed.kv_transfer import ensure_kv_transfer_initialized
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor import set_random_seed
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.model_executor.model_loader.tensorizer import TensorizerConfig
from vllm.platforms import current_platform
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sequence import (ExecuteModelRequest, IntermediateTensors,
                           SequenceGroupMetadata, SequenceGroupMetadataDelta)
from vllm.utils import (GiB_bytes, MemorySnapshot, bind_kv_cache,
                        memory_profiling)
from vllm.worker.cache_engine import CacheEngine
from vllm.worker.enc_dec_model_runner import EncoderDecoderModelRunner
from vllm.worker.model_runner import GPUModelRunnerBase, ModelRunner
from vllm.worker.pooling_model_runner import PoolingModelRunner
from vllm.worker.worker_base import (LocalOrDistributedWorkerBase, WorkerBase,
                                     WorkerInput, extract_previous_hidden_states)
from multiprocessing import shared_memory
import numpy as np
from vllm.core.shared_memory.manager import SharedMemoryManager

logger = init_logger(__name__)


class Worker(LocalOrDistributedWorkerBase):
    """A worker class that executes (a partition of) the model on a GPU.

    Each worker is associated with a single GPU. The worker is responsible for
    maintaining the KV cache and executing the model on the GPU. In case of
    distributed inference, each worker is assigned a partition of the model.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
        model_runner_cls: Optional[Type[GPUModelRunnerBase]] = None,
        shared_memory_manager=None,
    ) -> None:
        WorkerBase.__init__(self, vllm_config)
        self.parallel_config.rank = rank
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.is_driver_worker = is_driver_worker
        if self.model_config.trust_remote_code:
            # note: lazy import to avoid importing torch before initializing
            from vllm.utils import init_cached_hf_modules
            init_cached_hf_modules()

        # Return hidden states from target model if the draft model is an
        # mlp_speculator
        speculative_config = self.speculative_config
        model_config = self.model_config
        speculative_args = {} if speculative_config is None \
            or (speculative_config.draft_model_config.hf_config.model_type ==
                model_config.hf_config.model_type) \
            or (speculative_config.draft_model_config.hf_config.model_type
                not in ("medusa",
                        "mlp_speculator",
                        "eagle",
                        "deepseek_mtp",
                        "mimo_mtp")) \
            else {"return_hidden_states": True}

        ModelRunnerClass: Type[GPUModelRunnerBase] = ModelRunner
        if model_config.runner_type == "pooling":
            ModelRunnerClass = PoolingModelRunner
        elif self.model_config.is_encoder_decoder:
            ModelRunnerClass = EncoderDecoderModelRunner
        self.model_runner: GPUModelRunnerBase = ModelRunnerClass(
            vllm_config=self.vllm_config,
            kv_cache_dtype=self.cache_config.cache_dtype,
            is_driver_worker=is_driver_worker,
            **speculative_args,
        )
        if model_runner_cls is not None:
            self.model_runner = model_runner_cls(self.model_runner)

        # Uninitialized cache engine. Will be initialized by
        # initialize_cache.
        self.cache_engine: List[CacheEngine]
        # Initialize gpu_cache as pooling models don't initialize kv_caches
        self.gpu_cache: Optional[List[List[torch.Tensor]]] = None
        self._seq_group_metadata_cache: Dict[str, SequenceGroupMetadata] = {}

        # Buffers saved before sleep
        self._sleep_saved_buffers: Dict[str, torch.Tensor] = {}

        # Torch profiler. Enabled and configured through env vars:
        # VLLM_TORCH_PROFILER_DIR=/path/to/save/trace
        if envs.VLLM_TORCH_PROFILER_DIR:
            torch_profiler_trace_dir = envs.VLLM_TORCH_PROFILER_DIR
            logger.info("Profiling enabled. Traces will be saved to: %s",
                        torch_profiler_trace_dir)
            self.profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                with_stack=True,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    torch_profiler_trace_dir, use_gzip=True))
        else:
            self.profiler = None

        # 接收从executor传递的共享内存管理器实例
        self.shared_memory_manager: SharedMemoryManager = shared_memory_manager

    def start_profile(self):
        if self.profiler is None:
            raise RuntimeError("Profiler is not enabled.")
        self.profiler.start()

    def stop_profile(self):
        if self.profiler is None:
            raise RuntimeError("Profiler is not enabled.")
        self.profiler.stop()
        print(
            self.profiler.key_averages().table(sort_by="self_cuda_time_total"))

    def sleep(self, level: int = 1) -> None:
        free_bytes_before_sleep = torch.cuda.mem_get_info()[0]

        # Save the buffers before level 2 sleep
        if level == 2:
            model = self.model_runner.model
            self._sleep_saved_buffers = {
                name: buffer.cpu().clone()
                for name, buffer in model.named_buffers()
            }

        allocator = CuMemAllocator.get_instance()
        allocator.sleep(offload_tags=("weights", ) if level == 1 else tuple())
        free_bytes_after_sleep, total = torch.cuda.mem_get_info()
        freed_bytes = free_bytes_after_sleep - free_bytes_before_sleep
        used_bytes = total - free_bytes_after_sleep
        assert freed_bytes >= 0, "Memory usage increased after sleeping."
        logger.info(
            "Sleep mode freed %.2f GiB memory, "
            "%.2f GiB memory is still in use.", freed_bytes / GiB_bytes,
            used_bytes / GiB_bytes)

    def wake_up(self, tags: Optional[list[str]] = None) -> None:
        allocator = CuMemAllocator.get_instance()
        allocator.wake_up(tags=tags)

        # Restore the buffers after level 2 sleep
        if len(self._sleep_saved_buffers):
            model = self.model_runner.model
            for name, buffer in model.named_buffers():
                if name in self._sleep_saved_buffers:
                    buffer.data.copy_(self._sleep_saved_buffers[name].data)
            self._sleep_saved_buffers = {}

    def init_device(self) -> None:
        if self.device_config.device.type == "cuda":
            # torch.distributed.all_reduce does not free the input tensor until
            # the synchronization point. This causes the memory usage to grow
            # as the number of all_reduce calls increases. This env var disables
            # this behavior.
            # Related issue:
            # https://discuss.pytorch.org/t/cuda-allocation-lifetime-for-inputs-to-distributed-all-reduce/191573
            os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

            # This env var set by Ray causes exceptions with graph building.
            os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
            self.device = torch.device(f"cuda:{self.local_rank}")
            torch.cuda.set_device(self.device)

            _check_if_gpu_supports_dtype(self.model_config.dtype)
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            self.baseline_snapshot = MemorySnapshot()
        else:
            raise RuntimeError(
                f"Not support device type: {self.device_config.device}")
        # Initialize the distributed environment.
        init_worker_distributed_environment(self.vllm_config, self.rank,
                                            self.distributed_init_method,
                                            self.local_rank)
        # Set random seed.
        set_random_seed(self.model_config.seed)

    def load_model(self):
        if self.vllm_config.model_config.enable_sleep_mode:
            allocator = CuMemAllocator.get_instance()
            assert allocator.get_current_usage() == 0, (
                "Sleep mode can only be "
                "used for one instance per process.")
            context = allocator.use_memory_pool(tag="weights")
        else:
            from contextlib import nullcontext
            context = nullcontext()
        with context:
            self.model_runner.load_model()

    def save_sharded_state(
        self,
        path: str,
        pattern: Optional[str] = None,
        max_size: Optional[int] = None,
    ) -> None:
        self.model_runner.save_sharded_state(
            path,
            pattern=pattern,
            max_size=max_size,
        )

    def save_tensorized_model(
        self,
        tensorizer_config: TensorizerConfig,
    ) -> None:
        self.model_runner.save_tensorized_model(
            tensorizer_config=tensorizer_config, )

    @torch.inference_mode()
    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """Profiles the peak memory usage of the model to determine how many
        KV blocks may be allocated without OOMs.

        The engine will first conduct a profiling of the existing memory usage.
        Then, it calculate the maximum possible number of GPU and CPU blocks
        that can be allocated with the remaining free memory.

        Tip:
            You may limit the usage of GPU memory
            by adjusting the `gpu_memory_utilization` parameter.
        """
        # Profile the memory usage of the model and get the maximum number of
        # cache blocks that can be allocated with the remaining free memory.
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        free_memory_pre_profile, total_gpu_memory = torch.cuda.mem_get_info()

        # Execute a forward pass with dummy inputs to profile the memory usage
        # of the model.
        with memory_profiling(
                self.baseline_snapshot,
                weights_memory=self.model_runner.model_memory_usage) as result:
            self.model_runner.profile_run()

        self._assert_memory_footprint_increased_during_profiling()

        memory_for_current_instance = total_gpu_memory * \
            self.cache_config.gpu_memory_utilization
        available_kv_cache_memory = (memory_for_current_instance -
                                     result.non_kv_cache_memory)

        # Calculate the number of blocks that can be allocated with the
        # profiled peak memory.
        cache_block_size = self.get_cache_block_size_bytes()
        if cache_block_size == 0:
            num_gpu_blocks = 0
            num_cpu_blocks = 0
        else:
            num_gpu_blocks = int(available_kv_cache_memory // cache_block_size)
            num_cpu_blocks = int(self.cache_config.swap_space_bytes //
                                 cache_block_size)
        num_gpu_blocks = max(num_gpu_blocks, 0)
        num_cpu_blocks = max(num_cpu_blocks, 0)

        msg = (f"Memory profiling takes {result.profile_time:.2f} seconds\n"
               "the current vLLM instance can use "
               "total_gpu_memory "
               f"({(total_gpu_memory / GiB_bytes):.2f}GiB)"
               " x gpu_memory_utilization "
               f"({self.cache_config.gpu_memory_utilization:.2f})"
               f" = {(memory_for_current_instance / GiB_bytes):.2f}GiB\n"
               "model weights take "
               f"{(result.weights_memory / GiB_bytes):.2f}GiB;"
               " non_torch_memory takes "
               f"{(result.non_torch_increase / GiB_bytes):.2f}GiB;"
               " PyTorch activation peak memory takes "
               f"{(result.torch_peak_increase / GiB_bytes):.2f}GiB;"
               " the rest of the memory reserved for KV Cache is "
               f"{(available_kv_cache_memory / GiB_bytes):.2f}GiB.")

        logger.info(msg)
        # Final cleanup
        gc.collect()

        return num_gpu_blocks, num_cpu_blocks

    def _assert_memory_footprint_increased_during_profiling(self):
        # NOTE(woosuk): Here we assume that the other processes using the same
        # GPU did not change their memory usage during the profiling.
        free_gpu_memory, total = torch.cuda.mem_get_info()
        cuda_memory = total - free_gpu_memory
        assert self.baseline_snapshot.cuda_memory < cuda_memory, (
            "Error in memory profiling. "
            f"Initial used memory {self.baseline_snapshot.cuda_memory}, "
            f"currently used memory {cuda_memory}. "
            f"This happens when the GPU memory was "
            "not properly cleaned up before initializing the vLLM instance.")

    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        """Allocate GPU and CPU KV cache with the specified number of blocks.

        This also warms up the model, which may record CUDA graphs.
        """
        raise_if_cache_size_invalid(
            num_gpu_blocks, self.cache_config.block_size,
            self.cache_config.is_attention_free,
            self.model_config.max_model_len,
            self.parallel_config.pipeline_parallel_size)

        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

        if self.vllm_config.model_config.enable_sleep_mode:
            allocator = CuMemAllocator.get_instance()
            context = allocator.use_memory_pool(tag="kv_cache")
        else:
            from contextlib import nullcontext
            context = nullcontext()
        with context:
            self._init_cache_engine()
        self._warm_up_model()

    def _init_cache_engine(self):
        assert self.cache_config.num_gpu_blocks is not None
        self.cache_engine = [
            CacheEngine(self.cache_config, self.model_config,
                        self.parallel_config, self.device_config)
            for _ in range(self.parallel_config.pipeline_parallel_size)
        ]
        self.gpu_cache = [
            self.cache_engine[ve].gpu_cache
            for ve in range(self.parallel_config.pipeline_parallel_size)
        ]
        bind_kv_cache(self.compilation_config.static_forward_context,
                      self.gpu_cache)

    def _warm_up_model(self) -> None:
        # warm up sizes that are not in cudagraph capture sizes,
        # but users still want to compile for better performance,
        # e.g. for the max-num-batched token size in chunked prefill.
        warmup_sizes = self.vllm_config.compilation_config.compile_sizes.copy()
        if not self.model_config.enforce_eager:
            warmup_sizes = [
                x for x in warmup_sizes if x not in
                self.vllm_config.compilation_config.cudagraph_capture_sizes
            ]
        for size in sorted(warmup_sizes, reverse=True):
            logger.info("Compile and warming up model for size %d", size)
            self.model_runner._dummy_run(size)
        if not self.model_config.enforce_eager:
            self.model_runner.capture_model(self.gpu_cache)
        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.model_config.seed)

    @property
    def do_metadata_broadcast(self) -> bool:
        return self.parallel_config.tensor_parallel_size > 1

    @property
    def kv_cache(self) -> Optional[List[List[torch.Tensor]]]:
        return self.gpu_cache

    @torch.inference_mode()
    def prepare_worker_input(
            self, execute_model_req: ExecuteModelRequest) -> WorkerInput:
        virtual_engine = execute_model_req.virtual_engine
        num_steps = execute_model_req.num_steps
        num_seq_groups = len(execute_model_req.seq_group_metadata_list)
        # `blocks_to_swap_in` and `blocks_to_swap_out` are cpu tensors.
        # they contain parameters to launch cudamemcpyasync.
        blocks_to_swap_in = torch.tensor(execute_model_req.blocks_to_swap_in,
                                         device="cpu",
                                         dtype=torch.int64).view(-1, 2)
        blocks_to_swap_out = torch.tensor(execute_model_req.blocks_to_swap_out,
                                          device="cpu",
                                          dtype=torch.int64).view(-1, 2)
        # `blocks_to_copy` is a gpu tensor. The src and tgt of
        # blocks to copy are in the same device, and `blocks_to_copy`
        # can be used directly within cuda kernels.
        blocks_to_copy = torch.tensor(execute_model_req.blocks_to_copy,
                                      device=self.device,
                                      dtype=torch.int64).view(-1, 2)

        return WorkerInput(
            num_seq_groups=num_seq_groups,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
            blocks_to_shared_memory_upload=execute_model_req.blocks_to_shared_memory_upload,
            blocks_to_shared_memory_download=execute_model_req.blocks_to_shared_memory_download,
            virtual_engine=virtual_engine,
            num_steps=num_steps,
        )

    @torch.inference_mode()
    def execute_worker(self, worker_input: WorkerInput) -> None:
        virtual_engine = worker_input.virtual_engine
        # Issue cache operations.
        if (worker_input.blocks_to_swap_in is not None
                and worker_input.blocks_to_swap_in.numel() > 0):
            self.cache_engine[virtual_engine].swap_in(
                worker_input.blocks_to_swap_in)
        if (worker_input.blocks_to_swap_out is not None
                and worker_input.blocks_to_swap_out.numel() > 0):
            self.cache_engine[virtual_engine].swap_out(
                worker_input.blocks_to_swap_out)
        if (worker_input.blocks_to_copy is not None
                and worker_input.blocks_to_copy.numel() > 0):
            self.cache_engine[virtual_engine].copy(worker_input.blocks_to_copy)

        # 处理共享内存上传
        if (worker_input.blocks_to_shared_memory_upload is not None
                and len(worker_input.blocks_to_shared_memory_upload) > 0):
            # 格式: {request_id: physical_block_mapping}
            for request_id, physical_block_mapping in worker_input.blocks_to_shared_memory_upload.items():
                self.copy_block_to_sharememory(
                    virtual_engine,
                    request_id,
                    physical_block_mapping
                )

        # 处理共享内存下载
        if (worker_input.blocks_to_shared_memory_download is not None
                and len(worker_input.blocks_to_shared_memory_download) > 0):
            # 格式: {request_id: physical_block_mapping}
            for request_id, physical_block_mapping in worker_input.blocks_to_shared_memory_download.items():
                self.copy_block_from_sharememory(
                    virtual_engine,
                    request_id,
                    physical_block_mapping
                )

    def _get_cached_seq_group_metadata(
            self,
            seq_group_metadata_list: List[Union[SequenceGroupMetadata,
                                                SequenceGroupMetadataDelta]],
            finished_request_ids: List[str]) -> List[SequenceGroupMetadata]:
        """Return a list of cached Sequence Group Metadata after updating its
        state.

        It is used because scheduler only sends delta to workers to reduce
        the data payload size. The function also cleans up cache based on
        a given `finished_request_ids`.
        """
        new_seq_group_metadata_list = []
        for metadata_or_delta in seq_group_metadata_list:
            request_id = metadata_or_delta.request_id
            if request_id not in self._seq_group_metadata_cache:
                # The first prefill.
                assert isinstance(metadata_or_delta, SequenceGroupMetadata)
                self._seq_group_metadata_cache[request_id] = metadata_or_delta
            else:
                # The first prefill is already cached.
                if isinstance(metadata_or_delta, SequenceGroupMetadataDelta):
                    self._seq_group_metadata_cache[request_id].apply_delta(
                        metadata_or_delta)
                else:
                    # If metadata snapshot is sent again, it is
                    # preempted. Reset the cache because we need to start
                    # from scratch.
                    assert isinstance(metadata_or_delta, SequenceGroupMetadata)
                    self._seq_group_metadata_cache[
                        request_id] = metadata_or_delta

            new_seq_group_metadata_list.append(
                self._seq_group_metadata_cache[request_id])

        # Clean up finished ids
        for finished_id in finished_request_ids:
            del self._seq_group_metadata_cache[finished_id]

        return new_seq_group_metadata_list

    def _execute_model_spmd(
        self,
        execute_model_req: ExecuteModelRequest,
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Optional[List[SamplerOutput]]:
        if execute_model_req is not None:
            new_seq_group_metadata_list = self._get_cached_seq_group_metadata(
                execute_model_req.seq_group_metadata_list,
                execute_model_req.finished_requests_ids)

            execute_model_req.seq_group_metadata_list = (
                new_seq_group_metadata_list)
        output = super()._execute_model_spmd(execute_model_req,
                                             intermediate_tensors)
        return output

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.model_runner.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.model_runner.remove_lora(lora_id)

    def pin_lora(self, lora_id: int) -> bool:
        return self.model_runner.pin_lora(lora_id)

    def list_loras(self) -> Set[int]:
        return self.model_runner.list_loras()

    def add_prompt_adapter(
            self, prompt_adapter_request: PromptAdapterRequest) -> bool:
        return self.model_runner.add_prompt_adapter(prompt_adapter_request)

    def remove_prompt_adapter(self, prompt_adapter_id: int) -> bool:
        return self.model_runner.remove_lora(prompt_adapter_id)

    def pin_prompt_adapter(self, prompt_adapter_id: int) -> bool:
        return self.model_runner.pin_prompt_adapter(prompt_adapter_id)

    def list_prompt_adapters(self) -> Set[int]:
        return self.model_runner.list_prompt_adapters()

    @property
    def max_model_len(self) -> int:
        return self.model_config.max_model_len

    @property
    def vocab_size(self) -> int:
        return self.model_runner.vocab_size

    def get_cache_block_size_bytes(self) -> int:
        """Get the size of the KV cache block size in bytes.
        """
        return CacheEngine.get_cache_block_size(self.cache_config,
                                                self.model_config,
                                                self.parallel_config)

    # 重写execute_model方法，专门处理共享内存操作
    @torch.inference_mode()
    def execute_model(
        self,
        execute_model_req: Optional[ExecuteModelRequest] = None,
    ) -> Optional[List[SamplerOutput]]:
        """
        执行模型推理，支持共享内存操作的特殊处理
        
        当检测到共享内存操作时（空的seq_group_metadata_list但包含共享内存字段），
        会使用简化的处理流程，避免构建attention metadata导致的错误。
        对于remote worker，会检查broadcast data来判断是否为共享内存操作。
        
        Args:
            execute_model_req: 模型执行请求，包含序列组元数据和共享内存操作信息
            
        Returns:
            模型输出结果列表，共享内存操作返回空列表
        """
        # 检查是否为共享内存操作
        if (execute_model_req is not None and
            execute_model_req.seq_group_metadata_list is not None and
            len(execute_model_req.seq_group_metadata_list) == 0 and
            (execute_model_req.blocks_to_shared_memory_upload is not None or
             execute_model_req.blocks_to_shared_memory_download is not None)):

            # 这是一个共享内存操作请求，使用简化的处理流程
            return self._execute_shared_memory_operation(execute_model_req)

        # 对于remote worker，检查broadcast data是否为共享内存操作
        if (execute_model_req is None and not self.is_driver_worker and self.do_metadata_broadcast):
            # Remote worker情况：先查看broadcast data以判断是否为共享内存操作
            return self._handle_remote_worker_execution()

        # 否则使用默认的execute_model流程
        return super().execute_model(execute_model_req)

    def _handle_remote_worker_execution(self) -> Optional[List[SamplerOutput]]:
        """
        处理remote worker的执行逻辑，确保共享内存操作的正确分发
        
        remote worker需要通过broadcast接收driver worker的指令。
        此方法会检查broadcast data中是否包含共享内存操作，
        如果是则直接处理，否则重构完整的执行流程。
        
        Returns:
            模型输出结果列表，共享内存操作返回空列表
        """
        try:
            # 接收broadcast data
            broadcast_data = broadcast_tensor_dict(src=0)
            if not broadcast_data:
                return None

            # 检查是否包含共享内存操作
            has_shared_memory_op = (
                broadcast_data.get("blocks_to_shared_memory_upload") is not None or
                broadcast_data.get(
                    "blocks_to_shared_memory_download") is not None
            )

            if has_shared_memory_op and broadcast_data.get("num_seq_groups") == 0:
                # 这是共享内存操作，直接处理worker input
                worker_input = WorkerInput.from_broadcasted_tensor_dict(
                    broadcast_data)
                self.execute_worker(worker_input)
                return []
            else:
                # 这不是共享内存操作，但是我们已经消费了broadcast_data
                # 我们需要用这个数据重新构建inputs
                worker_input = WorkerInput.from_broadcasted_tensor_dict(
                    broadcast_data)
                model_input = (
                    self.model_runner.make_model_input_from_broadcasted_tensor_dict(
                        broadcast_data))

                kwargs = extract_previous_hidden_states(broadcast_data)

                # 执行标准流程
                self.execute_worker(worker_input)

                if worker_input.num_seq_groups == 0:
                    return []

                # 执行模型
                output = self.model_runner.execute_model(
                    model_input=model_input,
                    kv_caches=self.kv_cache[worker_input.virtual_engine]
                    if self.kv_cache is not None else None,
                    num_steps=worker_input.num_steps,
                    **kwargs,
                )
                return output

        except Exception as e:
            logger.error(f"Remote worker处理执行时出错: {e}")
            # 如果出错，回退到默认处理
            return super().execute_model(None)

    def _execute_shared_memory_operation(
        self,
        execute_model_req: ExecuteModelRequest
    ) -> Optional[List[SamplerOutput]]:
        """
        处理共享内存操作，绕过attention metadata构建
        
        共享内存操作不需要构建attention metadata，因为没有实际的序列数据处理。
        driver worker负责广播指令，remote worker接收并执行相应的共享内存操作。
        
        Args:
            execute_model_req: 包含共享内存操作信息的执行请求
            
        Returns:
            空列表，因为共享内存操作不产生模型输出
        """
        if self.is_driver_worker:
            # Driver worker: 准备并广播worker input
            worker_input = self.prepare_worker_input(execute_model_req)

            if self.do_metadata_broadcast:
                # 广播完整的worker input数据
                broadcast_data = worker_input.as_broadcastable_tensor_dict()
                broadcast_tensor_dict(broadcast_data, src=0)
        else:
            # Remote worker: 接收广播的数据并重建worker_input
            if self.do_metadata_broadcast:
                broadcast_data = broadcast_tensor_dict(src=0)
                if not broadcast_data:
                    return None
                worker_input = WorkerInput.from_broadcasted_tensor_dict(
                    broadcast_data)
            else:
                # 不应该到达这里，因为共享内存操作需要多worker协作
                logger.error("共享内存操作需要多worker协作，但do_metadata_broadcast为False")
                return []

        # 执行worker操作（包括共享内存操作）
        self.execute_worker(worker_input)

        # 共享内存操作不产生模型输出
        return []

    # 导出物理块到共享内存
    def export_physical_blocks_to_shared_memory(
        self,
        cache_engine: CacheEngine,
        block_ids: List[int],
        request_id: str,
    ) -> bool:
        """导出当前worker的物理块数据到manager并直接上传"""
        if not block_ids:
            return True

        # 获取tensor并行的rank信息
        tp_rank = get_tensor_model_parallel_rank()
        tp_world_size = get_tensor_model_parallel_world_size()

        logger.debug(f"Worker rank={tp_rank} 开始导出 {len(block_ids)} 个物理块")

        meta_list = []

        for block_id in block_ids:
            # 提取当前worker的KV数据
            layer_blocks = []
            for layer in cache_engine.gpu_cache:
                # 每个layer是一个tuple: (k_cache, v_cache)
                k_cache, v_cache = layer
                # 提取当前block的数据: [block_size, num_heads, head_size]
                # [block_size, num_heads, head_size]
                k_block = k_cache[block_id]
                # [block_size, num_heads, head_size]
                v_block = v_cache[block_id]

                # 重新排列为标准格式: [2, block_size, num_heads, head_size]
                kv_block = torch.stack([k_block, v_block], dim=0)
                layer_blocks.append(kv_block)

            # 堆叠所有layers: [num_layers, 2, block_size, num_heads, head_size]
            current_tensor = torch.stack(layer_blocks, dim=0)

            # 转换为numpy用于传输
            numpy_data = current_tensor.cpu().numpy()
            worker_data = {
                "block_id": block_id,
                "tp_rank": tp_rank,
                "tp_world_size": tp_world_size,
                "data": numpy_data,
                "shape": current_tensor.shape,
                "dtype": str(numpy_data.dtype),  # 使用numpy的dtype
                "device": "gpu"
            }

            # 直接上传碎片到manager
            success = self.shared_memory_manager.upload_physical_blocks_gpu(
                request_id, block_id, tp_rank, worker_data)

            if not success:
                logger.error(
                    f"Worker rank={tp_rank}: 上传block_id={block_id} 失败")
                return False

            logger.debug(f"Worker rank={tp_rank}: 成功上传block_id={block_id}")

            # 只有rank_0负责收集meta信息
            if tp_rank == 0:
                shm_name = self.shared_memory_manager.get_physical_block_name(
                    request_id, block_id, 0)
                meta = {
                    "block_id": block_id,
                    "shm_name": shm_name,
                    "shape": current_tensor.shape,
                    "dtype": str(numpy_data.dtype),  # 使用numpy的dtype
                    "nbytes": numpy_data.nbytes,
                }
                meta_list.append(meta)

        logger.debug(f"Worker rank={tp_rank} 导出完成，已上传 {len(block_ids)} 个物理块")

        if tp_rank == 0:
            self.shared_memory_manager.set_physical_blocks_meta(
                request_id, meta_list, tp_world_size)
        return True

    def copy_block_to_sharememory(
        self,
        virtual_engine: int,
        request_id: str,
        physical_block_mapping: dict[int, List[int]]
    ) -> bool:
        if not physical_block_mapping:
            return False

        for seq_id, block_ids in physical_block_mapping.items():
            success = self.export_physical_blocks_to_shared_memory(
                self.cache_engine[virtual_engine],
                block_ids,
                request_id=request_id
            )
            if not success:
                logger.error(
                    f"Worker rank={self.rank}: 导出seq_id={seq_id} 的块失败")
                return False

        return True

    # 从共享内存导入物理块
    def import_physical_blocks_from_shared_memory(
        self,
        cache_engine: CacheEngine,
        request_id: str,
        block_ids: List[int],
    ) -> bool:
        if not block_ids:
            return False

        kv_cache_data = self.shared_memory_manager.load_logical_block(
            request_id)
        if kv_cache_data is None:
            return False
        meta_list = kv_cache_data.get("physical_blocks_meta")
        if not meta_list:
            return False

        try:
            shared_memories = []

            for idx, meta in enumerate(meta_list):
                shm = shared_memory.SharedMemory(
                    name=meta["shm_name"], create=False)
                shared_memories.append(shm)
                raw_data = np.frombuffer(
                    shm.buf, dtype=np.float16).reshape(meta["shape"]).copy()
                logger.debug(
                    f"GPU 导入物理块: meta_shape={meta['shape']}, raw_data.shape={raw_data.shape}")

                # 确保使用正确的目标block_id
                if idx < len(block_ids):
                    dest_block_id = block_ids[idx]
                else:
                    dest_block_id = meta.get("block_id", block_ids[0])

                # 获取tensor并行信息
                tp_rank = get_tensor_model_parallel_rank()
                tp_world_size = get_tensor_model_parallel_world_size()

                logger.debug(
                    f"GPU import: tp_rank={tp_rank}, tp_world_size={tp_world_size}")

                for layer_idx in range(raw_data.shape[0]):
                    if layer_idx >= len(cache_engine.gpu_cache):
                        break

                    layer_data = raw_data[layer_idx]  # [2, flat_size]

                    block_size = cache_engine.block_size
                    num_kv_heads = cache_engine.num_kv_heads
                    head_size = cache_engine.head_size

                    # 计算当前GPU应该处理的数据范围
                    # 通用逻辑：适用于单GPU和多GPU情况
                    # 从flat_size推算total_heads
                    total_heads = layer_data.shape[1] // (
                        block_size * head_size)
                    heads_per_gpu = total_heads // tp_world_size

                    # 计算当前GPU的数据范围
                    head_start = tp_rank * heads_per_gpu
                    head_end = head_start + heads_per_gpu

                    # 计算在扁平化数据中的索引范围
                    flat_start = head_start * block_size * head_size
                    flat_end = head_end * block_size * head_size

                    # 提取当前GPU的数据部分
                    # [heads_per_gpu * block_size * head_size]
                    key_flat = layer_data[0][flat_start:flat_end]
                    # [heads_per_gpu * block_size * head_size]
                    value_flat = layer_data[1][flat_start:flat_end]

                    key_flat_tensor = torch.from_numpy(key_flat).to(
                        device=cache_engine.gpu_cache[layer_idx][0].device,
                        dtype=torch.float16
                    )
                    value_flat_tensor = torch.from_numpy(value_flat).to(
                        device=cache_engine.gpu_cache[layer_idx][1].device,
                        dtype=torch.float16
                    )

                    x = 16 // 2  # fp16 -> x=8

                    # Value处理：使用当前GPU的head数量
                    value_reshaped = value_flat_tensor.view(
                        num_kv_heads, head_size, block_size)
                    value_data = value_reshaped.permute(2, 0, 1)

                    # Key处理：使用当前GPU的head数量
                    key_temp = key_flat_tensor.view(
                        num_kv_heads, head_size // x, block_size, x)
                    key_temp = key_temp.permute(2, 0, 1, 3)
                    key_data = key_temp.reshape(
                        block_size, num_kv_heads, head_size)

                    k_cache, v_cache = cache_engine.gpu_cache[layer_idx]
                    k_cache[dest_block_id] = key_data
                    v_cache[dest_block_id] = value_data

            return True

        except Exception as e:
            logger.error(f"导入物理块时出错: {e}")
            return False
        finally:
            for shm in shared_memories:
                try:
                    shm.close()
                except:
                    pass

    def copy_block_from_sharememory(
        self,
        virtual_engine: int,
        request_id: str,
        physical_block_mapping: dict[int, List[int]]
    ) -> bool:
        """
        从共享内存导入物理块
        
        从共享内存读取CPU传来的完整KV缓存数据，
        根据当前worker的tensor parallel rank切割自己的部分并写入本地缓存。
        
        Args:
            virtual_engine: 虚拟引擎ID
            request_id: 请求ID
            physical_block_mapping: 物理块映射 {seq_id: [block_ids]}
            
        Returns:
            是否成功导入所有物理块
        """
        # 收集所有需要导入的block_ids
        all_block_ids = []
        for seq_id, block_ids in physical_block_mapping.items():
            all_block_ids.extend(block_ids)

        logger.debug(f"Worker rank={self.rank} 开始导入 {len(all_block_ids)} 个物理块")

        if not all_block_ids:
            return True

        # 导入所有物理块数据
        success = self.import_physical_blocks_from_shared_memory(
            self.cache_engine[virtual_engine], request_id, all_block_ids
        )

        if success:
            logger.debug(f"Worker rank={self.rank}: 成功导入 {len(all_block_ids)} 个物理块")
        else:
            logger.error(f"Worker rank={self.rank}: 导入物理块失败")

        return success

    def swap_block_gpu_cpu(
        self,
        execute_model_req: Optional[ExecuteModelRequest] = None
    ) -> None:
        _, worker_input, _ = self.prepare_input(
            execute_model_req=execute_model_req)
        virtual_engine = worker_input.virtual_engine
        # Issue cache operations.
        if (worker_input.blocks_to_swap_in is not None
                and worker_input.blocks_to_swap_in.numel() > 0):
            self.cache_engine[virtual_engine].swap_in(
                worker_input.blocks_to_swap_in)
        if (worker_input.blocks_to_swap_out is not None
                and worker_input.blocks_to_swap_out.numel() > 0):
            self.cache_engine[virtual_engine].swap_out(
                worker_input.blocks_to_swap_out)
        if (worker_input.blocks_to_copy is not None
                and worker_input.blocks_to_copy.numel() > 0):
            self.cache_engine[virtual_engine].copy(worker_input.blocks_to_copy)

def init_worker_distributed_environment(
    vllm_config: VllmConfig,
    rank: int,
    distributed_init_method: Optional[str] = None,
    local_rank: int = -1,
) -> None:
    """Initialize the distributed environment."""
    parallel_config = vllm_config.parallel_config
    set_custom_all_reduce(not parallel_config.disable_custom_all_reduce)

    init_distributed_environment(parallel_config.world_size, rank,
                                 distributed_init_method, local_rank)
    ensure_model_parallel_initialized(parallel_config.tensor_parallel_size,
                                      parallel_config.pipeline_parallel_size)

    ensure_kv_transfer_initialized(vllm_config)


def _check_if_gpu_supports_dtype(torch_dtype: torch.dtype):
    # Check if the GPU supports the dtype.
    if torch_dtype == torch.bfloat16:  # noqa: SIM102
        if not current_platform.has_device_capability(80):
            capability = current_platform.get_device_capability()
            gpu_name = current_platform.get_device_name()

            if capability is None:
                compute_str = "does not have a compute capability"
            else:
                version_str = capability.as_version_str()
                compute_str = f"has compute capability {version_str}"

            raise ValueError(
                "Bfloat16 is only supported on GPUs with compute capability "
                f"of at least 8.0. Your {gpu_name} GPU {compute_str}. "
                "You can use float16 instead by explicitly setting the "
                "`dtype` flag in CLI, for example: --dtype=half.")


def raise_if_cache_size_invalid(num_gpu_blocks, block_size, is_attention_free,
                                max_model_len, pipeline_parallel_size) -> None:
    if is_attention_free and num_gpu_blocks != 0:
        raise ValueError("No memory should be allocated for the cache blocks "
                         f"for an attention-free model, but {num_gpu_blocks} "
                         "blocks are allocated.")
    if not is_attention_free and num_gpu_blocks <= 0:
        raise ValueError("No available memory for the cache blocks. "
                         "Try increasing `gpu_memory_utilization` when "
                         "initializing the engine.")
    max_seq_len = block_size * (num_gpu_blocks // pipeline_parallel_size)
    if not is_attention_free and max_model_len > max_seq_len:
        raise ValueError(
            f"The model's max seq len ({max_model_len}) "
            "is larger than the maximum number of tokens that can be "
            f"stored in KV cache ({max_seq_len}). Try increasing "
            "`gpu_memory_utilization` or decreasing `max_model_len` when "
            "initializing the engine.")
