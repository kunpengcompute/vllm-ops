# Feature Introduction

Based on the inference timeline analysis, the overall optimization can be divided into three phases: data preprocessing, output postprocessing, and framework scheduling, covering Python-side optimization, data transmission optimization, and OS-side optimization.

## Data Preprocessing Optimization

### Optimizing Device-to-Host (D2H) Transmission

For the D2H process during FlashAttention computation in the preprocessing phase, the data is migrated to the GPU for computation. This reduces the data transmission between the CPU and GPU, saves transmission time, and achieves end-to-end acceleration of inference computation.

### Accelerating compute_slot_mapping Using Numba JIT

The `compute_slot_mapping` function called in `_prepare_inputs` is used to calculate the physical storage location (slot) of a token in the KV cache block table based on the request to which the token belongs and the absolute position of the token in the request, and then write the location to the `self.slot_mapping` tensor. The original implementation is based on `np.array` computation. This function is rewritten using Numba JIT to accelerate numerical computation on the CPU side.

### Splitting all_gather for CUDA Graph-based Acceleration

`all_gather` is an inter-card communication operation in multi-card inference, which is used for multi-card data aggregation and sharing. The native `all_gather` cannot be captured by CUDA Graph, which prevents multi-card inference from using static graph acceleration.

The optimization solution is to split `all_gather` into two steps: `reshape` and `clone`. On the premise that the inter-card communication results are consistent and the tensor parallelism logic is not affected, this allows related computation to be included in the CUDA Graph execution process, achieving static graph acceleration for multi-card inference.

## Output Postprocessing Optimization

### Merging np.array_split and Loop Logic in output_handler

In `output_handler`, the native logic independently performs array splitting (`np.array_split`) and subsequent `for` loop processing. This generates a large number of temporary arrays, resulting in significant memory overhead and scheduling overhead.

The optimization solution is to merge the array splitting and loop processing logic, and directly implement batch data processing by index. This reduces the overhead of array splitting, memory copying, and intermediate object creation, and improves the overall throughput efficiency in the postprocessing phase.

### Removing Unnecessary Synchronization and Asynchronously Updating output_token_ids in Asynchronous Scheduling

During `logprobs`-related logic processing, asynchronous scheduling involves redundant synchronization operations, which weakens the performance advantage of asynchronous execution.

The optimization method is to remove unnecessary synchronization points. As soon as the sampled token IDs are copied from the GPU to the CPU, `output_token_ids` is updated asynchronously. In this way, inference computing and result status update are performed in parallel, fully utilizing the performance of asynchronous scheduling.

### Optimizing _bookkeeping_sync Through Batch Vectorization

The `_bookkeeping_sync` function is used after model forward propagation and sampling to synchronize token IDs and `logprobs` from the GPU to the CPU, update internal states such as the requested token sequence and its length, and handle scenarios such as scheduling, discarding, and speculative decoding. In the original code, `num_tokens_no_spec[req_idx]` is repeatedly read in the loop, and there are a large number of Python-C cross-boundary calls. In addition, scalar-by-scalar assignment and scattered tensor and array updates are involved, resulting in low execution efficiency.

The optimization solution is to convert `num_tokens_no_spec` into a list for caching before the loop, collect the indexes and values to be updated in batches, use the advanced indexing function of NumPy to implement batch assignment, and use the vectorized update mode for tensors and arrays. This significantly reduces the interaction overhead between Python and the underlying C code, thereby improving the post-processing synchronization efficiency.

## Framework Scheduling Optimization

### Replacing sched_yield with time.sleep (1e-5)

vLLM natively uses `sched_yield()` to implement thread yielding. On the Kunpeng Arm platform, the context switching overhead is high and the number of calls is significantly higher than that on the x86 platform. This affects the inference pipeline stability.

The optimization method is to replace `sched_yield()` with `time.sleep(1e-5)` and use the lightweight thread scheduling mode. This reduces the thread switching overhead, improves the multi-core CPU utilization on the Arm platform, and makes the inference pipeline execution smoother.

## OS-Side Optimization

### Enabling the KQMalloc High-Performance Memory Allocator

During vLLM inference, there are frequent memory allocation and release operations. The default system memory allocator may suffer from problems such as lock contention, high allocation latency, and obvious performance jitter in Arm multi-core high-concurrency scenarios.

The optimization solution is to integrate the KQMalloc lock-free high-performance memory allocator to replace the default malloc implementation. This alleviates memory allocation contention in multi-core concurrency scenarios, improves the allocation and release efficiency of objects such as tensors and intermediate buffers, and significantly reduces the memory operation latency during high-concurrency inference.
