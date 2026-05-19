# 特性介绍
基于推理时间线（Timeline）分析，整体优化可划分为数据前处理、输出后处理、框架调度三大阶段；从优化维度主要分为Python侧优化、数据传输优化、操作系统侧优化三类。

## 数据前处理优化
### D2H 传输优化
针对前处理阶段的 FlashAttention 计算过程中的 D2H 过程，将传输数据迁移至 GPU 上完成计算，减少CPU、GPU间数据传输过程，节省传输耗时，实现推理计算端到端加速。
### Numba JIT 加速 compute_slot_mapping
`_prepare_inputs`中调用的`compute_slot_mapping`函数，用于根据每个 token 所属请求及其在请求内的绝对位置，计算 token 在 KV cache 块表中的物理存储位置（slot）并写入self.slot_mapping张量。原实现基于np.array计算，通过 Numba JIT 重写该函数，实现 CPU 侧数值计算加速。

### 拆分 all_gather 适配 CUDA Graph 加速
`all_gather` 为多卡推理中的卡间通信操作，用于实现多卡间数据聚合与共享。原生 `all_gather` 无法被 CUDA Graph 捕获，导致多卡推理无法利用静态图加速。

优化方案为将 `all_gather` 拆分为reshape + clone两步操作，在保证卡间通信结果一致、不影响张量并行逻辑的前提下，使相关计算纳入 CUDA Graph 执行流程，实现多卡推理静态图加速。

## 输出后处理优化
### output_handler 合并 np.array_split 与循环逻辑
`output_handler`中原生逻辑独立执行`np.array_split`数组分割与后续 for 循环处理，会产生大量临时数组，带来显著内存开销与调度损耗。

优化方案为合并数组分割与循环处理逻辑，直接按索引实现批量数据处理，减少数组分割、内存拷贝与中间对象创建开销，提升后处理阶段整体吞吐效率。

### 异步调度去同步，异步更新 output_token_ids
异步调度在处理 logprobs 相关逻辑时存在冗余同步操作，削弱异步执行性能优势。

优化方式为移除不必要同步点，在采样得到的 token ID 从 GPU 拷贝至 CPU 完成后，立即异步更新output_token_ids，实现推理计算与结果状态更新的并行重叠，充分发挥异步调度性能。

### _bookkeeping_sync 批量向量化优化
`_bookkeeping_sync`函数用于模型前向传播与采样完成后，将 token IDs、logprobs 从 GPU 同步至 CPU，更新请求 token 序列、长度等内部状态，并处理调度、丢弃与推测解码等场景。原代码在循环中重复读取`num_tokens_no_spec[req_idx]`，存在大量 Python-C 跨边界调用，且逐标量赋值、零散更新张量与数组，执行效率较低。

优化方案为在循环前预先将num_tokens_no_spec转为列表缓存，批量收集待更新索引与数值，利用 numpy 高级索引实现批量赋值，对张量与数组统一采用向量化更新方式，大幅减少 Python 与底层 C 的交互开销，提升后处理同步效率。

## 框架调度优化
### sched_yield 替换为 time.sleep (1e-5)
vLLM 原生使用`sched_yield()`实现线程礼让调度，在鲲鹏 ARM 平台下存在上下文切换开销大、调用次数显著高于 x86 等问题，影响推理流水线稳定性。

优化方式为将`sched_yield()`替换为`time.sleep(1e-5)`，采用轻量化线程调度方式，降低线程切换开销，提升 ARM 平台下 CPU 多核利用率，使推理流水线执行更平滑。

## 操作系统侧优化
### kqmalloc 高性能内存分配器优化
vLLM 推理过程存在高频内存分配与释放操作，系统默认内存分配器在 ARM 多核高并发场景下存在锁竞争、分配延迟高、性能抖动明显等问题。

优化方案为集成 kqmalloc 无锁高性能内存分配器，替换系统默认 malloc 实现，降低多核并发下的内存分配竞争，提升张量、中间缓冲区等对象的分配与释放效率，显著降低高并发推理时的内存操作时延。