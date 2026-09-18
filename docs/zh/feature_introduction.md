# 特性介绍

## opt优化

- **greedy sampler快路径**：当请求全部使用greedy sampling，且没有logprobs、penalty、allowed token、bad words或会改变argmax结果的logits processor时，直接执行argmax并返回结果，跳过通用采样流程中的无效计算。
- **block table脏行局部拷贝**：在主机侧记录block table被修改的最小连续行区间，提交到设备时只复制实际变化的行，减少调度热路径中的Host-to-Device数据传输。

## vLLM-MetaX metadata优化

- **prefill最大长度D2H同步优化**：FlashAttention metadata构造阶段复用已有的batch `max_seq_len`作为安全上界，替代`prefill_seq_lens.max().item()`，减少设备标量回传主机和`mcStreamSynchronize`同步等待。
- **Non-DCP cu_seqlens_k预计算优化**：Non-DCP forward会在调用FlashAttention前重新构造`cu_seqlens_k`，因此删除metadata builder中的重复`zero_`、类型转换、pad和cumsum操作，减少冗余GPU kernel及launch开销。

上述两项vLLM-MetaX优化不修改DCP路径、FlashAttention kernel输入或mcoplib API。

## 修订记录

| 文档编号 | 发布日期 | 修改说明 |
| --- | --- | --- |
| 02 | 2026-09-30 | v2.1.0版本修订更新。 |
| 01 | 2026-06-30 | v2.0.0版本第一次正式发布。 |
