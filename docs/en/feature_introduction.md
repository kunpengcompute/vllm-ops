# Feature Introduction

This document describes the standalone optimizations applied on top of the high-version feature backport: the opt optimizations and the vLLM-MetaX metadata optimizations.

## opt Optimization

- **greedy sampler fast path**: When all requests use greedy sampling and there are no logprobs, penalties, allowed tokens, bad words, or logits processors that would change the argmax result, argmax is executed directly and the result is returned, skipping the redundant computation in the general sampling path.
- **block table dirty-row local copy**: The minimum contiguous range of rows modified in the block table is recorded on the host side, and only the actually changed rows are copied when committing to the device, reducing Host-to-Device data transfers on the scheduling hot path.

## vLLM-MetaX metadata Optimization

- **prefill max length D2H sync optimization**: During FlashAttention metadata construction, the existing batch `max_seq_len` is reused as a safe upper bound to replace `prefill_seq_lens.max().item()`, reducing the device-to-host scalar copy and the `mcStreamSynchronize` synchronization wait.
- **Non-DCP cu_seqlens_k precomputation optimization**: The Non-DCP forward path rebuilds `cu_seqlens_k` immediately before calling FlashAttention. Therefore, the redundant `zero_`, type conversion, pad, and cumsum operations in the metadata builder are removed, reducing redundant GPU kernels and launch overhead.

The two vLLM-MetaX optimizations above do not modify the DCP path, the FlashAttention kernel inputs, or the mcoplib API.

## Revision History

| Document No. | Publish Date | Change Description |
| --- | --- | --- |
| 02 | 2026-09-30 | Revision for v2.1.0. |
| 01 | 2026-06-30 | First official release of v2.0.0. |
