# vLLM-MetaX优化补丁包

本目录包含面向vLLM-MetaX 0.17.0的FlashAttention metadata优化补丁。

## 目录说明

| 文件 | 说明 |
| --- | --- |
| `vllm-metax-flash-attn-metadata-optimizations.patch` | FlashAttention metadata构造阶段的开销优化补丁。 |

## 优化内容

- **prefill最大长度D2H同步优化**：FlashAttention metadata构造阶段复用已有的batch `max_seq_len`作为安全上界，替代`prefill_seq_lens.max().item()`，减少设备标量回传主机和`mcStreamSynchronize`同步等待。
- **Non-DCP cu_seqlens_k预计算优化**：Non-DCP forward会在调用FlashAttention前重新构造`cu_seqlens_k`，因此删除metadata builder中的重复`zero_`、类型转换、pad和cumsum操作，减少冗余GPU kernel及launch开销。

上述两项优化不修改DCP路径、FlashAttention kernel输入或mcoplib API。补丁修改`vllm_metax/v1/attention/backends/flash_attn.py`。

## 应用方法

补丁在源码编译前合入vLLM-MetaX 0.17.0源码树，随后源码安装生效。完整流程参见[用户指南](../../docs/zh/user_guide.md)。核心步骤如下：

```bash
cd <vLLM-MetaX 0.17.0 源码根目录>
patch -p1 --forward < <本目录绝对路径>/vllm-metax-flash-attn-metadata-optimizations.patch
```

补丁合入时若无`FAILED`或`Hunk`失败等特殊回显信息，则合入成功。优化说明参见[特性介绍](../../docs/zh/feature_introduction.md)。
