# Release Notes

## Version Mapping

### Product Version Information

<table><tbody>
<tr><th valign="top" width="42.17%"><p>Product Version</p></th>
<td valign="top" width="57.83%"><p>26.2.RC1</p></td></tr>
<tr><th valign="top" width="42.17%"><p>Software Name</p></th>
<td valign="top" width="57.83%"><p>vLLM High-Version Feature Backport</p></td></tr>
<tr><th valign="top" width="42.17%"><p>Software Package Version</p></th>
<td valign="top" width="57.83%"><p>v2.1.0</p></td></tr>
</tbody></table>

### Accelerator Compatibility

| Accelerator | vLLM | vLLM-MetaX |
| ------------ | ------------ | ------------ |
| MetaX C500 GPU | 0.17.0 | 0.17.0 |

## v2.1.0

### Update Description

For the inference scenario on the MetaX C500 GPU, based on the vLLM 0.15.0 container provided by MetaX, this release compiles vLLM 0.17.0 and vLLM-MetaX 0.17.0 from source, backports 38 feature points from vLLM 0.18.0, 0.19.0, and 0.20.0 to eliminate the performance disadvantages caused by version lag, and additionally applies opt optimizations (greedy sampler fast path, block table dirty-row local copy) and vLLM-MetaX FlashAttention metadata optimizations (prefill max length D2H sync optimization, Non-DCP cu_seqlens_k precomputation optimization) to further reduce the overhead on the inference hot path.

### Resolved Issues

None

### Known Issues

None

## Version-Matching Documents

### v2.1.0 Version-Matching Documents

| Document Name | Description | Delivery Method |
| ------------ | ------------ | --- |
| Release Notes | Provides basic information and feature updates of each release version. | Open-source repository |
| User Guide | Provides the usage description of the backport patches. | Open-source repository |
| Feature Introduction | Provides the description of the high-version feature backport and optimizations. | Open-source repository |

### How to Obtain Documents

You can browse and obtain the related documents by visiting the [open-source repository](https://gitcode.com/boostkit/vllm-ops).
