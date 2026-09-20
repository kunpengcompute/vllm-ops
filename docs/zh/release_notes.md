# 版本说明书

## 版本配套说明

### 产品版本信息

<table><tbody>
<tr><th valign="top" width="42.17%"><p>产品版本</p></th>
<td valign="top" width="57.83%"><p>26.2.RC1</p></td></tr>
<tr><th valign="top" width="42.17%"><p>软件名称</p></th>
<td valign="top" width="57.83%"><p>vLLM高版本特性回合</p></td></tr>
<tr><th valign="top" width="42.17%"><p>软件包版本</p></th>
<td valign="top" width="57.83%"><p>v2.1.0</p></td></tr>
</tbody></table>

### 与加速卡配套说明

| 加速卡 | vLLM | vLLM-MetaX |
| ------------ | ------------ | ------------ |
| 沐曦曦云C500 GPU | 0.17.0 | 0.17.0 |

## v2.1.0

### 更新说明

面向沐曦曦云C500 GPU的推理场景，基于沐曦提供的vLLM 0.15.0容器，从源码编译vLLM 0.17.0和vLLM-MetaX 0.17.0，回合vLLM 0.18.0、0.19.0、0.20.0中的38个功能点，以消除因版本滞后导致的性能劣势；同时叠加opt优化（greedy sampler快路径、block table脏行局部拷贝）和vLLM-MetaX FlashAttention metadata优化（prefill最大长度D2H同步优化、Non-DCP cu_seqlens_k预计算优化），进一步降低推理热路径的开销。

### 已解决的问题

无

### 遗留问题

无

## 版本配套文档

### v2.1.0版本配套文档

| 文档名称 | 内容简介 | 交付方式 |
| ------------ | ------------ | --- |
| 《版本说明书》 | 提供vLLM高版本特性回合每个发布版本的基础信息和特性更新信息。 | 开源仓 |
| 《用户指南》 | 提供vLLM高版本特性回合补丁使用说明。 | 开源仓 |
| 《特性介绍》 | 提供vLLM高版本特性回合及优化说明。 | 开源仓 |

### 获取文档的方法

您可以通过访问[开源仓](https://gitcode.com/boostkit/vllm-ops)浏览和获取相关文档。
