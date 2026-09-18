# vLLM独立opt优化补丁包

本目录包含面向vLLM 0.17.0的独立性能优化补丁，与高版本特性回合无关，可单独合入。

## 目录说明

| 文件 | 说明 |
| --- | --- |
| `vllm-0.17.0-opt.patch` | greedy sampler快路径与block table脏行局部拷贝合并优化补丁。 |

## 优化内容

- **greedy sampler快路径**：当请求全部使用greedy sampling且无logprobs、penalty、allowed token、bad words或会改变argmax结果的logits processor时，直接执行argmax并返回结果，跳过通用采样流程中的无效计算。
- **block table脏行局部拷贝**：在主机侧记录block table被修改的最小连续行区间，提交到设备时只复制实际变化的行，减少调度热路径中的Host-to-Device数据传输。

补丁修改`vllm/v1/sample/sampler.py`与`vllm/v1/worker/block_table.py`。

## 应用方法

补丁在源码编译前合入vLLM 0.17.0源码树，随后源码安装生效。完整流程参见[用户指南](../../docs/zh/user_guide.md)。核心步骤如下：

```bash
cd <vLLM 0.17.0 源码根目录>
patch -p1 --forward < <本目录绝对路径>/vllm-0.17.0-opt.patch
```

补丁合入时若无`FAILED`或`Hunk`失败等特殊回显信息，则合入成功。优化说明参见[特性介绍](../../docs/zh/feature_introduction.md)。
