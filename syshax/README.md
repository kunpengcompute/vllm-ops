# 项目介绍

vllm-ops是鲲鹏自研的向量检索加速组件，对接vllm使用。本仓库中提供了对接openEuler的sysHax插件的适配与优化。vllm-ops适用于鲲鹏920新型号处理器，支持NEON指令（128位宽）和SVE指令（256位宽）。



## Patch 列表

| Patch 文件名 | 说明 |
|---------------|------|
| **boostkit-vllm-ops-gpu-082.patch** | 适配 GPU 端运行 **vLLM 0.8.2** 版本，解决接口差异及依赖兼容问题。 |
| **boostkit-vllm-ops-cpu-gptq.patch** | 适配 CPU 端加载 **GPTQ 量化模型**，支持量化参数解析与推理执行。 |
| **boostkit-vllm-ops-op-speed.patch** | 优化算子执行性能，减少 CPU/GPU 端瓶颈，提高整体推理速度。 |

---

## 使用方法

此处以在一台 kunpeng 920新型号 + 沐曦C500 * 1的容器部署运行 tclf90_deepseek-r1-distill-qwen-32b-gptq-int8 为例。

模型获取地址 https://modelscope.cn/models/tclf90/deepseek-r1-distill-qwen-32b-gptq-int8

首先，通过沐曦官网，获取沐曦适配vllm的容器，使用该容器替换sysHax部署过程中的gpu侧的容器（此处适配版本为vllm 0.8.2）。

按照sysHax的指南，在本地完成gpu侧和cpu侧的容器部署

https://gitee.com/openeuler/sysHAX/blob/master/docs/sysHAX_online_deployment_guide.md

部署完成后，我们会获得vllm_gpu和vllm_cpu两个容器

进入vllm_gpu容器，输入

```bash
pip show vllm
```

获取得到vllm的安装路径

进入对应路径后，先对代码进行备份，然后输入
```bash
git apply boostkit-vllm-ops-gpu-082.patch
```

之后退出vllm_gpu容器，进入vllm_cpu容器，并从本仓库dev分支获取sysHAX cpu容器vllm源代码。

```bash
git apply boostkit-vllm-ops-cpu-gptq.patch #适配cpu侧读取gptq量化后的模型
git apply boostkit-vllm-ops-op-speed.patch #加速cpu侧推理速度
VLLM_TARGET_DEVICE=cpu pip install -v -e .
```

完成安装后，继续按照sysHAX的指南启动对应的实例即可。

参考启动方式：
gpu容器：
```bash
vllm serve your_model   --host 0.0.0.0   --port 8001   --dtype=half   --swap_space=16  --block_size=16   --preemption_mode=swap   --max_model_len=4096   --tensor-parallel-size 1   --gpu_memory_utilization=0.95 --enable-auto-pd-offload    --enforce-eager --use_greedy
```

cpu容器：
```
VLLM_USE_V1=0 NRC=4 INFERENCE_OP_MODE=fused OMP_NUM_THREADS=128 CUSTOM_CPU_AFFINITY="0-31,40-71,80-111,120-151" SYSHAX_QUANTIZE=q4_0 vllm serve your_model     --host 0.0.0.0     --port 8002     --dtype=half     --block_size=16     --preemption_mode=swap     --max_model_len=8192     --enable-auto-pd-offload  --tensor-parallel-size 1
```
注：cpu侧不要使用所有的核心，预留部分核心供gpu调度及部分计算使用。

---

**新增环境变量**

| 环境变量 | 可选值 |说明 |
|---------------|------|------|
| SYSHAX_QUANTIZE | q4_0 或 q8_0 | 可用于cpu容器的实例，选择量化方式，用于优化cpu侧推理性能 |
| NRC | 2 或 4 | 可用于cpu侧实例，提升矩阵计算过程的并行数，提升推理性能 |


**新增启动参数**
| 启动参数 | 可选值 |说明 |
|---------------|------|------|
| --use-greedy | 无 | 可同时用于gpu容器和cpu容器的实例，用于提升Sampler部分的性能 |

注:gpu侧默认不启用，启用需要将 **boost-kit-vllm-ops-gpu-082.patch** 文件中 1093行
```python
if isinstance(params, SamplingParams) and self.device_config.device.type == "cpu" and getattr(self, 'use_greedy', True):
```
替换成
```python
if isinstance(params, SamplingParams) and getattr(self, 'use_greedy', True):
```
后，重新添加patch
---

# 贡献指南
如果使用过程中有任何问题，或者需要反馈特性需求和bug报告，可以提交isssues联系我们，具体贡献方法可参考[这里](https://gitcode.com/boostkit/community/blob/master/docs/contributor/contributing.md)。


# 免责声明
此代码仓计划参与vllm软件开源，仅作cpu推理性能提升和，编码风格遵照原生开源软件，继承原生开源软件安全设计，不破坏原生开源软件设计及编码风格和方式，软件的任何漏洞与安全问题，均由相应的上游社区根据其漏洞和安全响应机制解决。请密切关注上游社区发布的通知和版本更新。鲲鹏计算社区对软件的漏洞及安全问题不承担任何责任。
