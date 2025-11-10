# boostkit-vllm-ops sysHAX相关patch集合

本目录包含对于openEuler的sysHAX插件的优化和功能适配
用于适配 GPU/CPU 环境以及优化算子运行速度。

## 📦 Patch 列表

| Patch 文件名 | 说明 |
|---------------|------|
| **boostkit-vllm-ops-gpu-082.patch** | 适配 GPU 端运行 **vLLM 0.8.2** 版本，解决接口差异及依赖兼容问题。 |
| **boostkit-vllm-ops-cpu-gptq.patch** | 适配 CPU 端加载 **GPTQ 量化模型**，支持量化参数解析与推理执行。 |
| **boostkit-vllm-ops-op-speed.patch** | 优化算子执行性能，减少 CPU/GPU 端瓶颈，提高整体推理速度。 |

---

## ⚙️ 使用方法

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

进入对应路径后，输入
```bash
git apply boostkit-vllm-ops-gpu-082.patch
```

之后退出vllm_gpu容器，进入vllm_cpu容器，并从仓库 https://gitee.com/openeuler/sysHAX-adapter 获取sysHAX cpu容器vllm源代码。


```bash
git apply boostkit-vllm-ops-cpu-gptq.patch #适配cpu侧读取gptq量化后的模型
git apply boostkit-vllm-ops-op-speed.patch #加速cpu侧推理速度
VLLM_TARGET_DEVICE=cpu pip install -v -e .
```

完成安装后，继续按照sysHAX的指南启动对应的实例即可。