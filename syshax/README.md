# 项目介绍

vllm-ops是鲲鹏自研的向量检索加速组件，对接vllm使用。本仓库中提供了对接openEuler的sysHax插件的适配与优化。vllm-ops适用于鲲鹏920新型号处理器，支持NEON指令（128位宽）和SVE指令（256位宽）。



## Patch 列表

| Patch 文件名 | 说明 |
|---------------|------|
| **boostkit-vllm-ops-gpu-082.patch** | 适配 GPU 端运行 **vLLM 0.8.2** 版本，解决接口差异及依赖兼容问题。 |
| **boostkit-vllm-ops-cpu-gptq.patch** | 适配 CPU 端加载 **GPTQ 量化模型**，支持量化参数解析与推理执行。 |
| **boostkit-vllm-ops-cpu-opt.patch** | 优化算子执行性能，减少 CPU/GPU 端瓶颈，提高整体推理速度。 |

---

## 使用方法

### 前置要求

参考 gpu/npu 厂商的使用教程，在已本地完成对应 gpu/npu 的 **操作系统 + 驱动 + 容器** 的安装部署，有可用的 gpu 容器镜像。

### 部署流程

此处以在一台 kunpeng 920新型号 + 沐曦C500 * 1的容器部署运行 tclf90_deepseek-r1-distill-qwen-32b-gptq-int8 为例。

此环境中已完成沐曦容器的部署，以下以 **vllm-gpu** 代指该容器。

#### 1、获取模型

从 [模型地址](https://modelscope.cn/models/tclf90/deepseek-r1-distill-qwen-32b-gptq-int8) 获取模型，参考链接中方式下载模型到本地目录。

参考下载方式：


```bash
pip install modelscope
modelscope download --model tclf90/deepseek-r1-distill-qwen-32b-gptq-int8 --local_dir /home/models/
```

下载完成后，下载路径中应存在 tclf90_deepseek-r1-distill-qwen-32b-gptq-int8 目录，进入目录后，确认模型的safetensors文件下载无误。

#### 2、获取容器

参考 [sysHax部署指南](https://gitee.com/openeuler/sysHAX/blob/v0.2.1/docs/sysHAX_online_deployment_guide_on_CPU+GPU.md)

获取得到 sysHAX cpu部分的容器（此处为syshax-vllm-cpu:0.2.1版本）

```bash
docker pull hub.oepkgs.net/neocopilot/syshax/syshax-vllm-cpu:0.2.1
```

#### 3、部署容器

此时机器上已经拥有 syshax-vllm-cpu:0.2.1 和 vllm-gpu 两个容器

部署GPU容器

```bash
docker run --name vllm_gpu \
    --ipc="shareable" \
    --shm-size=64g \
    --device=/dev/mxcd \
    --device=/dev/dri \
    -p 8001:8001 \
    -v /home/models:/home/models \
    -w /home/ \
    -itd vllm-gpu bash
```

在上述脚本中：

> --ipc="shareable"：允许容器共享IPC命名空间，可进行进程间通信。
> --shm-size=64g：设置容器共享内存为64G。
> --device=/dev/mxcd --device=/dev/dri：容器访问沐曦硬件需要的参数，使用其他算力卡时请修改为对应参数。
> -p 8001:8001：端口映射，将宿主机的8001端口与容器的8001端口进行映射，开发者可自行修改。
> -v /home/models:/home/models：目录挂载，将宿主机的 /home/models 映射到容器内的 /home/models 内，实现模型共享。开发者可自行修改映射目录。

部署cpu容器

```bash
docker run --name vllm_cpu \
    --ipc container:vllm_gpu \
    --shm-size=64g \
    --privileged \
    -p 8002:8002 \
    -v /home/models:/home/models \
    -w /home/ \
    -itd hub.oepkgs.net/neocopilot/syshax/syshax-vllm-cpu:0.2.1 bash
```

#### 4、使用补丁
进入vllm_gpu容器，输入

```bash
pip show vllm
```

获取得到vllm的安装路径
```bash
Location: /opt/conda/lib/python3.10/site-packages
```

进入对应路径后，先对代码进行备份，然后打上patch
```bash
cd /opt/conda/lib/python3.10/site-packages
cp -r vllm vllm.back
git apply boostkit-vllm-ops-gpu-082.patch
```

之后退出vllm_gpu容器，进入vllm_cpu容器，并从本仓库dev分支获取sysHAX cpu容器vllm源代码。

```bash
yum install -y libomp #安装libomp依赖包
git clone https://gitee.com/openeuler/sysHAX-adapter.git
cd sysHAX-adapter
git checkout tags/vllm_sysHAX
git apply ../boostkit-vllm-ops-cpu-gptq.patch #适配cpu侧读取gptq量化后的模型
git apply ../boostkit-vllm-ops-cpu-opt.patch #加速cpu侧推理速度
VLLM_TARGET_DEVICE=cpu pip install -v .
```

#### 5、启动服务

完成安装后，继续按照sysHAX的指南启动对应的 vllm 实例和 sysHAX 服务即可。

参考启动方式：

gpu容器：
```bash
VLLM_USE_V1=0 taskset -c 36-39,76-79,116-119,156-159 vllm serve /home/models/tclf90_deepseek-r1-distill-qwen-32b-gptq-int8/   --host 0.0.0.0   --port 8001   --dtype=half   --swap_space=16  --block_size=16   --preemption_mode=swap   --max_model_len=4096   --tensor-parallel-size 1   --gpu_memory_utilization=0.95 --enable-auto-pd-offload    --enforce-eager --use_greedy
```

cpu容器：
```bash
VLLM_USE_V1=0 NRC=4 INFERENCE_OP_MODE=fused OMP_NUM_THREADS=128 CUSTOM_CPU_AFFINITY="0-31,40-71,80-111,120-151" SYSHAX_QUANTIZE=q4_0 vllm serve /home/models/tclf90_deepseek-r1-distill-qwen-32b-gptq-int8/     --host 0.0.0.0     --port 8002     --dtype=half     --block_size=16     --preemption_mode=swap     --max_model_len=8192     --enable-auto-pd-offload  --tensor-parallel-size 1 --use_greedy
```

注：
1、cpu侧不要使用所有的核心，预留部分核心供gpu调度及部分计算使用。
2、/home/models/tclf90_deepseek-r1-distill-qwen-32b-gptq-int8/ 为模型存放路径，请替换成实际使用的模型及路径。
3、sysHAX启动方式推荐参考 [sysHAX部署指南](https://gitee.com/openeuler/sysHAX/blob/v0.2.1/docs/sysHAX_online_deployment_guide_on_CPU+GPU.md) 的源码部署模式进行部署，并参考指南中方式修改端口和并发量。


**新增环境变量**

| 环境变量 | 可选值 |说明 |
|---------------|------|------|
| SYSHAX_QUANTIZE | q4_0 或 q8_0 | 可用于cpu容器的实例，选择量化方式，用于优化cpu侧推理性能 |
| NRC | 2 或 4 | 可用于cpu侧实例，提升矩阵计算过程的并行数，提升推理性能 |

推荐使用组合为 **SYSHAX_QUANTIZE=q4_0 NRC=4** 或 **SYSHAX_QUANTIZE=q8_0 NRC=2**

**新增启动参数**
| 启动参数  |说明 |
|---------------|------|
| --use-greedy | 可同时用于gpu容器和cpu容器的实例，用于提升Sampler部分的性能 |
| --enable-auto-pd-offload | 支持pd分离，使用sysHAX时需要添加 |

---

# 通信矩阵

| 源设备 | 源ip地址 | 源端口 | 目的设备 | 目的IP地址 | 目的端口（侦听）| 协议 | 端口说明 | 侦听端口是否可更改 | 认证方式 | 所属平面 | 版本 | 特殊场景 | 
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 用户所在服务器| 用户服务器IP地址 | * | 运行sysHAX的服务器| 8000~9000| 运行sysHAX的IP地址 | HTTP/HTTPS | 接受用户推理请求 |  是 | N/A | N/A | 业务面 | 所有版本 | 无 |
| sysHAX所在服务器| sysHAX服务器IP地址 | * | 运行vllm-gpu的服务器| 8000~9000| 运行vllm-gpu的IP地址 | HTTP/HTTPS | 接受sysHAX转发的用户推理请求 |  是 | N/A | N/A | 业务面 | 所有版本 | 无 |
| sysHAX所在服务器| sysHAX服务器IP地址 | * | 运行vllm-cpu的服务器 | 8000~9000| 运行vllm-cpu的IP地址 | HTTP/HTTPS | 接受用户推理请求 |  是 | N/A | N/A | 业务面 | 所有版本 | 无 |



# 贡献指南
如果使用过程中有任何问题，或者需要反馈特性需求和bug报告，可以提交isssues联系我们，具体贡献方法可参考[这里](https://gitcode.com/boostkit/community/blob/master/docs/contributor/contributing.md)。


# 免责声明
此代码仓计划参与vllm软件开源，仅作cpu推理性能提升和gpu侧的框架适配，编码风格遵照原生开源软件，继承原生开源软件安全设计，不破坏原生开源软件设计及编码风格和方式，软件的任何漏洞与安全问题，均由相应的上游社区根据其漏洞和安全响应机制解决。请密切关注上游社区发布的通知和版本更新。鲲鹏计算社区对软件的漏洞及安全问题不承担任何责任。
