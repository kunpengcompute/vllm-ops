# vLLM推理框架优化适配指南

本文档基于鲲鹏920新型号处理器平台搭载沐曦曦云C500 GPU * 4环境，提供部署vllM框架运行Qwen3-32B模型的部署调优指导步骤。

## 环境要求

- 硬件平台：鲲鹏920新型号处理器，搭载沐曦曦云C500 GPU * 4  
- 系统要求：openEuler 24.03 LTS sp1，安装完成沐曦对应驱动、SDK及vLLM 0.11.0版本容器。
- 模型选择：[Qwen3-32B模型](https://modelscope.cn/models/Qwen/Qwen3-32B)

## 系统选项优化

### BIOS设置

设置BIOS CPU性能参数，设置路径如下。

CPU PERFORMANCE > MODE HPC/ PERFORMANCE
payload_size > 512B

### 安装高性能内存库

1. 请参见《[鲲鹏系统库 开发指南](https://www.hikunpeng.com/document/detail/zh/kunpengboostkithistory/230RC5/accel/kunpengaccel_ksl_16_0006.html)》，获取到KSL（Kunpeng System Library，鲲鹏系统库）软件包并安装。

2. 安装完成后以如下方式启用KQMalloc。

   ```bash
   LD_PRELOAD=/usr/local/ksl/libkqmalloc.so ./run_your_application
   ```

## 适配优化补丁

1. 进入沐曦提供的容器。

2. 拉取代码，此处以`/home/code`为例。

   ```bash
   mkdir -p /home/code
   cd /home/code
   git clone https://gitcode.com/boostkit/vllm-ops.git
   ```

3. 输入以下命令。

   ```bash
   pip show vllm
   ```

   获取到vLLM的安装目录。

   ```bash
   Name: vllm
   Version: 0.11.0+empty
   Summary: A high-throughput and memory-efficient inference and serving engine for LLMs
   Home-page: https://github.com/vllm-project/vllm
   Author: vLLM Team
   Author-email: 
   License: 
   Location: /opt/conda/lib/python3.12/site-packages
   Requires: aiohttp, blake3, cachetools, cbor2, cloudpickle, compressed-tensors, depyf, diskcache, einops, fastapi, filelock, gguf, lark, llguidance, lm-format-enforcer, mistral_common, msgspec, ninja, numpy, openai, openai-harmony, opencv-python-headless, outlines_core, partial-json-parser, pillow, prometheus-fastapi-instrumentator, prometheus_client, protobuf, psutil, py-cpuinfo, pybase64, pydantic, python-json-logger, pyyaml, pyzmq, regex, requests, scipy, sentencepiece, setproctitle, setuptools, six, tiktoken, tokenizers, tqdm, transformers, typing_extensions, watchfiles, xgrammar
   Required-by: 
   ```

4. 备份vLLM和vLLM-Metax。

   ```bash
   cp -r /opt/conda/lib/python3.12/site-packages/vllm /home/backend/vllm_back
   cp -r /opt/conda/lib/python3.12/site-packages/vllm /home/backend/vllm_metax
   ```

5. 进入软件包目录，合入补丁文件。

   ```bash
   cd /opt/conda/lib/python3.12/site-packages/
   patch -p1 < /home/code/vllm-ops/patch/0001-vllm_0.11.0-optimize-schedular.patch
   patch -p1 < /home/code/vllm-ops/patch/0002-vllm_0.11.0-optimize-sched_yield_on_arm.patch
   ···
   ```

   没有特殊回显信息，则补丁合入成功。

## 推理测试

本文提供的模型测试路径以`/home/models/Qwen3-32B/`为例，使用时请替换成实际使用的路径。

1. 设置环境变量。

   ```bash
   export MACA_SIGNAL_WAIT_MODE=2
   ```

2. 启动服务。

   ```bash
   LD_PRELOAD=/usr/local/ksl/libkqmalloc.so vllm  serve   /home/models/Qwen3-32B/     --host 0.0.0.0     --port 8206      --block_size=16     --max_model_len=9120   --tensor-parallel-size 4     --gpu_memory_utilization=0.95  --no-enable-prefix-caching --trust_remote_code --async-scheduling
   ```

3. 测试参考命令。

   ```bash
   vllm bench serve --backend vllm --model /home/models/Qwen3-32B --dataset-name random --random-input-len 4096 --random-output-len 1024 --request-rate 2 --num-prompts 2 --host 127.0.0.1 --port 8206
   ```
