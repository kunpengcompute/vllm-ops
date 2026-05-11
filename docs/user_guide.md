# vLLM推理框架优化适配指南

本文档是在鲲鹏920新型号平台上，搭载沐曦曦云C500 GPU * 4，部署vllm框架运行Qwen3-32B模型的部署调优指南

## 环境要求
硬件平台：鲲鹏920新型号平台上，搭载沐曦曦云C500 GPU * 4  
系统要求：openEuler 24.03 LTS sp1，安装完成沐曦对应驱动、SDK及vLLM 0.11.0版本容器。
模型选择：[Qwen3-32B模型]()

## 系统选项优化

### BIOS设置
CPU PERFORMANCE -> MODE HPC/ PERFORMANCE
payload_size -> 512B

### 高性能内存库
参考https://www.hikunpeng.com/document/detail/zh/kunpengboostkithistory/230RC5/accel/kunpengaccel_ksl_16_0006.html，获取到ksl软件包并安装

安装完成后以如下方式启用kqmalloc：
LD_PRELOAD=/usr/local/ksl/libkqmalloc.so ./run_your_application

## 优化patch适配
1. 进入沐曦提供的容器

拉取代码
```bash
mkdir -p /home/code
cd /home/code
git clone https://gitcode.com/boostkit/vllm-ops.git
```

输入 
```bash
pip show vllm
```
获取得到vllm的安装目录
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
备份vllm和vllm_metax
```bash
cp -r /opt/conda/lib/python3.12/site-packages/vllm /home/backend/vllm_back
cp -r /opt/conda/lib/python3.12/site-packages/vllm /home/backend/vllm_metax
```
进入vllm目录，打上vllm的patch
```bash
patch -pXXX < XXX.patch 
```
进入vllm_metax目录，打上vllm_metax的patch
```bash
patch -pXXX < XXX.patch
```
无反馈即为成功打上patch


5. 推理测试

参考测试方法：
模型路径：/xxx/Qwen3-32B

设置环境变量
```bash
export MACA_SIGNAL_WAIT_MODE=2
```

启动服务：
```bash
LD_PRELOAD=/usr/local/ksl/libkqmalloc.so vllm  serve   /home/models/Qwen3-32B/     --host 0.0.0.0     --port 8206      --block_size=16     --max_model_len=9120   --tensor-parallel-size 4     --gpu_memory_utilization=0.95  --no-enable-prefix-caching --trust_remote_code --async-scheduling
```
执行命令：
```bash
vllm bench serve --backend vllm --model /home/models/Qwen3-32B --dataset-name random --random-input-len 4096 --random-output-len 1024 --request-rate 2 --num-prompts 2 --host 127.0.0.1 --port 8002
```