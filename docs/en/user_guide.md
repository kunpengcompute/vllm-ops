# vLLM Inference Framework Optimization Guide

This document describes how to deploy and optimize the vLLM framework to run the Qwen3-32B model based on the new Kunpeng 920 processor model platform and four MetaX C500 GPUs.

## Environment Requirements

- Hardware platform: new Kunpeng 920 processor model + 4 × MetaX C500 GPU 
- Software platform: openEuler 24.03 LTS SP1, installed MetaX driver and SDK, and installed vLLM v0.11.0 container that can run LLM inference
- Model selection: [Qwen3-32B](https://modelscope.cn/models/Qwen/Qwen3-32B)

## System Option Optimization

### Setting the BIOS

Set the BIOS CPU performance parameters. The paths are as follows:

| BIOS Configuration Item | Recommended Value | Description |
| :--- | :--- | :--- |
| Max Payload Size | 512B | Maximum payload size. The size is in direct proportion to the transmission efficiency of the PCIe link. Setting this parameter to <code>512B</code> can improve the utilization of the PCIe link bandwidth. <br>Path: <code>Advanced</code> > <code>PCIe Configuration</code> > <code>CPU X PCIe Configuration</code> > <code>CPU X PCIe - Port Y</code> > <code>Max Payload Size</code> |
| Power Policy | Performance | Power policy. Setting this parameter to <code>Performance</code> allows the system to guarantee performance output. <br>Path: <code>Advanced</code> > <code>Power And Performance Configuration</code> > <code>Power Policy</code> |
| Performance Profile | HPC | Performance configuration file. Setting this parameter to <code>HPC</code> enables adaptation to heavy-load computing scenarios. <br>Path: <code>Advanced</code> > <code>Power And Performance Configuration</code> > <code>Performance Profile</code> |

### Installing the High-Performance Memory Library

1. Obtain and install the Kunpeng System Library (KSL) software package. For details, see [Kunpeng System Library Developer Guide](https://www.hikunpeng.com/document/detail/en/kunpengboostkithistory/230RC5/accel/kunpengaccel_ksl_16_0006.html).

2. After the installation is complete, enable KQMalloc as follows:

   ```bash
   LD_PRELOAD=/usr/local/ksl/libkqmalloc.so ./run_your_application
   ```

## Optimization Patches

1. Access the container provided by MetaX.

2. Pull code. The following uses `/home/code` as an example.

   ```bash
   mkdir -p /home/code
   cd /home/code
   git clone https://gitcode.com/boostkit/vllm-ops.git
   ```

3. Run the following command:

   ```bash
   pip show vllm
   ```

   The vLLM installation directory is obtained.

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

4. Back up vLLM and vLLM-metax.

   ```bash
   cp -r /opt/conda/lib/python3.12/site-packages/vllm /home/backend/vllm_back
   cp -r /opt/conda/lib/python3.12/site-packages/vllm /home/backend/vllm_metax
   ```

5. Go to the software package directory and apply the patches.

   ```bash
   cd /opt/conda/lib/python3.12/site-packages/
   patch -p1 < /home/code/vllm-ops/patch/0001-vllm_0.11.0-optimize-schedular.patch
   patch -p1 < /home/code/vllm-ops/patch/0002-vllm_0.11.0-optimize-sched_yield_on_arm.patch
   ……
   ```

   If no special command output is displayed, the patches are successfully applied.

## Inference Test

The model test path provided in this document uses `/home/models/Qwen3-32B/` as an example. Replace it with the actual path.

1. Set the environment variable.

   ```bash
   export MACA_SIGNAL_WAIT_MODE=2
   ```

2. Start the model inference service.  
   The reference command is as follows:

   ```bash
   LD_PRELOAD=/usr/local/ksl/libkqmalloc.so vllm  serve   /home/models/Qwen3-32B/     --host 0.0.0.0     --port 8206      --block_size=16     --max_model_len=9120   --tensor-parallel-size 4     --gpu_memory_utilization=0.95  --no-enable-prefix-caching --trust_remote_code --async-scheduling
   ```

3. Execute the performance test.  
   The reference command is as follows:

   ```bash
   vllm bench serve --backend vllm --model /home/models/Qwen3-32B --dataset-name random --random-input-len 4096 --random-output-len 1024 --request-rate 2 --num-prompts 2 --host 127.0.0.1 --port 8206
   ```
