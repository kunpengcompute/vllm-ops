# vLLM推理框架优化适配指南

本文档面向沐曦曦云C500 GPU的推理场景，使用沐曦提供的vLLM 0.15.0容器作为基础环境，从源码编译vLLM 0.17.0和vLLM-MetaX 0.17.0，合入vLLM 0.18.0、0.19.0、0.20.0的核心性能优化特性回合补丁，并叠加opt优化和vLLM-MetaX FlashAttention metadata优化，最终在4卡环境运行Qwen3-32B模型。

>![表示说明的图片](public_sys-resources/icon-note.gif) **说明：** 本指南全流程已在纯净vLLM 0.15.0容器上端到端实测通过，即从基础镜像编译出0.17.0，serve启动成功，推理跑通。

## 环境要求

- 硬件平台：搭载4张沐曦曦云C500 GPU。
- 系统要求：安装与容器匹配的沐曦驱动；使用沐曦提供的vLLM 0.15.0容器，容器内已包含MACA-SDK、PyTorch及源码编译依赖环境。基础容器镜像请自行准备。
- Python版本：3.12。
- 目标软件版本：vLLM 0.17.0、vLLM-MetaX 0.17.0；0.15.0容器仅作为编译基础环境，不能继续使用容器内预装的vLLM 0.15.0运行本补丁。
- 网络：编译过程需从[AtomGit](https://atomgit.com/boostkit/vllm-ops)与[GitHub](https://github.com)拉取源码，容器需能访问网络（见[步骤1](#step1)）。
- 模型选择：[Qwen3-32B模型](https://modelscope.cn/models/Qwen/Qwen3-32B)。

## 适配优化补丁

以下命令均在vLLM 0.15.0容器内执行。补丁在源码编译前合入vLLM 0.17.0和vLLM-MetaX 0.17.0源码树，随后源码安装生效。建议使用同一个终端按顺序完成所有步骤，以便复用环境变量。

<a id="step1"></a>

### 启动并进入沐曦提供的vLLM 0.15.0容器，安装编译前置依赖并准备环境变量

1. 前置依赖（基础镜像默认未装git、自带cmake版本过低，先补齐）。以下命令用于安装git和cmake。

   ```bash
   yum install -y git
   pip install cmake
   ```

2. 网络连通性：编译需访问网络。若容器内DNS超时/`git clone`卡住，通常是**宿主机未开启IP转发**，在**宿主机**执行`sysctl -w net.ipv4.ip_forward=1`后重试。github直连若过慢或不通，[步骤3](#step3)/[步骤4](#step4)已给出gh-proxy备选下载方式（第三方中转，仅建议在受限网络使用）。

3. 编译环境变量（注意PATH需包含`/opt/conda/bin`）。以下命令用于设置编译环境变量。

   ```bash
   export WORK_DIR=/home/code
   export MACA_PATH="/opt/maca"
   export CUCC_PATH="${MACA_PATH}/tools/cu-bridge"
   export CUDA_PATH="${HOME}/cu-bridge/CUDA_DIR"
   export CUCC_CMAKE_ENTRY=2
   export PATH=/opt/conda/bin:${MACA_PATH}/mxgpu_llvm/bin:${MACA_PATH}/bin:${CUCC_PATH}/tools:${CUCC_PATH}/bin:${PATH}
   export LD_LIBRARY_PATH=${MACA_PATH}/lib:${MACA_PATH}/ompi/lib:${MACA_PATH}/mxgpu_llvm/lib:${LD_LIBRARY_PATH}
   export SETUPTOOLS_SCM_PRETEND_VERSION=0.17.0
   ```

<a id="step2"></a>

### 拉取本仓库补丁合集

此处以`/home/code`为例。以下命令用于拉取本仓库补丁合集。

```bash
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"
git clone https://atomgit.com/boostkit/vllm-ops.git vllm-ops
export OPS_DIR="$WORK_DIR/vllm-ops"
```

<a id="step3"></a>

### 合入vLLM-MetaX优化补丁并源码编译vLLM-MetaX 0.17.0

1. 下载vLLM-MetaX 0.17.0源码。

   ```bash
   cd "$WORK_DIR"
   git clone --branch releases/v0.17.0 https://github.com/MetaX-MACA/vLLM-metax
   cd vLLM-metax
   ```

   备选：github直连不通时，使用gh-proxy代理下载（第三方中转，仅建议在受限网络使用）。

   ```bash
   curl -sSL -o metax.tgz https://gh-proxy.com/https://codeload.github.com/MetaX-MACA/vLLM-metax/tar.gz/refs/heads/releases/v0.17.0
   mkdir -p vLLM-metax && tar xzf metax.tgz -C vLLM-metax --strip-components=1
   cd vLLM-metax
   ```

2. 合入FlashAttention metadata优化补丁。

   ```bash
   patch -p1 --forward < "$OPS_DIR/patch/vllm-metax-feature-patches/vllm-metax-flash-attn-metadata-optimizations.patch"
   ```

3. 源码编译安装。

   ```bash
   python use_existing_metax.py
   pip install -r requirements/build.txt
   pip install . --no-build-isolation
   ```

<a id="step4"></a>

### 合入高版本特性回合补丁与opt优化补丁，并源码编译vLLM 0.17.0

1. 下载vLLM 0.17.0源码。

   ```bash
   cd "$WORK_DIR"
   git clone --depth 1 --branch releases/v0.17.0 https://github.com/vllm-project/vllm
   cd vllm
   ```

   备选：github直连不通时，使用gh-proxy代理下载（第三方中转，仅建议在受限网络使用）。

   ```bash
   curl -sSL -o vllm.tgz https://gh-proxy.com/https://codeload.github.com/vllm-project/vllm/tar.gz/refs/heads/releases/v0.17.0
   mkdir -p vllm && tar xzf vllm.tgz -C vllm --strip-components=1
   cd vllm
   ```

2. 按series顺序合入0001至0085高版本特性回合补丁。

   ```bash
   while read -r p; do
       patch -p1 --forward < "$OPS_DIR/patch/vllm-backport-feature-patches/$p"
   done < "$OPS_DIR/patch/vllm-backport-feature-patches/series"
   ```

3. 合入opt优化补丁（greedy sampler快路径与block table脏行局部拷贝）。

   ```bash
   patch -p1 --forward < "$OPS_DIR/patch/vllm-opt-patch/vllm-0.17.0-opt.patch"
   ```

4. 源码编译安装。

   ```bash
   python use_existing_torch.py
   pip install -r requirements/build.txt
   VLLM_TARGET_DEVICE=empty pip install . --no-build-isolation
   ```

补丁合入时若无`FAILED`或`Hunk`失败等特殊回显信息，则补丁合入成功。

<a id="step5"></a>

### 适配numpy版本

以下命令用于适配numpy版本。

```bash
pip install numpy==1.26.4
```

编译完成后可执行`pip show vllm vllm-metax`确认两者版本均为`0.17.0`。

## 推理测试

本文提供的模型测试路径以`/home/models/Qwen3-32B/`为例，使用时请替换成实际使用的路径。

1. 设置环境变量。

   ```bash
   export MACA_SIGNAL_WAIT_MODE=2
   ```

2. 启动服务。

   ```bash
   vllm serve /home/models/Qwen3-32B \
       --host 0.0.0.0 \
       --port 8206 \
       --max_model_len=2048 \
       --tensor-parallel-size 4 \
       --gpu_memory_utilization=0.9 \
       --no-enable-prefix-caching \
       --async-scheduling \
       --trust-remote-code
   ```

   回显`Application startup complete`表示服务启动成功。

3. 测试参考命令。

   ```bash
   vllm bench serve \
       --backend vllm \
       --model /home/models/Qwen3-32B \
       --base-url http://127.0.0.1:8206 \
       --endpoint /v1/completions \
       --dataset-name random \
       --random-input-len 100 \
       --random-output-len 100 \
       --num-prompts 4 \
       --max-concurrency 4 \
       --request-rate inf \
       --ignore-eos
   ```

## 常见问题排查

**问题**：执行git命令时提示`git: command not found`。

**原因**：基础镜像未预装git。

**答复**：执行`yum install -y git`安装git，详见[步骤1](#step1)前置依赖。

**问题**：编译时提示`CMake 3.26 is required, running 3.22`。

**原因**：基础镜像自带的cmake版本过低。

**答复**：执行`pip install cmake`安装新版cmake，详见[步骤1](#step1)前置依赖。

**问题**：容器内出现DNS超时，`git clone`或`curl`卡死。

**原因**：宿主机未开启IP转发，容器无法联网。

**答复**：在宿主机执行`sysctl -w net.ipv4.ip_forward=1`后重试。

**问题**：`github.com`直连超时或极慢。

**原因**：当前网络到github受限。

**答复**：优先使用`git clone`，受限时可用`https://gh-proxy.com/`前缀下载tar包（第三方中转），详见[步骤3](#step3)/[步骤4](#step4)。

**问题**：编译时提示`metadata-generation-failed`（版本推导失败）。

**原因**：tar包源码无`.git`，setuptools_scm推导不出版本。

**答复**：已在[步骤1](#step1)显式设置`SETUPTOOLS_SCM_PRETEND_VERSION=0.17.0`。

**问题**：serve启动时提示`No module named vllm.attention`。

**原因**：vLLM与vLLM-MetaX版本不匹配。

**答复**：确认两者都编译到0.17.0，可用`pip show`核对。

## 修订记录

| 文档编号 | 发布日期 | 修改说明 |
| --- | --- | --- |
| 02 | 2026-09-30 | v2.1.0版本修订更新。 |
| 01 | 2026-06-30 | v2.0.0版本第一次正式发布。 |
