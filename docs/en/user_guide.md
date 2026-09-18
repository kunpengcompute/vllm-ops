# vLLM Inference Framework Optimization Guide

This document targets the inference scenario on the MetaX C500 GPU. It uses the vLLM 0.15.0 container provided by MetaX as the base environment, compiles vLLM 0.17.0 and vLLM-MetaX 0.17.0 from source, applies the core performance optimization feature backport patches of vLLM 0.18.0, 0.19.0, and 0.20.0, additionally applies opt optimizations and vLLM-MetaX FlashAttention metadata optimizations, and finally runs the Qwen3-32B model on a 4-GPU environment.

>![note icon](public_sys-resources/icon-note.gif) **Note:** This entire flow has been verified end-to-end on a clean vLLM 0.15.0 container, that is, vLLM 0.17.0 was compiled from the base image, the serve command started successfully, and inference passed.

## Environment Requirements

- Hardware platform: 4 × MetaX C500 GPU.
- Software platform: install the MetaX driver matching the container; use the vLLM 0.15.0 container provided by MetaX, which already contains the MACA-SDK, PyTorch, and source-build dependencies. Prepare the base container image yourself.
- Python version: 3.12.
- Target software versions: vLLM 0.17.0 and vLLM-MetaX 0.17.0. The 0.15.0 container is only used as the build base environment; the vLLM 0.15.0 preinstalled in the container cannot be used to run these patches.
- Network: the build pulls source from [GitCode](https://gitcode.com/boostkit/vllm-ops) and [GitHub](https://github.com), so the container needs network access (see [step 1](#step1)).
- Model selection: [Qwen3-32B](https://modelscope.cn/models/Qwen/Qwen3-32B).

## Optimization Patches

The following commands are all executed inside the vLLM 0.15.0 container. The patches are applied to the vLLM 0.17.0 and vLLM-MetaX 0.17.0 source trees before source compilation, and then take effect after source installation. It is recommended to complete all steps in sequence in the same terminal to reuse the environment variables.

<a id="step1"></a>

### Start and access the vLLM 0.15.0 container provided by MetaX, install build prerequisites, and prepare the environment variables

1. Prerequisites (the base image ships without git and with a too-old cmake — install them first). The following commands install git and cmake.

   ```bash
   yum install -y git
   pip install cmake
   ```

2. Network connectivity: the build needs network access. If DNS times out or `git clone` hangs inside the container, the **host** usually has IP forwarding disabled. Run `sysctl -w net.ipv4.ip_forward=1` on the host and retry. If github is slow or unreachable, [step 3](#step3)/[step 4](#step4) provide an alternative gh-proxy download (third-party relay; recommended only in restricted networks).

3. Build environment variables (note: PATH must include `/opt/conda/bin`). The following commands set the build environment variables.

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

### Pull this patch collection

The following uses `/home/code` as an example. The following command pulls this patch collection.

```bash
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"
git clone https://gitcode.com/boostkit/vllm-ops.git vllm-ops
export OPS_DIR="$WORK_DIR/vllm-ops"
```

<a id="step3"></a>

### Apply the vLLM-MetaX optimization patch and compile vLLM-MetaX 0.17.0 from source

1. Download the vLLM-MetaX 0.17.0 source.

   ```bash
   cd "$WORK_DIR"
   git clone --branch releases/v0.17.0 https://github.com/MetaX-MACA/vLLM-metax
   cd vLLM-metax
   ```

   Alternative: when github is unreachable, use the gh-proxy relay (third-party relay; recommended only in restricted networks).

   ```bash
   curl -sSL -o metax.tgz https://gh-proxy.com/https://codeload.github.com/MetaX-MACA/vLLM-metax/tar.gz/refs/heads/releases/v0.17.0
   mkdir -p vLLM-metax && tar xzf metax.tgz -C vLLM-metax --strip-components=1
   cd vLLM-metax
   ```

2. Apply the FlashAttention metadata optimization patch.

   ```bash
   patch -p1 --forward < "$OPS_DIR/patch/vllm-metax-feature-patches/vllm-metax-flash-attn-metadata-optimizations.patch"
   ```

3. Build and install from source.

   ```bash
   python use_existing_metax.py
   pip install -r requirements/build.txt
   pip install . --no-build-isolation
   ```

<a id="step4"></a>

### Apply the high-version feature backport patches and the opt optimization patch, and compile vLLM 0.17.0 from source

1. Download the vLLM 0.17.0 source.

   ```bash
   cd "$WORK_DIR"
   git clone --depth 1 --branch releases/v0.17.0 https://github.com/vllm-project/vllm
   cd vllm
   ```

   Alternative: when github is unreachable, use the gh-proxy relay (third-party relay; recommended only in restricted networks).

   ```bash
   curl -sSL -o vllm.tgz https://gh-proxy.com/https://codeload.github.com/vllm-project/vllm/tar.gz/refs/heads/releases/v0.17.0
   mkdir -p vllm && tar xzf vllm.tgz -C vllm --strip-components=1
   cd vllm
   ```

2. Apply the 0001-0085 high-version feature backport patches in series order.

   ```bash
   while read -r p; do
       patch -p1 --forward < "$OPS_DIR/patch/vllm-backport-feature-patches/$p"
   done < "$OPS_DIR/patch/vllm-backport-feature-patches/series"
   ```

3. Apply the opt optimization patch (greedy sampler fast path + block table dirty-row local copy).

   ```bash
   patch -p1 --forward < "$OPS_DIR/patch/vllm-opt-patch/vllm-0.17.0-opt.patch"
   ```

4. Build and install from source.

   ```bash
   python use_existing_torch.py
   pip install -r requirements/build.txt
   VLLM_TARGET_DEVICE=empty pip install . --no-build-isolation
   ```

If no `FAILED` or `Hunk` failure messages are displayed during patching, the patches are successfully applied.

<a id="step5"></a>

### Adapt the numpy version

The following command adapts the numpy version.

```bash
pip install numpy==1.26.4
```

After the build completes, run `pip show vllm vllm-metax` to confirm both are version `0.17.0`.

## Inference Test

The model test path provided in this document uses `/home/models/Qwen3-32B/` as an example. Replace it with the actual path.

1. Set the environment variable.

   ```bash
   export MACA_SIGNAL_WAIT_MODE=2
   ```

2. Start the model inference service.

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

   The message `Application startup complete` indicates that the service has started successfully.

3. Execute the performance test.

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

## Troubleshooting

**Problem**: Running a git command reports `git: command not found`.

**Cause**: git is not preinstalled in the base image.

**Solution**: Run `yum install -y git` to install git; see [step 1](#step1) prerequisites.

**Problem**: The build reports `CMake 3.26 is required, running 3.22`.

**Cause**: The cmake bundled with the base image is too old.

**Solution**: Run `pip install cmake` to install a newer cmake; see [step 1](#step1) prerequisites.

**Problem**: DNS times out inside the container, or `git clone`/`curl` hangs.

**Cause**: IP forwarding is disabled on the host, so the container has no network access.

**Solution**: Run `sysctl -w net.ipv4.ip_forward=1` on the host, then retry.

**Problem**: `github.com` times out or is extremely slow.

**Cause**: The network path to github is restricted.

**Solution**: Prefer `git clone`; when restricted, prefix downloads with `https://gh-proxy.com/` (third-party relay); see [step 3](#step3)/[step 4](#step4).

**Problem**: The build reports `metadata-generation-failed` (version cannot be derived).

**Cause**: The tar source lacks `.git`, so setuptools_scm cannot derive the version.

**Solution**: The version is already set explicitly via `SETUPTOOLS_SCM_PRETEND_VERSION=0.17.0` in [step 1](#step1).

**Problem**: The serve command reports `No module named vllm.attention`.

**Cause**: The vLLM and vLLM-MetaX versions mismatch.

**Solution**: Ensure both are compiled to 0.17.0, which can be verified with `pip show`.

## Revision History

| Document No. | Publish Date | Change Description |
| --- | --- | --- |
| 02 | 2026-09-30 | Revision for v2.1.0. |
| 01 | 2026-06-30 | First official release of v2.0.0. |
