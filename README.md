# OpenVLA 图片前处理加速

将 OpenVLA 的图片前处理管线（bicubic resize + DINOv2/SigLIP 双路归一化）从 Python (PIL+NumPy) 迁移至 ARM NEON C++ 多核实现。1920×1080 全管线从 20.16ms 降至 1.30ms，加速 **15.5×**。

---

## 1. 项目简介

OpenVLA 是机器人视觉-语言-动作模型，输入摄像头图片和自然语言指令，输出机器人动作。每张图片在推理前需要经过前处理：缩放到 224×224，再做 DINOv2 和 SigLIP 双路归一化，拼成 `float32 [6, 224, 224]`。

原版 vLLM 用 PIL + NumPy 实现，瓶颈在 PIL resize（单线程，占 53.6%）和 `torch.from_numpy` 拷贝（占 31.3%）。本项目将整个前处理下沉到 C++，用 ARM NEON SIMD 指令单次遍历完成所有数值计算，OpenMP 多线程并行化 resize。一次调用：`uint8 [H,W,3]` 进，`float32 [6,224,224]` 出，中间不经过 Python 解释器，不分配临时数组。

详细技术说明见 `docs/openvla_preprocess_design.md`，完整数据见 `docs/openvla_preprocess_final_report.md`。

---

## 2. 目录结构

```text
├── vllm/                             # 放入 vLLM 源码树的文件
│   ├── csrc/cpu/
│   │   └── openvla_image_preprocess.cpp    # NEON kernel
│   └── vllm/transformers_utils/processors/
│       └── openvla.py                      # Python dispatch
│
├── deploy.sh                         # 一键部署脚本
├── test_openvla_kernel.py            # 端到端测试
│
├── docs/                             # 技术文档
│   ├── openvla_preprocess_design.md       # 设计文档（架构与技术细节）
│   ├── openvla_preprocess_final_report.md # 最终技术报告
│   └── figures/                           # 架构图（PNG 格式）
│       ├── data_layout_transform.png
│       ├── dataflow_diff.png
│       └── neon_normalize.png
│
└── workplace/                        # 独立验证脚本（不依赖 vLLM）
    ├── openvla_preprocess_pipeline.py     # 完整参考流水线
    ├── lib_preprocess.cpp                # 独立 .so 版本
    └── edition1/                          # 测试套件
```

---

## 3. 环境要求

| 项目 | 最低 | 推荐 |
|------|------|------|
| CPU | ARM aarch64 + NEON | Huawei Kunpeng 920|
| 操作系统 | Linux (aarch64) | openEuler 24.03|
| 编译器 | GCC 10+ | GCC 12+, `-march=armv8.2-a+fp16+dotprod` |
| Python | 3.10+ | 3.11 |
| 依赖 | `numactl-devel`, `ninja` | — |

x86 / macOS 走标量 fallback，功能正常但无 NEON 加速。

---

## 4. 运行前检查清单

- [ ] **ARM 架构**：`uname -m` 应输出 `aarch64`
- [ ] **编译器支持 NEON**：`gcc -march=armv8.2-a+fp16+dotprod -E - < /dev/null` 无报错
- [ ] **numactl-devel 已安装**：`rpm -q numactl-devel` 或 `dpkg -l | grep numactl`
- [ ] **ninja 已安装**：`which ninja`
- [ ] **vLLM 源码已克隆**：`git clone https://github.com/vllm-project/vllm.git`

---

## 5. 快速开始

### 方式一：自动化部署

```bash
cd <本项目根目录>
bash deploy.sh <vLLM源码目录>

# 例如: bash deploy.sh /home/user/vllm
```

`deploy.sh` 自动完成文件复制和 cmake 补丁。

### 方式二：手动部署

**① 复制文件**

```bash
cp vllm/csrc/cpu/openvla_image_preprocess.cpp <vllm>/csrc/cpu/
cp vllm/vllm/transformers_utils/processors/openvla.py <vllm>/vllm/transformers_utils/processors/
```

**② 修改 cmake**

在 `<vllm>/cmake/cpu_extension.cmake` 中找到 `"csrc/cpu/shm.cpp"` 一行，其后追加：

```cmake
        "csrc/cpu/openvla_image_preprocess.cpp"
```

**③ 编译**

```bash
cd <vllm>
pip3 install -e . --no-build-isolation
```

---

## 6. 路径约定

| 占位符 | 含义 | 示例 |
|--------|------|------|
| `<本项目根目录>` | 本项目的根目录 | `/home/user/openvla_preprocess` |
| `<vllm>` | vLLM 源码根目录 | `/home/user/vllm` |
| `<vllm>/csrc/cpu/` | vLLM C++ CPU 扩展目录 | `/home/user/vllm/csrc/cpu/` |
| `<vllm>/vllm/.../processors/` | vLLM processors 目录 | `/home/user/vllm/vllm/transformers_utils/processors/` |
| `<vllm>/cmake/cpu_extension.cmake` | vLLM CPU 扩展 cmake | `/home/user/vllm/cmake/cpu_extension.cmake` |

部署后 vLLM 源码树新增/修改的文件：

```text
<vllm>/
├── csrc/cpu/
│   └── openvla_image_preprocess.cpp    ← 新增
├── cmake/
│   └── cpu_extension.cmake            ← 追加一行
└── vllm/transformers_utils/processors/
    └── openvla.py                      ← 替换
```

---

## 7. 测试验证方式

### 集成测试

```bash
python3 test_openvla_kernel.py
```

预期：16 项全部通过。

### 性能基准

```bash
python3 test_openvla_kernel.py --bench
```

### 手动验证算子已注册

```bash
python3 -c "
from vllm.transformers_utils.processors.openvla import preprocess_openvla_image
import torch
print(torch.ops._C.openvla_fused_preprocess)
"
```

> 必须 import `openvla` 模块才能触发算子注册（`import vllm` 不会自动加载 processor）。

预期输出包含 `_C.openvla_fused_preprocess`。报 AttributeError 则算子注册失败（走 Python fallback）。

### 独立验证（不依赖 vLLM）

```bash
cd workplace
g++ -std=c++14 -march=armv8.2-a+fp16+dotprod -fopenmp -O3 -shared -fPIC \
    -o libpreprocess.so lib_preprocess.cpp
python3 edition1/e2e_verify.py
```

---

## 参考

- [最终技术报告](docs/openvla_preprocess_final_report.md) — 完整性能数据
- [设计文档](docs/openvla_preprocess_design.md) — 架构和技术细节
