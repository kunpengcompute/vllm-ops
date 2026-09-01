# workplace

开发验证脚本。最终部署方案见 `../vllm/` 和 `../deploy.sh`。

## 独立脚本

| 脚本 | 说明 |
|------|------|
| `openvla_preprocess_pipeline.py` | 完整参考流水线，独立于 vLLM 可运行，支持 `--validate` / `--bench` / `--profile` |
| `lib_preprocess.cpp` | 独立 `.so` 版本，ctypes 直调 NEON kernel，不依赖 vLLM |

## edition1/

早期测试套件，通过 ctypes 直调 `libpreprocess.so`，不依赖 vLLM。

| 脚本 | 说明 |
|------|------|
| `bench_all.py` | 多尺寸多线程性能基准 |
| `e2e_verify.py` | 10 种尺寸确定性正确性验证 |
| `step_breakdown.py` | 原版 7 步逐步骤延迟分解 |
| `test_torch_op.py` | JIT 编译 C++ kernel 为 torch 算子（概念验证） |

