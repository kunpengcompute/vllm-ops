# Introduction to vLLM-ops

## Latest Updates

- [2026-06-30]: Based on the new Kunpeng 920 processor model, optimizes the performance based on the vLLM v0.11.0 container environment provided by MetaX.

## Project Introduction

vLLM-ops is a performance improvement patch repository for large language model (LLM) inference based on the new Kunpeng 920 processor model and MetaX C500. It uses Python code optimization on the CPU side to reduce data transmission between the CPU and GPU, and OS-side optimization to improve throughput. This project optimizes the performance based on the vLLM v0.11.0 container environment provided by MetaX.

## Directory Structure

```text
vllm-ops/
├── patch                                                                    # Patch file directory              
│   ├── 0001-vllm_0.11.0-optimize-schedular.patch                            
│   ├── 0002-vllm_0.11.0-optimize-sched_yield_on_arm.patch                   
│   ├── 0003-vllm_0.11.0-optimize-JIT.patch                                  
│   ├── 0004-vllm_0.11.0-optimize-batch_update_np_array.patch  
│   ├── 0005-vllm_0.11.0-refactor-extract_all_gather_for_cuda_graph.patch  
│   ├── 0006-vllm_0.11.0-optimize-reduce_numpy_split_operations.patch                
│   └── 0007-vllm_metax_0.11.0-dev-move_compute_to_gpu.patch
├── docs
|   └── en                                                                    # English document directory
│      ├── feature_introduction.md                                            # Feature description document
│      ├── menu_vllm_ops.md                                                   # Document guide
│      ├── release_notes.md                                                   # Basic information and feature updates of each release version
│      └── user_guide.md                                                      # User Guide
├── LICENSE                                                                   # Open-source license file
├── CC-BY                                                                     # Open-source document license file
└── README_en.md                                                              # Project introduction
```

## Release Notes

For details about the vLLM-ops version description, see [Release Notes](./docs/en/release_notes.md).

## Documents

|  Document Name|Description  |
| ------------ | ------------ |
| [Release Notes](./docs/en/release_notes.md) | Provides basic information and feature updates of each vLLM-ops version. |
|  [Feature Introduction](./docs/en/feature_introduction.md)|  Provides vLLM-ops optimization description.|
|  [User Guide](./docs/en/user_guide.md)|  Provides vLLM-ops optimization usage description.|

## Contribution Statement

We welcome your contributions to the community. If you have any questions/suggestions or want to provide feedback on feature requirements and bug reports, you can submit issues. For details, see the [contribution guideline](https://gitcode.com/boostkit/community/blob/master/docs/contributor/contributing.md). You are also welcome to share insights in [Discussions](https://gitcode.com/boostkit/community/discussions). Thank you for your support.

## Disclaimer

This code repository contributes to the vLLM and vLLM-metax open-source components. It strictly adheres to the coding style and methods, as well as security design of the native open-source software. Any vulnerability and security issues of the software shall be resolved by the corresponding upstream communities according to their response mechanisms. Please pay attention to the notifications and version updates released by the upstream communities. The Kunpeng computing community does not assume any responsibility for software vulnerabilities and security issues.

## License

This project is released under the Apache License 2.0. For details, see [LICENSE](./LICENSE).
The documents of this project are licensed under CC-BY 4.0. For details, see [LICENSE](./docs/LICENSE).

## Acknowledgments

vLLM-ops is jointly developed by the following Huawei department:

Kunpeng Computing BoostKit Development Dept

Thank you to everyone in the community for your PRs. We warmly welcome contributions to vLLM-ops!
