# Introduction to vLLM High-Version Feature Backport

English|[简体中文](./README.md)

## Latest Updates

- [2026-09-30]: Released a patch collection for the inference scenario on the MetaX C500 GPU. Based on the vLLM 0.15.0 container, it compiles vLLM 0.17.0 and vLLM-MetaX 0.17.0 from source and backports the core performance optimization features of vLLM 0.18.0, 0.19.0, and 0.20.0.
- [2026-06-30]: Released an optimization patch collection for community edition vLLM 0.11.0 and MetaX edition vLLM-MetaX 0.11.0-dev.

## Project Introduction

The vLLM High-Version Feature Backport project targets the inference scenario on the MetaX C500 GPU. It uses the vLLM 0.15.0 container provided by MetaX as the base environment, compiles vLLM 0.17.0 and vLLM-MetaX 0.17.0 from source inside the container, backports 38 feature points from vLLM 0.18.0, 0.19.0, and 0.20.0 to eliminate the performance disadvantages caused by version lag, and additionally applies opt optimizations and vLLM-MetaX FlashAttention metadata optimizations to further improve inference performance.

## Directory Structure

```text
vllm-ops/
├── patch                                                                           # Patch file directory
│   ├── vllm-backport-feature-patches                                               # Main high-version feature backport patch package
│   │   ├── 0001-benchmarks__kernels__cpu__benchmark_cpu_attn.py.patch
│   │   ├── ...                                                                      # File-level backport patches 0002 to 0084
│   │   ├── 0085-vllm__v1__worker__gpu_ubatch_wrapper.py.patch
│   │   ├── manifest.csv                                                            # Mapping between patches and source files
│   │   ├── series                                                                  # Patch application order
│   │   └── README.md                                                               # Main patch package description
│   ├── vllm-opt-patch                                                              # Standalone vLLM opt optimization patch package
│   │   ├── vllm-0.17.0-opt.patch                                                   # greedy sampler and block table merge patch
│   │   └── README.md                                                               # opt patch package description
│   └── vllm-metax-feature-patches                                                  # vLLM-MetaX optimization patch package
│       ├── vllm-metax-flash-attn-metadata-optimizations.patch                      # FlashAttention metadata optimization patch
│       └── README.md                                                               # vLLM-MetaX patch package description
├── docs
│   └── en                                                                          # English document directory
│       ├── feature_introduction.md                                                 # Feature description
│       ├── menu_vllm_ops.md                                                         # Document guide
│       ├── release_notes.md                                                         # Release Notes
│       └── user_guide.md                                                           # User Guide
├── LICENSE                                                                         # Open-source license file
├── CC-BY                                                                           # Open-source document license file
└── README_en.md                                                                    # Project introduction
```

## Release Notes

For details about the version description, see [Release Notes](./docs/en/release_notes.md).

## Documents

| Document Name | Description |
| ------------ | ------------ |
| [Release Notes](./docs/en/release_notes.md) | Provides basic information and feature updates of each release version. |
| [Feature Introduction](./docs/en/feature_introduction.md) | Provides the description of the high-version feature backport and optimizations. |
| [User Guide](./docs/en/user_guide.md) | Provides the usage description of the backport patches. |

## Contribution Statement

We welcome your contributions to the community. If you have any questions/suggestions or want to provide feedback on feature requirements and bug reports, you can submit issues. For details, see the [contribution guideline](https://gitcode.com/boostkit/community/blob/master/docs/contributor/contributing.md). You are also welcome to share insights in [Discussions](https://gitcode.com/boostkit/community/discussions). Thank you for your support.

## Disclaimer

This code repository contributes to the vLLM and vLLM-MetaX open-source components. It strictly adheres to the coding style and methods, as well as security design of the native open-source software. Any vulnerability and security issues of the software shall be resolved by the corresponding upstream communities according to their response mechanisms. Please pay attention to the notifications and version updates released by the upstream communities. This project does not assume any responsibility for software vulnerabilities and security issues.

## License

This project is released under the Apache License 2.0. For details, see [LICENSE](./LICENSE).
The documents of this project are licensed under CC-BY 4.0. For details, see [LICENSE](./docs/LICENSE).

## Acknowledgments

Thank you to everyone in the community for your PRs. We warmly welcome contributions!
