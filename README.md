# vLLM高版本特性回合介绍

简体中文|[English](./README_en.md)

## 最新消息

- [2026.09.30]：面向沐曦曦云C500 GPU的推理场景，发布基于vLLM 0.15.0容器、源码编译vLLM 0.17.0和vLLM-MetaX 0.17.0，并回合vLLM 0.18.0、0.19.0、0.20.0核心性能优化特性的补丁合集。
- [2026.06.30]：发布针对社区版vLLM 0.11.0及沐曦版vLLM-MetaX 0.11.0-dev的优化补丁合集。

## 项目介绍

vLLM高版本特性回合项目面向沐曦曦云C500 GPU的推理场景。项目使用沐曦提供的vLLM 0.15.0容器作为基础环境，在容器内从源码编译vLLM 0.17.0和vLLM-MetaX 0.17.0；回合vLLM 0.18.0、0.19.0、0.20.0中的38个功能点，以消除因版本滞后导致的性能劣势；同时叠加opt优化和vLLM-MetaX FlashAttention metadata优化，进一步提升推理性能。

## 目录结构

```text
vllm-ops/
├── patch                                                                           # 补丁文件目录
│   ├── vllm-backport-feature-patches                                               # vLLM高版本特性回合主补丁包
│   │   ├── 0001-benchmarks__kernels__cpu__benchmark_cpu_attn.py.patch
│   │   ├── ...                                                                      # 0002至0084文件级回合补丁
│   │   ├── 0085-vllm__v1__worker__gpu_ubatch_wrapper.py.patch
│   │   ├── manifest.csv                                                            # 补丁与源文件映射
│   │   ├── series                                                                  # 补丁应用顺序
│   │   └── README.md                                                               # 主补丁包说明
│   ├── vllm-opt-patch                                                              # vLLM独立opt优化补丁包
│   │   ├── vllm-0.17.0-opt.patch                                                   # greedy sampler和block table合并补丁
│   │   └── README.md                                                               # opt补丁包说明
│   └── vllm-metax-feature-patches                                                  # vLLM-MetaX优化补丁包
│       ├── vllm-metax-flash-attn-metadata-optimizations.patch                      # FlashAttention metadata优化补丁
│       └── README.md                                                               # vLLM-MetaX补丁包说明
├── docs
│   └── zh                                                                          # 中文文档目录
│       ├── feature_introduction.md                                                 # 特性说明
│       ├── menu_vllm_ops.md                                                        # 文档指南
│       ├── release_notes.md                                                        # 版本说明书
│       └── user_guide.md                                                           # 用户指南
├── LICENSE                                                                         # 开源许可证文件
├── CC-BY                                                                           # 开源文档许可证文件
└── README.md                                                                       # 项目说明文档
```

## 版本说明

vLLM高版本特性回合的版本说明，具体请参见《[版本说明书](./docs/zh/release_notes.md)》。

## 学习文档

| 资源名称 | 资源简介 |
| ------------ | ------------ |
| [版本说明书](./docs/zh/release_notes.md) | 提供vLLM高版本特性回合每个发布版本的基础信息和特性更新信息。 |
| [特性介绍](./docs/zh/feature_introduction.md) | 提供vLLM高版本特性回合及优化说明。 |
| [用户指南](./docs/zh/user_guide.md) | 提供vLLM高版本特性回合补丁使用说明。 |

## 贡献声明

欢迎大家为社区做贡献，如果使用过程中有任何问题/建议，或者需要反馈特性需求和bug报告，可以提交issues联系我们，具体贡献方法可参考[这里](https://atomgit.com/boostkit/community/blob/master/docs/contributor/contributing.md)。同时也欢迎大家在[讨论专区](https://atomgit.com/boostkit/community/discussions)展开讨论交流。感谢您的支持。

## 免责声明

此代码仓计划参与vLLM和vLLM-MetaX开源组件，编码风格遵照开源软件，继承开源软件安全设计，不破坏开源软件设计及编码风格和方式，软件的任何漏洞与安全问题，均由相应的上游社区根据其漏洞和安全响应机制解决。请密切关注上游社区发布的通知和版本更新。本项目对软件的漏洞及安全问题不承担任何责任。

## 许可证书

本项目采用Apache License 2.0，详见[LICENSE](./LICENSE)文件。
本项目文档适用CC-BY 4.0许可证，具体请参见[LICENSE](./docs/LICENSE)文件。

## 致谢

感谢来自社区的每一个PR，欢迎贡献vLLM高版本特性回合！
