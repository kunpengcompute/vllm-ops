# vLLM-ops介绍

## 项目介绍

vLLM-ops是针对鲲鹏920新型号处理作为机头搭载沐曦曦云C500 GPU时进行的推理性能提升，采用了CPU侧Python代码优化，减少CPU和GPU之间数据传输，OS侧调优等手段提升吞吐。本项目针对社区版vLLM 0.11.0及沐曦版vLLM-metax 0.11.0-dev的输出优化补丁。

## 目录结构

```text
vllm-ops/
├── patch                                                                    # 补丁文件目录               
│   ├── 0001-vllm_0.11.0-optimize-schedular.patch                            
│   ├── 0002-vllm_0.11.0-optimize-sched_yield_on_arm.patch                   
│   ├── 0003-vllm_0.11.0-optimize-JIT.patch                                  
│   ├── 0004-vllm_0.11.0-optimize-batch_update_np_array.patch  
│   ├── 0005-vllm_0.11.0-refactor-extract_all_gather_for_cuda_graph.patch  
│   ├── 0006-vllm_0.11.0-optimize-reduce_numpy_split_operations.patch                
│   └── 0007-vllm_metax_0.11.0-dev-move_compute_to_gpu.patch
├── docs
|   └── zh                                                                    # 中文文档目录
│      ├── feature_introduction.md                                            # 特性说明文档
│      ├── menu_vllm_ops.md                                                   # 文档指南
│      ├── release_notes.md                                                   # 每个发布版本的基础信息和特性更新信息
│      └── user_guide.md                                                      # 用户指南
├── LICENSE                                                                   # 开源许可证文件
├── CC-BY                                                                     # 开源文档许可证文件
└── README.md                                                                 # 项目说明文档
```

## 版本说明

vLLM-ops本身的版本说明，具体请参见《[版本说明书](./docs/zh/release_notes.md)》。

## 学习文档

|  资源名称 |资源简介   |
| ------------ | ------------ |
| [版本说明书](./docs/zh/release_notes.md)  | 提供vLLM-ops每个发布版本的基础信息和特性更新信息。  |
|  [特性介绍](./docs/zh/feature_introduction.md) |  提供vLLM-ops优化说明。 |
|  [用户指南](./docs/zh/user_guide.md) |  提供vLLM-ops优化使用说明。 |

## 贡献声明

欢迎大家为社区做贡献，如果使用过程中有任何问题/建议，或者需要反馈特性需求和bug报告，可以提交issues联系我们，具体贡献方法可参考[这里](https://gitcode.com/boostkit/community/blob/master/docs/contributor/contributing.md)。同时也欢迎大家在[讨论专区](https://gitcode.com/boostkit/community/discussions)展开讨论交流。感谢您的支持。

## 免责声明

此代码仓计划参与vLLM和vLLM-metax开源组件，编码风格遵照原生开源软件，继承原生开源软件安全设计，不破坏原生开源软件设计及编码风格和方式，软件的任何漏洞与安全问题，均由相应的上游社区根据其漏洞和安全响应机制解决。请密切关注上游社区发布的通知和版本更新。鲲鹏计算社区对软件的漏洞及安全问题不承担任何责任。

## 许可证书

本项目采用Apache License 2.0，详见[LICENSE](./LICENSE)文件 
本项目文档适用CC-BY 4.0许可证，具体请参见[LICENSE](./CC-BY)文件。

## 致谢

vLLM-ops由华为公司的下列部门联合贡献：

鲲鹏计算Boostkit开发部

感谢来自社区的每一个PR，欢迎贡献vLLM-ops！
