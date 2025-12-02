# 项目介绍

vllm在鲲鹏平台上推理性能加速

本目录包含针对vllm及相关插件的功能增强及性能优化补丁
这些补丁可以独立或组合使用，用于适配对应环境以及优化插件性能。

# 目录结构与模块说明

## syshax — sysHAX 加速模块

该目录包含920少卡 sysHAX 场景下的补丁，包括：


| 文件 | 作用 |
|------|------|
| **boostkit-vllm-ops-gpu-082.patch** | 适配 GPU 侧 vLLM 0.8.2 使用 sysHAX 调度模块 |
| **boostkit-vllm-ops-cpu-gptq.patch** | CPU 侧 GPTQ INT4/INT8 权重读取与执行加速补丁 |
| **boostkit-vllm-ops-cpu-opt.patch** | CPU 推理算子加速优化相关补丁 |
| **README.md** | sysHAX 加速模块的详细使用说明 |


# 贡献指南
如果使用过程中有任何问题，或者需要反馈特性需求和bug报告，可以提交isssues联系我们，具体贡献方法可参考[这里](https://gitcode.com/boostkit/community/blob/master/docs/contributor/contributing.md)。


# 免责声明
此代码仓计划参与vllm软件开源，编码风格遵照原生开源软件，继承原生开源软件安全设计，不破坏原生开源软件设计及编码风格和方式，软件的任何漏洞与安全问题，均由相应的上游社区根据其漏洞和安全响应机制解决。请密切关注上游社区发布的通知和版本更新。鲲鹏计算社区对软件的漏洞及安全问题不承担任何责任。
