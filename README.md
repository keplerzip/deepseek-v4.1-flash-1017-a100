# deepseek-v4.1-flash-1017-a100

DeepSeek-V4.1-Flash-1017 的 8×A100 80GB 离线部署、SM80 适配和验收源码。API 模型名固定为 **`DeepSeek-Flash`**；默认 **TP8 / DP1 / EP8 / PP1、C32、524288 token 上下文、8192 token 调度预算、DSpark k5**，保留原生图文配置。

本公开源码快照已整合 V3 的内核测试依赖、断网 IPv4 分布式启动、8005 访问策略、原生缓存 usage、多轮消息和多图/工具图片关联修复。**仓库不包含模型权重或 Docker 镜像。** 离线目标机需要另备已核验的完整 R2 运行镜像、NVIDIA 驱动和 Container Toolkit；仅克隆仓库不足以启动模型，也不意味着可以从零重建同一个镜像。

模型从 [ModelScope 官方仓库](https://www.modelscope.cn/models/deepseek-ai/DeepSeek-V4.1-Flash-1017) 准备，固定 revision 和每文件 SHA256 见 [模型清单](manifests/model-files.json)。不混用旧版权重，不转换为其他社区量化格式。

## 2026-10-19 A100 性能记录

以下数字来自目标机最终验收报告的**用户 OCR 转录**，报告总状态和配置检查均为 PASS。原始 `summary.json` / 请求证据尚未收到，未标为公开源码维护者独立复现。部署主机身份与原始运行编号已省略，完整显示数值见 [结构化性能记录](docs/benchmark-20261019.json)。

硬件为 8×NVIDIA A100-SXM4-80GB、每卡 81920 MiB、NV12/NVSwitch、驱动 580.159.04、MIG Disabled、128 个逻辑 CPU、约 2 TiB 主存。配置与上述 TP8 / C32 / 524288 / DSpark k5 基线一致。

| 模式 | 单发 decode 中位数 tok/s | 单发端到端中位数 tok/s | 单发 TTFT 中位数 秒 | C32 总 tok/s | C32 端到端 P95 秒 | C32 TTFT P95 秒 |
|---|---:|---:|---:|---:|---:|---:|
| off：关闭思考 | 155.520 | 148.521 | 0.294 | 1318.582 | 34.615 | 4.147 |
| high：高强度思考 | 148.733 | 143.493 | 0.262 | 1282.933 | 30.772 | 2.250 |

主测速采用短输入、固定输出预算，正式单发样本取中位数并排除预热；TPS 包含思考 token。decode 与 TTFT 分开计时，端到端 TPS 包含完整请求开销。C32 总吞吐是整个负载阶段总输出 / 墙钟时间；P95 是单请求端到端延迟，包含排队、预填充和生成。OCR 未给出的实际请求数、预热次数与短请求输出预算未用默认值补造。

| C1 诊断负载 | 清洁 / 实际样本 | decode 中位数 tok/s | TTFT 中位数 秒 | DSpark 加权接受率 | 每轮接受草稿 token |
|---|---:|---:|---:|---:|---:|
| code / high | 10 / 10 | 123.205 | 0.225 | 39.02% | 1.951 |
| code / off | 10 / 10 | 177.040 | 0.229 | 48.38% | 2.419 |
| text / high | 10 / 10 | 128.847 | 0.236 | 30.12% | 1.506 |
| text / off | 10 / 10 | 136.036 | 0.213 | 31.92% | 1.596 |

接受率按正式清洁样本 `Σaccepted / Σproposed` 加权；每轮接受量为 `Σaccepted / Σdrafts`，不含主模型 bonus token。代码诊断与主测速负载不同，不能互相替代；没有同模型关闭投机的 A/B，不宣称 DSpark 净加速比。

- **单条近满 512K 检索：报告 PASS。** 实际输入 523626 token，输出预算 512，合计预算 524138 / 524288，TTFT 101.290 秒；检查开头、25%、50%、75% 和末尾五处检索。预算不表示实际生成了 512 token。
- 基础功能报告 PASS：模型身份、off/high SSE、单图/三图 OCR、反序工具图片关联、冷热缓存计数、C32 工具调用、原生 DSpark k5 计数。
- **37 项扩展报告 PASS：23 项缓存、14 项视觉。** 覆盖三协议 JSON/SSE、多轮历史、换图隔离、动态尾 system、长图文前缀缓存、5/8 图顺序、反序工具图片 ID 和 Responses `view_image`。
- **32 条满 512K 窗口同时驻留：`NOT_PROVEN`。New API 账单/折扣倍率实链路：`NOT_RUN`。** 视觉质量结论限上述任务，不延伸到所有视觉任务、999 图压力或原生音视频。

## 配置与访问

| 项目 | 默认配置 |
|---|---|
| 并行 | TP8 / DP1 / EP8 / PP1 |
| 上下文与调度 | 524288 token 输入输出合计；C32；8192 token 分块预填预算 |
| 投机 | 模型原生 DSpark，k5；主模型及草稿 decode Graph |
| KV 与主存 | `fp8_ds_mla`；GPU 预算 0.92；Engram CPU offload；前缀缓存 |
| 图文 | 官方视觉处理器；无额外 processor override；图片配置上限 999，仍受上下文和显存限制 |
| 接口 | Chat Completions、Responses、Anthropic Messages；唯一 API 模型名 `DeepSeek-Flash` |
| 入口 | `127.0.0.1:8005` 与默认 Docker bridge 网关 `:8005`；按已核实 bridge 来源放行；模型服务免 API Key |
| 隔离 | 模型容器 `--network none`、共享 Unix socket；`--pull never`；不访问远程图像 |

Docker 客户端可用 `http://host.docker.internal:8005/v1`，Linux 需配置 `host.docker.internal:host-gateway`。新增/更改 Docker 网络后重启本项目刷新来源清单；Docker 自身 `Internal=true` 路由隔离仍有效。前端按真实 TCP 对端判定来源，不信任转发头。

原生多模态范围是文字和图片输入、文字输出。图片使用内联 data URI；Codex 通用媒体类型下的 PNG/JPEG 先检查 base64 和图片签名。工具结果图片保留调用 ID 关联，Responses 省略图片 detail 时默认 auto，Messages 保留尾 system 的多轮前缀。

V4 增加 [OpenAI / Codex / Claude 思考参数兼容](docs/REASONING.md)：OpenAI 的 `none/minimal/low/medium/high/xhigh/max` 均有明确映射，Claude 保留其五档 effort 与 enabled/adaptive/disabled 输入。`medium/xhigh` 不再因原生三档校验报错；显式关闭不会被自动标志反向开启，Claude enabled-thinking 的 token 预算传给原生采样器。

[Responses 流式兼容说明](docs/RESPONSES-STREAMING.md) 记录 item / call ID 修复及 V5 的已知回归：多次思考与回答切换时，严格输出形状检查可能中止原本能结束的流。V6 在验证文本与工具语义一致后保留流式分段，并用标准失败事件保留真实错误原因。实际 Codex App 与 New API 完整链路仍需目标机验收。

## 部署

详细步骤见 [离线部署与验收](DEPLOY.md)。以下命令假设已导入 [固定运行镜像](manifests/runtime.env) 并已准备完整官方模型目录：

```bash
bash verify-package.sh
bash profiles.sh select baseline
bash deploy.sh "$PWD/models/DeepSeek-V4.1-Flash-1017"
```

以普通部署用户执行，脚本内部使用 `sudo -n docker`。部署不在宿主安装 Python、pip、CUDA 或其他新依赖。首次核验读取全部模型文件；后续模型状态不变时复用本机校验凭据。启动会检查 GPU、驱动、NVLink、资源、实际 KV 容量、SM80 数值/Graph、EP8、图文与工具功能，失败不降低配置。

```bash
curl --fail http://127.0.0.1:8005/v1/models
bash status.sh
```

## 测速和验收

```bash
bash tests/run.sh smoke
bash tests/run.sh benchmark
bash tests/diagnose-speed.sh
bash tests/run.sh long-context
bash tests/r39-alignment.sh
```

测试期间会暂停本项目入口，使用 Unix socket 避免外部请求污染；结束后恢复入口。记录保存在本地 `runtime/`。单独 benchmark 套件默认输出 512 token、单发 5 次、C32 阶段 64 请求；需要其他参数时显式传入。`long-context` 与短请求 TPS 独立，不能用短请求 decode 速度估算 512K 首 token 时间。

`bash tests/run.sh full-window-c32` 是额外的 32 条近满窗口压力入口；排队完成全部请求不等于同时驻留，缺少物理 KV 驻留证据时仍返回 `NOT_PROVEN`。

## 源码校验与构建范围

[PUBLIC_SOURCE_SHA256SUMS](PUBLIC_SOURCE_SHA256SUMS) 描述本公开快照；它与原离线交付包的 base/combined/archive 身份不同。不要将 Git 文件覆盖到已封存交付目录并继续沿用旧清单。公开版已将内核检查点接入源码清单校验，完成单卡检查后才能记录检查点；只复用已核对的数值报告，不跳过未通过的 EP8。

公开的 SM80/协议覆盖源码和构建脚本用于已有对应审计基底的维护者。构建仍依赖另外准备的固定完整本地基底，且重建镜像有新的 digest，需要独立审查身份与目标 GPU 验收。说明见 [构建依赖与来源](docs/SOURCE-BUILD.md)。

## 许可证与来源

部署脚本与适配按 [Apache-2.0](LICENSE) 提供；上游归因见 [NOTICE](NOTICE)。vLLM/backport 及相关补丁保留原 SPDX/版权头和 [vLLM 许可证](licenses/vLLM-LICENSE)，模型及官方编码器的许可见 [DeepSeek 许可证](licenses/DeepSeek-LICENSE)。运行镜像中其他软件仍分别遵循其许可证。本仓库不重新分发模型权重或运行镜像。
