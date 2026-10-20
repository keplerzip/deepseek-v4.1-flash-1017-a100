# Responses SSE 与 Codex 子代理

V5 修复已在固定 R2 镜像和目标 A100 上复现的 Responses SSE 标识不一致：`response.output_item.added/done` 的 item ID 与最终 `response.completed.response.output` 不同。工具调用的 `call_id` 也存在同类问题。客户端按标识归并事件时可能因此无法正确关联最终输出。

修复直接作用于原生 Responses generator：只将原生最终 item 的 ID、工具 call ID 与已发送的 done item 对齐，保留原生最终对象、状态、logprobs、函数 namespace、参数和输出顺序。usage、缓存 token 计数及原有图文/思考转换沿用原逻辑。达到输出预算、最终状态为 incomplete 时发送 `response.incomplete`；后台流回放识别该终止事件。

## 证据与边界

目标 A100 在 V4 上的八项受控问候测试：JSON 四项通过；SSE 四项回答内容正确但协议检查失败，唯一失败类型为 `DONE_COMPLETED_ID_MISMATCH`。测试覆盖普通文本/任务信封、none/medium；这证明部署侧存在该缺陷，不单独证明它是实际 App 子代理异常的全部原因。

本地固定 R2 镜像的十二项回放使用真实 DeepSeek parser、Responses generator 与 HTTP SSE 序列化器，输入为合成 engine delta。原十项在未修复版本均复现标识漂移；扩展后的十二项在未修复版本均被检查拒绝，修复后全部通过。除多工具、namespace、思考过滤、长度上限、输出顺序、缓存 usage、存储与后台回放，还新增真实 logprobs 保留及输出形状不一致时的错误终止检查。这是 CPU 协议检查，没有加载权重或执行 GPU 推理。

离线 V5 安装后会重启现有引擎，保留 TP8、C32、512K、DSpark k5、8005、DeepSeek-Flash 和原图文配置；模型文件未变时继续复用校验收据。Ready 后自动执行六项跨协议思考参数检查及相同八项真实 JSON/SSE 受控请求。任一请求或流检查失败，验收返回非零。输出报告位于 `runtime/responses-v5/live-stream.json`。

## 手动复查

在项目子目录执行，无需重启：

```bash
source scripts/common.sh
resolve_image
image_python /package/tests/subagent_diagnostic.py --unix-socket /state/api.sock --output /state/subagent-live.json --require-pass
```

只发送固定测试任务，报告不采集真实 App 会话或完整思考文本。`NO_FAILURE_OBSERVED` 仅表示这八项通过，`subagent_end_to_end` 仍为 `NOT_RUN`；更新后还需在 Codex App 中执行原本失败的子代理任务。

Codex 客户端的子代理调度与 API 服务的模型推理是两层能力。普通函数工具可由客户端负责调度；OpenAI 的服务端 `multi_agent` beta 属于另一套托管能力。本项目没有实现该托管功能，也未声称完整兼容 arbitrary-input custom tools 或专用 agent 消息类型。参考：[Codex 子代理配置](https://learn.chatgpt.com/docs/agent-configuration/subagents)、[Responses 托管 multi-agent](https://developers.openai.com/api/docs/guides/responses-multi-agent)、[OpenAI 自定义工具](https://developers.openai.com/api/docs/guides/function-calling#custom-tools)。
