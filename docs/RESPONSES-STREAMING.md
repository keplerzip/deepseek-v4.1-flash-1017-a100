# Responses SSE 与 Codex 子代理

V6 保留流式 item / call ID 修复，并修正 V5 对合法多段思考输出的误拦截。模型、TP8 / C32 / 512K、DSpark k5、图文处理、8005、缓存用量字段及访问策略不变。

## V5 回归

同一受控输出在思考和回答之间多次切换时，增量 parser 产生四个 reasoning/message item，最终全量 parser 聚合为两个。V5 只按类型/数量匹配，因而发送通用 error 后关闭；V4 对同输出发送 completed。真实 Codex CLI 0.163.0 会忽略该通用 error，最后显示 `stream closed before response.completed`。

八项独立问候请求没有覆盖此情况。故障也可在真正创建子代理、父代理收到创建及等待结果的受控流程中复现，不能把问候测试通过等同于子代理调用链通过。

## V6 行为

正常一对一输出只绑定已发出的 item ID / call ID，保留原生最终状态、logprobs、函数 namespace、参数、缓存和其他元数据。分段不同但全部可见文本、思考文本和有序函数语义相同时，按已发出的分段顺序生成最终输出，保留可无损映射的原生元数据。

真实引擎错误或无法无损对齐时发送 `response.failed`，包含实际已发出的 partial item ID、原生 usage 和明确错误原因。长度上限仍为 `response.incomplete`，不会为了避免客户端报错而伪造 completed。后台生产、存储及回放识别所有这些终止状态。错误日志只包含请求 ID、输出项数量和固定错误码，不记录模型正文、思考内容、工具参数或凭据。

分段 logprobs 只有在实际 delta 列表与原生完整列表严格一致时才能无损切分；无法证明一致的请求返回明确失败，避免重复或丢弃概率数据。普通单段 logprobs 保留原生行为。带不可安全拆分的 annotations、summary 或未知内容类型也会明确失败。

## 离线安装

V6 增量支持精确的原 V4 或 V5 交付，使用原 R2 镜像，无需联网或新增宿主依赖。包内安装器先核对基线及源码，保存实际原版本，再重启加载已有本地权重；完整匹配的既有数值证据可复用，缺证仍走原门禁。回退恢复本次安装前的实际版本。原 V4 / V5 封存包不被改写。

```bash
cd /ai/services/deepseek-v4.1-flash-1017-a100
tar -xzf DeepSeek-Flash-R3-Hotfix-20261020-v6.tar.gz
bash DeepSeek-Flash-R3-Hotfix-20261020-v6/apply-and-start.sh
```

Ready 后自动执行六项跨协议思考参数检查及八项真实 JSON/SSE 受控请求。任一请求或流检查失败，验收返回非零。结果位于 `runtime/responses-v6/`；这些直连引擎检查不经过用户实际 New API 配置。

## 验证范围

固定 R2 的 25 组检查包含 23 个真实原生 parser / generator / HTTP SSE 序列化或明确故障注入场景，以及 2 组明确标记的 SDK 元数据 fixture。覆盖普通问候、多段思考、隐藏思考、长度限制、命名空间多函数、DSML 分片边界、logprobs、失败输出 ID、usage、存储和真实后台生产/回放。输入为合成 engine delta，没有加载权重或执行 GPU 推理。

[脱敏调用链证据](validation/v6-subagent-fixtures.json) 记录真实 Codex CLI 0.163.0 的实际 spawn、子请求、wait 和父续接：同输入 V4 正常，V5 精确复现断流，V6 正常。V6 另核对父代理收到子代理实际最后答复。经新建隔离 New API `v1.0.0-rc.45` 的原生 Responses 渠道转发，同样观察到 V5 失败、V6 正常。模型输出均为合成 fixture，不读取用户现有网关配置或凭据，也不向外网发请求。

真实 Codex CLI 的受控回放使用隔离网络、空配置及固定输出；子代理流程由客户端真正执行 spawn 和等待，模型回复仍为原生生成器的合成 fixture。它与目标模型推理、用户 Codex App / New API 完整链路是不同层级的证据。实际目标 App 仍需更新后复测，不声明任意模型任务、所有工具类型或所有网关配置均已通过。

## 手动复查

在项目子目录执行，无需重启：

```bash
source scripts/common.sh
resolve_image
image_python /package/tests/subagent_diagnostic.py --unix-socket /state/api.sock --output /state/subagent-live.json --require-pass
```

只发送固定测试任务，不采集真实 App 会话或完整思考。`NO_FAILURE_OBSERVED` 只表示这八项通过；`subagent_end_to_end` 仍为 `NOT_RUN`。

Codex 客户端的子代理调度与 API 服务的模型推理是两层能力。普通函数工具由客户端调度；本项目没有实现 OpenAI 托管 `multi_agent` beta，也未宣称完整支持 arbitrary-input custom tools 或专用 agent 消息类型。参考：[Codex 子代理配置](https://learn.chatgpt.com/docs/agent-configuration/subagents)、[Responses 托管 multi-agent](https://developers.openai.com/api/docs/guides/responses-multi-agent)、[OpenAI 自定义工具](https://developers.openai.com/api/docs/guides/function-calling#custom-tools)。
