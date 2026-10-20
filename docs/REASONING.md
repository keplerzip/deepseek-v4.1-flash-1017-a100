# OpenAI、Codex 与 Claude 思考参数

V4 修复客户端选择 `medium` 或 `xhigh` 时的 `reasoning_effort must be low, high, max` 错误。原生 1017 编码器接受 1–100 的相对思考强度；本项目在现有 tokenizer 适配层转换客户端名称，不更换模型或编码器。

| 客户端档位 | DeepSeek 思考强度 |
|---|---:|
| `none` | 关闭思考 |
| `minimal` | 25 |
| `low` | 50，保留官方定义 |
| `medium` | 63 |
| `high` | 75，保留官方定义与默认值 |
| `xhigh` | 90 |
| `max` | 100，保留官方定义 |

`minimal/medium/xhigh` 的数字是本项目的兼容策略，表示相对强度；不保证与其他模型同名档位的质量、耗时或 token 数相同。官方原生 `low/high/max` 与默认请求的提示保持原样。合法整数 1–100 仍可通过 `chat_template_kwargs.reasoning_effort` 使用，布尔值、浮点数和越界值会报错。

依据：[官方 1017 编码说明](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/main/encoding/README.md)、[OpenAI Codex 档位](https://developers.openai.com/api/docs/models/gpt-5.3-codex)、[Claude effort 参数](https://platform.claude.com/docs/en/build-with-claude/effort)。各客户端可选档位仍由客户端自身版本决定。

## Chat Completions

`POST http://127.0.0.1:8005/v1/chat/completions`：

```json
{
  "model": "DeepSeek-Flash",
  "messages": [{"role": "user", "content": "分析这个问题。"}],
  "reasoning_effort": "medium",
  "max_tokens": 2048
}
```

可选 `none/minimal/low/medium/high/xhigh/max`。也接受扩展字段 `chat_template_kwargs.reasoning_effort`。两处同时给出时，原生 vLLM 的非空顶层 `reasoning_effort` 优先；建议只传一处。`stream: true` 使用相同路径。

## Responses / Codex

`POST http://127.0.0.1:8005/v1/responses`：

```json
{
  "model": "DeepSeek-Flash",
  "input": "分析这个问题。",
  "reasoning": {"effort": "xhigh"},
  "max_output_tokens": 2048
}
```

可选同一组七档。Codex 配置的模型名应为 `DeepSeek-Flash`，提供方使用 Responses；在客户端切换 `medium/high/xhigh` 后，由请求中的 `reasoning.effort` 驱动映射，无需改服务配置。

## Messages / Claude Code

`POST http://127.0.0.1:8005/v1/messages`：

```json
{
  "model": "DeepSeek-Flash",
  "messages": [{"role": "user", "content": "分析这个问题。"}],
  "thinking": {"type": "adaptive"},
  "output_config": {"effort": "medium"},
  "max_tokens": 2048
}
```

Claude 协议的 `output_config.effort` 接受 `low/medium/high/xhigh/max`；`none/minimal` 是 OpenAI 档位，不扩展到 Claude 的原生枚举。Claude Code 连接时模型名同样必须设置为 `DeepSeek-Flash`。

- `thinking.type=enabled` 或 `adaptive` 启用 DeepSeek 思考，`disabled` 关闭；省略 `thinking` 时保持既有 Messages 默认关闭行为。`adaptive` 在此兼容为启用 DeepSeek 思考，不模拟 Claude 专有的按任务自适应策略。
- `thinking: {"type": "enabled", "budget_tokens": 1024}` 会把 token 预算传给 vLLM 原生 `thinking_token_budget`，不再静默丢弃。预算必须是正整数且小于 `max_tokens`；`adaptive` 不设置固定预算，`disabled` 不传预算。输出总上限仍由 `max_tokens` 控制。
- token 预算和表中的 1–100 思考强度是两个不同参数。预算截断由引擎及其 DSpark 路径执行，CPU 协议测试证明参数传递；精确的 GPU 生成行为需要目标机验证。

## 显式关闭、缓存与更新

显式 `chat_template_kwargs.thinking=false` 优先于自动添加的 `enable_thinking=true`，不会因为选择了 `high/xhigh` 被反向开启。只有未指定 `thinking` 时才使用 `enable_thinking`；`reasoning_effort=none` 始终关闭。Messages 默认的关闭标志也会保持生效；要使用其 effort 强度，请显式发送 `enabled/adaptive`。

同一档位的相同历史仍可命中前缀缓存；切换档位会改变 prompt 开头的强度值，不能承诺跨档位复用同一个前缀。图片、工具图片 ID 关联、usage 字段及三协议 JSON/SSE 传输沿用现有实现。

固定 R2 镜像无需重建；已有离线部署请使用 V4 增量包，在原部署目录运行包内安装入口。修复通过只读覆盖层生效，需要重启引擎并重新加载权重。原模型校验凭据在文件状态不变时继续使用，不全量重读模型 SHA，也不重新下载权重。
