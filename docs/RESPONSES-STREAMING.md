# V7 Responses 流式与子代理兼容

V7 最小回移 [vLLM PR #59307](https://github.com/vllm-project/vllm/pull/59307)（合并提交 `7ca31ef5993dc4426d912f32b8eb3cd70fb7a078`），最终响应复用同一次流式解析完成的 output items。保留原生 item ID、parser call ID、函数 namespace、参数、文本分段及逐消息 logprobs；零参数调用也发送 arguments.done。

从 [PR #59324](https://github.com/vllm-project/vllm/pull/59324)（`80dd705c1d422d04fcbd65d061dfedbf738ff573`）只回移 Simple streaming 最后一项的长度截断状态。真正的引擎错误继续发送 response.failed；达到输出上限发送 response.incomplete。后台存储与回放识别这些终态。

## 修正的回归

V5 以输出类型和数量匹配两次解析，V6 又要求增量解析与重新解析全文的语义逐字相等。原生解析器对多段思考边界空白、工具参数规范化存在差异，因而 V6 会误拦截实际请求并记录 stream_final_semantic_mismatch。V7 删除这套判断，不用放宽字符串比较来掩盖差异。

另有局部兼容措施：只对当前 DeepSeekV41 Responses parser 实例关闭提前参数增量，使用其已有的原生 invoke 最终转换。包装参数或重复参数不会留下无法撤回的 JSON 前缀。参数预览延迟到该调用 flush；文本和思考仍流式。此措施经过固定案例验证，不能表述为上游已合并的 DeepSeek 专用修复，也不保证模型总能选择正确工具。

## 保留的行为

TP8 / DP1 / EP8 / PP1、C32、524288 上下文、8192 调度预算、DSpark k5、原生图文配置、8005、本机和 Docker bridge 访问策略保持原值。权重、官方 encoder、渲染器、GPU 内核及缓存用量计算不替换。四个既有图文/思考兼容模块逐字节保留；新增两个原生协议依赖后，共挂载七个已校验模块。

原生 usage、cached_tokens、reasoning 隐藏规则、transfer metadata 和 response 存储仍使用原路径。V7 没有新的 GPU 性能测量；README 中既有实测数据属于先前明确配置的验收，不能作为 V7 速度保证。

## 离线安装与回滚

增量适用于精确的原 V4、V5 或 V6 解压交付；复用已加载的 R2 镜像及本地权重。宿主仍需原交付已有的 Bash/GNU 工具与 sudo Docker。Python 与全部运行库均在原镜像内，Docker 使用 --pull never 和 --network none；补丁校验不重新读取权重或镜像归档内容。

```bash
cd /ai/services/deepseek-v4.1-flash-1017-a100
tar -xzf DeepSeek-Flash-R3-Hotfix-20261020-v7.tar.gz
bash DeepSeek-Flash-R3-Hotfix-20261020-v7/apply-and-start.sh
```

安装前校验基线和完整源码，保存实际原版本；匹配的已有数值证据可复用，缺证仍执行原门禁。安装会重启服务，从已有本地权重重新加载。任一验收失败返回非零，报告位于项目 runtime/responses-v7/；不会自动退回有已知回归的旧版本。

回滚恢复本次更新前的实际 V4/V5/V6 版本，删除 V7 新增文件后启动原服务：

```bash
bash DeepSeek-Flash-R3-Hotfix-20261020-v7/apply-and-start.sh --rollback
```

仅应用并验证源码、暂不启动可用 --apply-only；该模式会停止服务，不能视为 Ready。历史封存包不修改。

## 验证范围

原镜像断网 CPU 回归覆盖原图文/协议九项、思考兼容六十五项，以及流式 parser/generator/原生 SSE 序列化回归。流式测试包含 publisher tokenizer 派生的固定 token 分块、多段思考空白、隐藏思考、包装及重复参数、namespace、并行调用、零参数、logprobs、长度截断、明确错误注入、存储及后台回放。测试禁止流式路径调用第二次全文解析，并检查 done/final 字段及原生 call_id；合成输出不算模型推理。

[脱敏子代理链路证据](validation/v7-subagent-fixtures.json) 记录真实 Codex CLI 0.163.0 的 spawn → 子请求 → wait → 父续接。直连与新建隔离 New API v1.0.0-rc.45 各跑三组：思考边界换行、arguments 包装、input 包装。六组均通过，父代理收到子代理最终确认。模型输出由固定 fixture 提供，未访问用户网关配置或真实会话。CLI 回传子代理最后一段 assistant 内容，不代表回传所有中间段。

Ready 后自动执行六项跨协议思考检查、八项问候 JSON/SSE 检查，以及十二组真实模型工具往返：普通单函数和 namespace 并行函数，none/medium/xhigh，JSON/SSE；参数含字符串与整数，工具结果反序回传，第二轮严格检查 call_id、ack、count 对应。工具仅在本地模拟，不执行命令或写业务文件。任意思考和正文不作为工具验收报告保存。

目标 A100 的真实工具验收由安装脚本执行，构建机没有 GPU。用户实际 Codex App → 现有 New API → A100 全链路仍需复测，不能因 CPU 或 CLI fixture 通过就标为已签收。此次没有承诺任意任务、全部网关设置或所有工具类型兼容。

手动重跑工具往返：

```bash
cd /ai/services/deepseek-v4.1-flash-1017-a100/deepseek-v4.1-flash-1017-a100
source scripts/common.sh
resolve_image
image_python /package/tests/tool_chain_live.py --unix-socket /state/api.sock --output /state/responses-v7/live-tool-chain.json
```

Codex 客户端工具调度与服务端模型推理是不同层。本项目没有实现 OpenAI 托管 multi_agent beta 或所有 arbitrary-input custom tools。协议依据：[OpenAI Responses 事件规范](https://developers.openai.com/api/reference/resources/responses/streaming-events)、[Codex 子代理配置](https://learn.chatgpt.com/docs/agent-configuration/subagents)。
