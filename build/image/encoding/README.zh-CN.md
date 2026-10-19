# 1017 编码兼容修复

`source-reference.py` 从 ModelScope 的固定 revision
`f04a978641fa9bfe6f448ec5da3b2e1c2520a6ad` 下载，SHA256 已与完整快照清单核对。
它是测试参考，不直接作为 vLLM tokenizer 使用。原始来源为
[DeepSeek 官方编码器](https://www.modelscope.cn/models/deepseek-ai/DeepSeek-V4.1-Flash-1017/resolve/f04a978641fa9bfe6f448ec5da3b2e1c2520a6ad/encoding/encoding.py)，
许可证见项目 `research/model-metadata/LICENSE`。

两个安装文件保留原 R1.9 的 vLLM 适配与合法协议扩展：

- 编码器移植纯工具结果不新建用户轮、结构化 content 保留、context 先合并再排序。
- renderer 仅对 `deepseek_v41` 稳定排列连续 `role=tool` 回复，使相同的排序后消息同时进入
  同步/异步的图片收集及 tokenizer。排序 key 与官方工具结果排序一致；未识别 id 使用相同
  默认 key，保持它们彼此的输入顺序。普通 user/system/assistant 消息不跨边界重排，其他模型路径不变。
- 不整体替换官方图片处理器：vLLM 已负责收集及编码图片，tokenizer 需要接受框架产生的图片占位符。

`install.py` 对参考文件、补丁内容和目标基底逐项核对 SHA256，先全部验证，再写入目标；可重复执行。
`--site` 用于构建测试，省略则安装到容器当前 Python 的 site-packages。

标准库测试同时验证官方参考差分、合法扩展和真实 renderer 方法的同步/异步路径。renderer 中的
`parse_chat_messages` 与 tokenizer 使用受控替身，核对图片收集顺序及最终工具占位顺序。
生成的完整 prompt 和图片 URL 顺序还与1017官方多模态编码器逐项对比。这些测试证明
调度前的协议映射，不代表实际 ViT 数值、图像内容理解或八卡 GPU 推理已验收。OpenAI/Responses/
Messages 的 HTTP 转换、异步请求、真实图片处理仍须目标机验收。

结构化 `content_blocks` 是内部编码器格式；HTTP 图片工具结果应使用其协议允许的 content 数组。
本修复处理标准连续工具回复，不新增非标准 `user`/`tool` 任意交错历史的媒体重排语义。
