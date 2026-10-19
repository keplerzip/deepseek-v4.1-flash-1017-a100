# 可选 DSpark context combine 图捕获

底层权重仍是 ModelScope `deepseek-ai/DeepSeek-V4.1-Flash-1017`；最终 API 唯一模型名称由启动器设为 `DeepSeek-Flash`，此补丁不修改模型名称或编码。

这是 [vLLM PR #59511](https://github.com/vllm-project/vllm/pull/59511) 的针对性移植，固定合并提交 `388a610dfb5f462261cc1ec27bc2e3bebc0249ca`。R1.9 已将 context K/V 预计算纳入草稿图；此补丁再把三个目标层辅助隐状态的 `main_proj`/norm 放到同一 FULL 草稿图中。没有改动草稿采样、接受判定、C32、524288 上下文、视觉塔、图像预算、工具协议或原始权重。

补丁默认关闭。仅适用于本项目确切 R1.9 文件，安装前与回滚前均检查原始文件和负载 SHA256。安装器不启用环境变量。

```bash
python3 /opt/dsv411017/image/speed/install.py
# 随下一次容器启动传入：
# -e VLLM_DSV41_CAPTURE_CONTEXT_COMBINE=1
# 关闭时设为0；需要完全还原代码时：
python3 /opt/dsv411017/image/speed/install.py --rollback
```

启用时只接受 DeepSeek-V4.1、DSpark 和 FULL decode graphs；其他配置报错。缓冲在构造时分配，以进入内存预算；C32/k5 的192行、3×5120维、BF16缓冲为每卡5.625MiB。每次 replay 前重新复制实际辅助状态；短批次与占位输入保留 R1.9 的 PAD slot 清理；超过图容量的 prefill 尾部仍在图外处理。多个草稿 KV group 映射保持不变。关闭与 eager 路径保留原始投影行数，避免无意改变原 Marlin/BF16 分派。

上游性能来自 4×GB200、TP4、合成100%接受率的控制实验：C1 ITL降低2%至6%，C8/C64没有可测收益。该实验用于定位 CPU 发射开销，不能作为 A100、1017、真实接受率或用户40–60TPS负载的提速证明。此移植未在GPU执行。正式启用后必须测真实请求的decode/E2E/TTFT、逐位置接受率、C32、长上下文、图文工具历史及实际KV容量，再决定默认值。

验证命令：

```bash
python3 /opt/dsv411017/image/speed/tests/test_context_combine.py
# 仅有真实GPU时运行。此脚本只验适配方法的图回放，不替代整模型验收：
python3 /opt/dsv411017/image/speed/tests/test_context_combine.py --gpu
```

CPU 测试覆盖开关关闭路径、图内投影路由、混合 prefill 尾部、多个 KV group、陈旧 slot 清理、安装与回滚。可选 GPU 检查覆盖不同输入、较小批次与长尾的固定地址回放；使用小型随机投影，不加载模型，不能验证生产 Marlin kernel 或完整多模态质量。
