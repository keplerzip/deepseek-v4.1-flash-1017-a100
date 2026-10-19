# 源码范围与镜像构建

## 推荐运行方式

公开源码已合并 V3 的启动、协议和测试修复。直接部署使用 `manifests/runtime.env` 固定的 R2 镜像，配合完整官方 ModelScope 权重，参见 [DEPLOY.md](../DEPLOY.md)。源码仓库不分发镜像、权重或旧离线包。

`verify-package.sh` 只检查本次发布的 `PUBLIC_SOURCE_SHA256SUMS`。该清单代表公开源码字节，不代表旧 R3 离线交付、旧补丁或 Docker 镜像的清单。下载文件的官方 SHA256、上游源码的 base SHA 和既有镜像身份仍保留真实来源，不随源码脱敏而改写成虚假的兼容关系。

## 历史构建的前置条件

现有构建脚本是已有完整运行环境之上的离线适配构建。仓库没有从系统基础镜像安装完整 CUDA、PyTorch、vLLM 及 SM80 依赖的从零构建配方，也没有可供 `docker pull` 的基础镜像地址。**仅克隆本仓库无法从零重建可部署镜像。**

| 构建脚本 | 必须已经存在的本地基础镜像 | 固定身份 |
|---|---|---|
| `build/build-runtime.sh` | `deepseek-v4.1-flash-a100:20260913-r1` | `sha256:29e46788ca2667042d06b496e92f6d38c7b50bee16f5b1d211e9229e68907403` |
| `build/build-speed-runtime.sh` | `deepseek-v4.1-flash-1017-a100:20261018-r1` | `sha256:10743e6faaa326f4049b4e661e172fc5be5346a50894242784b136131c9569ca` |

这些身份记录用于拒绝未知基础环境，不能通过重新打 tag 绕过。脚本使用 `--pull=false --network=none` 构建，并用 `--pull never --network none` 运行验证；不会在线补装缺少的依赖。

构建需要准备端已有 Python 3、Bash、GNU 工具、`sudo -n docker` 和足够磁盘空间。模型元数据不再随公开仓库复制；两条构建通路都要求显式提供已经下载的官方模型目录，并只读挂载到容器内固定位置：

```bash
export DSV411017_MODEL_DIR=/data/DeepSeek-V4.1-Flash-1017
python3 build/download_model.py --dest "$DSV411017_MODEL_DIR" --verify-only

# 仅在分别具备上表对应固定基础镜像的构建环境执行：
bash build/build-runtime.sh
bash build/build-speed-runtime.sh
```

`DSV411017_MODEL_DIR` 会规范化为真实路径；不接受 Docker 挂载分隔符、换行或制表符。模型挂载为 `/models/DeepSeek-V4.1-Flash-1017:ro`，不会复制模型进构建上下文。`build/test_installed_runtime.py` 通过该目录核对官方编码、token ID、原生视觉配置和请求转换。R1 阶段使用 `--expected-stage r1`，仅把尚未安装的 R2 speed 层标记为不适用；R2 阶段仍校验该层全部已安装源码 SHA。

脚本还执行原有 CLI、CPU 单测、协议/编码一致性、离线传输和镜像归档完整性检查。R2 另检查 context-combine CPU 路径及 TP 通信模块导入。SM80 编译、真实 GPU 数值、CUDA Graph、EP8、整模型启动、KV 容量和测速是不同验收项目，不能用 CPU 导入通过替代。

**公开源码重建出的镜像不保证与历史镜像字节一致。** 构建会写出新的镜像归档与本地清单；源码调整也可能改变镜像 ID。当前部署检查和单卡检查点对受信镜像身份有明确约束。维护者须审核新构建的内容、来源与清单，形成新的发布身份，并完成完整目标 GPU 验收后才能将其纳入支持范围。不要把新 ID 直接替换进旧清单并宣称兼容；主部署路线仍是复用固定 R2。

## 源码测试

以下检查使用 Python 标准库、Bash 或本地 socket 的小型 fixture，不读取权重、不运行模型，也不证明 GPU 速度：

```bash
python3 build/test_api_access_v2.py
python3 build/test_r39_alignment_v2.py
python3 build/test_r39_overlay_mount_v2.py
python3 build/test_kernel_checkpoint_v3.py
python3 build/test_kernel_resume_workflow_v3.py
```

有固定 R2 镜像时，可另外执行真实容器内的 CPU 协议回归。它保留预期失败的原生基线和修复后的结果，不发送模型推理请求：

```bash
python3 build/test_r39_protocol_overlay_v2.py --evidence-dir build-state/native-protocol
```

真实双进程 CPU Gloo 复现可用 `python3 build/test_torchrun_offline_ep8.py`。该负向用例会限时终止旧 launcher 参数产生的 DNS 重试，并验证显式 static / 数字 IP / 固定 master port 的修复；CPU Gloo 通过不代表八卡 NCCL 已通过。

模型已启动后的验收命令和报告位置见 [DEPLOY.md](../DEPLOY.md)。测试生成的本机状态、原始响应和报告位于被 Git 忽略的 `runtime/` 或 `build-state/`，公开前需要单独核对内容。

## 来源与许可

适配源码遵循根目录 `LICENSE` 与 `NOTICE`。第三方源码保留原版权头和 SPDX 声明，vLLM 许可见 `build/image/vllm-LICENSE`；官方编码器许可见 `licenses/DeepSeek-LICENSE`。上游 commit、base/patched SHA、参考文件与合成图片来源分别记录在相应 manifest/provenance 文件中。

公开源码清单应在发布文件冻结后生成。源码修改后先重跑相关测试，再生成新的源码发布清单；保留原镜像和官方模型的既有身份，不用旧的机器验收日志为修改后的源码签收。
