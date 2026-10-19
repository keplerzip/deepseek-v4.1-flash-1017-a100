# 部署与验收

本仓库发布部署源码、适配层和测试，不包含 Docker 镜像、模型权重、旧离线交付包或机器运行记录。主部署路线是复用 `manifests/runtime.env` 固定的 R2 镜像，并挂载从 ModelScope 下载的官方 1017 完整目录。仅克隆仓库不足以启动模型。

固定配置为 8×A100-SXM4-80GB、TP8 / DP1 / EP8、C32、524288 token 上下文、8192 token 调度预算、DSpark k5。唯一 API 模型名为 `DeepSeek-Flash`，唯一服务端口为 `8005`。原生多模态范围为文字和图像输入、文字输出。

## 准备资源

目标机须已有 NVIDIA 驱动、Docker、NVIDIA Container Toolkit、Bash、GNU 基础工具，以及当前用户可用的 `sudo -n docker`。不要使用 `sudo bash deploy.sh`。Engram 卸载需要充足主存；启动检查要求至少 512 GiB available RAM，并会核对实际 GPU、驱动及拓扑。

先核对发布源码：

```bash
bash verify-package.sh
```

联网准备端下载固定 ModelScope 快照；请使用至少约 600 GiB 可用空间的模型盘：

```bash
python3 build/download_model.py --dest /data/DeepSeek-V4.1-Flash-1017
python3 build/download_model.py --dest /data/DeepSeek-V4.1-Flash-1017 --verify-only
```

下载器使用 `manifests/model-files.json` 中固定的 revision、下载地址和官方 SHA256。模型目录必须保留分片、索引、配置、分词器及 `encoding/` 等全部小文件。不要把旧模型分片混入该目录。下载器需要准备端 Python 3 标准库；离线目标机通过镜像中的 Python 执行验证和推理。

R2 镜像须来自已有可信交付或已经保存的原镜像。仓库没有镜像下载地址，也不承诺任意同名镜像可兼容。若需要从已有镜像归档导入，先核对清单中的真实归档 SHA256：

```bash
source manifests/runtime.env
IMAGE_ARCHIVE=/data/runtime-image.tar
printf '%s  %s\n' "$IMAGE_ARCHIVE_SHA256" "$IMAGE_ARCHIVE" | sha256sum --check -
sudo -n docker load --input "$IMAGE_ARCHIVE"
sudo -n docker image inspect --format '{{.Id}}' "$IMAGE_TAG"
```

部署脚本还会核对镜像身份，拒绝不匹配的镜像。已加载固定 R2 时无需再次复制归档。历史镜像构建要求见 [SOURCE-BUILD.md](docs/SOURCE-BUILD.md)。

## 启动

在源码根目录执行：

```bash
bash deploy.sh /data/DeepSeek-V4.1-Flash-1017
bash status.sh
sudo -n docker exec dsv411017-engine /usr/bin/python3 /opt/dsv411017/scripts/runtime.py health
```

部署会验证官方模型文件、四个只读协议覆盖层、单卡数值和 CUDA Graph、EP8 通信、网络隔离、实际 KV 容量、基础图文功能及前端访问。失败会保留 `runtime/` 证据并返回非零，不自动减少上下文或并发。单卡检查点只有在报告、源码、依赖、镜像与硬件记录均符合要求时才复用；EP8 必须实际通过。

引擎使用 `--network none`，前端把本机及已核实 Docker bridge 来源的请求转到同一 Unix socket。API 监听 `127.0.0.1:8005` 和默认 Docker bridge 网关 `:8005`，不监听任意 LAN 地址。此范围内不要求 API Key。容器客户端使用 `http://host.docker.internal:8005/v1`，Linux 容器需配置 `host.docker.internal:host-gateway`；更改 Docker 网络后应重启本部署以刷新来源清单。

```bash
bash stop.sh
bash start.sh
# 配置或源码改变后重新执行启动验收：
bash restart.sh
```

## 测速与多模态验收

引擎健康后按顺序执行，测试期间不要向该模型发送其他请求：

```bash
bash tests/run.sh smoke
bash tests/run.sh benchmark --warmups 1 --single-samples 5 --load-requests 64 --output-tokens 512
bash tests/diagnose-speed.sh --warmups 1 --samples 10 --output-tokens 1024
bash tests/run.sh long-context --timeout 7200
bash tests/r39-alignment.sh --timeout 7200
```

结果分别保存在 `runtime/acceptance-*/report.json`、`runtime/speed-diagnosis-*/report.json` 和 `runtime/r39-alignment-*/report.json`；命令会打印本次精确路径。基准报告包含单发 decode/端到端 TPS、TTFT 和 C32 总吞吐；诊断报告保留每请求时序及原生 DSpark 接受计数。37 项协议验收另外保留原始请求、JSON/SSE 响应与 usage，覆盖三协议冷热/多轮缓存、5/8 图顺序、工具图片关联和 `view_image`。

单条近满 512K 通过不能证明 32 条满窗口同时驻留；如需单独评估，可显式运行 `bash tests/run.sh full-window-c32 --timeout 7200`，以实际容量和请求结果判断。New API 账单、折扣倍率与真实网关链路需要在网关另行验收。公开性能记录来自用户 OCR 转录，不作为本仓库源码在新机器上的速度保证。

本源码版没有携带旧离线包的一键补丁入口；不要把旧入口或旧 `package.sha256` 拷贝过来当作本源码树的校验凭据。当前源码完整性以 `PUBLIC_SOURCE_SHA256SUMS` 为准。

## 离线移交

分别准备并移交：完整发布源码、固定 R2 的原镜像归档及其 SHA256、完整官方模型目录及官方文件清单。源码和镜像、权重可以放在不同磁盘；`deploy.sh` 接受权重目录参数，挂载时只读使用，不复制进源码目录。

断网目标机先用 `bash verify-package.sh` 核对源码，再按上面的归档校验/导入步骤准备镜像，最后运行 `bash deploy.sh /实际模型目录`。首次模型校验会读取全部模型文件，之后只有文件状态符合要求才会复用本机校验凭据。勿将其他机器的运行凭据当作本机验收结果。
