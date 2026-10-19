#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../scripts/common.sh"
resolve_image
signature="$IMAGE_ID|$(cat "$STATE_DIR/kernels.signature")"
[[ ! -f "$STATE_DIR/tp-ar.signature" || $(cat "$STATE_DIR/tp-ar.signature") != "$signature" ]] || exit 0
run="tp-ar-$(date -u +%Y%m%dT%H%M%S)-$$"
mkdir -p "$STATE_DIR/$run"
docker run --rm --pull never --network none --gpus all --ipc host --ulimit memlock=-1:-1 \
  "${USER_ARGS[@]}" -v "$STATE_DIR:/state" --entrypoint /usr/bin/python3 "$IMAGE_ID" \
  -m torch.distributed.run --nnodes 1 --node-rank 0 --nproc-per-node 8 \
  --rdzv-backend static --master-addr 127.0.0.1 --master-port 29500 --local-addr 127.0.0.1 \
  /opt/dsv411017/tests/performance_tp_ar.py --output "/state/$run/reports" \
  2>&1 | tee "$STATE_DIR/$run/test.log"
printf '%s\n' "$signature" > "$STATE_DIR/tp-ar.signature"
