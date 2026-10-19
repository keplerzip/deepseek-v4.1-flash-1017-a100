#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../scripts/common.sh"
resolve_image
signature="$IMAGE_ID|$(cat "$STATE_DIR/kernels.signature")"
[[ ! -f "$STATE_DIR/context-graph.signature" || $(cat "$STATE_DIR/context-graph.signature") != "$signature" ]] || exit 0
run="context-graph-$(date -u +%Y%m%dT%H%M%S)-$$"
mkdir -p "$STATE_DIR/$run"
docker run --rm --pull never --network none --gpus 'device=0' --ipc host --ulimit memlock=-1:-1 \
  "${USER_ARGS[@]}" -v "$STATE_DIR:/state" --entrypoint /usr/bin/python3 "$IMAGE_ID" \
  /opt/dsv411017/image/speed/tests/test_context_combine.py --gpu --output "/state/$run/report.json" \
  2>&1 | tee "$STATE_DIR/$run/test.log"
printf '%s\n' "$signature" > "$STATE_DIR/context-graph.signature"
