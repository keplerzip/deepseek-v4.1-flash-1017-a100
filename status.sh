#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/scripts/common.sh"
docker ps -a --filter "label=dsv411017.owner=$OWNER" --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}'
if owned "$ENGINE"; then
  docker logs --tail 40 "$ENGINE"
  engine_health && printf 'ENGINE_HEALTH=PASS\n' || printf 'ENGINE_HEALTH=NOT_READY\n'
fi
[[ ! -f "$STATE_DIR/current-run" ]] || printf 'Current reports: %s/runs/%s\n' "$STATE_DIR" "$(cat "$STATE_DIR/current-run")"
