#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../scripts/common.sh"
owned "$ENGINE" && engine_health || fail 'Start a healthy engine first'
exec 9>"$STATE_DIR/operation.lock"
flock -n 9 || fail 'Another deployment operation is active'
was_running=false
if owned "$FRONTEND" && [[ $(docker inspect --format '{{.State.Running}}' "$FRONTEND") == true ]]; then
  was_running=true
  docker stop -t 10 "$FRONTEND" >/dev/null
fi
restore_ingress() { if [[ "$was_running" == true ]]; then docker start "$FRONTEND" >/dev/null; fi; }
trap restore_ingress EXIT
run=${DSV411017_REPORT_ID:-"r39-alignment-$(date -u +%Y%m%dT%H%M%S)-$$"}
[[ "$run" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]{0,159}$ ]] || fail 'Invalid DSV411017_REPORT_ID'
mkdir "$STATE_DIR/$run" || fail 'Report directory already exists'
docker exec "$ENGINE" /usr/bin/python3 /opt/dsv411017/hotfix-tests/r39_alignment.py \
  --unix-socket /state/api.sock --output "/state/$run/report.json" --model DeepSeek-Flash "$@"
printf 'R3.9 cache/multi-image alignment report: %s/%s/report.json\n' "$STATE_DIR" "$run"
