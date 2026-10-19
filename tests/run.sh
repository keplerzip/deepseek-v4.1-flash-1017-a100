#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../scripts/common.sh"
suite=${1:-smoke}
[[ "$suite" == smoke || "$suite" == benchmark || "$suite" == long-context || "$suite" == full || "$suite" == full-window-c32 ]] || fail 'Usage: bash tests/run.sh [smoke|benchmark|long-context|full|full-window-c32] [test options]'
if (( $# > 0 )); then shift; fi
owned "$ENGINE" && engine_health || fail 'Start a healthy engine first'
exec 9>"$STATE_DIR/operation.lock"
flock -n 9 || fail 'Another deployment operation is active'
# Pausing only this project's ingress makes the benchmark reviewable. Direct
# Extra generation traffic is detected by metric accounting. A prefill-only
# request that is then cancelled may still disturb timing without detection.
was_running=false
if owned "$FRONTEND" && [[ $(docker inspect --format '{{.State.Running}}' "$FRONTEND") == true ]]; then
  was_running=true
  docker stop -t 10 "$FRONTEND" >/dev/null
fi
restore_ingress() { if [[ "$was_running" == true ]]; then docker start "$FRONTEND" >/dev/null; fi; }
trap restore_ingress EXIT
run=${DSV411017_REPORT_ID:-"acceptance-$suite-$(date -u +%Y%m%dT%H%M%S)-$$"}
[[ "$run" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]{0,159}$ ]] || fail 'Invalid DSV411017_REPORT_ID'
mkdir "$STATE_DIR/$run" || fail 'Report directory already exists; use a new report ID'
docker exec "$ENGINE" /usr/bin/python3 /opt/dsv411017/tests/acceptance.py \
  --unix-socket /state/api.sock --suite "$suite" --output "/state/$run/report.json" "$@"
printf 'Report: %s/%s/report.json\n' "$STATE_DIR" "$run"
