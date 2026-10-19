#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../scripts/common.sh"
owned "$ENGINE" && engine_health || fail 'Start a healthy accepted engine first'
exec 9>"$STATE_DIR/operation.lock"
flock -n 9 || fail 'Another operation is active'
was_running=false
if owned "$FRONTEND" && [[ $(docker inspect --format '{{.State.Running}}' "$FRONTEND") == true ]]; then
  was_running=true
  docker stop -t 10 "$FRONTEND" >/dev/null
fi
restore_ingress() { if [[ "$was_running" == true ]]; then docker start "$FRONTEND" >/dev/null; fi; }
trap restore_ingress EXIT
run_id=$(cat "$STATE_DIR/current-run")
run=${DSV411017_REPORT_ID:-"speed-diagnosis-$(date -u +%Y%m%dT%H%M%S)-$$"}
[[ "$run" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]{0,159}$ ]] || fail 'Invalid DSV411017_REPORT_ID'
mkdir "$STATE_DIR/$run" || fail 'Report directory already exists; use a new report ID'
cp "$STATE_DIR/speed-profile.json" "$STATE_DIR/$run/profile.json"
docker exec "$ENGINE" /usr/bin/python3 /opt/dsv411017/tests/diagnose_speed.py \
  --unix-socket /state/api.sock --observations "/state/runs/$run_id/observations" \
  --runtime-report "/state/runs/$run_id/capacity-after-smoke.json" \
  --output "/state/$run/report.json" "$@"
printf 'Report and per-request DSpark/latency evidence: %s/%s\n' "$STATE_DIR" "$run"
