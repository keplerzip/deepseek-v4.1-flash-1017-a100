#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/scripts/common.sh"
[[ $# -le 1 ]] || fail 'Usage: bash deploy.sh [MODEL_DIRECTORY]'
resolve_image
resolve_model "${1:-}"
exec 9>"$STATE_DIR/operation.lock"
flock -n 9 || fail 'Another deployment operation is active'
if ! image_python /opt/dsv411017/scripts/runtime.py check; then
  printf 'Verifying every ModelScope file against the pinned publisher SHA256.\n'
  image_python /opt/dsv411017/scripts/runtime.py verify
fi
printf '%s\n' "$MODEL_DIR" > "$STATE_DIR/model.path"
flock -u 9
exec bash "$ROOT/start.sh"
