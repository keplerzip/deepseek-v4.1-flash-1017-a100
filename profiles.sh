#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/scripts/common.sh"
resolve_image
operation=${1:-list}
[[ "$operation" == list || "$operation" == select || "$operation" == show ]] || fail 'Usage: bash profiles.sh list|show|select PROFILE'
exec 9>"$STATE_DIR/operation.lock"
flock -n 9 || fail 'Another operation is active'
if [[ "$operation" == select ]]; then
  [[ $# == 2 ]] || fail 'Usage: bash profiles.sh select PROFILE'
  if owned "$ENGINE" && [[ $(docker inspect --format '{{.State.Running}}' "$ENGINE") == true ]]; then fail 'Stop this engine before selecting a different profile'; fi
  image_python /opt/dsv411017/scripts/profiles.py select "$2"
elif [[ "$operation" == show ]]; then
  [[ -f "$STATE_DIR/speed-profile.json" ]] || fail 'No prepared profile yet'
  cat "$STATE_DIR/speed-profile.json"
else
  image_python /opt/dsv411017/scripts/profiles.py list
fi
