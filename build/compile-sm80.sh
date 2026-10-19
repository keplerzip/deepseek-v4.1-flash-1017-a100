#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../scripts/common.sh"
resolve_image
mkdir -p "$ROOT/build-state/sm80"
for revision in 11 12 13 14 19; do
  docker run --rm --pull never --network none "${USER_ARGS[@]}" \
    -v "$ROOT/build:/compile:ro" -v "$ROOT/build-state/sm80:/reports" -v "$STATE_DIR:/state" \
    --entrypoint /usr/bin/python3 "$IMAGE_ID" \
    "/compile/compile_r${revision}_kernels.py" --output "/reports/r${revision}.json" \
    > "$ROOT/build-state/sm80/r${revision}.log" 2>&1
done
printf 'SM80 compile-only reports: %s/build-state/sm80\nThese checks do not execute GPU kernels.\n' "$ROOT"
