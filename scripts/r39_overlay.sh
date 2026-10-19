#!/usr/bin/env bash
# Source only; Python and all runtime imports remain in the pinned image.
prepare_r39_overlay() {
  local project_root=$1 relative count=0 seen="|"
  local list="$project_root/manifests/r39-overlay-files.txt"
  [[ -f "$list" && ! -L "$list" ]] || { printf 'Missing R3.9 protocol overlay list.\n' >&2; return 1; }
  R39_OVERLAY_ARGS=()
  while IFS= read -r relative || [[ -n "$relative" ]]; do
    case "$relative" in
      vllm/tokenizers/deepseek_v41.py|vllm/entrypoints/anthropic/serving.py|vllm/entrypoints/openai/responses/protocol.py|dsv41_guard.py) ;;
      *) printf 'Unexpected protocol overlay path: %s\n' "$relative" >&2; return 1 ;;
    esac
    case "$seen" in
      *"|$relative|"*) printf 'Duplicate protocol overlay path.\n' >&2; return 1 ;;
    esac
    seen="$seen$relative|"
    [[ -f "$project_root/source/r39-overlay/$relative" && ! -L "$project_root/source/r39-overlay/$relative" ]] || { printf 'Protocol overlay source is missing or unsafe.\n' >&2; return 1; }
    [[ $(realpath -e -- "$project_root/source/r39-overlay/$relative") == "$project_root/source/r39-overlay/$relative" ]] || { printf 'Protocol overlay parent path is unsafe.\n' >&2; return 1; }
    R39_OVERLAY_ARGS+=(-v "$project_root/source/r39-overlay/$relative:/usr/local/lib/python3.12/dist-packages/$relative:ro")
    count=$((count + 1))
  done < "$list"
  [[ "$count" == 4 ]] || { printf 'Protocol overlay must contain all four pinned files.\n' >&2; return 1; }
}
