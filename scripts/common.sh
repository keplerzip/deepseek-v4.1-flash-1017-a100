#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
STATE_DIR="$ROOT/runtime"
ENGINE=dsv411017-engine
FRONTEND=dsv411017-api
API_PORT=${DSV411017_API_PORT:-8005}
[[ "$API_PORT" == 8005 ]] || { printf "API port is fixed at 8005 to match R3.9.\n" >&2; exit 2; }
unset R3_API_KEY VLLM_API_KEY
OWNER=deepseek-v41-1017-offline
umask 077
mkdir -p "$STATE_DIR" "$STATE_DIR/home" "$STATE_DIR/tmp" "$STATE_DIR/cache"
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
docker() { command sudo -n docker "$@"; }
[[ "$API_PORT" =~ ^[0-9]+$ ]] && (( API_PORT >= 1024 && API_PORT <= 65535 )) || fail 'Invalid DSV411017_API_PORT'

resolve_image() {
  [[ -f "$ROOT/manifests/runtime.env" ]] || fail 'Build/package the runtime image first'
  source "$ROOT/manifests/runtime.env"
  [[ "$IMAGE_ID" =~ ^sha256:[a-f0-9]{64}$ && "$IMAGE_TAG" =~ ^deepseek-v4.1-flash-1017-a100:20261018-r[12]$ ]] || fail 'Invalid runtime manifest'
  local actual allowed accepted=false
  for allowed in ${IMAGE_ALLOWED_IDS:-$IMAGE_ID}; do [[ "$allowed" =~ ^sha256:[a-f0-9]{64}$ ]] || fail 'Invalid allowed image identity'; done
  actual=$(docker image inspect --format '{{.Id}}' "$IMAGE_TAG" 2>/dev/null || true)
  for allowed in ${IMAGE_ALLOWED_IDS:-$IMAGE_ID}; do [[ "$actual" != "$allowed" ]] || accepted=true; done
  if [[ "$accepted" != true ]]; then
    local archive="$ROOT/images/runtime-image.tar" archive_hash
    [[ -f "$archive" ]] || fail 'Pinned image is missing; place images/runtime-image.tar from the runtime package here'
    archive_hash=$(sha256sum "$archive")
    [[ ${archive_hash%% *} == "$IMAGE_ARCHIVE_SHA256" ]] || fail 'Runtime archive SHA256 mismatch'
    docker load -i "$archive"
    actual=$(docker image inspect --format '{{.Id}}' "$IMAGE_TAG")
    for allowed in ${IMAGE_ALLOWED_IDS:-$IMAGE_ID}; do [[ "$actual" != "$allowed" ]] || accepted=true; done
  fi
  [[ "$accepted" == true ]] || fail 'Loaded image identity differs from the verified runtime'
  # Classic overlay2 Docker exposes a config ID; the containerd image store
  # may expose the archived OCI index ID. Both are tied to the same archive.
  IMAGE_ID=$actual
}

resolve_model() {
  local requested=${1:-}
  if [[ -z "$requested" && -f "$STATE_DIR/model.path" ]]; then requested=$(cat "$STATE_DIR/model.path"); fi
  [[ -n "$requested" ]] || requested="$ROOT/models/DeepSeek-V4.1-Flash-1017"
  [[ -d "$requested" ]] || fail "Model directory missing: $requested"
  MODEL_DIR=$(realpath -e -- "$requested")
  [[ "$MODEL_DIR" != *$'\n'* && "$MODEL_DIR" != *:* && "$MODEL_DIR" != *,* ]] || fail 'Unsupported delimiter in model path'
}

USER_ARGS=(--user "$(id -u):$(id -g)" --workdir /state -e HOME=/state/home -e TMPDIR=/state/tmp
  -e VLLM_HOST_IP=127.0.0.1 -e GLOO_SOCKET_IFNAME=lo -e VLLM_API_KEY= -e R3_API_KEY=
  -e XDG_CACHE_HOME=/state/cache -e HF_HOME=/state/cache/hf
  -e TRITON_CACHE_DIR=/state/cache/triton -e TORCHINDUCTOR_CACHE_DIR=/state/cache/inductor
  -e TORCH_EXTENSIONS_DIR=/state/cache/torch-extensions -e CUDA_CACHE_PATH=/state/cache/cuda
  -e HTTP_PROXY= -e HTTPS_PROXY= -e ALL_PROXY= -e http_proxy= -e https_proxy= -e all_proxy=)
for dsv411017_group in $(id -G); do USER_ARGS+=(--group-add "$dsv411017_group"); done
source "$ROOT/scripts/r39_overlay.sh"
prepare_r39_overlay "$ROOT"
USER_ARGS+=("${R39_OVERLAY_ARGS[@]}" -e PYTHONDONTWRITEBYTECODE=1)

image_python() {
  local -a mount_model=()
  [[ -z ${MODEL_DIR:-} ]] || mount_model=(-v "$MODEL_DIR:/models/DeepSeek-V4.1-Flash-1017:ro")
  docker run --rm --pull never --network none --read-only --tmpfs /tmp:rw,size=2g \
    "${USER_ARGS[@]}" -v "$ROOT:/package:ro" -v "$STATE_DIR:/state" "${mount_model[@]}" \
    --entrypoint /usr/bin/python3 "$IMAGE_ID" "$@"
}

owned() {
  [[ $(docker inspect --format '{{index .Config.Labels "dsv411017.owner"}} {{index .Config.Labels "dsv411017.root"}}' "$1" 2>/dev/null || true) == "$OWNER $ROOT" ]]
}

remove_owned() {
  local name=$1
  if docker inspect "$name" >/dev/null 2>&1; then
    owned "$name" || fail "Container name belongs to another deployment: $name"
    docker stop -t 90 "$name" >/dev/null
    docker rm "$name" >/dev/null
  fi
}

engine_health() {
  docker exec "$ENGINE" /usr/bin/python3 /opt/dsv411017/scripts/runtime.py health >/dev/null 2>&1
}

prepare_api_policy() {
  local -a network_ids=()
  docker network ls --filter driver=bridge --format '{{.ID}}' > "$STATE_DIR/docker-network-ids.txt"
  mapfile -t network_ids < "$STATE_DIR/docker-network-ids.txt"
  (( ${#network_ids[@]} )) || fail 'Docker bridge network inventory is empty'
  docker network inspect "${network_ids[@]}" > "$STATE_DIR/docker-networks.json.tmp"
  mv -- "$STATE_DIR/docker-networks.json.tmp" "$STATE_DIR/docker-networks.json"
  docker run --rm --pull never --network host --read-only --tmpfs /tmp:rw,size=64m \
    "${USER_ARGS[@]}" -v "$ROOT:/package:ro" -v "$STATE_DIR:/state" \
    --entrypoint /usr/bin/python3 "$IMAGE_ID" /package/scripts/build_api_listeners.py \
    --networks /state/docker-networks.json --output /state/api-listeners.json --port "$API_PORT"
}

start_frontend() {
  prepare_api_policy
  docker run -d --pull never --name "$FRONTEND" --init --restart no \
    --label "dsv411017.owner=$OWNER" --label "dsv411017.root=$ROOT" \
    --network host --read-only --tmpfs /tmp:rw,size=64m "${USER_ARGS[@]}" \
    --log-opt max-size=20m --log-opt max-file=3 \
    -v "$STATE_DIR:/state:ro" -v "$ROOT:/package:ro" --entrypoint /usr/bin/python3 "$IMAGE_ID" \
    /package/scripts/frontend.py --policy /state/api-listeners.json >/dev/null
}

ensure_frontend() {
  if docker inspect "$FRONTEND" >/dev/null 2>&1; then
    owned "$FRONTEND" || fail 'Frontend name belongs to another deployment'
    [[ $(docker inspect --format '{{.Config.Image}}' "$FRONTEND") == "$IMAGE_ID" ]] || fail 'Frontend uses a different image; restart this deployment'
    if [[ $(docker inspect --format '{{.State.Running}}' "$FRONTEND") != true ]]; then docker start "$FRONTEND" >/dev/null; fi
  else
    start_frontend
  fi
  local attempt
  for attempt in {1..15}; do
    [[ $(docker inspect --format '{{.State.Running}}' "$FRONTEND") == true ]] || break
    if docker exec "$FRONTEND" /usr/bin/python3 /opt/dsv411017/scripts/runtime.py frontend-health --port "$API_PORT" >/dev/null 2>&1; then return 0; fi
    sleep 1
  done
  docker logs --tail 30 "$FRONTEND" >&2 || true
  fail 'Frontend did not become healthy; check API port conflicts and its logs'
}

ready_signature() {
  local config_sha profile_sha container_id started_at package_sha
  config_sha=$(sha256sum "$STATE_DIR/runtime.active.json")
  config_sha=${config_sha%% *}
  profile_sha=$(sha256sum "$STATE_DIR/speed-profile.json")
  profile_sha=${profile_sha%% *}
  container_id=$(docker inspect --format '{{.Id}}' "$ENGINE")
  started_at=$(docker inspect --format '{{.State.StartedAt}}' "$ENGINE")
  package_sha=$(sha256sum "$ROOT/PUBLIC_SOURCE_SHA256SUMS")
  package_sha=${package_sha%% *}
  printf '%s|%s|%s|%s|%s|%s\n' "$IMAGE_ID" "$container_id" "$started_at" "$config_sha" "$profile_sha" "$package_sha"
}
