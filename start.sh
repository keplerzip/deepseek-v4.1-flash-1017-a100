#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/scripts/common.sh"
resolve_image
resolve_model
exec 9>"$STATE_DIR/operation.lock"
flock -n 9 || fail 'Another deployment operation is active'
image_python /opt/dsv411017/scripts/profiles.py prepare > "$STATE_DIR/profile-prepare.log"
image_python /opt/dsv411017/scripts/profiles.py env > "$STATE_DIR/profile-env.nul"
mapfile -d '' -t PROFILE_ENV < "$STATE_DIR/profile-env.nul"
image_python /opt/dsv411017/scripts/runtime.py check
image_python /package/tests/check_r39_protocols.py --output /state/r39-protocol-compat.json
if docker inspect "$ENGINE" >/dev/null 2>&1; then
  owned "$ENGINE" || fail 'Engine name belongs to another deployment'
  if engine_health; then
    [[ $(docker inspect --format '{{.Config.Image}}' "$ENGINE") == "$IMAGE_ID" ]] || fail 'Running engine uses another runtime image'
    [[ -f "$STATE_DIR/ready.signature" && $(cat "$STATE_DIR/ready.signature") == "$(ready_signature)" ]] || fail 'This engine/configuration has not completed acceptance; use restart.sh to rerun all startup gates'
    ensure_frontend
    printf 'Existing accepted engine and API are healthy.\n'
    exit 0
  fi
  [[ $(docker inspect --format '{{.State.Running}}' "$ENGINE") == false ]] || fail 'Engine is running but not ready; inspect status/logs before restarting'
  remove_owned "$ENGINE"
fi
if docker inspect "$FRONTEND" >/dev/null 2>&1; then remove_owned "$FRONTEND"; fi
# Hardware checks use the bundled tools; no host Python/nvidia-smi package is needed.
docker run --rm --pull never --gpus all --network none --read-only --tmpfs /tmp:rw,size=1g \
  "${USER_ARGS[@]}" -v "$STATE_DIR:/state" "$IMAGE_ID" probe --output /state/hardware.json
bash "$ROOT/tests/kernels.sh"
tp_ar_gate=$(image_python /opt/dsv411017/scripts/profiles.py requires --gate tp-ar)
context_gate=$(image_python /opt/dsv411017/scripts/profiles.py requires --gate context-graph)
[[ "$tp_ar_gate" == required || "$tp_ar_gate" == skip ]] || fail 'Invalid TP correctness gate decision'
[[ "$context_gate" == required || "$context_gate" == skip ]] || fail 'Invalid context graph correctness gate decision'
if [[ "$tp_ar_gate" == required ]]; then bash "$ROOT/tests/tp-ar.sh"; fi
if [[ "$context_gate" == required ]]; then bash "$ROOT/tests/context-graph.sh"; fi
run_id="$(date -u +%Y%m%dT%H%M%S)-$$"
observations="/state/runs/$run_id/observations"
mkdir -p "$STATE_DIR/runs/$run_id/observations"
printf '%s\n' "$run_id" > "$STATE_DIR/current-run"
if [[ -S "$STATE_DIR/api.sock" ]]; then rm -- "$STATE_DIR/api.sock"; fi
[[ ! -e "$STATE_DIR/api.sock" ]] || fail 'Unexpected file at the engine Unix socket path'
docker run -d --pull never --name "$ENGINE" --init --restart no \
  --label "dsv411017.owner=$OWNER" --label "dsv411017.root=$ROOT" \
  --network none --gpus all --ipc host --ulimit memlock=-1:-1 --ulimit stack=67108864:67108864 \
  --read-only --tmpfs /tmp:rw,size=8g "${USER_ARGS[@]}" "${PROFILE_ENV[@]}" \
  --log-opt max-size=100m --log-opt max-file=5 \
  -e "DSV41_OBSERVATION_DIR=$observations" \
  -v "$MODEL_DIR:/models/DeepSeek-V4.1-Flash-1017:ro" -v "$STATE_DIR:/state" \
  -v "$STATE_DIR/runtime.active.json:/config/runtime.json:ro" \
  -v "$ROOT/scripts/network_proof.py:/opt/dsv411017/scripts/network_proof.py:ro" \
  -v "$ROOT/scripts/serve_with_usage.py:/opt/dsv411017/scripts/serve_with_usage.py:ro" \
  -v "$ROOT/tests:/opt/dsv411017/hotfix-tests:ro" \
  --entrypoint /usr/bin/python3 "$IMAGE_ID" /opt/dsv411017/scripts/serve_with_usage.py serve --config /config/runtime.json >/dev/null
trap 'docker logs --timestamps "$ENGINE" > "$STATE_DIR/runs/$run_id/engine.log" 2>&1 || true' EXIT
deadline=$((SECONDS + ${DSV411017_START_TIMEOUT:-7200}))
until engine_health; do
  [[ $(docker inspect --format '{{.State.Running}}' "$ENGINE") == true ]] || fail "Engine exited; see runtime/runs/$run_id/engine.log"
  ((SECONDS < deadline)) || fail 'Engine startup timed out; logs retained and no context/concurrency downgrade applied'
  sleep 5
done
network_mode=$(docker inspect --format '{{.HostConfig.NetworkMode}}' "$ENGINE")
[[ "$network_mode" == none ]] || fail 'Engine Docker NetworkMode must remain none'
docker exec "$ENGINE" /usr/bin/python3 /opt/dsv411017/scripts/network_proof.py \
  --docker-network-mode "$network_mode" --output "/state/runs/$run_id/network.json"
docker exec "$ENGINE" /usr/bin/python3 /opt/dsv411017/scripts/runtime.py capacity \
  --config /config/runtime.json --observations "$observations" --output "/state/runs/$run_id/capacity.json"
docker exec "$ENGINE" /usr/bin/python3 /opt/dsv411017/tests/acceptance.py \
  --unix-socket /state/api.sock --suite smoke --output "/state/runs/$run_id/smoke.json"
docker exec "$ENGINE" /usr/bin/python3 /opt/dsv411017/scripts/runtime.py capacity \
  --config /config/runtime.json --observations "$observations" --require-replay \
  --output "/state/runs/$run_id/capacity-after-smoke.json"
ensure_frontend
# Same no-key local and host.docker.internal access contract as R3.9.
docker run --rm --pull never --network host --read-only --tmpfs /tmp:rw,size=64m \
  "${USER_ARGS[@]}" -v "$ROOT:/package:ro" -v "$STATE_DIR:/state" \
  --entrypoint /usr/bin/python3 "$IMAGE_ID" /package/scripts/api_access_check.py \
  --output "/state/runs/$run_id/api-local.json"
docker run --rm --pull never --network bridge --add-host host.docker.internal:host-gateway \
  --read-only --tmpfs /tmp:rw,size=64m "${USER_ARGS[@]}" \
  -v "$ROOT:/package:ro" -v "$STATE_DIR:/state" --entrypoint /usr/bin/python3 "$IMAGE_ID" \
  /package/scripts/api_access_check.py --docker-client --output "/state/runs/$run_id/api-docker.json"
ready_signature > "$STATE_DIR/ready.signature"
printf 'Ready: http://127.0.0.1:%s/v1\nModel: DeepSeek-Flash\nDSpark k5 / global C32 / 524288 tokens / original vision\nProfile: runtime/speed-profile.json\nMeasured KV capacity: runtime/runs/%s/capacity-after-smoke.json\n' "$API_PORT" "$run_id"
