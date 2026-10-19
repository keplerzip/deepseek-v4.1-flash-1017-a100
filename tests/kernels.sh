#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../scripts/common.sh"
resolve_image
(cd -- "$ROOT" && sha256sum --check --quiet manifests/kernel-test-support.sha256)
closure=$(sha256sum "$ROOT/manifests/kernel-test-support.sha256")
closure=${closure%% *}
docker run --rm --pull never --network none --gpus all --entrypoint nvidia-smi "$IMAGE_ID" \
  --query-gpu=index,name,memory.total,driver_version,mig.mode.current,uuid --format=csv,noheader,nounits \
  > "$STATE_DIR/current-gpus.csv.tmp"
mv -- "$STATE_DIR/current-gpus.csv.tmp" "$STATE_DIR/current-gpus.csv"
gpu_signature=$(cat "$STATE_DIR/current-gpus.csv")
signature="$IMAGE_ID|$gpu_signature|kernel-test-support=$closure"
[[ ! -f "$STATE_DIR/kernels.signature" || $(cat "$STATE_DIR/kernels.signature") != "$signature" ]] || { printf 'Reusing complete checks for this image, numerical sources and GPU/driver set.\n'; exit 0; }
run="kernel-check-$(date -u +%Y%m%dT%H%M%S)-$$"
mkdir -p "$STATE_DIR/$run/ep8"
reuse_single=false
if image_python /package/scripts/kernel_checkpoint.py --project /package --image-id "$IMAGE_ID" \
    --gpu-signature /state/current-gpus.csv --output /state/single-gpu-checkpoint.json check; then
  reuse_single=true
else
  checkpoint_status=$?
  [[ "$checkpoint_status" == 10 ]] || fail 'Single-GPU checkpoint verifier failed unexpectedly'
fi
if [[ "$reuse_single" == true ]]; then
  printf 'Reusing validated single-GPU PASS report; skipping its compilation/numerical/graph tests. Continuing at EP8.\n'
else
  printf 'No matching complete single-GPU PASS report; executing required checks.\n'
  docker run --rm --pull never --network none --read-only --tmpfs /tmp:rw,size=1g \
    "${USER_ARGS[@]}" -v "$ROOT:/deploy:ro" -v "$STATE_DIR:/state" \
    --entrypoint /usr/bin/python3 "$IMAGE_ID" /deploy/tests/check_kernel_dependencies.py \
    --output "/state/$run/dependencies.json" \
    2>&1 | tee "$STATE_DIR/$run/dependencies.log"
  docker run --rm --pull never --network none --gpus 'device=0' --ipc host --ulimit memlock=-1:-1 \
    "${USER_ARGS[@]}" -v "$ROOT:/deploy:ro" -v "$STATE_DIR:/state" --entrypoint /usr/bin/python3 "$IMAGE_ID" \
    /deploy/tests/performance_kernels.py --output "/state/$run/kernels.json" \
    2>&1 | tee "$STATE_DIR/$run/kernels.log"
  image_python /package/scripts/kernel_checkpoint.py --project /package --image-id "$IMAGE_ID" \
    --gpu-signature /state/current-gpus.csv --output /state/single-gpu-checkpoint.json check \
    --record "/state/$run/kernels.json"
fi
printf 'EP8: static IPv4 rendezvous at 127.0.0.1:29500, Gloo interface lo.\n'
docker run --rm --pull never --network none --gpus all --ipc host --ulimit memlock=-1:-1 \
  "${USER_ARGS[@]}" -v "$ROOT:/deploy:ro" -v "$STATE_DIR:/state" --entrypoint /usr/bin/python3 "$IMAGE_ID" \
  -m torch.distributed.run --nnodes 1 --node-rank 0 --nproc-per-node 8 \
  --rdzv-backend static --master-addr 127.0.0.1 --master-port 29500 --local-addr 127.0.0.1 \
  /deploy/tests/performance_ep8.py "/state/$run/ep8" \
  2>&1 | tee "$STATE_DIR/$run/ep8.log"
printf '%s\n' "$signature" > "$STATE_DIR/kernels.signature"
