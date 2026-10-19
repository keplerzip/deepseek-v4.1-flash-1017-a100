#!/usr/bin/env bash
# Reproducible, network-disabled R2 build on the pinned, complete local R1.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
[[ -n ${DSV411017_MODEL_DIR:-} ]] || { echo 'Set DSV411017_MODEL_DIR to an existing complete official 1017 model directory.' >&2; exit 2; }
IFS= read -r -d '' model_dir < <(realpath -e -z -- "$DSV411017_MODEL_DIR") || { echo 'DSV411017_MODEL_DIR cannot be resolved.' >&2; exit 2; }
[[ -d "$model_dir" && "$model_dir" != *:* && "$model_dir" != *,* && "$model_dir" != *$'\n'* && "$model_dir" != *$'\r'* && "$model_dir" != *$'\t'* ]] || { echo 'Model directory is missing or contains unsupported mount delimiters.' >&2; exit 2; }
for model_file in config.json tokenizer.json tokenizer_config.json encoding/encoding.py; do
  [[ -f "$model_dir/$model_file" ]] || { echo "Official model file missing: $model_file" >&2; exit 2; }
done
[[ -d "$model_dir/encoding/tests" ]] || { echo 'Official encoding/tests directory is required.' >&2; exit 2; }
model_mount=(-v "$model_dir:/models/DeepSeek-V4.1-Flash-1017:ro")
base=deepseek-v4.1-flash-1017-a100:20261018-r1
expected=sha256:10743e6faaa326f4049b4e661e172fc5be5346a50894242784b136131c9569ca
actual=$(sudo -n docker image inspect --format '{{.Id}}' "$base")
[[ "$actual" == "$expected" ]] || { echo 'Pinned R1 image identity mismatch' >&2; exit 1; }
tag=deepseek-v4.1-flash-1017-a100:20261018-r2
mkdir -p "$root/build-state" "$root/images"
sudo -n docker build --pull=false --network=none --progress=plain -f "$root/build/image/Dockerfile.speed" -t "$tag" "$root" \
  2>&1 | tee "$root/build-state/image-build.log"
sudo -n docker image inspect "$tag" > "$root/build-state/image-inspect.json"
sudo -n docker run --rm --pull never --network none --read-only --tmpfs /tmp:rw,size=2g \
  --entrypoint /usr/bin/python3 -v "$root:/project:ro" "$tag" \
  /project/build/validate_profiles.py --project /project > "$root/build-state/cli-test.log" 2>&1
sudo -n docker run --rm --pull never --network none --read-only --tmpfs /tmp:rw,size=2g \
  --entrypoint /usr/bin/python3 -v "$root:/project:ro" "$tag" \
  -m unittest discover -s /project/tests -p 'test_*.py' > "$root/build-state/cpu-tests.log" 2>&1
sudo -n docker run --rm --pull never --network none --read-only --tmpfs /tmp:rw,size=2g \
  --entrypoint /usr/bin/python3 -v "$root:/project:ro" "${model_mount[@]}" "$tag" \
  /project/build/test_installed_runtime.py --model /models/DeepSeek-V4.1-Flash-1017 --expected-stage r2 \
  > "$root/build-state/installed-runtime.log" 2>&1
sudo -n docker run --rm --pull never --network none --read-only --tmpfs /tmp:rw,size=2g \
  --entrypoint /usr/bin/python3 -v "$root/build-state:/evidence" "$tag" \
  /opt/dsv411017/image/speed/tests/test_context_combine.py --output /evidence/context-combine-cpu.json \
  > "$root/build-state/context-combine-cpu.log" 2>&1
sudo -n docker run --rm --pull never --network none --read-only --tmpfs /tmp:rw,size=2g \
  --entrypoint /usr/bin/python3 "$tag" \
  /opt/dsv411017/tests/performance_tp_ar.py --import-check \
  > "$root/build-state/tp-ar-import.log" 2>&1
python3 "$root/build/test_offline_transport.py" --image "$tag" > "$root/build-state/offline-transport.log" 2>&1
sudo -n docker save "$tag" > "$root/images/runtime-image.tar.partial"
python3 "$root/build/verify_image.py" --archive "$root/images/runtime-image.tar.partial" --tag "$tag" \
  --inspect "$root/build-state/image-inspect.json" --output "$root/manifests/image-integrity.json" \
  > "$root/build-state/image-integrity.log"
mv -- "$root/images/runtime-image.tar.partial" "$root/images/runtime-image.tar"
image_id=$(sudo -n docker image inspect --format '{{.Id}}' "$tag")
archive_sha=$(sha256sum "$root/images/runtime-image.tar")
archive_sha=${archive_sha%% *}
allowed=$(python3 - "$root/manifests/image-integrity.json" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
print(' '.join(sorted({p['image_id'],p['image_config_id'],p['platform_manifest_id']} - {None})))
PY
)
printf 'IMAGE_TAG=%s\nIMAGE_ID=%s\nIMAGE_ALLOWED_IDS="%s"\nIMAGE_ARCHIVE_SHA256=%s\n' "$tag" "$image_id" "$allowed" "$archive_sha" > "$root/manifests/runtime.env"
printf 'R2 runtime image verified: %s\nGPU inference and TPS remain pending target acceptance.\n' "$image_id"
