#!/usr/bin/env bash
# Connected build host only. The actual Docker build is network-disabled.
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
base=deepseek-v4.1-flash-a100:20260913-r1
expected=sha256:29e46788ca2667042d06b496e92f6d38c7b50bee16f5b1d211e9229e68907403
actual=$(sudo -n docker image inspect --format '{{.Id}}' "$base")
[[ "$actual" == "$expected" ]] || { echo 'Base image identity mismatch; use the reviewed local SM80 runtime.' >&2; exit 1; }
[[ -f "$root/build/image/encoding/install.py" && -f "$root/tests/acceptance.py" ]] || { echo '1017 encoding overlays and acceptance tests must be present before building.' >&2; exit 1; }
tag=deepseek-v4.1-flash-1017-a100:20261018-r1
mkdir -p "$root/build-state" "$root/images"
sudo -n docker build --pull=false --network=none --progress=plain -f "$root/build/image/Dockerfile" -t "$tag" "$root" \
  2>&1 | tee "$root/build-state/image-build.log"
sudo -n docker image inspect "$tag" > "$root/build-state/image-inspect.json"
sudo -n docker run --rm --pull never --network none "$tag" cli-test \
  > "$root/build-state/cli-test.log" 2>&1
sudo -n docker run --rm --pull never --network none --read-only --tmpfs /tmp:rw,size=2g \
  --entrypoint /usr/bin/python3 -v "$root:/project:ro" "$tag" \
  -m unittest discover -s /project/tests -p 'test_*.py' \
  > "$root/build-state/cpu-tests.log" 2>&1
sudo -n docker run --rm --pull never --network none --read-only --tmpfs /tmp:rw,size=2g \
  --entrypoint /usr/bin/python3 -v "$root:/project:ro" "${model_mount[@]}" "$tag" \
  /project/build/test_installed_runtime.py --model /models/DeepSeek-V4.1-Flash-1017 --expected-stage r1 \
  > "$root/build-state/installed-runtime.log" 2>&1
python3 "$root/build/test_offline_transport.py" --image "$tag" \
  > "$root/build-state/offline-transport.log" 2>&1
sudo -n docker save "$tag" > "$root/images/runtime-image.tar.partial"
python3 "$root/build/verify_image.py" --archive "$root/images/runtime-image.tar.partial" \
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
printf 'Runtime image verified: %s\nGPU inference and speed: pending target acceptance.\n' "$image_id"
