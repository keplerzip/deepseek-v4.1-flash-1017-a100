#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
cd -- "$root"
[[ -s PUBLIC_SOURCE_SHA256SUMS ]] || {
  printf 'PUBLIC_SOURCE_SHA256SUMS is missing or empty. Use a complete published source release.\n' >&2
  exit 1
}
sha256sum --check --strict --quiet PUBLIC_SOURCE_SHA256SUMS
printf 'PUBLIC_SOURCE_SHA256=PASS\n'
