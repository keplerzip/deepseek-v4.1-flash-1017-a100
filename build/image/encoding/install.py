"""Install the reviewed 1017 protocol changes into an exact R1.9 runtime."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sysconfig


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def install(site: Path, root: Path | None = None) -> dict:
    root = root or Path(__file__).resolve().parent
    manifest = json.loads((root / "manifest.json").read_text())
    reference = root / "source-reference.py"
    if digest(reference) != manifest["reference"]["sha256"]:
        raise RuntimeError("1017 reference hash mismatch")

    pending = []
    for row in manifest["files"]:
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError("Invalid encoding patch path")
        source = root / "patched" / relative
        target = site / relative
        if digest(source) != row["patched_sha256"]:
            raise RuntimeError("Encoding payload hash mismatch: " + row["path"])
        if not target.is_file() or digest(target) not in {
            row["base_sha256"], row["patched_sha256"]
        }:
            raise RuntimeError("Unexpected encoding base bytes: " + row["path"])
        # Validate all payloads and destinations before modifying any file.
        data = source.read_bytes()
        compile(data, str(target), "exec")
        pending.append((target, data))

    for target, data in pending:
        temporary = target.with_name(target.name + ".1017.tmp")
        try:
            temporary.write_bytes(data)
            temporary.chmod(target.stat().st_mode & 0o777)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
    receipt = {
        "id": manifest["id"],
        "reference": manifest["reference"],
        "files": [{"path": r["path"], "sha256": r["patched_sha256"]}
                  for r in manifest["files"]],
        "gpu_validated": False,
    }
    (root / "installed.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", type=Path,
                        default=Path(sysconfig.get_paths()["purelib"]))
    arguments = parser.parse_args()
    receipt = install(arguments.site)
    print("Installed 1017 protocol patches:", len(receipt["files"]))
