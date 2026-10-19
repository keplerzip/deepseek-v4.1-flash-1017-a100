"""Install or restore the opt-in DSpark context-combine capture overlay."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sysconfig


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def install(site: Path, root: Path | None = None, rollback: bool = False) -> dict:
    root = root or Path(__file__).resolve().parent
    manifest = json.loads((root / "manifest.json").read_text())
    pending = []
    for row in manifest["files"]:
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError("Invalid speed-overlay path")
        baseline = root / row["baseline_source"]
        patched = root / "patched" / relative
        if digest(baseline) != row["base_sha256"]:
            raise RuntimeError("Speed-overlay baseline hash mismatch")
        if digest(patched) != row["patched_sha256"]:
            raise RuntimeError("Speed-overlay payload hash mismatch")
        target = site / relative
        if not target.is_file() or digest(target) not in {
            row["base_sha256"], row["patched_sha256"]
        }:
            raise RuntimeError("Unexpected speed-overlay base bytes: " + str(relative))
        source = baseline if rollback else patched
        data = source.read_bytes()
        compile(data, str(target), "exec")
        pending.append((target, data))

    # Check every destination and both directions before replacing any file.
    for target, data in pending:
        temporary = target.with_name(target.name + ".speed.tmp")
        try:
            temporary.write_bytes(data)
            temporary.chmod(target.stat().st_mode & 0o777)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
    return {
        "id": manifest["id"],
        "action": "rollback" if rollback else "install",
        "feature_enabled": False,
        "enable_env": manifest["enable_env"],
        "files": [{"path": row["path"], "sha256": row[
            "base_sha256" if rollback else "patched_sha256"]}
            for row in manifest["files"]],
        "gpu_validated": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", type=Path,
                        default=Path(sysconfig.get_paths()["purelib"]))
    parser.add_argument("--rollback", action="store_true")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    receipt = install(args.site, rollback=args.rollback)
    if args.receipt:
        args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
