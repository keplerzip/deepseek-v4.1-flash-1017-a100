"""Run installed reasoning protocol checks in the pinned offline CPU image."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, default=ROOT / "build-state/reasoning-compatibility")
    args = parser.parse_args()
    output = args.evidence_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT / "manifests/r39-overlay-provenance.json"
    manifest = json.loads(manifest_path.read_text())
    command = ["sudo", "-n", "docker", "run", "--rm", "--pull", "never", "--network", "none",
               "--read-only", "--tmpfs", "/tmp:rw,size=2g", "-e", "HOME=/tmp", "-e", "XDG_CACHE_HOME=/tmp/cache",
               "-e", "CUDA_VISIBLE_DEVICES=", "-e", "PYTHONDONTWRITEBYTECODE=1",
               "-v", str(ROOT) + ":/package:ro", "-v", str(output) + ":/audit-output"]
    hashes = {}
    rows = manifest["files"]
    if len(rows) != 4:
        raise RuntimeError("Exactly four reviewed source overlays are required")
    for row in rows:
        source = ROOT / row["overlay_path"]
        if source.is_symlink() or not source.is_file():
            raise RuntimeError("Unsafe or missing overlay source")
        hashes[row["overlay_path"]] = hashlib.sha256(source.read_bytes()).hexdigest()
        command += ["-v", str(source) + ":" + row["site_packages_path"] + ":ro"]
    command += ["--entrypoint", "/usr/bin/python3", manifest["native_image_tag"],
                "/package/tests/check_reasoning_compat.py", "--output", "/audit-output/report.json"]
    print("Running installed Chat/Responses/Messages reasoning compatibility without network or GPU.", flush=True)
    report_path = output / "report.json"
    if report_path.exists():
        report_path.unlink()
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    log_path = output / "check.log"
    log_path.write_text(result.stdout)
    if not report_path.is_file():
        raise RuntimeError("CPU compatibility check produced no report: " + result.stdout[-6000:])
    report = json.loads(report_path.read_text())
    for relative in ("tests/check_reasoning_compat.py", "build/test_reasoning_compat.py",
                     "manifests/r39-overlay-provenance.json", "manifests/r39-overlay-files.txt"):
        hashes[relative] = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
    evidence = {
        "schema": 1, "status": report["status"], "exit_code": result.returncode,
        "image": manifest["native_image_tag"], "gpu_executed": False, "weights_loaded": False,
        "network_used": False, "real_inference_executed": False,
        "checks": len(report.get("checks", [])),
        "scope": report["scope"], "source_sha256": hashes,
        "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "log_sha256": hashlib.sha256(log_path.read_bytes()).hexdigest(),
        "finished_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    (output / "verification.json").write_text(json.dumps(evidence, indent=2) + "\n")
    if result.returncode != 0 or report["status"] != "PASS":
        failed = [row for row in report.get("checks", []) if row["status"] != "PASS"]
        raise RuntimeError("Installed reasoning compatibility failed: " + json.dumps(failed, ensure_ascii=False))
    print(json.dumps({"status": "PASS", "checks": evidence["checks"], "gpu_executed": False,
                      "verification": str(output / "verification.json")}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(type(error).__name__ + ": " + str(error), file=sys.stderr)
        raise SystemExit(1)
