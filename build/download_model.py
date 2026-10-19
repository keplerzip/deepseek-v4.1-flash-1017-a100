#!/usr/bin/env python3
"""Download/verify a fixed ModelScope snapshot using only Python's standard library.

Examples (build host, never the offline target):
  python3 build/download_model.py --dest /data/DeepSeek-V4.1-Flash-1017 --metadata-only
  python3 build/download_model.py --dest /data/DeepSeek-V4.1-Flash-1017
  python3 build/download_model.py --dest /data/DeepSeek-V4.1-Flash-1017 --verify-only
No destination file is overwritten or removed on failure.
"""
import argparse
import concurrent.futures
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
CHUNK = 8 * 1024**2
MODEL_ID = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
WEIGHT_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf",
                   ".onnx", ".h5", ".msgpack", ".npy", ".npz"}


class ContractError(RuntimeError):
    """A fixed snapshot/path/HTTP contract was violated; retain local evidence."""


def timestamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    if path.is_symlink() or temporary.is_symlink():
        raise ContractError("Symlink state file refused")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(value, output, ensure_ascii=False, indent=2)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(CHUNK), b""):
            result.update(data)
    return result.hexdigest()


def safe_path(destination, name):
    relative = PurePosixPath(name)
    if (not isinstance(name, str) or not name or "\\" in name or "\x00" in name
            or relative.is_absolute() or ".." in relative.parts
            or relative.as_posix() != name or not relative.parts):
        raise ContractError("Unsafe or noncanonical manifest path")
    path = destination.joinpath(*relative.parts)
    current = destination
    for component in relative.parts:
        current = current / component
        if current.is_symlink():
            raise ContractError("Symlink snapshot path refused: " + name)
    if not path.resolve().is_relative_to(destination.resolve()):
        raise ContractError("Snapshot path escapes the destination")
    return path


def load_manifest(path):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    revision = manifest.get("modelscope_revision", "")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ContractError("Manifest must pin a 40-character ModelScope commit")
    if manifest.get("model_id", MODEL_ID) != MODEL_ID:
        raise ContractError("Manifest model ID differs from DeepSeek-V4.1-Flash-1017")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ContractError("Manifest contains no file entries")
    names = set()
    for item in files:
        name, size = item.get("path"), item.get("size")
        if not isinstance(name, str):
            raise ContractError("Manifest file path must be a string")
        safe_path(Path("/manifest-check"), name)
        if name in names:
            raise ContractError("Duplicate manifest path: " + name)
        names.add(name)
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ContractError("Invalid manifest file size: " + name)
        if not re.fullmatch(r"[0-9a-f]{64}", item.get("sha256", "")):
            raise ContractError("Invalid publisher SHA256: " + name)
        url = urllib.parse.urlsplit(item.get("modelscope_url", ""))
        expected_path = "/models/" + MODEL_ID + "/resolve/" + revision + "/" + name
        if (url.scheme != "https" or url.hostname not in {"modelscope.cn", "www.modelscope.cn"}
                or url.username or url.password or url.port not in {None, 443}
                or url.query or url.fragment
                or urllib.parse.unquote(url.path) != expected_path):
            raise ContractError("URL must resolve the pinned ModelScope model and file: " + name)
    # A file cannot also be a parent directory of another manifest entry.
    for name in names:
        if any(parent.as_posix() in names for parent in PurePosixPath(name).parents
               if parent != PurePosixPath(".")):
            raise ContractError("Manifest file/directory collision: " + name)
    if any(name + ".partial" in names for name in names):
        raise ContractError("Manifest collides with a partial-download filename")
    return manifest


def response_end(status, headers, offset, expected):
    if headers.get("Content-Encoding", "identity").lower() not in {"identity", ""}:
        raise ContractError("Compressed HTTP responses cannot be byte-resumed")
    if status == 200:
        if offset:
            raise ContractError("Server ignored Range; partial file retained")
        end = expected
    elif status == 206:
        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", headers.get("Content-Range", ""))
        if not match:
            raise ContractError("Missing or invalid Content-Range")
        start, last, total = map(int, match.groups())
        if start != offset or total != expected or not start <= last < total:
            raise ContractError("Content-Range differs from the pinned size or offset")
        end = last + 1
    else:
        raise ContractError("Unexpected HTTP response status")
    length = headers.get("Content-Length")
    if length is not None and (not length.isdigit() or int(length) != end - offset):
        raise ContractError("Content-Length differs from the response range")
    return end


def run(args):
    manifest = load_manifest(args.manifest)
    items = [item for item in manifest["files"]
             if not args.metadata_only or PurePosixPath(item["path"]).suffix.lower() not in WEIGHT_SUFFIXES]
    if not items:
        raise ContractError("No files selected")
    destination = args.dest.absolute()
    if destination.is_symlink() or destination.resolve() != destination:
        raise ContractError("Destination must be a physical, canonical directory")
    if args.verify_only and not destination.is_dir():
        raise ContractError("Verification destination does not exist")
    destination.mkdir(parents=True, exist_ok=True)
    state = destination.parent / (destination.name + ".download-state")
    if state.is_symlink():
        raise ContractError("Symlink state directory refused")
    state.mkdir(exist_ok=True)
    lock_path = state / "download.lock"
    if lock_path.is_symlink():
        raise ContractError("Symlink download lock refused")
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ContractError("Another downloader/verifier holds this destination lock") from None
        return process(args, manifest, items, destination, state)


def process(args, manifest, items, destination, state):
    total = sum(item["size"] for item in items)
    selected_names = {item["path"] for item in items}
    progress = {}
    mutex = threading.Lock()
    stop = threading.Event()
    started = timestamp()
    reserve = int(args.reserve_gib * 1024**3)

    def update(name, **fields):
        with mutex:
            progress.setdefault(name, {}).update(fields)

    def save(status="RUNNING", failures=None):
        with mutex:
            rows = {key: dict(value) for key, value in progress.items()}
        result = {"status": status, "source": "ModelScope", "model_id": MODEL_ID,
                  "modelscope_revision": manifest["modelscope_revision"],
                  "destination": str(destination), "metadata_only": args.metadata_only,
                  "verify_only": args.verify_only, "selected_files": len(items),
                  "snapshot_files": len(manifest["files"]),
                  "complete_snapshot": status == "PASS" and len(selected_names) == len(manifest["files"]),
                  "total_bytes": total, "observed_bytes": sum(row.get("bytes", 0) for row in rows.values()),
                  "verified_files": sum(row.get("status") == "verified" for row in rows.values()),
                  "started_at": started, "updated_at": timestamp(), "files": rows,
                  "scope": "Selected file sizes and publisher SHA256; not GPU inference or performance"}
        if failures:
            result["failures"] = failures
        atomic_json(state / "progress.json", result)
        if status != "RUNNING":
            atomic_json(state / "result.json", result)
        return result

    local_bytes = 0
    for item in items:
        path = safe_path(destination, item["path"])
        partial = safe_path(destination, item["path"] + ".partial")
        for candidate in (path, partial):
            if candidate.exists() and not candidate.is_file():
                raise ContractError("Existing snapshot entry is not a regular file: " + item["path"])
        if path.exists() and path.stat().st_size != item["size"]:
            raise ContractError("Existing final file has the wrong size: " + item["path"])
        length = path.stat().st_size if path.exists() else partial.stat().st_size if partial.exists() else 0
        if length > item["size"]:
            raise ContractError("Partial file exceeds the pinned size: " + item["path"])
        local_bytes += length
        update(item["path"], bytes=length, status="pending")
    if not args.verify_only:
        required = total - local_bytes + reserve
        free = shutil.disk_usage(destination).free
        if free < required:
            message = (f"Insufficient space: available={free} required={required} "
                       f"remaining_snapshot={total-local_bytes} reserve={reserve} bytes; no downloads started")
            save("FAIL", [message])
            raise ContractError(message)

    def check_file(item):
        name, expected = item["path"], item["size"]
        path = safe_path(destination, name)
        partial = safe_path(destination, name + ".partial")
        if stop.is_set():
            return
        if path.exists():
            update(name, status="verifying")
            if sha256(path) != item["sha256"]:
                raise ContractError("Existing final file failed SHA256; retained: " + name)
            update(name, status="verified", bytes=expected, sha256=item["sha256"])
            print("VERIFIED " + name, flush=True)
            return
        if args.verify_only:
            raise ContractError("Missing final file: " + name)
        path.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(1, args.retries + 1):
            try:
                offset = partial.stat().st_size if partial.exists() else 0
                while offset < expected:
                    if stop.is_set():
                        return
                    if shutil.disk_usage(destination).free < reserve + min(CHUNK, expected-offset):
                        raise ContractError("Space reserve reached; partial files retained")
                    headers = {"User-Agent": "DeepSeek-V4.1-Flash-1017-offline/1",
                               "Accept-Encoding": "identity"}
                    if offset:
                        headers["Range"] = f"bytes={offset}-"
                    request = urllib.request.Request(item["modelscope_url"], headers=headers)
                    with urllib.request.urlopen(request, timeout=args.timeout) as response:
                        final_url = urllib.parse.urlsplit(response.geturl())
                        if final_url.scheme != "https" or final_url.username or final_url.password:
                            raise ContractError("Insecure redirect refused")
                        end = response_end(response.status, response.headers, offset, expected)
                        update(name, status="downloading", attempt=attempt, bytes=offset)
                        with partial.open("ab" if offset else "wb") as stream:
                            while offset < end:
                                data = response.read(min(CHUNK, end-offset))
                                if not data:
                                    raise OSError("Incomplete HTTP body")
                                if shutil.disk_usage(destination).free < reserve + len(data):
                                    raise ContractError("Space reserve reached; partial files retained")
                                stream.write(data)
                                offset += len(data)
                                update(name, bytes=offset)
                                if stop.is_set():
                                    return
                            if response.read(1):
                                raise ContractError("HTTP body exceeds its declared range")
                            stream.flush()
                            os.fsync(stream.fileno())
                # Zero-length entries still receive a real file and SHA256 verification.
                if not partial.exists():
                    partial.touch(exist_ok=False)
                update(name, status="verifying")
                if partial.stat().st_size != expected or sha256(partial) != item["sha256"]:
                    raise ContractError("Publisher SHA256 mismatch; partial retained: " + name)
                if path.exists():
                    raise ContractError("Final file appeared during download; both files retained: " + name)
                partial.replace(path)
                update(name, status="verified", bytes=expected, sha256=item["sha256"])
                print("VERIFIED " + name, flush=True)
                return
            except ContractError:
                raise
            except (OSError, urllib.error.URLError) as error:
                status = getattr(error, "code", None)
                update(name, status="retry", attempt=attempt,
                       error_type=type(error).__name__, http_status=status)
                # URLs and exception strings may contain signed CDN credentials.
                print(f"RETRY {name} attempt={attempt} type={type(error).__name__} HTTP={status}", flush=True)
                if status in {400, 401, 403, 404, 410, 416} or attempt == args.retries:
                    raise ContractError("Download retries exhausted or permanent HTTP error: " + name) from None
                if stop.wait(min(5, attempt)):
                    return

    failures = []
    save()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(check_file, item): item["path"] for item in items}
        while pending:
            completed, _ = concurrent.futures.wait(pending, timeout=10,
                                                   return_when=concurrent.futures.FIRST_COMPLETED)
            for future in completed:
                name = pending.pop(future)
                try:
                    future.result()
                except Exception as error:
                    # Only our own contract messages are safe to persist verbatim.
                    message = str(error) if isinstance(error, ContractError) else type(error).__name__
                    failures.append(name + ": " + message)
                    update(name, status="failed", error=message)
                    stop.set()
            result = save()
            if not completed:
                print(f'PROGRESS {result["observed_bytes"]}/{total} bytes; '
                      f'{result["verified_files"]}/{len(items)} verified', flush=True)
    result = save("FAIL" if failures else "PASS", failures)
    if failures:
        raise ContractError("Selected snapshot failed; see " + str(state / "result.json"))
    print("MODEL_DOWNLOAD=PASS " + json.dumps({key: result[key] for key in
          ("metadata_only", "verify_only", "complete_snapshot", "verified_files", "total_bytes")}), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", required=True, type=Path, help="Physical snapshot directory")
    parser.add_argument("--manifest", type=Path, default=ROOT / "manifests/model-files.json")
    parser.add_argument("--metadata-only", action="store_true", help="Skip model weight payload extensions")
    parser.add_argument("--verify-only", action="store_true", help="Verify final files without network access")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--reserve-gib", type=float, default=50, help="Free-space reserve after remaining downloads")
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16 or not 1 <= args.retries <= 10 or not 1 <= args.timeout <= 120:
        parser.error("workers must be 1..16, retries 1..10, timeout 1..120 seconds")
    if not math.isfinite(args.reserve_gib) or args.reserve_gib < 0:
        parser.error("reserve-gib must be finite and nonnegative")
    try:
        return run(args)
    except (ContractError, OSError, ValueError) as error:
        message = str(error) if isinstance(error, ContractError) else type(error).__name__
        print("MODEL_DOWNLOAD=FAIL " + message, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
