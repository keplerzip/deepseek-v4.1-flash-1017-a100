#!/usr/bin/env python3
"""Real pinned-R2, network-none, CPU-only launcher regression; never runs EP8/GPU.

The baseline intentionally retains the current EP8 launcher's missing master-port.
Its Gloo store timeout is shortened to keep this negative test bounded.  The fixed
case runs two real CPU workers and checks an all-reduce result, not GPU kernels.
"""
import argparse
import concurrent.futures
import datetime
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
import uuid


IMAGE = "deepseek-v4.1-flash-1017-a100:20261018-r2"
ALLOWED_IMAGE = "sha256:a63405a72ce202bb8347f17fc2432be0df484b794fba0224fd210b6daff004f6"
SCRIPT = Path(__file__).resolve()
FIXED_FLAGS = ["--nnodes", "1", "--node-rank", "0", "--rdzv-backend", "static",
               "--master-addr", "127.0.0.1", "--master-port", "29500",
               "--local-addr", "127.0.0.1"]


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def source_probe():
    import torch
    import torch.distributed.run as run
    from torch.distributed.elastic.rendezvous.api import RendezvousStoreInfo
    from torch.distributed.elastic.rendezvous import static_tcp_rendezvous as static
    from torch.distributed.elastic.rendezvous import dynamic_rendezvous as dynamic
    parsed = run.parse_args(["--nproc-per-node", "2", "--master-addr", "127.0.0.1", "/audit.py"])
    print(json.dumps({
        "torch_version": torch.__version__, "hostname": socket.gethostname(),
        "interfaces": socket.if_nameindex(), "hosts": Path("/etc/hosts").read_text(),
        "ipv4_routes": Path("/proc/net/route").read_text(),
        "ipv6_routes": Path("/proc/net/ipv6_route").read_text(),
        "parsed_baseline": vars(parsed),
        "source_files": {m.__name__: {"path": m.__file__, "sha256": hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()}
                         for m in [run, static, dynamic]},
        "run_source": inspect.getsource(run.run),
        "bootstrap_source": inspect.getsource(RendezvousStoreInfo.build),
        "static_source": inspect.getsource(static.StaticTCPRendezvous.next_rendezvous),
        "gpu_executed": False, "actual_model_inference": False,
    }, ensure_ascii=False))


def worker(case):
    import torch
    import torch.distributed as dist
    rank = int(os.environ["RANK"])
    path = Path("/evidence") / case / f"rank-{rank}.json"
    result = {
        "rank": rank, "world_size": int(os.environ["WORLD_SIZE"]),
        "master_addr": os.environ["MASTER_ADDR"], "master_port": os.environ["MASTER_PORT"],
        "hostname": socket.gethostname(), "interfaces": socket.if_nameindex(),
        "hosts": Path("/etc/hosts").read_text(),
        "gloo_socket_ifname": os.environ.get("GLOO_SOCKET_IFNAME"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_initialized_before": torch.cuda.is_initialized(),
        "torch_version": torch.__version__, "status": "STARTED", "gpu_executed": False,
        "actual_model_inference": False,
    }
    try:
        result["master_resolves"] = [x[4][0] for x in socket.getaddrinfo(result["master_addr"], int(result["master_port"]))]
    except socket.gaierror as exc:
        result["master_resolve_error"] = {"errno": exc.errno, "message": str(exc)}
    dump(path, result)
    try:
        dist.init_process_group("gloo", timeout=datetime.timedelta(seconds=8))
        value = torch.tensor(float(rank + 1), device="cpu")
        dist.all_reduce(value)
        if value.item() != 3.0:
            raise AssertionError(f"all_reduce got {value.item()}, expected 3.0")
        dist.barrier()
        result.update(status="PASS", all_reduce_sum=value.item(), backend=dist.get_backend())
        dist.destroy_process_group()
    except Exception as exc:
        result.update(status="FAIL", exception=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        result["cuda_initialized_after"] = torch.cuda.is_initialized()
        dump(path, result)


def launch_case(base, case, flags, timeout, gloo_lo):
    folder = base / case
    folder.mkdir()
    name = "torchrun-cpu-audit-" + uuid.uuid4().hex[:12]
    cmd = ["sudo", "-n", "docker", "run", "--rm", "--pull", "never", "--name", name,
           "--network", "none", "--read-only", "--tmpfs", "/tmp:rw,size=128m",
           "--user", f"{os.getuid()}:{os.getgid()}", "--workdir", "/evidence",
           "-e", "CUDA_VISIBLE_DEVICES=", "-e", "OMP_NUM_THREADS=1", "-e", "PYTHONDONTWRITEBYTECODE=1",
           "-v", f"{SCRIPT}:/audit.py:ro", "-v", f"{base}:/evidence"]
    if gloo_lo:
        cmd += ["-e", "GLOO_SOCKET_IFNAME=lo"]
    cmd += ["--entrypoint", "/usr/bin/python3", ALLOWED_IMAGE,
            "-m", "torch.distributed.run", "--nproc-per-node", "2", *flags, "/audit.py", "--worker", case]
    started = time.monotonic()
    timed_out = False
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    container = None
    for _ in range(20):
        probe = subprocess.run(["sudo", "-n", "docker", "inspect", "--format",
                                '{"hostname":{{json .Config.Hostname}},"network_mode":{{json .HostConfig.NetworkMode}}}', name],
                               capture_output=True, text=True, timeout=5)
        if probe.returncode == 0:
            container = json.loads(probe.stdout)
            break
        time.sleep(0.1)
    try:
        output, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        subprocess.run(["sudo", "-n", "docker", "rm", "-f", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
        output, _ = proc.communicate(timeout=10)
    finally:
        subprocess.run(["sudo", "-n", "docker", "rm", "-f", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
    (folder / "docker.log").write_text(output)
    ranks = [json.loads(p.read_text()) for p in sorted(folder.glob("rank-*.json"))]
    return {"case": case, "command": cmd, "exit_code": proc.returncode,
            "timed_out": timed_out, "elapsed_seconds": round(time.monotonic() - started, 3),
            "hostname_resolution_warning": bool(re.search(r"IPv[46].*retrieved|gai error|Name or service not known|Temporary failure in name resolution", output)),
            "container": container, "workers": ranks, "log": str(folder / "docker.log")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-probe", action="store_true")
    parser.add_argument("--worker")
    parser.add_argument("--output", type=Path, default=SCRIPT.parent.parent / "build-state/torchrun-offline-ep8-final")
    args = parser.parse_args()
    if args.source_probe:
        source_probe()
        return
    if args.worker:
        worker(args.worker)
        return
    base = args.output.resolve()
    base.mkdir(parents=True, exist_ok=False)
    image_id = subprocess.check_output(["sudo", "-n", "docker", "image", "inspect", IMAGE, "--format", "{{.Id}}"], text=True).strip()
    if image_id != ALLOWED_IMAGE:
        raise RuntimeError(f"Expected pinned R2 {ALLOWED_IMAGE}, got {image_id}")
    source_output = subprocess.check_output([
        "sudo", "-n", "docker", "run", "--rm", "--pull", "never", "--network", "none", "--read-only", "--tmpfs", "/tmp:rw,size=128m",
        "-e", "CUDA_VISIBLE_DEVICES=", "-e", "PYTHONDONTWRITEBYTECODE=1", "-v", f"{SCRIPT}:/audit.py:ro",
        "--entrypoint", "/usr/bin/python3", image_id, "/audit.py", "--source-probe"], text=True)
    source = json.loads(source_output)
    dump(base / "source.json", source)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        baseline = pool.submit(launch_case, base, "baseline", ["--master-addr", "127.0.0.1"], 22, False)
        fixed = pool.submit(launch_case, base, "explicit-static-loopback", FIXED_FLAGS, 35, True)
        cases = [baseline.result(), fixed.result()]
    fixed_result = cases[1]
    baseline_result = cases[0]
    fixed_pass = (fixed_result["exit_code"] == 0 and not fixed_result["timed_out"]
                  and len(fixed_result["workers"]) == 2 and all(
                      x["status"] == "PASS" and x["master_addr"] == "127.0.0.1"
                      and x["master_port"] == "29500" and x["all_reduce_sum"] == 3.0
                      and x["backend"] == "gloo" and x["gloo_socket_ifname"] == "lo"
                      and [y[1] for y in x["interfaces"]] == ["lo"]
                      and not x["cuda_initialized_before"] and not x["cuda_initialized_after"]
                      for x in fixed_result["workers"]))
    baseline_log = Path(baseline_result["log"]).read_text()
    # In 2.13 this failure occurs in agent rendezvous before workers are spawned.
    reproduced = (baseline_result["exit_code"] != 0 and baseline_result["hostname_resolution_warning"]
                  and baseline_result["container"] is not None
                  and baseline_result["container"]["network_mode"] == "none"
                  and re.search(r"\(" + re.escape(baseline_result["container"]["hostname"])
                                + r", [0-9]+\).*gai error: -3", baseline_log) is not None
                  and len(baseline_result["workers"]) == 0)
    report = {"schema": 1, "suite": "torchrun-network-none-cpu-regression", "status": "PASS" if fixed_pass and reproduced else "FAIL",
              "finished_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "image": IMAGE, "image_id": image_id, "torch_version": source["torch_version"],
              "source_sha256": {str(SCRIPT): hashlib.sha256(SCRIPT.read_bytes()).hexdigest()},
              "scope": "Real pinned R2 Docker network-none launcher; two real CPU Gloo workers, no model weights or GPU kernels",
              "gpu_executed": False, "actual_model_inference": False, "nccl_tested": False,
              "network_mode": "none", "baseline_reproduced": reproduced, "fixed_cpu_gloo_pass": fixed_pass,
              "cases": cases,
              "conclusion": "Torch 2.13 changes omitted-master-port single-node launches from static to c10d; dynamic bootstrap uses unresolved container hostname. Explicit static numeric master port and local address avoid that path; Gloo is pinned to loopback."}
    dump(base / "verification.json", report)
    print(json.dumps({k: report[k] for k in ["status", "image_id", "torch_version", "baseline_reproduced", "fixed_cpu_gloo_pass", "gpu_executed", "actual_model_inference"]}, indent=2))
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
