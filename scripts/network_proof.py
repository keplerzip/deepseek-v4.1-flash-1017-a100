"""Prove this process's network namespace has only lo and no usable default route.

Interface identity comes from socket.if_nameindex(), not sysfs directory names.
Sysfs includes control files such as bonding_masters and is diagnostic only.
The host launcher must separately verify Docker inspect NetworkMode == none.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
import re
import socket


RTF_UP = 0x1
RTF_REJECT = 0x200


def _hex(value, width=None):
    if not re.fullmatch(r"[0-9A-Fa-f]+", value) or (width is not None and len(value) != width):
        raise ValueError("Malformed hexadecimal route field: " + repr(value))
    return int(value, 16)


def parse_ipv4_routes(text):
    lines = text.splitlines()
    if not lines or lines[0].split()[:4] != ["Iface", "Destination", "Gateway", "Flags"]:
        raise ValueError("IPv4 route header absent or malformed")
    header = lines[0].split()
    if "Mask" not in header:
        raise ValueError("IPv4 route mask column absent")
    rows = []
    for line in lines[1:]:
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != len(header):
            raise ValueError("Malformed IPv4 route row: " + repr(line))
        row = dict(zip(header, fields))
        destination = _hex(row["Destination"], 8)
        mask = _hex(row["Mask"], 8)
        _hex(row["Gateway"], 8)
        flags = _hex(row["Flags"])
        is_default = destination == 0 and mask == 0
        rows.append({"interface": row["Iface"], "destination_hex": row["Destination"],
                     "gateway_hex": row["Gateway"], "mask_hex": row["Mask"], "flags_hex": row["Flags"],
                     "is_default": is_default,
                     "valid_default": is_default and bool(flags & RTF_UP) and not bool(flags & RTF_REJECT)})
    return rows


def parse_ipv6_routes(text):
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 10:
            raise ValueError("Malformed IPv6 route row: " + repr(line))
        destination, prefix, source, source_prefix, gateway, metric, refcnt, use, flag_text, interface = fields
        dest = _hex(destination, 32)
        prefix_value = _hex(prefix, 2)
        _hex(source, 32)
        source_prefix_value = _hex(source_prefix, 2)
        _hex(gateway, 32)
        for value in (metric, refcnt, use):
            _hex(value)
        flags = _hex(flag_text)
        if prefix_value > 128 or source_prefix_value > 128:
            raise ValueError("IPv6 route prefix exceeds 128")
        is_default = dest == 0 and prefix_value == 0
        rows.append({"interface": interface, "destination_hex": destination,
                     "prefix_hex": prefix, "gateway_hex": gateway, "flags_hex": flag_text,
                     "is_default": is_default,
                     "valid_default": is_default and bool(flags & RTF_UP) and not bool(flags & RTF_REJECT)})
    return rows


def _sysfs_diagnostic(interfaces):
    diagnostic = {"source": "/sys/class/net", "used_for_isolation_decision": False, "entries": []}
    try:
        for path in sorted(Path("/sys/class/net").iterdir()):
            row = {"name": path.name, "is_directory": path.is_dir(),
                   "in_socket_if_nameindex": path.name in interfaces}
            try:
                row["ifindex"] = int((path / "ifindex").read_text().strip())
            except (OSError, ValueError) as exc:
                row["ifindex_error"] = type(exc).__name__ + ": " + str(exc)
            diagnostic["entries"].append(row)
    except OSError as exc:
        diagnostic["error"] = type(exc).__name__ + ": " + str(exc)
    return diagnostic


def collect_network_evidence():
    report = {"schema": 2, "status": "FAIL", "interfaces": [], "if_nameindex": [],
              "interface_source": "socket.if_nameindex() in this process network namespace",
              "ipv4_route_source": "/proc/self/net/route", "ipv6_route_source": "/proc/self/net/ipv6_route",
              "ipv4_routes": [], "ipv6_routes": [], "observation_errors": [], "failures": [],
              "docker_network_mode": "CHECK_SEPARATELY_WITH_DOCKER_INSPECT",
              "scope": "Actual process netns only lo and no valid IPv4/IPv6 default route; UDS frontend unchanged"}
    try:
        report["netns_before"] = os.readlink("/proc/self/ns/net")
    except OSError as exc:
        report["observation_errors"].append("netns_before: " + str(exc))
    try:
        observed = socket.if_nameindex()
        if not all(isinstance(index, int) and index > 0 and isinstance(name, str) and name
                   for index, name in observed):
            raise ValueError("Invalid interface identity from if_nameindex")
        if len({index for index, _ in observed}) != len(observed) or len({name for _, name in observed}) != len(observed):
            raise ValueError("Duplicate interface identities from if_nameindex")
        report["if_nameindex"] = [[index, name] for index, name in observed]
        report["interfaces"] = sorted(name for _, name in observed)
    except (OSError, ValueError, TypeError) as exc:
        report["observation_errors"].append("if_nameindex: " + str(exc))
    for family, path, parser in (
        ("ipv4", "/proc/self/net/route", parse_ipv4_routes),
        ("ipv6", "/proc/self/net/ipv6_route", parse_ipv6_routes),
    ):
        try:
            raw = Path(path).read_text()
            report[family + "_routes_raw"] = raw
            report[family + "_routes"] = parser(raw)
            report[family + "_route_available"] = True
        except FileNotFoundError as exc:
            report[family + "_route_available"] = False
            if family == "ipv4":
                report["observation_errors"].append(family + ": " + str(exc))
            else:
                report["ipv6_unavailable_reason"] = "Kernel IPv6 route table absent: " + str(exc)
        except (OSError, ValueError) as exc:
            report["observation_errors"].append(family + ": " + str(exc))
    try:
        report["netns_after"] = os.readlink("/proc/self/ns/net")
    except OSError as exc:
        report["observation_errors"].append("netns_after: " + str(exc))
    report["sysfs_diagnostic"] = _sysfs_diagnostic(report["interfaces"])
    if report["observation_errors"]:
        report["failures"].append("Network evidence could not be collected/parsed completely")
    if report["interfaces"] != ["lo"]:
        report["failures"].append("Actual network namespace must contain exactly lo")
    if report.get("netns_before") != report.get("netns_after"):
        report["failures"].append("Network namespace changed while collecting evidence")
    routes = report["ipv4_routes"] + report["ipv6_routes"]
    report["default_route"] = any(row["valid_default"] for row in routes)
    if report["default_route"]:
        report["failures"].append("A valid default route exists in this process network namespace")
    if any(row["interface"] not in report["interfaces"] for row in routes):
        report["failures"].append("Route table interface absent from namespace interface snapshot")
    report["routes"] = report.get("ipv4_routes_raw", "")
    report["status"] = "PASS" if not report["failures"] else "FAIL"
    report["observed_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    return report


class NetworkIsolationError(RuntimeError):
    def __init__(self, evidence):
        self.evidence = evidence
        super().__init__("Network isolation failed: " + json.dumps(evidence, ensure_ascii=False, sort_keys=True))


def require_network_none(report=None):
    evidence = collect_network_evidence() if report is None else report
    if evidence["status"] != "PASS":
        raise NetworkIsolationError(evidence)
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--docker-network-mode", help="Actual NetworkMode read by host Docker inspect, if provided")
    args = parser.parse_args()
    report = collect_network_evidence()
    if args.docker_network_mode is not None:
        report["docker_network_mode"] = args.docker_network_mode
        if args.docker_network_mode != "none":
            report["failures"].append("Host Docker inspect NetworkMode must equal none")
            report["status"] = "FAIL"
    serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(serialized, encoding="utf-8")
        temporary.replace(args.output)
    print(serialized, end="", flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
