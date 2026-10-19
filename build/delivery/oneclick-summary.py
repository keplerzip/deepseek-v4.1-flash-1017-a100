#!/usr/bin/env python3
"""Summarize one explicit offline deployment run; standard library only.

Reads metadata and inference reports, never model tensor payloads. PASS covers
the declared tests; 32 simultaneous full-window KV residency stays NOT_PROVEN.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import statistics
import sys
import datetime

PROJECT = "deepseek-v4.1-flash-1017-a100"
WEIGHT_MODEL = "DeepSeek-V4.1-Flash-1017"
API_MODEL = "DeepSeek-Flash"
STAGES = ("integrity", "extract", "package", "image", "stop", "profile", "deploy", "config",
          "smoke", "benchmark", "diagnosis", "long-context")
SUITES = ("smoke", "benchmark", "diagnosis", "long-context")
SMOKE_NAMES = ("model-identity", "text-SSE-off", "text-SSE-high", "single-image-OCR",
               "multi-image-OCR", "reverse-tool-result-image-association", "prefix-cache-cold-hot-details",
               "C32-auto-tool-calls", "DSpark-native-k5-counters")
BENCH_FIELDS = ("single_decode_tokens_per_second_median", "single_e2e_tokens_per_second_median",
                "single_ttft_seconds_median", "single_e2e_seconds_median",
                "c32_aggregate_tokens_per_second", "c32_latency_p50_seconds",
                "c32_latency_p95_seconds", "c32_ttft_p95_seconds")


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def number(value, minimum=0):
    return type(value) in (int, float) and math.isfinite(value) and value >= minimum


def exact(actual, expected):
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(exact(actual[k], v) for k, v in expected.items())
    return actual == expected


def issue(result, severity, message):
    result.setdefault("issues", []).append({"severity": severity, "message": message})


def read_json(path):
    raw = path.read_bytes()
    value = parse_json(raw)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object: " + str(path))
    return value, hashlib.sha256(raw).hexdigest(), len(raw)


def parse_json(raw):
    def reject_constant(value):
        raise ValueError("Non-finite JSON number: " + value)
    return json.loads(raw, parse_constant=reject_constant)


def valid_accounting(audit):
    if not isinstance(audit, dict) or audit.get("issues") != []:
        return False
    expected, observed = audit.get("expected_generation_tokens"), audit.get("observed_generation_tokens")
    return number(expected, 1) and number(observed, 1) and expected == observed


def check_config(base, model_directory=None):
    project = base / PROJECT
    model_directory = (Path(model_directory) if model_directory is not None else base / "models" / WEIGHT_MODEL).resolve()
    result = {"status": "FAIL", "scope": "Actual configuration and original model metadata only; no GPU inference",
              "gpu_executed": False, "model_directory": str(model_directory), "issues": [], "evidence": {}}
    paths = {"active": project / "runtime/runtime.active.json", "profile": project / "runtime/speed-profile.json",
             "resolved": project / "runtime/runtime.resolved.json",
             "model": model_directory / "config.json"}
    values = {}
    for key, path in paths.items():
        try:
            values[key], digest, size = read_json(path)
            result["evidence"][key] = {"path": str(path), "sha256": digest, "bytes": size}
        except Exception as error:
            issue(result, "FAIL" if key == "model" else "INCOMPLETE", "无法读取实际配置 %s: %s" % (key, error))
    active = values.get("active")
    if active is not None:
        expected = {"model": "/models/" + WEIGHT_MODEL, "served_model_name": API_MODEL,
            "tensor_parallel_size": 8, "data_parallel_size": 1, "enable_expert_parallel": True,
            "pipeline_parallel_size": 1, "max_num_seqs": 32, "max_model_len": 524288,
            "max_num_batched_tokens": 8192, "limit_mm_per_prompt": {"image": 999},
            "disable_chunked_mm_input": True, "enable_chunked_prefill": True,
            "enable_prefix_caching": True, "kv_cache_dtype": "fp8_ds_mla",
            "engram_config": {"cpu_offload": True}}
        for key, wanted in expected.items():
            if not exact(active.get(key), wanted):
                issue(result, "FAIL", "实际运行配置不匹配: " + key)
        for key in ("hf_overrides", "mm_processor_kwargs"):
            if key in active:
                issue(result, "FAIL", "实际配置包含原生视觉覆盖项: " + key)
        if active.get("language_model_only", False) is not False:
            issue(result, "FAIL", "实际配置启用了纯语言模型模式")
        spec = active.get("speculative_config") or {}
        if not isinstance(spec, dict) or spec.get("method") != "dspark" or not exact(spec.get("num_speculative_tokens"), 5):
            issue(result, "FAIL", "实际投机解码不是原生 DSpark k5")
        graph = active.get("compilation_config") or {}
        if not isinstance(graph, dict) or graph.get("cudagraph_mode") != "FULL_DECODE_ONLY":
            issue(result, "FAIL", "实际 decode CUDA Graph 配置不匹配")
        result["configuration"] = active
    profile = values.get("profile")
    if profile is not None:
        if profile.get("name") != "baseline" or active is None or profile.get("configuration") != active:
            issue(result, "FAIL", "实际 profile 不是与运行配置一致的 baseline")
        if not exact(profile.get("global_concurrency"), 32) or not exact(profile.get("max_model_len"), 524288):
            issue(result, "FAIL", "profile 的全局并发或上下文不匹配")
    if "resolved" in values and active is not None and values["resolved"] != active:
        issue(result, "FAIL", "实际进程解析配置与 runtime.active.json 不一致")
    model = values.get("model")
    if model is not None:
        vision, text = model.get("vision_config") or {}, model.get("text_config") or {}
        if not isinstance(vision, dict) or not exact(vision.get("max_image_tokens"), 1024):
            issue(result, "FAIL", "实际模型原生视觉预算不是 1024 token")
        if not isinstance(text, dict) or not exact(text.get("dspark_block_size"), 5):
            issue(result, "FAIL", "实际模型 DSpark block size 不是 5")
        maximum = text.get("max_position_embeddings") if isinstance(text, dict) else None
        if type(maximum) is not int or maximum < 524288:
            issue(result, "FAIL", "实际模型原生位置上限不能支持 524288")
        result["native_vision_tokens"] = vision.get("max_image_tokens") if isinstance(vision, dict) else None
    current = project / "runtime/current-run"
    if current.is_file():
        result["observed_launch_id"] = current.read_text().strip()
    result["status"] = "PASS" if not result["issues"] else "FAIL"
    return result


def stage_records(run, result):
    rows = []
    try:
        lines = (run / "steps.tsv").read_text().splitlines()
    except Exception as error:
        issue(result, "INCOMPLETE", "无法读取阶段记录: " + str(error))
        lines = []
    seen = set()
    for index, line in enumerate(lines, 1):
        parts = line.split("\t")
        if len(parts) != 5:
            issue(result, "INCOMPLETE", "steps.tsv 第 %d 行不完整" % index)
            continue
        name, started, finished, code, log = parts
        try:
            code = int(code)
        except ValueError:
            issue(result, "INCOMPLETE", "阶段退出码无效: " + name)
            code = None
        if name in seen:
            issue(result, "FAIL", "阶段记录重复，不能忽略前次失败: " + name)
        seen.add(name)
        if name not in STAGES:
            issue(result, "FAIL", "未知阶段: " + name)
        if not started or not finished:
            issue(result, "INCOMPLETE", "阶段时间记录不完整: " + name)
        if PurePosixPath(log).name != log or not log:
            issue(result, "FAIL", "阶段日志不是文件名: " + name)
        if code is not None and code != 0:
            issue(result, "FAIL", "阶段 %s 退出码为 %s" % (name, code))
        rows.append({"id": name, "started_at": started, "finished_at": finished, "exit_code": code, "log": log})
    for name in STAGES:
        if name not in seen:
            issue(result, "INCOMPLETE", "阶段未完成: " + name)
    return rows


def copy_report(base, run, context, suite, result):
    path = (context.get("report_paths") or {}).get(suite)
    record = {"status": "MISSING", "source_relative_path": path, "evidence_relative_path": "reports/" + suite + ".json"}
    result["reports"][suite] = record
    if not isinstance(path, str) or not path:
        issue(result, "INCOMPLETE", "context 未指定报告: " + suite)
        return None
    relative = PurePosixPath(path)
    project = (base / PROJECT).resolve()
    if relative.is_absolute() or ".." in relative.parts:
        issue(result, "FAIL", "报告路径越界: " + suite)
        return None
    source = project / path
    if source.is_symlink() or not source.resolve().is_relative_to(project):
        issue(result, "FAIL", "报告路径指向工程外: " + suite)
        return None
    try:
        raw = source.read_bytes()
        destination = run / record["evidence_relative_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
        record.update(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
        report = parse_json(raw)
        if not isinstance(report, dict):
            raise ValueError("Report is not an object")
    except Exception as error:
        issue(result, "INCOMPLETE", "无法读取/解析报告 %s: %s" % (suite, error))
        return None
    record["status"] = report.get("status", "UNKNOWN")
    if report.get("status") != "PASS":
        severity = "INCOMPLETE" if report.get("status") in (None, "RUNNING") else "FAIL"
        issue(result, severity, "%s 报告状态为 %s" % (suite, report.get("status")))
    if report.get("contamination"):
        issue(result, "FAIL", "%s 报告存在计数污染: %s" % (suite, report["contamination"]))
    if report.get("model") != API_MODEL or not exact(report.get("schema"), 1):
        issue(result, "FAIL", "%s 报告模型名或 schema 不匹配" % suite)
    if not report.get("finished_at"):
        issue(result, "INCOMPLETE", suite + " 报告尚未完成")
    if suite != "diagnosis":
        if report.get("suite") != suite or not exact(report.get("context"), 524288) or not exact(report.get("concurrency"), 32):
            issue(result, "FAIL", suite + " 报告测试范围不匹配")
        audit = report.get("global_accounting")
        if not isinstance(audit, dict):
            issue(result, "INCOMPLETE", suite + " 缺少全局计数核账")
        elif not valid_accounting(audit):
            issue(result, "FAIL", suite + " 全局 generation token 核账失败")
    return report


def test_by_name(report, name, result):
    matches = [row for row in report.get("tests", []) if isinstance(row, dict) and row.get("name") == name]
    if len(matches) != 1:
        issue(result, "INCOMPLETE", "缺少或重复验收项: " + name)
        return {"name": name, "status": "MISSING"}
    row = matches[0]
    if row.get("status") != "PASS":
        issue(result, "FAIL", "验收项失败 %s: %s" % (name, row.get("error", row.get("status"))))
    return row


def extract_benchmark(report, result):
    parameters = result.get("parameters") or {}
    for mode in ("off", "high"):
        test = test_by_name(report, "benchmark-" + mode, result)
        source = (report.get("benchmarks") or {}).get(mode)
        if not isinstance(source, dict):
            issue(result, "INCOMPLETE", "缺少 benchmark 汇总: " + mode)
            result["benchmarks"][mode] = {"status": "MISSING", "signoff": False}
            continue
        row = {key: source.get(key) for key in BENCH_FIELDS}
        row.update(mode=mode, status=test.get("status"), signoff=False)
        result["benchmarks"][mode] = row
        local_issues = []
        for key in BENCH_FIELDS:
            if not number(source.get(key)):
                local_issues.append("缺少有效指标 " + key)
        samples = source.get("samples") or []
        phases = source.get("phases") or {}
        counts = {}
        for phase, concurrency in (("warmup", 1), ("single", 1), ("c32", 32)):
            wanted = phases.get(phase) or {}
            selected = [r for r in samples if r.get("phase") == phase]
            good = [r for r in selected if r.get("status") == "PASS"]
            counts[phase] = {"requested": wanted.get("requests"), "received": len(selected), "successful": len(good)}
            minimum = 32 if phase == "c32" else 1
            requested = parameters.get({"warmup": "warmups", "single": "single_samples", "c32": "load_requests"}[phase])
            if (type(wanted.get("requests")) is not int or wanted["requests"] < minimum
                    or not exact(wanted.get("concurrency"), concurrency) or wanted["requests"] != requested):
                local_issues.append("阶段样本/并发配置异常: " + phase)
            if len(good) != wanted.get("requests") or len(selected) != len(good):
                local_issues.append("阶段包含失败或缺失样本: " + phase)
            if any(r.get("mode") != mode or type(r.get("completion_tokens")) is not int or r["completion_tokens"] < 128
                   or r["completion_tokens"] != parameters.get("output_tokens") for r in good):
                local_issues.append("阶段实际模式/token 异常: " + phase)
        audit = source.get("accounting") or {}
        if not valid_accounting(audit):
            local_issues.append("该模式计数核账缺失或污染")
        dspark = audit.get("dspark") or {}
        drafts, proposed, accepted = (dspark.get(key) for key in ("drafts", "draft_tokens", "accepted_tokens"))
        if not (number(drafts, 1) and number(proposed, 1) and number(accepted)
                and accepted <= proposed <= 5 * drafts):
            local_issues.append("该模式 DSpark k5 计数缺失或不一致")
        row.update(sample_counts=counts, dspark_counters_all_phases=dspark,
                   dspark_acceptance_rate_all_phases=accepted / proposed if number(proposed) and proposed > 0 and number(accepted) else None,
                   dspark_scope="Includes warmup, single and C32 phases; not C1-only acceptance", issues=local_issues)
        row["signoff"] = report.get("status") == "PASS" and not report.get("contamination") and test.get("status") == "PASS" and not local_issues
        for message in local_issues:
            issue(result, "FAIL", "benchmark %s: %s" % (mode, message))


def extract_diagnosis(report, result):
    parameters = result.get("parameters") or {}
    expected = report.get("samples_per_case")
    if type(expected) is not int or expected < 1:
        issue(result, "INCOMPLETE", "diagnosis 缺少正式样本数")
    if (expected != parameters.get("diagnosis_samples") or report.get("output_tokens") != parameters.get("output_tokens")
            or report.get("warmups_per_case") != parameters.get("warmups")):
        issue(result, "FAIL", "diagnosis 样本/输出/预热参数与此次一键请求不同")
    workloads = report.get("workloads") or []
    if not {"text", "code"}.issubset(set(workloads)) or not {"off", "high"}.issubset(set(report.get("modes") or [])):
        issue(result, "FAIL", "diagnosis 未覆盖文本/代码及 off/high")
    if report.get("expect_dspark") is not True or not exact(report.get("expected_draft_k"), 5):
        issue(result, "FAIL", "diagnosis 不是 DSpark k5 测量")
    samples = report.get("samples") or []
    for workload in workloads:
        for mode in ("off", "high"):
            key = workload + "/" + mode
            selected = [s for s in samples if s.get("phase") == "sample" and s.get("workload") == workload and s.get("mode") == mode]
            clean = [s for s in selected if s.get("status") == "PASS" and s.get("timing_valid") is True
                     and (s.get("accounting") or {}).get("status") == "CLEAN" and not s["accounting"].get("issues")]
            source = (report.get("summary") or {}).get(key) or {}
            valid = len(clean) == expected and len(selected) == expected and source.get("measurement_status") == "CLEAN"
            totals = {name: 0.0 for name in ("drafts", "draft_tokens", "accepted_tokens")}
            valid_counters = 0
            for sample in clean:
                audit, timing = sample["accounting"], sample.get("timing") or {}
                if not audit.get("activity_observed") or any(not number(audit.get(name)) for name in totals):
                    valid = False
                    continue
                if not (audit["drafts"] > 0 and 0 <= audit["accepted_tokens"] <= audit["draft_tokens"] <= 5 * audit["drafts"] and audit["draft_tokens"] > 0):
                    valid = False
                    continue
                if any(not number(timing.get(name)) for name in ("decode_tokens_per_second", "e2e_tokens_per_second", "ttft_seconds")):
                    valid = False
                    continue
                completion = timing.get("completion_tokens")
                if (type(completion) is not int or completion != parameters.get("output_tokens")
                        or audit.get("generation_tokens_expected") != completion
                        or audit.get("generation_tokens_observed") != completion):
                    valid = False
                    continue
                for name in totals:
                    totals[name] += audit[name]
                valid_counters += 1
            def median(name):
                values = [(s.get("timing") or {}).get(name) for s in clean]
                return statistics.median(values) if values and all(number(value) for value in values) else None
            result["diagnostics"][key] = {"status": "PASS" if valid and report.get("status") == "PASS" else "NOT_SIGNOFF",
                "sample_phase": "sample", "requested_samples": expected, "observed_samples": len(selected),
                "clean_samples": len(clean), "failed_or_contaminated_samples": len(selected) - len(clean),
                "decode_tps_median": median("decode_tokens_per_second"), "e2e_tps_median": median("e2e_tokens_per_second"),
                "ttft_seconds_median": median("ttft_seconds"), "dspark_counters": totals if valid_counters else None,
                "counter_samples": valid_counters,
                "weighted_acceptance_rate": totals["accepted_tokens"] / totals["draft_tokens"] if valid_counters and totals["draft_tokens"] > 0 else None,
                "accepted_per_draft": totals["accepted_tokens"] / totals["drafts"] if valid_counters and totals["drafts"] > 0 else None,
                "acceptance_scope": "Sum accepted / sum proposed across clean non-warmup C1 samples only",
                "speculation_speedup_proven": False}
            if not valid:
                issue(result, "FAIL", "diagnosis 正式样本缺失/失败/污染: " + key)


def extract_smoke(report, result):
    for name in SMOKE_NAMES:
        test = test_by_name(report, name, result)
        row = {"status": test.get("status"), "error": test.get("error"), "details": test.get("details")}
        result["functional_tests"][name] = row
    identity = (result["functional_tests"]["model-identity"].get("details") or {}).get("ids")
    if identity != [API_MODEL]:
        issue(result, "FAIL", "模型列表未证明唯一 DeepSeek-Flash 身份")
    result["multimodal_scope"] = "Representative single-image/three-image OCR and reverse-arrival image/tool association; not proof of every visual task's quality"


def extract_long_context(report, result):
    test = test_by_name(report, "independent-near-524288-token-retrieval", result)
    details = test.get("details") or {}
    keys = ("requested_context", "prompt_token_target", "actual_prompt_tokens", "prompt_tokens", "output_budget",
            "input_plus_output_budget", "retrieval_positions", "ttft_seconds", "e2e_seconds", "completion_tokens")
    row = {key: details.get(key) for key in keys}
    normalize = lambda text: re.sub(r"\s+", "", text).strip("`\"'") if isinstance(text, str) else None
    actual, expected = normalize(details.get("actual")), normalize(details.get("expected"))
    row.update(status=test.get("status"), retrieval_passed=bool(expected and actual == expected),
               scope="Single near-full context retrieval; output_budget is not actual output tokens")
    result["long_context"] = row
    prompt = row["actual_prompt_tokens"]
    good = (test.get("status") == "PASS" and exact(row["requested_context"], 524288)
            and exact(row["prompt_token_target"], 523776) and type(prompt) is int and 522752 <= prompt <= 523776
            and row["prompt_tokens"] == prompt and exact(row["output_budget"], 512)
            and row["input_plus_output_budget"] == prompt + 512
            and row["retrieval_positions"] == [0, .25, .5, .75, 1] and row["retrieval_passed"])
    if not good:
        issue(result, "FAIL", "512K 单请求检索证据不完整或不匹配")


def build_summary(base, run, exit_code, interrupted, model_directory=None):
    result = {"schema": 1, "status": "RUNNING", "generated_at": now(), "stage_exit_code": exit_code, "exit_code": exit_code,
              "interrupted": interrupted, "issues": [], "reports": {}, "benchmarks": {}, "diagnostics": {},
              "functional_tests": {}, "long_context": {}, "full_window_C32_residency": "NOT_PROVEN",
              "acceptance_scope": "Deployment, C1/short-input C32, one near-512K request and representative text/image/tool checks; not 32 full windows simultaneously resident"}
    try:
        context, _, _ = read_json(run / "context.json")
    except Exception as error:
        issue(result, "INCOMPLETE", "无法读取 context.json: " + str(error))
        context = {}
    for key in ("run_id", "started_at", "requested_config", "parameters", "launch_id"):
        result[key] = context.get(key)
    if not isinstance(context.get("run_id"), str) or not context["run_id"]:
        issue(result, "INCOMPLETE", "缺少此次一键运行 ID")
    parameters = context.get("parameters")
    if not isinstance(parameters, dict):
        issue(result, "INCOMPLETE", "缺少此次一键测速参数")
    else:
        for name, minimum in (("warmups", 1), ("single_samples", 1), ("diagnosis_samples", 1),
                              ("load_requests", 32), ("output_tokens", 128)):
            if type(parameters.get(name)) is not int or parameters[name] < minimum:
                issue(result, "FAIL", "此次一键测速参数无效: " + name)
    result["steps"] = stage_records(run, result)
    if exit_code != 0:
        issue(result, "FAIL", "一键流程退出码: " + str(exit_code))
    if interrupted:
        issue(result, "FAIL", "流程被信号中断: " + interrupted)
    result["configuration_check"] = check_config(base, model_directory)
    for item in result["configuration_check"]["issues"]:
        issue(result, item["severity"], item["message"])
    launch = result["configuration_check"].get("observed_launch_id")
    recorded_launch = context.get("launch_id")
    launch_path = run / "launch-id.txt"
    if launch_path.exists():
        try:
            recorded_launch = launch_path.read_text().strip()
            result["launch_id"] = recorded_launch
            if context.get("launch_id") and context["launch_id"] != recorded_launch:
                issue(result, "FAIL", "context 与 launch-id.txt 的启动 ID 不同")
        except Exception as error:
            issue(result, "INCOMPLETE", "无法读取此次引擎启动 ID: " + str(error))
    if not launch or not recorded_launch:
        issue(result, "INCOMPLETE", "缺少当前引擎与此次一键运行的启动 ID 绑定证据")
    elif recorded_launch != launch:
        issue(result, "FAIL", "当前引擎启动 ID 与此次一键记录不同")
    image_path = run / "image-id.txt"
    if image_path.exists():
        result["loaded_image_identity"] = image_path.read_text().strip()
    model_source_path = run / "model-source.txt"
    if model_source_path.exists():
        result["model_source_directory"] = model_source_path.read_text().strip()
    for suite in SUITES:
        report = copy_report(base, run, context, suite, result)
        if report is None:
            continue
        try:
            {"smoke": extract_smoke, "benchmark": extract_benchmark, "diagnosis": extract_diagnosis,
             "long-context": extract_long_context}[suite](report, result)
        except Exception as error:
            issue(result, "FAIL", "报告 %s 结构异常: %s" % (suite, error))
    result["status"] = ("FAIL" if any(i["severity"] == "FAIL" for i in result["issues"])
                        else "INCOMPLETE" if result["issues"] else "PASS")
    result["exit_code"] = exit_code or (0 if result["status"] == "PASS" else 1)
    for row in result["benchmarks"].values():
        row["signoff"] = row["signoff"] and result["status"] == "PASS"
    for row in result["diagnostics"].values():
        row["signoff"] = row["status"] == "PASS" and result["status"] == "PASS"
    return result


def format_number(value, percent=False):
    if not number(value):
        return "未取得"
    return ("%.2f%%" % (100 * value)) if percent else ("%.3f" % value)


def render(result):
    lines = ["DeepSeek-Flash 一键部署与测速汇总", "状态: " + result.get("status", "FAIL"),
        "运行ID: " + str(result.get("run_id") or "未取得"),
        "要求配置: TP8/DP1/EP8/PP1，C32，524288，8192调度预算，原生图文，DSpark k5。",
        "实际配置检查: " + result.get("configuration_check", {}).get("status", "未取得"),
        "模型来源目录: " + str(result.get("model_source_directory") or result.get("configuration_check", {}).get("model_directory") or "未取得"),
        "32条满窗口同时驻留: NOT_PROVEN（不属于已通过声明）。", "",
        "模式 | 单发decode tok/s | 单发端到端tok/s | 单发TTFT秒 | C32总tok/s | C32 P95秒 | C32 TTFT P95秒 | 签收",
        "--- | --- | --- | --- | --- | --- | --- | ---"]
    for mode in ("off", "high"):
        row = result.get("benchmarks", {}).get(mode, {})
        values = [row.get(k) for k in ("single_decode_tokens_per_second_median", "single_e2e_tokens_per_second_median",
            "single_ttft_seconds_median", "c32_aggregate_tokens_per_second", "c32_latency_p95_seconds", "c32_ttft_p95_seconds")]
        lines.append(" | ".join([mode] + [format_number(v) for v in values] + ["通过" if row.get("signoff") else "不可签收"]))
    lines.extend(["", "C1诊断分组 | 清洁/实际样本 | decode tok/s | TTFT秒 | DSpark接受率 | 每轮接受token | 签收",
                  "--- | --- | --- | --- | --- | --- | ---"])
    for key, row in sorted(result.get("diagnostics", {}).items()):
        lines.append(" | ".join([key, "%s/%s" % (row.get("clean_samples"), row.get("observed_samples")),
            format_number(row.get("decode_tps_median")), format_number(row.get("ttft_seconds_median")),
            format_number(row.get("weighted_acceptance_rate"), True), format_number(row.get("accepted_per_draft")),
            "PASS" if row.get("signoff") else "NOT_SIGNOFF"]))
    lines.extend(["接受率按正式清洁样本 Σaccepted/Σproposed 加权；高接受率不证明净提速。",
                  "off=关闭思考；high=高强度思考。TPS采用短输入、固定输出；512K检索独立验收，不混入TPS。",
                  "TPS包含思考token；TTFT与持续decode分开，C32合计不能当单发速度。", "", "功能验收:"])
    for name in SMOKE_NAMES:
        lines.append("  %s: %s" % (name, result.get("functional_tests", {}).get(name, {}).get("status", "未执行")))
    long = result.get("long_context", {})
    lines.extend(["", "单条512K检索: %s；实际输入 %s token；输出预算 %s；检索 %s；TTFT %s秒" % (
        long.get("status", "未执行"), long.get("actual_prompt_tokens", "未取得"), long.get("output_budget", "未取得"),
        "通过" if long.get("retrieval_passed") else "未通过或未执行", format_number(long.get("ttft_seconds"))),
        "图像验收限单图/三图OCR和图片工具关联；不代表全部视觉任务质量已证明。", "", "阶段:"])
    observed = {row["id"]: row for row in result.get("steps", [])}
    for name in STAGES:
        row = observed.get(name)
        lines.append("  %s: %s" % (name, "退出码 " + str(row.get("exit_code")) if row else "未执行"))
    if result.get("issues"):
        lines.extend(["", "未签收原因:"])
        lines.extend("  [%s] %s" % (item["severity"], item["message"]) for item in result["issues"][:50])
    lines.extend(["", "完整证据: summary.json 和 reports/*.json（各报告SHA256记录在summary.json）。"])
    return "\n".join(lines) + "\n"


def write_output(run, result):
    run.mkdir(parents=True, exist_ok=True)
    text = render(result)
    for name, body in (("summary.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n"), ("summary.txt", text)):
        path = run / name
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(body)
        temporary.replace(path)
    print(text, end="")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--model-directory", type=Path,
                        help="Read original config.json from this mounted model directory; defaults to BASE/models/" + WEIGHT_MODEL)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--exit-code", type=int, default=0)
    parser.add_argument("--interrupted")
    parser.add_argument("--check-config", action="store_true")
    args = parser.parse_args(argv)
    if args.check_config:
        try:
            result = check_config(args.base.resolve(), args.model_directory)
        except Exception as error:
            result = {"status": "FAIL", "gpu_executed": False, "issues": [{"severity": "FAIL", "message": str(error)}]}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["status"] == "PASS" else 1
    if args.run is None:
        parser.error("--run is required unless --check-config is used")
    run = args.run.resolve()
    try:
        result = build_summary(args.base.resolve(), run, args.exit_code, args.interrupted, args.model_directory)
    except Exception as error:
        result = {"schema": 1, "status": "FAIL", "generated_at": now(), "exit_code": args.exit_code,
            "interrupted": args.interrupted, "full_window_C32_residency": "NOT_PROVEN",
            "issues": [{"severity": "FAIL", "message": "汇总器异常: " + type(error).__name__ + ": " + str(error)}]}
    write_output(run, result)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
