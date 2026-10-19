"""CPU-only fixtures and tests for the offline one-click report summarizer."""
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "build/delivery/oneclick-summary.py"
spec = importlib.util.spec_from_file_location("oneclick_summary", SCRIPT)
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)
STAMP = "2026-10-18T03:04:05+00:00"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def accounting(tokens=256):
    return {"expected_generation_tokens": tokens, "observed_generation_tokens": float(tokens), "issues": [],
            "counter_deltas": {"vllm:generation_tokens_total": float(tokens)},
            "dspark": {"drafts": 1000., "draft_tokens": 5000., "accepted_tokens": 2000.}}


def successful_reports():
    """Return independent, small reports shaped like the real acceptance tools."""
    def acceptance(suite):
        return {"schema": 1, "status": "PASS", "suite": suite, "model": summary.API_MODEL,
                "context": 524288, "concurrency": 32, "started_at": STAMP, "finished_at": STAMP,
                "contamination": [], "tests": [], "global_accounting": accounting()}
    smoke = acceptance("smoke")
    details = {
        "model-identity": {"ids": [summary.API_MODEL]},
        "text-SSE-off": {"answer": "437", "reasoning_tokens": 0},
        "text-SSE-high": {"answer": "437", "reasoning_tokens": 128},
        "single-image-OCR": {"images": 1, "expected": "739261", "actual": "739261"},
        "multi-image-OCR": {"images": 3, "expected": "739261,628150,517049", "actual": "739261,628150,517049"},
        "reverse-tool-result-image-association": {"expected": "A=394726;B=850163", "actual": "A=394726;B=850163"},
        "prefix-cache-cold-hot-details": {"hot_cache_hit_ratio": .8},
        "C32-auto-tool-calls": {"concurrency": 32, "successful_requests": 32},
        "DSpark-native-k5-counters": {"accounting": accounting(), "sample": {"completion_tokens": 256}},
    }
    smoke["tests"] = [{"name": name, "status": "PASS", "seconds": .1, "details": details[name]}
                      for name in summary.SMOKE_NAMES]
    benchmark = acceptance("benchmark")
    benchmark["benchmarks"] = {}
    for mode, rate in (("off", 60.), ("high", 45.)):
        samples, phases = [], {}
        for phase, count, concurrency in (("warmup", 1, 1), ("single", 2, 1), ("c32", 32, 32)):
            phases[phase] = {"requests": count, "concurrency": concurrency, "wall_seconds": 16.}
            for index in range(count):
                samples.append({"phase": phase, "mode": mode, "index": index, "status": "PASS",
                    "request_id": "%s-%s-%s" % (mode, phase, index), "prompt_tokens": 92,
                    "completion_tokens": 256, "reasoning_tokens": 0 if mode == "off" else 128,
                    "cached_tokens": 0, "decode_tokens_per_second": rate, "e2e_tokens_per_second": rate - 5,
                    "ttft_seconds": .5, "e2e_seconds": 5., "decode_seconds": 4., "decode_measured_tokens": 252,
                    "first_token_batch_size": 4, "token_bearing_events": 63, "finish_reason": "length"})
        benchmark["benchmarks"][mode] = {"mode": mode, "phases": phases, "samples": samples,
            "accounting": accounting(35 * 256), "single_decode_tokens_per_second_median": rate,
            "single_e2e_tokens_per_second_median": rate - 5, "single_ttft_seconds_median": .5,
            "single_e2e_seconds_median": 5., "c32_total_output_tokens": 8192,
            "c32_aggregate_tokens_per_second": 512., "c32_latency_p50_seconds": 10.,
            "c32_latency_p95_seconds": 14., "c32_ttft_p95_seconds": 3.}
        benchmark["tests"].append({"name": "benchmark-" + mode, "status": "PASS", "seconds": 48.,
                                   "details": {"mode": mode}})
    diagnosis = {"schema": 1, "status": "PASS", "model": summary.API_MODEL, "started_at": STAMP,
        "finished_at": STAMP, "workloads": ["text", "code"], "modes": ["off", "high"],
        "warmups_per_case": 1, "samples_per_case": 2, "output_tokens": 256,
        "expected_draft_k": 5, "expect_dspark": True, "samples": [], "summary": {}}
    for workload, base_rate in (("text", 60.), ("code", 80.)):
        for mode in ("off", "high"):
            key = workload + "/" + mode
            for phase, count in (("warmup", 1), ("sample", 2)):
                for index in range(count):
                    rate = 999. if phase == "warmup" else base_rate + index * 4 - (10 if mode == "high" else 0)
                    drafts, proposed, accepted = ((10., 50., 10.) if index == 0 else (20., 100., 80.))
                    diagnosis["samples"].append({"workload": workload, "mode": mode, "phase": phase,
                        "index": index, "status": "PASS", "timing_valid": True,
                        "timing": {"decode_tokens_per_second": rate, "e2e_tokens_per_second": rate - 5,
                                   "ttft_seconds": .25 + index / 2, "prompt_tokens": 92, "completion_tokens": 256},
                        "accounting": {"status": "CLEAN", "issues": [], "activity_observed": True,
                            "generation_tokens_expected": 256, "generation_tokens_observed": 256.,
                            "drafts": drafts, "draft_tokens": proposed, "accepted_tokens": accepted},
                        "graphs": {"status": "OBSERVED", "new_records": 0},
                        "analysis": {"speculation_speedup_proven": False}})
            diagnosis["summary"][key] = {"samples": 2, "failed": 0, "contaminated": 0,
                "decode_tps_median": 999., "measurement_status": "CLEAN"}
    long = acceptance("long-context")
    long["tests"] = [{"name": "independent-near-524288-token-retrieval", "status": "PASS", "seconds": 123.,
        "details": {"requested_context": 524288, "prompt_token_target": 523776, "actual_prompt_tokens": 523700,
            "prompt_tokens": 523700, "output_budget": 512, "input_plus_output_budget": 524212,
            "retrieval_positions": [0, .25, .5, .75, 1], "expected": "A,B,C,D,E", "actual": "A, B, C, D, E",
            "ttft_seconds": 120., "e2e_seconds": 123., "completion_tokens": 20}}]
    return {"smoke": smoke, "benchmark": benchmark, "diagnosis": diagnosis, "long-context": long}


def configure_fixture(base):
    """Write tiny actual-state metadata, without weights, Docker or GPU access."""
    base = Path(base)
    project = base / summary.PROJECT
    config = {"model": "/models/" + summary.WEIGHT_MODEL, "served_model_name": summary.API_MODEL,
        "tensor_parallel_size": 8, "data_parallel_size": 1, "enable_expert_parallel": True,
        "pipeline_parallel_size": 1, "max_num_seqs": 32, "max_model_len": 524288,
        "max_num_batched_tokens": 8192, "limit_mm_per_prompt": {"image": 999},
        "disable_chunked_mm_input": True, "enable_chunked_prefill": True, "enable_prefix_caching": True,
        "kv_cache_dtype": "fp8_ds_mla", "engram_config": {"cpu_offload": True},
        "speculative_config": {"method": "dspark", "num_speculative_tokens": 5},
        "compilation_config": {"cudagraph_mode": "FULL_DECODE_ONLY"}}
    for name in ("runtime.active.json", "runtime.resolved.json"):
        write_json(project / "runtime" / name, config)
    write_json(project / "runtime/speed-profile.json", {"name": "baseline", "configuration": copy.deepcopy(config),
        "global_concurrency": 32, "max_model_len": 524288})
    (project / "runtime/current-run").write_text("launch-fixture\n")
    write_json(base / "models" / summary.WEIGHT_MODEL / "config.json",
        {"vision_config": {"max_image_tokens": 1024},
         "text_config": {"dspark_block_size": 5, "max_position_embeddings": 1048576}})
    return config


def complete_fixture(base, run):
    """Write a full successful metadata-only run; return its report dictionaries."""
    base, run = Path(base), Path(run)
    configure_fixture(base)
    run.mkdir(parents=True, exist_ok=True)
    reports = successful_reports()
    context = {"run_id": "oneclick-fixture", "started_at": STAMP,
        "requested_config": {"model": summary.API_MODEL, "tp": 8, "dp": 1, "concurrency": 32, "context": 524288},
        "parameters": {"warmups": 1, "single_samples": 2, "diagnosis_samples": 2,
                       "load_requests": 32, "output_tokens": 256, "timeout": 3600},
        "report_paths": {suite: "runtime/oneclick-fixture-" + suite + "/report.json" for suite in summary.SUITES}}
    write_json(run / "context.json", context)
    (run / "steps.tsv").write_text("".join("%s\t%s\t%s\t0\t%s.log\n" % (name, STAMP, STAMP, name) for name in summary.STAGES))
    (run / "launch-id.txt").write_text("launch-fixture\n")
    (run / "image-id.txt").write_text("sha256:fixture-pinned-image\n")
    for suite, report in reports.items():
        write_json(base / summary.PROJECT / context["report_paths"][suite], report)
    return reports


class OneclickSummaryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.base = Path(self.directory.name) / "delivery"
        self.run = self.base / "results/run"
        self.reports = complete_fixture(self.base, self.run)

    def report_path(self, suite):
        context = json.loads((self.run / "context.json").read_text())
        return self.base / summary.PROJECT / context["report_paths"][suite]

    def replace_report(self, suite):
        write_json(self.report_path(suite), self.reports[suite])

    def build(self, code=0, signal=None):
        return summary.build_summary(self.base, self.run, code, signal)

    def test_complete_run_preserves_evidence_and_measurement_scope(self):
        result = self.build()
        self.assertEqual(result["issues"], [])
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(result["full_window_C32_residency"], "NOT_PROVEN")
        self.assertEqual(result["launch_id"], "launch-fixture")
        self.assertEqual(result["configuration_check"]["gpu_executed"], False)
        self.assertEqual(result["benchmarks"]["off"]["single_decode_tokens_per_second_median"], 60)
        self.assertTrue(all(row["signoff"] for row in result["benchmarks"].values()))
        row = result["diagnostics"]["text/off"]
        self.assertEqual(row["decode_tps_median"], 62.)
        self.assertEqual(row["weighted_acceptance_rate"], .6)
        self.assertEqual(row["accepted_per_draft"], 3.)
        self.assertTrue(row["signoff"])
        self.assertFalse(row["speculation_speedup_proven"])
        self.assertTrue(result["long_context"]["retrieval_passed"])
        for suite in summary.SUITES:
            source = self.report_path(suite).read_bytes()
            self.assertEqual((self.run / "reports" / (suite + ".json")).read_bytes(), source)
            self.assertEqual(result["reports"][suite]["sha256"], hashlib.sha256(source).hexdigest())
        text = summary.render(result)
        self.assertIn("C32 TTFT P95秒", text)
        self.assertIn("32条满窗口同时驻留: NOT_PROVEN", text)

    def test_partial_failed_benchmark_retains_numbers_without_signoff(self):
        report = self.reports["benchmark"]
        report["status"] = "FAIL"
        report["tests"][0]["status"] = "FAIL"
        report["benchmarks"]["off"]["samples"][-1]["status"] = "FAIL"
        self.replace_report("benchmark")
        result = self.build()
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["stage_exit_code"], 0)
        self.assertEqual(result["exit_code"], 1)
        self.assertEqual(result["benchmarks"]["off"]["single_decode_tokens_per_second_median"], 60)
        self.assertFalse(any(row["signoff"] for row in result["benchmarks"].values()))

    def test_contamination_excluded_from_diagnostic_median_and_signature(self):
        report = self.reports["diagnosis"]
        row = next(s for s in report["samples"] if s["workload"] == "text" and s["mode"] == "off" and s["phase"] == "sample")
        row["timing"]["decode_tokens_per_second"] = 999999.
        row["accounting"].update(status="CONTAMINATED", issues=["external traffic"])
        self.replace_report("diagnosis")
        result = self.build()
        self.assertEqual(result["status"], "FAIL")
        row = result["diagnostics"]["text/off"]
        self.assertEqual(row["decode_tps_median"], 64)
        self.assertEqual(row["weighted_acceptance_rate"], .8)
        self.assertEqual(row["status"], "NOT_SIGNOFF")
        self.assertFalse(row["signoff"])
        self.assertIn("NOT_SIGNOFF", summary.render(result))

    def test_top_level_pollution_blocks_all_signoff(self):
        self.reports["benchmark"]["contamination"] = ["generation mismatch"]
        self.replace_report("benchmark")
        result = self.build()
        self.assertEqual(result["status"], "FAIL")
        self.assertFalse(any(row["signoff"] for row in result["diagnostics"].values()))

    def test_missing_report_stays_incomplete_with_no_invented_zero(self):
        self.report_path("benchmark").unlink()
        result = self.build()
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertEqual(result["benchmarks"], {})
        self.assertIn("未取得", summary.render(result))
        self.assertEqual(result["exit_code"], 1)

    def test_missing_or_empty_accounting_does_not_pass(self):
        self.reports["smoke"]["global_accounting"] = {}
        self.replace_report("smoke")
        self.assertEqual(self.build()["status"], "FAIL")

    def test_wrong_physical_model_native_vision_budget_fails(self):
        path = self.base / "models" / summary.WEIGHT_MODEL / "config.json"
        model = json.loads(path.read_text())
        model["vision_config"]["max_image_tokens"] = 256
        write_json(path, model)
        result = self.build()
        self.assertEqual(result["status"], "FAIL")
        self.assertFalse(any(row["signoff"] for row in result["benchmarks"].values()))

    def test_actual_parallelism_or_resolved_mismatch_fails(self):
        path = self.base / summary.PROJECT / "runtime/runtime.resolved.json"
        config = json.loads(path.read_text())
        config["tensor_parallel_size"] = 4
        write_json(path, config)
        self.assertEqual(self.build()["status"], "FAIL")

    def test_required_resolved_configuration_missing_is_incomplete(self):
        (self.base / summary.PROJECT / "runtime/runtime.resolved.json").unlink()
        self.assertEqual(self.build()["status"], "INCOMPLETE")

    def test_launch_id_mismatch_or_missing_prevents_signoff(self):
        (self.run / "launch-id.txt").write_text("another-launch\n")
        self.assertEqual(self.build()["status"], "FAIL")
        (self.run / "launch-id.txt").unlink()
        self.assertEqual(self.build()["status"], "INCOMPLETE")

    def test_retrieval_failure_or_wrong_actual_input_fails(self):
        self.reports["long-context"]["tests"][0]["details"]["actual"] = "wrong"
        self.replace_report("long-context")
        result = self.build()
        self.assertEqual(result["status"], "FAIL")
        self.assertFalse(result["long_context"]["retrieval_passed"])

    def test_unknown_duplicate_and_missing_stages_cannot_pass(self):
        path = self.run / "steps.tsv"
        original = path.read_text()
        path.write_text(original + "smoke\t%s\t%s\t1\tsmoke.log\n" % (STAMP, STAMP))
        self.assertEqual(self.build()["status"], "FAIL")
        path.write_text("\n".join(original.splitlines()[:-1]) + "\n")
        self.assertEqual(self.build()["status"], "INCOMPLETE")

    def test_no_external_report_path_or_symlink(self):
        path = self.run / "context.json"
        context = json.loads(path.read_text())
        context["report_paths"]["smoke"] = "../outside.json"
        write_json(path, context)
        self.assertEqual(self.build()["status"], "FAIL")

    def test_zero_acceptance_is_real_data_and_missing_counters_are_not_zero(self):
        report = self.reports["diagnosis"]
        for row in report["samples"]:
            if row["workload"] == "text" and row["mode"] == "off":
                row["accounting"]["accepted_tokens"] = 0.
            if row["workload"] == "code" and row["mode"] == "high":
                row["accounting"].pop("draft_tokens")
        self.replace_report("diagnosis")
        result = self.build()
        self.assertEqual(result["diagnostics"]["text/off"]["weighted_acceptance_rate"], 0.)
        self.assertIsNone(result["diagnostics"]["code/high"]["weighted_acceptance_rate"])
        self.assertIsNone(result["diagnostics"]["code/high"]["dspark_counters"])
        self.assertEqual(result["status"], "FAIL")

    def test_interrupt_retains_code_and_writes_both_reports(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            code = summary.main(["--base", str(self.base), "--run", str(self.run), "--exit-code", "143", "--interrupted", "TERM"])
        result = json.loads((self.run / "summary.json").read_text())
        self.assertEqual(code, 1)
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["exit_code"], 143)
        self.assertEqual(result["interrupted"], "TERM")
        self.assertEqual((self.run / "summary.txt").read_text(), output.getvalue())

    def test_cli_check_config_is_not_gpu_validation(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            code = summary.main(["--base", str(self.base), "--run", str(self.run), "--check-config"])
        result = json.loads(output.getvalue())
        self.assertEqual(code, 0)
        self.assertFalse(result["gpu_executed"])
        self.assertEqual(result["status"], "PASS")

    def test_malformed_report_preserves_bytes_and_cannot_pass(self):
        self.report_path("smoke").write_text("{broken")
        result = self.build()
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertEqual((self.run / "reports/smoke.json").read_text(), "{broken")

    def test_sample_counts_and_output_budget_must_match_run_parameters(self):
        context_path = self.run / "context.json"
        context = json.loads(context_path.read_text())
        context["parameters"]["output_tokens"] = 1024
        context["parameters"]["diagnosis_samples"] = 10
        write_json(context_path, context)
        result = self.build()
        self.assertEqual(result["status"], "FAIL")
        self.assertFalse(result["benchmarks"]["off"]["signoff"])

    def test_nonfinite_json_report_is_rejected_and_kept_as_raw_evidence(self):
        self.report_path("smoke").write_text('{"value":NaN}')
        result = self.build()
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertEqual((self.run / "reports/smoke.json").read_text(), '{"value":NaN}')

    def external_model(self):
        external = Path(self.directory.name) / "usb-weights" / summary.WEIGHT_MODEL
        external.parent.mkdir(parents=True, exist_ok=True)
        (self.base / "models" / summary.WEIGHT_MODEL).rename(external)
        return external

    def test_external_model_directory_keeps_config_checks_and_read_only_evidence(self):
        external = self.external_model()
        original = (external / "config.json").read_bytes()
        (self.run / "model-source.txt").write_text("/media/USB/" + summary.WEIGHT_MODEL + "\n")
        result = summary.build_summary(self.base, self.run, 0, None, external)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["configuration_check"]["evidence"]["model"]["path"], str(external / "config.json"))
        self.assertEqual(result["configuration_check"]["model_directory"], str(external))
        self.assertEqual(result["model_source_directory"], "/media/USB/" + summary.WEIGHT_MODEL)
        self.assertIn(result["model_source_directory"], summary.render(result))
        self.assertEqual((external / "config.json").read_bytes(), original)
        self.assertFalse((self.base / "models" / summary.WEIGHT_MODEL).exists())
        self.assertEqual(list(external.iterdir()), [external / "config.json"])

    def test_external_missing_or_wrong_metadata_fails_without_default_fallback(self):
        external = Path(self.directory.name) / "missing-weights"
        result = summary.build_summary(self.base, self.run, 0, None, external)
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["configuration_check"]["status"], "FAIL")
        external = self.external_model()
        config_path = external / "config.json"
        config = json.loads(config_path.read_text())
        config["vision_config"]["max_image_tokens"] = 256
        write_json(config_path, config)
        result = summary.build_summary(self.base, self.run, 0, None, external)
        self.assertEqual(result["status"], "FAIL")
        self.assertFalse(any(row["signoff"] for row in result["benchmarks"].values()))

    def test_cli_external_model_option_reaches_config_and_full_summary(self):
        external = self.external_model()
        with contextlib.redirect_stdout(io.StringIO()) as output:
            code = summary.main(["--base", str(self.base), "--model-directory", str(external), "--check-config"])
        config_result = json.loads(output.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(config_result["model_directory"], str(external))
        with contextlib.redirect_stdout(io.StringIO()):
            code = summary.main(["--base", str(self.base), "--run", str(self.run), "--model-directory", str(external)])
        self.assertEqual(code, 0)
        result = json.loads((self.run / "summary.json").read_text())
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["configuration_check"]["model_directory"], str(external))


if __name__ == "__main__":
    unittest.main()
