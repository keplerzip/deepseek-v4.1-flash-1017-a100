"""CPU-only deployment contract regressions; no Docker, network or model payloads.

Run: python3 -B -m unittest discover -s tests -p test_deployment.py -v
"""
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("deployment_runtime", ROOT / "scripts/runtime.py")
runtime = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime)


class TemporaryCase(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="dsv411017-contract-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def write_json(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
        return path


class ConfigurationTests(TemporaryCase):
    def setUp(self):
        super().setUp()
        self.config = json.loads((ROOT / "configs/runtime.json").read_text())

    def load(self, value):
        return runtime.load_config(self.write_json(self.root / "config.json", value))

    def test_requested_context_parallelism_and_native_vision(self):
        config = self.load(self.config)
        self.assertEqual(config["max_model_len"], 512 * 1024)
        self.assertEqual(config["max_num_seqs"], 32)
        self.assertEqual(config["tensor_parallel_size"], 8)
        self.assertEqual(config["data_parallel_size"], 1)
        self.assertEqual(config["kv_cache_dtype"], "fp8_ds_mla")
        self.assertNotIn("hf_overrides", config)
        self.assertNotIn("mm_processor_kwargs", config)

    def test_context_concurrency_and_parallelism_cannot_silently_downgrade(self):
        changes = {"max_model_len": 512000, "max_num_seqs": 16,
                   "tensor_parallel_size": 4, "data_parallel_size": 2,
                   "pipeline_parallel_size": 2, "enable_expert_parallel": False,
                   "kv_cache_dtype": "auto", "engram_config": {"cpu_offload": False},
                   "disable_chunked_mm_input": False, "enable_prefix_caching": False}
        for key, value in changes.items():
            with self.subTest(key=key):
                config = copy.deepcopy(self.config)
                config[key] = value
                with self.assertRaises(ValueError):
                    self.load(config)

    def test_protected_numbers_require_exact_types(self):
        for key, value in (("max_num_seqs", 32.0), ("data_parallel_size", True),
                           ("tensor_parallel_size", "8")):
            with self.subTest(key=key):
                config = copy.deepcopy(self.config)
                config[key] = value
                with self.assertRaises(ValueError):
                    self.load(config)

    def test_old_visual_token_override_is_rejected(self):
        for key, value in (("hf_overrides", {"vision_max_n_token": 1020}),
                           ("mm_processor_kwargs", {})):
            with self.subTest(key=key):
                config = copy.deepcopy(self.config)
                config[key] = value
                with self.assertRaises(ValueError):
                    self.load(config)

    def test_invalid_draft_or_decode_graph_is_rejected(self):
        for method, k in (("ngram", 5), ("dspark", 6), ("dspark", 0)):
            with self.subTest(method=method, k=k):
                config = copy.deepcopy(self.config)
                config["speculative_config"].update(method=method, num_speculative_tokens=k)
                with self.assertRaises(ValueError):
                    self.load(config)
        config = copy.deepcopy(self.config)
        config["compilation_config"]["cudagraph_mode"] = "NONE"
        with self.assertRaises(ValueError):
            self.load(config)

    def test_invalid_scheduler_budget_is_rejected(self):
        for budget in (0, 192, 1024, 32769):
            with self.subTest(budget=budget):
                config = copy.deepcopy(self.config)
                config["max_num_batched_tokens"] = budget
                with self.assertRaises(ValueError):
                    self.load(config)

    def test_cli_preserves_uds_and_boolean_semantics(self):
        arguments = runtime.argv(self.load(self.config))
        self.assertEqual(arguments[0], str(runtime.MODEL_PATH))
        self.assertEqual(arguments[arguments.index("--uds") + 1], "/state/api.sock")
        self.assertIn("--disable-chunked-mm-input", arguments)
        self.assertIn("--enable-expert-parallel", arguments)
        self.assertNotIn("--hf-overrides", arguments)
        speculative = json.loads(arguments[arguments.index("--speculative-config") + 1])
        self.assertEqual(speculative["num_speculative_tokens"], 5)


class ModelReceiptTests(TemporaryCase):
    def setUp(self):
        super().setUp()
        self.model = self.root / "model"
        self.model.mkdir()
        config = {"text_config": {"dspark_block_size": 5},
                  "vision_config": {"max_image_tokens": 1024}}
        self.write_json(self.model / "config.json", config)
        (self.model / "tiny.safetensors").write_bytes(b"test fixture; not real weights")
        self.manifest = self.root / "manifest.json"
        self.receipt = self.root / "receipt.json"
        self.refresh_manifest()

    def refresh_manifest(self):
        files = []
        for path in sorted(self.model.iterdir()):
            data = path.read_bytes()
            files.append({"path": path.name, "size": len(data),
                          "sha256": hashlib.sha256(data).hexdigest()})
        self.write_json(self.manifest, {"modelscope_model_id": "deepseek-ai/" + runtime.MODEL,
                                      "modelscope_revision": "a" * 40, "files": files})

    def verify(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return runtime.verify_model(self.model, self.manifest, self.receipt)

    def test_verified_snapshot_can_be_reused_without_rehash(self):
        verified = self.verify()
        self.assertEqual(verified["status"], "PASS")
        self.assertFalse(verified["gpu_executed"])
        self.assertIsNone(runtime.check_receipt(self.model, self.manifest, self.receipt))

    def test_same_size_edit_invalidates_receipt_even_when_mtime_is_restored(self):
        self.verify()
        path = self.model / "tiny.safetensors"
        previous = path.stat()
        path.write_bytes(b"X" * previous.st_size)
        os.utime(path, ns=(previous.st_atime_ns, previous.st_mtime_ns))
        with self.assertRaises(ValueError):
            runtime.check_receipt(self.model, self.manifest, self.receipt)

    def test_replacing_file_with_identical_bytes_invalidates_inode_fingerprint(self):
        self.verify()
        path = self.model / "tiny.safetensors"
        old = path.stat()
        replacement = self.model / "replacement"
        replacement.write_bytes(path.read_bytes())
        os.utime(replacement, ns=(old.st_atime_ns, old.st_mtime_ns))
        replacement.replace(path)
        with self.assertRaises(ValueError):
            runtime.check_receipt(self.model, self.manifest, self.receipt)

    def test_manifest_change_invalidates_receipt(self):
        self.verify()
        self.manifest.write_text(self.manifest.read_text() + "\n")
        with self.assertRaises(ValueError):
            runtime.check_receipt(self.model, self.manifest, self.receipt)

    def test_receipt_status_hash_and_fingerprint_changes_are_rejected(self):
        self.verify()
        original = json.loads(self.receipt.read_text())
        for field, value in (("status", "RUNNING"), ("status", "FAIL"),
                             ("manifest_sha256", "0" * 64), ("fingerprint", "0" * 64)):
            with self.subTest(field=field, value=value):
                changed = copy.deepcopy(original)
                changed[field] = value
                self.write_json(self.receipt, changed)
                with self.assertRaises(ValueError):
                    runtime.check_receipt(self.model, self.manifest, self.receipt)

    def test_wrong_sha_writes_fail_receipt_and_cannot_be_reused(self):
        manifest = json.loads(self.manifest.read_text())
        manifest["files"][0]["sha256"] = "0" * 64
        self.write_json(self.manifest, manifest)
        with self.assertRaises(ValueError):
            self.verify()
        self.assertEqual(json.loads(self.receipt.read_text())["status"], "FAIL")
        with self.assertRaises(ValueError):
            runtime.check_receipt(self.model, self.manifest, self.receipt)

    def test_changed_native_vision_or_draft_block_is_rejected(self):
        for key, field, value in (("text_config", "dspark_block_size", 6),
                                  ("vision_config", "max_image_tokens", 512)):
            with self.subTest(field=field):
                config = {"text_config": {"dspark_block_size": 5},
                          "vision_config": {"max_image_tokens": 1024}}
                config[key][field] = value
                self.write_json(self.model / "config.json", config)
                self.refresh_manifest()
                with self.assertRaises(ValueError):
                    self.verify()
                self.assertEqual(json.loads(self.receipt.read_text())["status"], "FAIL")

    def test_symlink_payload_is_rejected(self):
        path = self.model / "tiny.safetensors"
        external = self.root / "outside"
        external.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(external)
        with self.assertRaises(ValueError):
            self.verify()


class CapacityTests(TemporaryCase):
    def setUp(self):
        super().setUp()
        self.observations = self.root / "observations"
        self.observations.mkdir()
        self.output = self.root / "report.json"
        self.config = json.loads((ROOT / "configs/runtime.json").read_text())

    def records(self, windows=1, pool=None, extra=0):
        # 11 blocks/window, intentionally not a multiple of TP=8. TP shards
        # describe one logical KV pool and must never multiply full windows.
        groups = [{"full_window_blocks": 7}, {"full_window_blocks": 4}]
        capacity = {"kind": "capacity", "pid": 10, "data_parallel_rank": 0, "time_ns": 2,
                    "max_model_len": 524288, "configured_max_num_seqs": 32,
                    "tensor_parallel_size": 8, "data_parallel_size": 1,
                    "groups": groups, "blocks_per_full_window": 11,
                    "pool_blocks": pool if pool is not None else windows * 11 + extra,
                    "full_windows": windows}
        return [capacity, *[{"kind": "dspark", "pid": 100 + rank,
                             "num_speculative_tokens": 5} for rank in range(8)],
                {"kind": "graph-capture", "pid": 100},
                {"kind": "graph-replay", "pid": 100}]

    def evaluate(self, records, require_replay=True, expected_pass=True):
        for index, row in enumerate(records):
            self.write_json(self.observations / f"{index}.json", row)
        with contextlib.redirect_stdout(io.StringIO()):
            if expected_pass:
                runtime.capacity(self.observations, self.config, self.output, require_replay)
            else:
                with self.assertRaises(SystemExit) as caught:
                    runtime.capacity(self.observations, self.config, self.output, require_replay)
                self.assertEqual(caught.exception.code, 1)
        report = json.loads(self.output.read_text())
        self.assertEqual(report["status"], "PASS" if expected_pass else "FAIL")
        return report

    def test_one_window_is_not_multiplied_by_eight_workers(self):
        report = self.evaluate(self.records(1, extra=10))
        self.assertEqual(report["measured_full_window_capacity"], 1)
        self.assertFalse(report["capacity_sufficient_for_32_full_windows"])
        self.assertFalse(report["full_window_C32_proven"])

    def test_exact_32_window_capacity_is_still_not_a_load_test(self):
        report = self.evaluate(self.records(32))
        self.assertEqual(report["measured_full_window_capacity"], 32)
        self.assertTrue(report["capacity_sufficient_for_32_full_windows"])
        self.assertFalse(report["full_window_C32_proven"])

    def test_bad_block_sum_and_rounded_window_count_are_rejected(self):
        for mutate in (lambda row: row.update(blocks_per_full_window=10),
                       lambda row: row.update(pool_blocks=21, full_windows=2)):
            with self.subTest(mutate=mutate):
                records = self.records()
                mutate(records[0])
                report = self.evaluate(records, expected_pass=False)
                self.assertTrue(any("accounting" in error or "mismatch" in error
                                    for error in report["errors"]))

    def test_pool_smaller_than_one_window_is_rejected(self):
        report = self.evaluate(self.records(windows=0, pool=10), expected_pass=False)
        self.assertIn("KV pool cannot hold even one requested full window", report["errors"])

    def test_wrong_observed_topology_is_rejected(self):
        for key, value in (("tensor_parallel_size", 4), ("data_parallel_size", 2),
                           ("max_model_len", 512000), ("configured_max_num_seqs", 64)):
            with self.subTest(key=key):
                records = self.records()
                records[0][key] = value
                report = self.evaluate(records, expected_pass=False)
                self.assertIn("Wrong observed runtime: " + key, report["errors"])

    def test_latest_capacity_record_wins_for_same_worker(self):
        records = self.records(2)
        old = copy.deepcopy(records[0])
        old.update(time_ns=1, full_windows=0, pool_blocks=0)
        records.append(old)
        report = self.evaluate(records)
        self.assertEqual(report["measured_full_window_capacity"], 2)
        self.assertEqual(len(report["capacity_records"]), 1)

    def test_missing_worker_or_draft_configuration_is_rejected(self):
        records = self.records()
        records = [row for row in records if row.get("pid") != 107]
        report = self.evaluate(records, expected_pass=False)
        self.assertEqual(report["dspark_workers"], 7)
        records = self.records()
        records[1]["num_speculative_tokens"] = 6
        report = self.evaluate(records, expected_pass=False)
        self.assertIn("DSpark k=5 was not loaded", report["errors"])

    def test_replay_is_required_after_smoke(self):
        records = [row for row in self.records() if row["kind"] != "graph-replay"]
        report = self.evaluate(records, expected_pass=False)
        self.assertIn("Missing CUDA Graph replay telemetry after smoke requests", report["errors"])

    def test_no_telemetry_fails_closed(self):
        report = self.evaluate([], expected_pass=False)
        self.assertIn("Missing measured capacity records", report["errors"])


if __name__ == "__main__":
    unittest.main()
