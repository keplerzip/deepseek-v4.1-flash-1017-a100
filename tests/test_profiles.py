"""CPU contract checks for reviewed profiles and per-DP capacity accounting.

All observations in this module are synthetic fixtures; they are never retained
as GPU acceptance or model-performance evidence.
"""
import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runtime = import_file("profile_test_runtime", ROOT / "scripts/runtime.py")
with mock.patch.dict(sys.modules, {"runtime": runtime}):
    profiles = import_file("profile_test_resolver", ROOT / "scripts/profiles.py")
acceptance = import_file("profile_test_acceptance", ROOT / "tests/acceptance.py")


class TemporaryCase(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="dsv411017-profile-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.base = json.loads((ROOT / "configs/runtime.json").read_text())

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path

    def load(self, config):
        return runtime.load_config(self.write(self.root / "config.json", config))

    def project(self, mutate):
        folder = self.root / "project"
        self.write(folder / "configs/runtime.json", self.base)
        catalog = json.loads((ROOT / "configs/profiles.json").read_text())
        mutate(catalog)
        self.write(folder / "configs/profiles.json", catalog)
        return folder


class ProfileTests(TemporaryCase):
    def test_all_eight_profiles_preserve_the_fixed_capabilities(self):
        catalog = json.loads((ROOT / "configs/profiles.json").read_text())
        self.assertEqual(set(catalog["profiles"]), {
            "baseline", "tp8-quiet-eplb", "tp8-fast-align", "tp8-custom-ar",
            "tp8-context-graph", "tp4-dp2", "tp8-batch2048", "tp8-fast-candidate"})
        self.assertEqual(catalog["constraints"], {"global_concurrency": 32, "context": 524288,
            "native_vision_tokens": 1024, "served_model_name": "DeepSeek-Flash"})
        for name in catalog["profiles"]:
            with self.subTest(profile=name):
                record = profiles.resolve(ROOT, name)
                profiles.validate_record(record, self.root)
                config = record["configuration"]
                self.assertEqual(config["served_model_name"], "DeepSeek-Flash")
                self.assertEqual(config["model"], "/models/DeepSeek-V4.1-Flash-1017")
                self.assertEqual(config["max_model_len"], 524288)
                self.assertEqual(config["data_parallel_size"] * config["max_num_seqs"], 32)
                self.assertEqual(config["tensor_parallel_size"] * config["data_parallel_size"], 8)
                self.assertTrue(config["enable_expert_parallel"])
                self.assertTrue(config["disable_chunked_mm_input"])
                self.assertTrue(config["enable_chunked_prefill"])
                self.assertEqual(config["limit_mm_per_prompt"], {"image": 999})
                self.assertNotIn("hf_overrides", config)
                self.assertNotIn("mm_processor_kwargs", config)
                self.assertFalse(config.get("language_model_only", False))
                self.assertEqual(config["speculative_config"]["method"], "dspark")
                self.assertEqual(config["speculative_config"]["num_speculative_tokens"], 5)
                self.assertFalse(config["speculative_config"]["enable_adaptive_verification"])
                self.assertEqual(record["global_concurrency"], 32)
                self.assertEqual(record["gpu_acceptance"], "NOT_RUN_BY_SELECTION")
                self.assertEqual(record["tps_gain"], "NOT_MEASURED")
                if config["disable_custom_all_reduce"] is False:
                    self.assertIn("tp-ar", record["gates"])
                if record["environment"]["VLLM_DSV41_CAPTURE_CONTEXT_COMBINE"] == "1":
                    self.assertIn("context-graph", record["gates"])

    def test_tp4_dp2_has_two_local_pools_of_sixteen(self):
        config = profiles.resolve(ROOT, "tp4-dp2")["configuration"]
        loaded = self.load(config)
        self.assertEqual((loaded["tensor_parallel_size"], loaded["data_parallel_size"],
                          loaded["data_parallel_size_local"], loaded["max_num_seqs"]), (4, 2, 2, 16))
        self.assertGreaterEqual(loaded["compilation_config"]["max_cudagraph_capture_size"], 16 * 6)

    def test_dp2_rejects_wrong_global_concurrency_or_incomplete_local_pools(self):
        base = profiles.resolve(ROOT, "tp4-dp2")["configuration"]
        for key, value in (("max_num_seqs", 32), ("max_num_seqs", 8),
                           ("max_num_seqs", 16.0), ("data_parallel_size_local", 1),
                           ("data_parallel_size_local", None), ("data_parallel_size_local", 2.0),
                           ("data_parallel_size_local", True)):
            with self.subTest(key=key, value=value):
                config = copy.deepcopy(base)
                if value is None:
                    config.pop(key, None)
                else:
                    config[key] = value
                with self.assertRaises(ValueError):
                    self.load(config)

    def test_profile_cannot_silently_enable_text_only_loading(self):
        project = self.project(lambda catalog: catalog["profiles"]["baseline"]["config"].update(language_model_only=True))
        with self.assertRaises(ValueError):
            record = profiles.resolve(project, "baseline")
            profiles.validate_record(record, self.root)

    def test_profile_cannot_relax_context_vision_or_public_model_identity(self):
        for key, value in (("max_model_len", 512000), ("served_model_name", "DeepSeek-V4.1-Flash-1017"),
                           ("served_model_name", ["DeepSeek-Flash", "other"]),
                           ("hf_overrides", {"vision_max_n_token": 1020}),
                           ("mm_processor_kwargs", {"max_image_tokens": 512})):
            with self.subTest(key=key):
                project = self.project(lambda catalog: catalog["profiles"]["baseline"]["config"].update({key: value}))
                with self.assertRaises(ValueError):
                    record = profiles.resolve(project, "baseline")
                    profiles.validate_record(record, self.root)

    def test_environment_and_required_gpu_gates_fail_closed(self):
        mutations = [
            lambda c: c["base_environment"].update(UNREVIEWED_ENV="1"),
            lambda c: c["base_environment"].update(VLLM_DETERMINISTIC_MOE_ALIGN="2"),
            lambda c: c["profiles"]["tp8-custom-ar"].update(gates=[]),
            lambda c: c["profiles"]["tp8-context-graph"].update(gates=[]),
            lambda c: c["profiles"]["baseline"].update(gates=["unknown"]),
        ]
        names = ["baseline", "baseline", "tp8-custom-ar", "tp8-context-graph", "baseline"]
        for mutate, name in zip(mutations, names):
            with self.subTest(profile=name):
                project = self.project(mutate)
                with self.assertRaises(ValueError):
                    profiles.resolve(project, name)


class DP2CapacityTests(TemporaryCase):
    def setUp(self):
        super().setUp()
        self.config = profiles.resolve(ROOT, "tp4-dp2")["configuration"]
        self.observations = self.root / "observations"
        self.output = self.root / "capacity-fixture.json"

    def records(self, pools=((16,), (16,)), remainder=0):
        records = []
        for dp_rank, workers in enumerate(pools):
            for worker, windows in enumerate(workers):
                records.append({"kind": "capacity", "pid": 200 + dp_rank * 10 + worker,
                    "data_parallel_rank": dp_rank, "time_ns": 2,
                    "max_model_len": 524288, "configured_max_num_seqs": 16,
                    "tensor_parallel_size": 4, "data_parallel_size": 2,
                    "groups": [{"full_window_blocks": 7}, {"full_window_blocks": 4}],
                    "blocks_per_full_window": 11, "pool_blocks": windows * 11 + remainder,
                    "full_windows": windows})
        for rank in range(8):
            records.extend([
                {"kind": "dspark", "pid": 100 + rank, "data_parallel_rank": rank // 4,
                 "num_speculative_tokens": 5},
                {"kind": "graph-capture", "pid": 100 + rank, "data_parallel_rank": rank // 4},
                {"kind": "graph-replay", "pid": 100 + rank, "data_parallel_rank": rank // 4},
            ])
        return records

    def evaluate(self, records, passes=True):
        self.observations.mkdir(exist_ok=True)
        for path in self.observations.glob("*.json"):
            path.unlink()
        for number, row in enumerate(records):
            self.write(self.observations / (str(number) + ".json"), row)
        with contextlib.redirect_stdout(io.StringIO()):
            if passes:
                runtime.capacity(self.observations, self.config, self.output, require_replay=True)
            else:
                with self.assertRaises(SystemExit) as caught:
                    runtime.capacity(self.observations, self.config, self.output, require_replay=True)
                self.assertEqual(caught.exception.code, 1)
        report = json.loads(self.output.read_text())
        self.assertEqual(report["status"], "PASS" if passes else "FAIL")
        return report

    def test_capacity_takes_each_pool_minimum_then_sums_without_tp_multiplication(self):
        report = self.evaluate(self.records(((18, 17, 19, 20), (16, 21, 22, 23))))
        self.assertEqual(report["per_dp_full_window_capacity"], {"0": 17, "1": 16})
        self.assertEqual(report["measured_full_window_capacity"], 33)
        self.assertTrue(report["capacity_sufficient_for_32_full_windows"])
        self.assertFalse(report["full_window_C32_proven"])

    def test_partial_windows_cannot_be_combined_across_dp_pools(self):
        # Each pool has 15 complete windows plus 10/11 of a window. Combining
        # pool bytes before flooring would incorrectly report 31 instead of 30.
        report = self.evaluate(self.records(((15,), (15,)), remainder=10))
        self.assertEqual(report["per_dp_full_window_capacity"], {"0": 15, "1": 15})
        self.assertEqual(report["measured_full_window_capacity"], 30)
        self.assertFalse(report["capacity_sufficient_for_32_full_windows"])

    def test_twenty_four_plus_eight_does_not_prove_balanced_c32_full_windows(self):
        report = self.evaluate(self.records(((24,), (8,))))
        self.assertEqual(report["measured_full_window_capacity"], 32)
        self.assertFalse(report["capacity_sufficient_for_32_full_windows"])
        self.assertFalse(report["full_window_C32_proven"])

    def test_sixteen_per_pool_is_capacity_only_and_never_a_load_test(self):
        report = self.evaluate(self.records())
        self.assertEqual(report["measured_full_window_capacity"], 32)
        self.assertTrue(report["capacity_sufficient_for_32_full_windows"])
        self.assertFalse(report["full_window_C32_proven"])

    def test_missing_pool_capacity_fails_even_with_all_eight_draft_workers(self):
        report = self.evaluate(self.records(((32,),)), passes=False)
        self.assertEqual(report["dspark_workers"], 8)
        self.assertFalse(report["capacity_sufficient_for_32_full_windows"])

    def test_unexpected_pool_rank_cannot_leave_sufficient_capacity_true(self):
        records = self.records()
        for row in records:
            if row["kind"] == "capacity" and row["data_parallel_rank"] == 1:
                row["data_parallel_rank"] = 2
        report = self.evaluate(records, passes=False)
        self.assertFalse(report["capacity_sufficient_for_32_full_windows"])

    def test_pool_rank_boolean_is_not_an_integer_rank(self):
        records = self.records()
        records[0]["data_parallel_rank"] = False
        report = self.evaluate(records, passes=False)
        self.assertFalse(report["capacity_sufficient_for_32_full_windows"])

    def test_per_pool_runtime_concurrency_and_topology_must_match(self):
        for key, value in (("configured_max_num_seqs", 32), ("tensor_parallel_size", 8),
                           ("data_parallel_size", 1), ("max_model_len", 512000)):
            with self.subTest(field=key):
                records = self.records()
                records[0][key] = value
                report = self.evaluate(records, passes=False)
                self.assertFalse(report["capacity_sufficient_for_32_full_windows"])

    def context_graph_records(self):
        records = self.records()
        for row in records:
            if row["kind"] not in ("graph-capture", "graph-replay"):
                continue
            row.update(manager="DFlashCudaGraphManager", manager_id=10000 + row["pid"],
                       draft_context_combine_captured=True)
            if row["kind"] == "graph-capture":
                row.update(memory_estimation_probe=False,
                           captured_full_graphs=[{"mode": "FULL", "num_tokens": 5}])
            else:
                row["descriptor"] = {"mode": "FULL", "num_tokens": 5}
        return records

    def test_context_combine_requires_real_full_draft_replay_on_all_eight_workers(self):
        with mock.patch.dict(os.environ, {"VLLM_DSV41_CAPTURE_CONTEXT_COMBINE": "1"}):
            report = self.evaluate(self.context_graph_records())
        self.assertEqual(report["context_combine_full_draft_replay_workers"], 8)
        self.assertEqual(report["dspark_workers"], 8)
        self.assertTrue(report["capacity_sufficient_for_32_full_windows"])
        self.assertFalse(report["full_window_C32_proven"])

    def test_context_combine_rejects_target_wrong_manager_probe_capture_only_and_missing_replay(self):
        # Each variant preserves other capacity/worker evidence, so a target
        # graph, profiling graph, or partial draft replay cannot pass this gate.
        cases = (
            "target-graph", "wrong-manager-id", "wrong-replay-manager", "capture-only",
            "one-worker-no-replay", "memory-estimation-probe", "capture-not-full",
            "replay-not-full", "capture-combine-disabled", "replay-combine-disabled",
        )
        for case in cases:
            with self.subTest(case=case):
                records = self.context_graph_records()
                if case == "capture-only":
                    records = [row for row in records if row["kind"] != "graph-replay"]
                elif case == "one-worker-no-replay":
                    records = [row for row in records if not (row["kind"] == "graph-replay" and row["pid"] == 107)]
                else:
                    for row in records:
                        if row["kind"] not in ("graph-capture", "graph-replay"):
                            continue
                        if case == "target-graph":
                            row["manager"] = "CudaGraphManager"
                        elif case == "wrong-manager-id" and row["kind"] == "graph-replay":
                            row["manager_id"] += 1
                        elif case == "wrong-replay-manager" and row["kind"] == "graph-replay":
                            row["manager"] = "CudaGraphManager"
                        elif case == "memory-estimation-probe" and row["kind"] == "graph-capture":
                            row["memory_estimation_probe"] = True
                        elif case == "capture-not-full" and row["kind"] == "graph-capture":
                            row["captured_full_graphs"][0]["mode"] = "PIECEWISE"
                        elif case == "replay-not-full" and row["kind"] == "graph-replay":
                            row["descriptor"]["mode"] = "PIECEWISE"
                        elif case == "capture-combine-disabled" and row["kind"] == "graph-capture":
                            row["draft_context_combine_captured"] = False
                        elif case == "replay-combine-disabled" and row["kind"] == "graph-replay":
                            row["draft_context_combine_captured"] = False
                with mock.patch.dict(os.environ, {"VLLM_DSV41_CAPTURE_CONTEXT_COMBINE": "1"}):
                    report = self.evaluate(records, passes=False)
                self.assertLess(report["context_combine_full_draft_replay_workers"], 8)
                self.assertFalse(report["capacity_sufficient_for_32_full_windows"])
                self.assertIn("Context-combine profile needs actual FULL draft graph capture/replay on every worker", report["errors"])


class ReadySignatureTests(TemporaryCase):
    def test_signature_changes_with_actual_container_start_and_reviewed_profile(self):
        source = (ROOT / "scripts/common.sh").read_text()
        function = source.split("ready_signature() {", 1)[1].split("\n}", 1)[0]
        function = "ready_signature() {" + function + "\n}\n"
        # Execute only the signature function. No common.sh initialization,
        # sudo, Docker invocation or actual runtime state is involved.
        harness = """docker() {
  if [[ "$3" == '{{.Id}}' ]]; then printf '%s\\n' "$TEST_CONTAINER_ID";
  elif [[ "$3" == '{{.State.StartedAt}}' ]]; then printf '%s\\n' "$TEST_STARTED_AT";
  else return 20; fi
}
""" + function + "ready_signature\n"
        self.write(self.root / "runtime.active.json", self.base)
        self.write(self.root / "speed-profile.json", {"name": "baseline", "environment": {"TEST": "0"}})
        source_manifest = self.root / "PUBLIC_SOURCE_SHA256SUMS"
        source_manifest.write_text("Synthetic public source manifest, revision A.\n")
        env = {**os.environ, "ROOT": str(self.root), "STATE_DIR": str(self.root), "IMAGE_ID": "sha256:fixture-image",
               "ENGINE": "fixture-engine", "TEST_CONTAINER_ID": "fixture-container", "TEST_STARTED_AT": "start-A"}

        def signature(changes=None):
            return subprocess.check_output(["bash", "-eu", "-c", harness], env={**env, **(changes or {})}, text=True).strip()

        original = signature()
        self.assertEqual(len(original.split("|")), 6)
        self.assertEqual(original, signature())
        for change in ({"IMAGE_ID": "other-image"}, {"TEST_CONTAINER_ID": "other-container"}, {"TEST_STARTED_AT": "start-B"}):
            self.assertNotEqual(original, signature(change))
        config = copy.deepcopy(self.base)
        config["max_num_batched_tokens"] = 2048
        self.write(self.root / "runtime.active.json", config)
        self.assertNotEqual(original, signature())
        self.write(self.root / "runtime.active.json", self.base)
        self.write(self.root / "speed-profile.json", {"name": "baseline", "environment": {"TEST": "1"}})
        self.assertNotEqual(original, signature())
        self.write(self.root / "speed-profile.json", {"name": "baseline", "environment": {"TEST": "0"}})
        source_manifest.write_text("Synthetic public source manifest, revision B.\n")
        self.assertNotEqual(original, signature())


class RecordedCapacityEvidenceTests(TemporaryCase):
    def capacity_report(self, tp=4, dp=2, pool_windows=((16,), (16,)), remainder=0):
        per_pool, records = {}, []
        for rank, workers in enumerate(pool_windows):
            per_pool[str(rank)] = min(workers)
            for worker, windows in enumerate(workers):
                records.append({"kind": "capacity", "pid": 100 + rank * 10 + worker, "time_ns": 2,
                    "data_parallel_rank": rank, "tensor_parallel_size": tp, "data_parallel_size": dp,
                    "configured_max_num_seqs": 32 // dp, "max_model_len": 524288,
                    "groups": [{"full_window_blocks": 7}, {"full_window_blocks": 4}],
                    "blocks_per_full_window": 11, "pool_blocks": windows * 11 + remainder, "full_windows": windows})
        return {"status": "PASS", "gpu_executed": True, "configured_context": 524288,
                "configured_scheduler_concurrency": 32, "capacity_records": records,
                "per_dp_full_window_capacity": per_pool,
                "measured_full_window_capacity": sum(per_pool.values())}

    def evidence(self, report):
        path = self.write(self.root / "recorded-capacity.json", report)
        return acceptance.recorded_capacity_evidence(path)

    def test_tp8_and_dp2_reports_are_accepted_without_residency_claims(self):
        for report in (self.capacity_report(8, 1, ((32,),)),
                       self.capacity_report(pool_windows=((18, 17, 19, 20), (16, 21, 22)))):
            with self.subTest(report=report):
                evidence = self.evidence(report)
                self.assertEqual(evidence["status"], "VALID_RECORDED_CAPACITY")
                self.assertTrue(evidence["capacity_sufficient_for_32_full_windows"])
                self.assertEqual(evidence["recomputed_full_window_capacity"], report["measured_full_window_capacity"])
                self.assertEqual(evidence["simultaneous_full_window_residency_status"], "NOT_PROVEN")

    def test_total_thirty_two_does_not_override_an_eight_window_pool(self):
        evidence = self.evidence(self.capacity_report(pool_windows=((24,), (8,))))
        self.assertEqual(evidence["status"], "VALID_RECORDED_CAPACITY")
        self.assertEqual(evidence["recomputed_full_window_capacity"], 32)
        self.assertFalse(evidence["capacity_sufficient_for_32_full_windows"])
        self.assertEqual(evidence["per_dp_full_window_capacity"], {"0": 24, "1": 8})

    def test_capacity_report_rejects_combined_fractional_windows_and_tp_multiplication(self):
        report = self.capacity_report(pool_windows=((15,), (15,)), remainder=10)
        self.assertFalse(self.evidence(report)["capacity_sufficient_for_32_full_windows"])
        for inflated in (31, 32, 120):
            with self.subTest(inflated=inflated):
                report["measured_full_window_capacity"] = inflated
                evidence = self.evidence(report)
                self.assertEqual(evidence["status"], "UNVERIFIED")
                self.assertIsNone(evidence["capacity_sufficient_for_32_full_windows"])
                self.assertEqual(evidence["recomputed_full_window_capacity"], 30)

    def test_capacity_report_rejects_incomplete_mixed_and_boolean_topologies(self):
        for field, value in (("data_parallel_rank", False), ("data_parallel_rank", 2),
                             ("configured_max_num_seqs", 32), ("tensor_parallel_size", 8),
                             ("data_parallel_size", 1), ("full_windows", 16.0)):
            with self.subTest(field=field, value=value):
                report = self.capacity_report()
                report["capacity_records"][0][field] = value
                evidence = self.evidence(report)
                self.assertEqual(evidence["status"], "UNVERIFIED")
                self.assertIsNone(evidence["capacity_sufficient_for_32_full_windows"])
        report = self.capacity_report(pool_windows=((32,),))
        self.assertEqual(self.evidence(report)["status"], "UNVERIFIED")

    def test_full_window_suite_integrates_dp2_evidence_but_keeps_residency_unproven(self):
        path = self.write(self.root / "dp2-capacity.json", self.capacity_report())
        suite = acceptance.Suite.__new__(acceptance.Suite)
        suite.args = types.SimpleNamespace(capacity_report=path)
        suite.report = {}
        suite.save = lambda: None
        # Stop before tokenization or requests: the test only exercises the
        # report initialization and real capacity-evidence integration.
        suite.near_full_prompt = mock.Mock(side_effect=RuntimeError("stop-before-inference"))
        with self.assertRaisesRegex(RuntimeError, "stop-before-inference"):
            suite.full_window_c32()
        detail = suite.report["full_window_c32"]
        self.assertEqual(detail["capacity_evidence"]["status"], "VALID_RECORDED_CAPACITY")
        self.assertEqual(detail["capacity_evidence"]["topology"]["data_parallel_size"], 2)
        self.assertEqual(detail["simultaneous_full_window_residency_status"], "NOT_PROVEN")


class StartupGateTests(TemporaryCase):
    def run_startup_fixture(self, decision, extra_environment=None):
        # Run the actual startup shell script with isolated fake common/tools.
        # A gate-reader error must stop before any engine launch, even though
        # every other fake prerequisite would pass.
        project = self.root / decision
        (project / "scripts").mkdir(parents=True)
        (project / "tests").mkdir()
        (project / "start.sh").write_text((ROOT / "start.sh").read_text())
        common = r'''ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
STATE_DIR="$ROOT/runtime"
mkdir -p "$STATE_DIR"
ENGINE=fixture-engine
FRONTEND=fixture-frontend
IMAGE_ID=fixture-image
MODEL_DIR="$ROOT/models"
OWNER=fixture-owner
API_PORT=8005
USER_ARGS=()
fail() { printf '%s\n' "$*" >&2; exit 32; }
resolve_image() { :; }
resolve_model() { :; }
owned() { return 0; }
engine_health() { return 0; }
ensure_frontend() { printf 'frontend\n' >> "$STATE_DIR/startup-events"; }
remove_owned() { :; }
ready_signature() { printf 'fixture-signature\n'; }
image_python() {
  if [[ "$1" == /package/tests/check_r39_protocols.py ]]; then
    [[ "$2" == --output && "$3" == /state/r39-protocol-compat.json && "$#" == 3 ]] || return 24
    printf 'protocol-compat\n' >> "$STATE_DIR/startup-events"
    return "${TEST_PROTOCOL_STATUS:-0}"
  fi
  [[ "$1" == /opt/dsv411017/scripts/profiles.py || "$1" == /opt/dsv411017/scripts/runtime.py ]] || return 24
  case "$2" in
    prepare|env|check) return 0 ;;
    requires)
      [[ "$1" == /opt/dsv411017/scripts/profiles.py && "$3" == --gate && ( "$4" == tp-ar || "$4" == context-graph ) ]] || return 24
      if [[ "$TEST_GATE_DECISION" == error ]]; then return 23; fi
      printf '%s\n' "$TEST_GATE_DECISION" ;;
    *) return 24 ;;
  esac
}
docker() {
  "$TEST_PYTHON" "$ROOT/mock-docker.py" "$@"
}
'''
        (project / "scripts/common.sh").write_text(common)
        (project / "mock-docker.py").write_text(r'''import os,sys
from pathlib import Path
args=sys.argv[1:]
state=Path(__file__).parent/'runtime'
def event(name):
    with (state/'startup-events').open('a') as stream:stream.write(name+'\n')
if args[0]=='inspect':
    if '--format' not in args:sys.exit(1)
    assert args==['inspect','--format','{{.HostConfig.NetworkMode}}','fixture-engine'],args
    print(os.environ.get('TEST_NETWORK_MODE','none'))
elif args[0]=='run':
    if '-d' in args:
        assert args[args.index('--network')+1]=='none'
        assert '/opt/dsv411017/scripts/serve_with_usage.py' in args
        assert 'serve' in args and '--config' in args
        (state/'engine-started').touch();event('engine-started')
    elif '/package/scripts/api_access_check.py' in args:
        mode=args[args.index('--network')+1]
        if '--docker-client' in args:
            assert mode=='bridge' and '--add-host' in args and 'host.docker.internal:host-gateway' in args
            event('api-docker')
            sys.exit(int(os.environ.get('TEST_DOCKER_ACCESS_STATUS','0')))
        assert mode=='host';event('api-local')
        sys.exit(int(os.environ.get('TEST_LOCAL_ACCESS_STATUS','0')))
    else:
        assert 'probe' in args and '--output' in args and args[args.index('--network')+1]=='none',args
        event('hardware-probe')
elif args[0]=='exec':
    assert args[1:3]==['fixture-engine','/usr/bin/python3'],args
    if args[3]=='/opt/dsv411017/scripts/network_proof.py':
        assert args[4:6]==['--docker-network-mode','none'] and args[6]=='--output',args
        event('network-proof')
    elif args[3]=='/opt/dsv411017/scripts/runtime.py':
        assert args[4]=='capacity',args
        event('capacity-after-smoke' if '--require-replay' in args else 'capacity')
    elif args[3]=='/opt/dsv411017/tests/acceptance.py':
        assert '--unix-socket' in args and '/state/api.sock' in args and args[args.index('--suite')+1]=='smoke',args
        event('smoke')
    else:raise AssertionError('Unexpected production exec path: '+repr(args))
elif args[0]=='logs':event('engine-log')
else:raise AssertionError('Unexpected synthetic Docker operation: '+repr(args))
''')
        for name in ("kernels", "tp-ar", "context-graph"):
            (project / ("tests/" + name + ".sh")).write_text("printf '%s\\n' " + name + " >> \"" + str(project / "runtime/gates") + "\"\n")
        result = subprocess.run(["bash", str(project / "start.sh")], text=True, capture_output=True,
                                env={**os.environ, "TEST_GATE_DECISION": decision,
                                     "TEST_PYTHON": sys.executable, **(extra_environment or {})})
        return result, project

    def test_gate_reader_error_and_unknown_decision_prevent_engine_launch(self):
        for decision in ("error", "unrecognized"):
            with self.subTest(decision=decision):
                result, project = self.run_startup_fixture(decision)
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertFalse((project / "runtime/engine-started").exists())
                self.assertIn("protocol-compat", (project / "runtime/startup-events").read_text().splitlines())

    def test_required_gpu_gates_execute_and_explicit_skip_is_allowed(self):
        for decision in ("required", "skip"):
            with self.subTest(decision=decision):
                result, project = self.run_startup_fixture(decision)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertTrue((project / "runtime/engine-started").exists())
                gates = (project / "runtime/gates").read_text().splitlines()
                self.assertEqual(gates, ["kernels", "tp-ar", "context-graph"] if decision == "required" else ["kernels"])
                events = (project / "runtime/startup-events").read_text().splitlines()
                self.assertEqual(events, ["protocol-compat", "hardware-probe", "engine-started", "network-proof",
                                          "capacity", "smoke", "capacity-after-smoke", "frontend", "api-local",
                                          "api-docker", "engine-log"])
                self.assertEqual((project / "runtime/ready.signature").read_text(), "fixture-signature\n")

    def test_protocol_compatibility_failure_prevents_any_engine_launch(self):
        result, project = self.run_startup_fixture("required", {"TEST_PROTOCOL_STATUS": "17"})
        self.assertEqual(result.returncode, 17, result.stdout + result.stderr)
        self.assertFalse((project / "runtime/engine-started").exists())
        self.assertFalse((project / "runtime/gates").exists())
        self.assertFalse((project / "runtime/ready.signature").exists())

    def test_wrong_engine_network_mode_never_creates_ready_signature(self):
        result, project = self.run_startup_fixture("required", {"TEST_NETWORK_MODE": "bridge"})
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue((project / "runtime/engine-started").exists())
        self.assertFalse((project / "runtime/ready.signature").exists())
        self.assertNotIn("network-proof", (project / "runtime/startup-events").read_text().splitlines())

    def test_local_and_docker_access_failures_never_create_ready_signature(self):
        for key in ("TEST_LOCAL_ACCESS_STATUS", "TEST_DOCKER_ACCESS_STATUS"):
            with self.subTest(stage=key):
                result, project = self.run_startup_fixture(key, {key: "19", "TEST_GATE_DECISION": "required"})
                self.assertEqual(result.returncode, 19, result.stdout + result.stderr)
                self.assertTrue((project / "runtime/engine-started").exists())
                self.assertFalse((project / "runtime/ready.signature").exists())
                self.assertIn("api-local", (project / "runtime/startup-events").read_text().splitlines())


if __name__ == "__main__":
    unittest.main()
