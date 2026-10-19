"""CPU fixtures for the speed diagnostic protocol; no GPU result is created."""
import copy
import contextlib
import http.server
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import types
import unittest

import diagnose_speed as diag


def snapshot(generated=0, drafts=0, proposed=0, accepted=0, settled=True):
    return {"settled": settled, "totals": {
        "vllm:generation_tokens_total": generated,
        "vllm:spec_decode_num_drafts_total": drafts,
        "vllm:spec_decode_num_draft_tokens_total": proposed,
        "vllm:spec_decode_num_accepted_tokens_total": accepted}, "accepted_positions": {}}


class DiagnosticsTests(unittest.TestCase):
    def test_metrics_preserve_positions_across_engine_labels(self):
        parsed = diag.parse_metrics('\n'.join([
            'vllm:generation_tokens_total{engine="0"} 128',
            'vllm:generation_tokens_total{engine="1"} 256',
            'vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",position="1"} 12',
            'vllm:spec_decode_num_accepted_tokens_per_pos_total{position="1",engine="1"} 23',
            'vllm:spec_decode_num_accepted_tokens_per_pos_total{position="0"} 41',
        ]))
        self.assertEqual(parsed["totals"]["vllm:generation_tokens_total"], 384)
        self.assertEqual(parsed["accepted_positions"], {"0": 41, "1": 35})

    def test_clean_accounting_and_round_sensitivity(self):
        audit = diag.account(snapshot(), snapshot(1024, 321, 1605, 702), 1024)
        self.assertEqual(audit["status"], "CLEAN")
        self.assertAlmostEqual(audit["accepted_per_draft"], 702 / 321)
        stats = {"decode_tokens_per_second": 60, "decode_seconds": 17, "completion_tokens": 1024}
        analysis = diag.round_analysis(stats, audit, 60, 100)
        self.assertFalse(analysis["speculation_speedup_proven"])
        self.assertAlmostEqual(analysis["effective_tokens_per_draft_round_estimate"], 1 + 702 / 321)
        self.assertAlmostEqual(analysis["hypothetical_target"]["required_round_time_reduction_fraction"], .4)

    def test_busy_external_traffic_counter_reset_and_missing_activity(self):
        tests = [
            (snapshot(), snapshot(1025, 321, 1605, 702), 1024),
            (snapshot(), snapshot(1024, 321, 1605, 702, False), 1024),
            (snapshot(2000), snapshot(1024, 321, 1605, 702), 1024),
            (snapshot(), snapshot(1024), 1024),
        ]
        for before, after, expected in tests:
            with self.subTest(after=after):
                audit = diag.account(before, after, expected)
                self.assertEqual(audit["status"], "CONTAMINATED")
                self.assertEqual(diag.round_analysis({"decode_tokens_per_second": 60}, audit)["status"], "NOT_ATTRIBUTABLE")
        # A separately launched baseline may deliberately disable speculation.
        self.assertEqual(diag.account(snapshot(), snapshot(1024), 1024, expect_dspark=False)["status"], "CLEAN")

    def test_invalid_speculative_counters_and_position_mismatch(self):
        after = snapshot(1024, 10, 51, 45)
        self.assertEqual(diag.account(snapshot(), after, 1024)["status"], "CONTAMINATED")
        after = snapshot(1024, 321, 1605, 702)
        after["accepted_positions"] = {"0": 701}
        self.assertEqual(diag.account(snapshot(), after, 1024)["status"], "CONTAMINATED")

    def test_private_multimodal_request_is_copied_without_history_changes(self):
        client = diag.Client("http://127.0.0.1", "DeepSeek-Flash", 10)
        template = {"model": "DeepSeek-Flash", "messages": [{"role": "user", "content": [
            {"type": "text", "text": "Existing private history"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}]}],
            "tools": [{"type": "function", "function": {"name": "read_panel"}}],
            "temperature": .8, "stop": ["END"]}
        original = copy.deepcopy(template)
        body = diag.make_body(client, "request", "high", "sample", 0, "seed", 1024, template)
        self.assertEqual(template, original)
        self.assertEqual(body["messages"], template["messages"])
        self.assertEqual(body["tools"], template["tools"])
        self.assertEqual(body["chat_template_kwargs"], {"thinking": True, "reasoning_effort": "high"})
        self.assertEqual(body["temperature"], 0)
        self.assertNotIn("stop", body)
        self.assertTrue(body["return_token_ids"])

    def test_exact_first_batch_and_reasoning_inclusive_timing(self):
        result = {"token_batches": [{"seconds": 2, "count": 4}, {"seconds": 3, "count": 60}, {"seconds": 4, "count": 64}],
            "usage": {"prompt_tokens": 50, "completion_tokens": 128, "total_tokens": 178,
                      "completion_tokens_details": {"reasoning_tokens": 80}},
            "first_visible_seconds": 2.1, "elapsed_seconds": 4.2, "finish_reason": "length", "text": "fixture", "reasoning": "fixture"}
        stats = diag.timing(result, 128)
        self.assertEqual(stats["decode_measured_tokens"], 124)
        self.assertEqual(stats["decode_tokens_per_second"], 62)
        self.assertEqual(stats["reasoning_tokens"], 80)
        self.assertEqual(stats["ttft_seconds"], 2)
        self.assertEqual(stats["token_batches"], result["token_batches"])
        bad = copy.deepcopy(result)
        bad["token_batches"][-1]["count"] = 63
        with self.assertRaisesRegex(ValueError, "sum"):
            diag.timing(bad, 128)

    def test_graph_records_do_not_claim_replay_frequency(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            before = diag.inspect_graphs(path)
            (path / "capture.json").write_text(json.dumps({"kind": "graph-capture", "pid": 1, "manager_id": 2}))
            (path / "replay.json").write_text(json.dumps({"kind": "graph-replay", "pid": 1, "manager_id": 2}))
            first = diag.graph_delta(before, diag.inspect_graphs(path))
            self.assertEqual(first["new_first_shape_replay_records"], 1)
            after = diag.inspect_graphs(path)
            repeated = diag.graph_delta(after, diag.inspect_graphs(path))
            self.assertEqual(repeated["new_first_shape_replay_records"], 0)
            self.assertIn("cannot establish replay frequency", repeated["limit"])

    def test_real_local_http_fixture_retains_metrics_and_timing(self):
        state = {"generated": 0, "drafts": 0, "proposed": 0, "accepted": 0}

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_GET(self):
                mapping = {"generation_tokens": "generated", "spec_decode_num_drafts": "drafts",
                           "spec_decode_num_draft_tokens": "proposed", "spec_decode_num_accepted_tokens": "accepted"}
                body = "\n".join("vllm:%s_total %d" % (name, state[key]) for name, key in mapping.items())
                body += "\nvllm:num_requests_running 0\nvllm:num_requests_waiting 0\n"
                self.send_response(200)
                self.end_headers()
                self.wfile.write(body.encode())

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                self.assert_request = body
                for name, value in (("generated", 128), ("drafts", 40), ("proposed", 200), ("accepted", 87)):
                    state[name] += value
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for number, count in enumerate((4, 60, 63 if state.get("bad_tokens") else 64)):
                    event = {"id": "cpu-fixture", "model": diag.MODEL, "choices": [{"index": 0,
                        "delta": {"content": "CPU fixture"}, "token_ids": list(range(count)),
                        "finish_reason": "length" if number == 2 else None}]}
                    self.wfile.write(("data: " + json.dumps(event) + "\n\n").encode())
                    self.wfile.flush()
                    time.sleep(.01)
                usage = {"model": diag.MODEL, "choices": [], "usage": {"prompt_tokens": 42,
                    "completion_tokens": 128, "total_tokens": 170}}
                self.wfile.write(("data: " + json.dumps(usage) + "\n\ndata: [DONE]\n\n").encode())

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as folder:
                output = Path(folder) / "cpu-protocol-fixture.json"
                args = types.SimpleNamespace(output=output, api_key_file=None, base_url="http://127.0.0.1:%d" % server.server_port,
                    model=diag.MODEL, timeout=5, unix_socket=None, request_file=None, workloads=["code"], modes=["off"],
                    warmups=0, samples=1, output_tokens=128, seed="test", expected_draft_k=5, expect_dspark=True,
                    observations=None, runtime_report=None, settle_seconds=.1, baseline_tps=60, target_tps=100)
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(diag.run(args), 0)
                report = json.loads(output.read_text())
                self.assertEqual(report["status"], "PASS")
                row = report["samples"][0]
                self.assertEqual(row["timing"]["decode_measured_tokens"], 124)
                self.assertEqual(row["timing"]["prompt_tokens"], 42)
                self.assertEqual(row["accounting"]["status"], "CLEAN")
                self.assertEqual(row["analysis"]["status"], "ACCOUNTED_CLIENT_ESTIMATE")
                raw = Path(row["metrics_after"]["raw_metrics_path"])
                self.assertIn("vllm:generation_tokens_total 128", raw.read_text())
                self.assertNotIn("CPU fixture", output.read_text())
                state["bad_tokens"] = True
                args.output = Path(folder) / "cpu-rejected-token-fixture.json"
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(diag.run(args), 1)
                rejected = json.loads(args.output.read_text())
                self.assertEqual(rejected["status"], "FAIL")
                failed = rejected["samples"][0]
                self.assertFalse(failed["timing_valid"])
                self.assertEqual(failed["received_usage"]["completion_tokens"], 128)
                self.assertEqual(sum(batch["count"] for batch in failed["received_token_batches"]), 127)
                self.assertEqual(failed["accounting"]["status"], "CLEAN")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
