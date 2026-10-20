"""Synthetic transport and protocol fixtures; no GPU or real inference."""
import copy
import contextlib
import io
import sys
from unittest import mock
import http.server
import importlib.util
import json
from pathlib import Path
import socketserver
import tempfile
import threading
import unittest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('subagent_probe', ROOT / 'tests/subagent_diagnostic.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
TEXT = '你好，很高兴见到你！'


def message(ident='msg_a', text=TEXT):
    return {'type': 'message', 'id': ident, 'role': 'assistant', 'status': 'completed',
            'content': [{'type': 'output_text', 'text': text}]}


def final(items=None, status='completed'):
    return {'id': 'resp_fixture', 'model': probe.MODEL, 'status': status,
            'usage': {'input_tokens': 20, 'output_tokens': 10},
            'output': [message()] if items is None else items}


def events(final_id='msg_a', text=TEXT, status='completed', terminal='response.completed'):
    return [
        {'type': 'response.output_item.added', 'output_index': 0,
         'item': dict(message(text=''), status='in_progress')},
        {'type': 'response.output_text.delta', 'output_index': 0, 'item_id': 'msg_a', 'delta': text[:3]},
        {'type': 'response.output_text.delta', 'output_index': 0, 'item_id': 'msg_a', 'delta': text[3:]},
        {'type': 'response.output_item.done', 'output_index': 0, 'item': message(text=text)},
        {'type': terminal, 'response': final([message(final_id, text)], status)},
    ]


class StreamContractTests(unittest.TestCase):
    def run_events(self, values):
        audit = probe.StreamAudit()
        for value in values:
            audit.feed(value)
        return audit

    def test_valid_unicode_delta_and_final_identity(self):
        result = self.run_events(events()).contract()
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(result['stream_text']['text'], TEXT)
        self.assertEqual(bytes.fromhex(result['stream_text']['utf8_hex']).decode(), TEXT)

    def test_detects_native_final_id_regeneration_with_same_text(self):
        result = self.run_events(events(final_id='msg_different')).contract()
        self.assertEqual(result['status'], 'FAIL')
        self.assertIn('DONE_COMPLETED_ID_MISMATCH', {x['kind'] for x in result['issues']})
        self.assertEqual(result['stream_text']['text'], TEXT)

    def test_function_call_id_drift_is_detected(self):
        item = {'type': 'function_call', 'id': 'fc_a', 'call_id': 'call_a',
                'name': 'spawn_agent', 'arguments': '{}'}
        changed = dict(item, call_id='call_b')
        result = self.run_events([
            {'type': 'response.output_item.added', 'output_index': 0, 'item': item},
            {'type': 'response.output_item.done', 'output_index': 0, 'item': item},
            {'type': 'response.completed', 'response': final([changed])},
        ]).contract()
        self.assertIn('DONE_COMPLETED_CALL_ID_MISMATCH', {x['kind'] for x in result['issues']})

    def test_budget_limit_is_not_mislabeled_as_successful_completion(self):
        result = self.run_events(events(status='incomplete')).contract()
        self.assertIn('TERMINAL_EVENT_STATUS_MISMATCH', {x['kind'] for x in result['issues']})
        good = self.run_events(events(status='incomplete', terminal='response.incomplete')).contract()
        self.assertEqual(good['status'], 'PASS')

    def test_no_terminal_is_detected(self):
        result = self.run_events(events()[:-1]).contract()
        self.assertIn('NO_TERMINAL_RESPONSE_EVENT', {x['kind'] for x in result['issues']})

    def test_reasoning_text_is_represented_only_by_metadata(self):
        secret = 'Synthetic chain of thought, never place in diagnostic output.'
        item = {'type': 'reasoning', 'id': 'rs_a',
                'summary': [{'type': 'summary_text', 'text': secret}]}
        audit = self.run_events([
            {'type': 'response.output_item.added', 'output_index': 0, 'item': item},
            {'type': 'response.output_item.done', 'output_index': 0, 'item': item},
            {'type': 'response.completed', 'response': final([item])},
        ])
        self.assertNotIn(secret, json.dumps(audit.events))
        self.assertEqual(audit.contract()['status'], 'PASS')

    def test_quote_only_and_envelope_echo_are_not_accepted_as_greetings(self):
        self.assertEqual(probe.answer_check(final([message(text='“')]))['status'], 'ONLY_PUNCTUATION')
        raw = json.dumps(probe.TASK, ensure_ascii=False)
        self.assertEqual(probe.answer_check(final([message(text=raw)]))['status'], 'TASK_ENVELOPE_ECHO')
        self.assertEqual(probe.answer_check(final([message(text='')], 'incomplete'))['status'],
                         'INCONCLUSIVE_OUTPUT_LIMIT')


class UnixHTTPTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'api.sock'
        self.observed = []
        self.final_id = "msg_a"
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'

            def log_message(self, *args):
                pass

            def do_POST(self):
                data = self.rfile.read(int(self.headers['Content-Length']))
                request = json.loads(data)
                outer.observed.append(request)
                self.send_response(200)
                if request['stream']:
                    self.send_header('Content-Type', 'text/event-stream')
                    self.send_header('Transfer-Encoding', 'chunked')
                    self.end_headers()
                    for event in events(final_id=outer.final_id):
                        raw = ('data: ' + json.dumps(event, ensure_ascii=False) + '\n\n').encode()
                        # Split even multibyte UTF-8 across HTTP chunks.
                        for pos in range(0, len(raw), 7):
                            block = raw[pos:pos + 7]
                            self.wfile.write(('%x\r\n' % len(block)).encode() + block + b'\r\n')
                    self.wfile.write(b'0\r\n\r\n')
                    self.wfile.flush()
                else:
                    raw = json.dumps(final(), ensure_ascii=False).encode()
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Content-Length', str(len(raw)))
                    self.end_headers()
                    self.wfile.write(raw)

        class Server(socketserver.UnixStreamServer):
            allow_reuse_address = True

        self.server = Server(str(self.path), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    def test_strict_live_gate_rejects_drifting_stream_ids(self):
        self.final_id = "msg_different"
        output = Path(self.temp.name) / "report.json"
        argv = ["probe", "--unix-socket", str(self.path), "--output", str(output), "--require-pass"]
        with mock.patch.object(sys, "argv", argv), mock.patch.object(probe, "json_get", side_effect=[{}, {"data": [{"id": probe.MODEL}]}]), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(probe.main(), 1)
        report = json.loads(output.read_text())
        self.assertEqual(len(report["checks"]), 8)
        self.assertEqual(report["status"], "ISSUES_DETECTED")
        self.assertEqual(sum(row["status"] == "ISSUES_DETECTED" for row in report["checks"]), 4)
        self.assertTrue(all(row["answer_check"]["status"] == "PASS" for row in report["checks"]))

    def test_real_chunked_http_sse_transport_preserves_unicode(self):
        result = probe.one_case(self.path, 5, True, True, 'medium')
        self.assertEqual(result['status'], 'PASS', result)
        self.assertEqual(result['response']['output_text']['text'], TEXT)
        self.assertEqual(result['stream_contract']['status'], 'PASS')
        self.assertEqual(self.observed[0], probe.make_request(True, True, 'medium'))

    def test_real_json_http_transport_and_controlled_payload(self):
        result = probe.one_case(self.path, 5, False, False, 'none')
        self.assertEqual(result['status'], 'PASS', result)
        self.assertEqual(result['answer_check']['text']['text'], TEXT)
        self.assertFalse(self.observed[0]['store'])
        self.assertEqual(self.observed[0]['tools'], [])


if __name__ == '__main__':
    unittest.main(verbosity=2)
