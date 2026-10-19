"""CPU fixtures for the V2 acceptance harness; never GPU/model validation."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
import io


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / 'tests/r39_alignment.py'
SPEC = importlib.util.spec_from_file_location('r39_alignment_v2', MODULE)
alignment = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(alignment)


def usage_fixture(protocol, cached=1024):
    if protocol == 'chat':
        return {'prompt_tokens': 2048, 'completion_tokens': 8, 'total_tokens': 2056,
                'prompt_tokens_details': {'cached_tokens': cached}}
    if protocol == 'responses':
        return {'input_tokens': 2048, 'output_tokens': 8, 'total_tokens': 2056,
                'input_tokens_details': {'cached_tokens': cached}}
    return {'input_tokens': 2048 - cached, 'output_tokens': 8, 'cache_read_input_tokens': cached,
            'cache_creation_input_tokens': 0}


def response_fixture(protocol, expected, cached=1024, stream=False):
    """Synthetic protocol bytes only, deliberately independent of the GPU."""
    usage = usage_fixture(protocol, cached)
    text = json.dumps(expected)
    if protocol == 'chat':
        if stream:
            return [{'id': 'fixture-chat', 'model': alignment.MODEL,
                     'choices': [{'index': 0, 'delta': {'content': text}, 'finish_reason': None}]},
                    {'id': 'fixture-chat', 'model': alignment.MODEL,
                     'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}], 'usage': usage},
                    {'type': 'done'}]
        return {'id': 'fixture-chat', 'model': alignment.MODEL,
                'choices': [{'message': {'content': text}, 'finish_reason': 'stop'}], 'usage': usage}
    if protocol == 'responses':
        response = {'id': 'fixture-response', 'model': alignment.MODEL, 'status': 'completed',
                    'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': text}]}], 'usage': usage}
        return [{'type': 'response.completed', 'response': response}] if stream else response
    if stream:
        return [{'type': 'message_start', 'message': {'id': 'fixture-message', 'model': alignment.MODEL,
                                                    'usage': usage}},
                {'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': text}},
                {'type': 'message_delta', 'delta': {'stop_reason': 'end_turn'}, 'usage': {'output_tokens': 8}},
                {'type': 'message_stop'}]
    return {'id': 'fixture-message', 'model': alignment.MODEL, 'stop_reason': 'end_turn',
            'content': [{'type': 'text', 'text': text}], 'usage': usage}


def sse_fixture(events):
    parts = []
    for event in events:
        if event.get('type') == 'done':
            parts.append('data: [DONE]\n\n')
        else:
            prefix = ('event: ' + event['type'] + '\n') if event.get('type') else ''
            parts.append(prefix + 'data: ' + json.dumps(event) + '\n\n')
    return ''.join(parts).encode()


def successful_report_fixture():
    """Complete 37-case workflow fixture. Every metric is synthetic CPU data."""
    report = {'schema': 1, 'suite': 'r39-alignment', 'model': alignment.MODEL,
              'context': 524288, 'concurrency': 32, 'status': 'PASS',
              'started_at': '2026-10-19T00:00:00+00:00', 'finished_at': '2026-10-19T00:00:01+00:00',
              'newapi_billing_end_to_end': 'NOT_RUN', 'gpu_executed': False,
              'gpu_validation': 'NOT_INDEPENDENTLY_CHECKED', 'target_gpu_required': True,
              'fixture_only': True, 'execution_mode': 'CPU_SYNTHETIC_WORKFLOW_FIXTURE',
              'required_cases': list(alignment.REQUIRED_CASES), 'cases': []}
    for name, kind, protocol, fmt, content in alignment.CASE_SPECS:
        row = {'name': name, 'kind': kind, 'protocol': protocol, 'format': fmt,
               'status': 'PASS', 'errors': [], 'cold_cached_tokens': None, 'hot_cached_tokens': None,
               'started_at': report['started_at'], 'finished_at': report['finished_at']}
        if kind == 'cache':
            expected = {'b_value': 60 if content == 'changed-image' else 65} if content in ('image', 'changed-image', 'long-image') else {'check': 73019}
            cold_usage, hot_usage = usage_fixture(protocol, 0), usage_fixture(protocol, 1024)
            row.update(cold_cached_tokens=0, hot_cached_tokens=1024,
                       cold_usage=cold_usage, hot_usage=hot_usage, hot_input_total=2048,
                       hot_cache_hit_ratio=0.5, min_hot_cached_tokens=512 if content == 'dynamic-tail-system' else 1)
            requests = [{'phase': 'cold-json', 'format': 'json', 'cached_tokens': 0,
                         'raw_usage': cold_usage, 'input_total': 2048},
                        {'phase': 'hot-' + fmt, 'format': fmt, 'cached_tokens': 1024,
                         'raw_usage': hot_usage, 'input_total': 2048}]
            if content == 'changed-image':
                row.update(original_image_cached_tokens=0, changed_image_cached_tokens=0,
                           changed_image_input_total=2048)
                requests.insert(1, {'phase': 'changed-image-json', 'format': 'json', 'cached_tokens': 0,
                                    'raw_usage': cold_usage, 'input_total': 2048})
            if content == 'dynamic-tail-system' and fmt == 'sse':
                requests.insert(1, {'phase': 'followup-json', 'format': 'json', 'cached_tokens': 1024,
                                    'raw_usage': hot_usage, 'input_total': 2048})
            if content != 'dynamic-tail-system':
                row.update(multi_turn_usage=hot_usage, multi_turn_cached_tokens=1024,
                           multi_turn_input_total=2048, multi_turn_format=fmt)
                requests.append({'phase': 'multi-turn-' + fmt, 'format': fmt, 'cached_tokens': 1024,
                                 'raw_usage': hot_usage, 'input_total': 2048})
        else:
            if content.startswith('ordered-'):
                count = int(content.rsplit('-', 1)[1])
                expected = {'numbers': alignment.NUMBERS[:count]}
                row['image_count'] = count
            else:
                codex = content == 'codex-view-image'
                expected = {'call_a': [2], 'call_b': [1]} if codex else {'call_a': [2, 1], 'call_b': [3, 2]}
                row.update(image_count=2 if codex else 4, tool_call_id_binding=True,
                           tool_call_order=['call_a', 'call_b'], tool_result_arrival_order=['call_b', 'call_a'])
            row.update(actual=expected, usage=usage_fixture(protocol))
            requests = [{'phase': 'vision-' + fmt, 'format': fmt, 'cached_tokens': 1024,
                         'raw_usage': usage_fixture(protocol), 'input_total': 2048}]
        row['expected'] = expected
        row['requests'] = [dict(request, case=name, protocol=protocol, status='PASS', expected=expected,
                                actual=expected, usage=request['raw_usage'], seconds=0.01,
                                response_ids=['CPU-FIXTURE-ONLY'], evidence_file='CPU-FIXTURE-ONLY',
                                raw_response_file='CPU-FIXTURE-ONLY', fixture_only=True) for request in requests]
        report['cases'].append(row)
    report.update(passed_cases=len(report['cases']), failed_cases=0, not_run_cases=0,
                  inference_requests_executed=sum(len(row['requests']) for row in report['cases']))
    return report


class AlignmentTests(unittest.TestCase):
    def test_required_cases_and_report_fixture(self):
        self.assertEqual(len(alignment.REQUIRED_CASES), 37)
        self.assertEqual(len(set(alignment.REQUIRED_CASES)), 37)
        report = successful_report_fixture()
        self.assertEqual(tuple(row['name'] for row in report['cases']), alignment.REQUIRED_CASES)
        self.assertTrue(report['fixture_only'])
        self.assertFalse(report['gpu_executed'])
        self.assertEqual(report['newapi_billing_end_to_end'], 'NOT_RUN')

    def test_all_protocols_json_and_sse(self):
        for protocol in alignment.PATHS:
            for stream in (False, True):
                with self.subTest(protocol=protocol, stream=stream):
                    raw = response_fixture(protocol, {'check': 73019}, stream=stream)
                    if stream:
                        raw = alignment.parse_sse(sse_fixture(raw))
                    normalized = alignment.normalize(protocol, {'stream': stream}, raw)
                    self.assertEqual(normalized['actual'], {'check': 73019})
                    self.assertEqual(normalized['cached_tokens'], 1024)
                    self.assertEqual(normalized['input_total'], 2048)

    def test_messages_usage_merges_without_double_counting(self):
        events = response_fixture('messages', {'check': 73019}, stream=True)
        events.insert(-1, {'type': 'message_delta', 'delta': {},
                           'usage': {'input_tokens': 1024, 'cache_read_input_tokens': 1024, 'output_tokens': 8}})
        value = alignment.normalize('messages', {'stream': True}, events)
        self.assertEqual(value['input_total'], 2048)
        self.assertEqual(value['usage']['cache_creation_input_tokens'], 0)
        # Missing creation count remains missing; no invented cache write.
        del events[0]['message']['usage']['cache_creation_input_tokens']
        value = alignment.normalize('messages', {'stream': True}, events)
        self.assertNotIn('cache_creation_input_tokens', value['usage'])

    def test_chat_implicit_usage_requires_final_usage(self):
        events = response_fixture('chat', {'check': 73019}, stream=True)
        self.assertEqual(alignment.normalize('chat', {'stream': True}, events)['cached_tokens'], 1024)
        del events[1]['usage']
        with self.assertRaisesRegex(ValueError, 'Missing final usage'):
            alignment.normalize('chat', {'stream': True}, events)

    def test_rejects_truncated_failed_or_wrong_model_stream(self):
        raw = sse_fixture(response_fixture('chat', {'check': 73019}, stream=True))
        with self.assertRaisesRegex(ValueError, 'Truncated'):
            alignment.parse_sse(raw[:-1])
        with self.assertRaisesRegex(ValueError, 'SSE error'):
            alignment.parse_sse(b'event: response.failed\ndata: {"type":"response.failed"}\n\n')
        with self.assertRaisesRegex(ValueError, 'event/type'):
            alignment.parse_sse(b'event: message_start\ndata: {"type":"message_stop"}\n\n')
        events = response_fixture('chat', {'check': 73019}, stream=True)
        events[0]['model'] = 'wrong-model'
        with self.assertRaisesRegex(ValueError, 'model mismatch'):
            alignment.normalize('chat', {'stream': True}, events)
        for protocol in alignment.PATHS:
            events = response_fixture(protocol, {'check': 73019}, stream=True)
            events = events[:-1]
            with self.subTest(protocol=protocol), self.assertRaises((ValueError, IndexError)):
                alignment.normalize(protocol, {'stream': True}, events)

    def test_usage_rejects_fabricated_invalid_counts(self):
        for protocol in alignment.PATHS:
            with self.subTest(protocol=protocol):
                usage = usage_fixture(protocol)
                if protocol == 'messages':
                    usage['cache_read_input_tokens'] = True
                else:
                    usage['prompt_tokens_details' if protocol == 'chat' else 'input_tokens_details']['cached_tokens'] = True
                with self.assertRaises(ValueError):
                    alignment.cache_usage(protocol, usage)
        usage = usage_fixture('chat', 3000)
        with self.assertRaises(ValueError):
            alignment.cache_usage('chat', usage)

    def test_strict_answer_types(self):
        self.assertFalse(alignment.strict_equal({'numbers': [True]}, {'numbers': [1]}))
        self.assertFalse(alignment.strict_equal({'check': 73019.0}, {'check': 73019}))
        self.assertFalse(alignment.strict_equal({'check': 73019, 'extra': 1}, {'check': 73019}))

    def test_hit_requires_real_increase_and_dynamic_minimum(self):
        cold = {'status': 'PASS', 'cached_tokens': 0}
        hot = {'status': 'PASS', 'cached_tokens': 1024}
        alignment.validate_hit(cold, hot, 512)
        with self.assertRaises(ValueError):
            alignment.validate_hit(hot, hot)
        with self.assertRaises(ValueError):
            alignment.validate_hit(cold, {'status': 'PASS', 'cached_tokens': 511}, 512)

    def test_asset_provenance_and_hashes(self):
        manifest = alignment.verify_assets()
        self.assertEqual(len(manifest['files']), 5)
        self.assertEqual(len(manifest['reference_sources']), 4)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / 'provenance.json').write_text(json.dumps(manifest))
            for row in manifest['files']:
                target = directory / row['path']
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((alignment.ASSETS / row['path']).read_bytes())
            alignment.verify_assets(directory)
            (directory / manifest['files'][0]['path']).write_bytes(b'changed')
            with self.assertRaises(ValueError):
                alignment.verify_assets(directory)

    def test_ordered_five_eight_and_tool_call_binding(self):
        for protocol in alignment.PATHS:
            for count in (5, 8):
                body, expected = alignment.ordered_body(protocol, count, count == 8)
                self.assertEqual(expected, {'numbers': alignment.NUMBERS[:count]})
                content = body['input' if protocol == 'responses' else 'messages'][0]['content']
                self.assertEqual(len([item for item in content if item['type'] in ('input_image', 'image_url', 'image')]), count)
            body, expected = alignment.reversed_tools_body(protocol)
            self.assertEqual(expected, {'call_a': [2, 1], 'call_b': [3, 2]})
            if protocol == 'responses':
                results = [item for item in body['input'] if item.get('type') == 'function_call_output']
                ids = [item['call_id'] for item in results]
            elif protocol == 'chat':
                ids = [item['tool_call_id'] for item in body['messages'] if item['role'] == 'tool']
            else:
                ids = [item['tool_use_id'] for item in body['messages'][2]['content'] if item['type'] == 'tool_result']
            self.assertEqual(ids, ['call_b', 'call_a'])
        body, expected = alignment.reversed_tools_body('responses', codex=True)
        for result in body['input'][3:5]:
            self.assertEqual(len(result['output']), 1)
            image = result['output'][0]
            self.assertEqual(image['detail'], 'high')
            self.assertTrue(image['image_url'].startswith('data:application/octet-stream;base64,'))

    def test_cache_changed_image_and_tag_position(self):
        for protocol in alignment.PATHS:
            original = alignment.cache_body(protocol, 'fixed-fixture-tag', 'chart_before.png')
            changed = alignment.cache_body(protocol, 'fixed-fixture-tag', 'chart_after.png')
            content = original['input' if protocol == 'responses' else 'messages'][0]['content']
            altered = changed['input' if protocol == 'responses' else 'messages'][0]['content']
            self.assertEqual(content[0], altered[0])
            self.assertNotEqual(content[1], altered[1])
            self.assertEqual(content[2], altered[2])
            self.assertIn('fixed-fixture-tag', content[0]['text'])
            self.assertEqual(original['model'], 'DeepSeek-Flash')

    def test_real_uds_transport_fixture_preserves_usage_on_bad_answer(self):
        # Real local Unix-socket I/O, fixture bytes; no GPU or model invocation.
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            sock = directory / 'fixture.sock'
            server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            server.bind(str(sock)); server.listen(1)
            def handler():
                connection, _ = server.accept()
                with connection:
                    incoming = b''
                    while b'\r\n\r\n' not in incoming:
                        incoming += connection.recv(65536)
                    headers, content = incoming.split(b'\r\n\r\n', 1)
                    length = next(int(line.split(b':', 1)[1]) for line in headers.split(b'\r\n')
                                  if line.lower().startswith(b'content-length:'))
                    while len(content) < length:
                        content += connection.recv(65536)
                    raw = json.dumps(response_fixture('chat', {'wrong': True})).encode()
                    connection.sendall(b'HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: ' +
                                       str(len(raw)).encode() + b'\r\nConnection: close\r\n\r\n' + raw)
                server.close()
            thread = threading.Thread(target=handler, daemon=True)
            thread.start()
            transport = alignment.Transport(sock, 10, directory / 'report.json')
            row = transport.call('chat', alignment.cache_body('chat', 'CPU-ONLY'), 'fixture-transport', 'json', {'check': 73019})
            thread.join(timeout=10)
            self.assertEqual(row['status'], 'FAIL')
            self.assertEqual(row['raw_usage']['prompt_tokens_details']['cached_tokens'], 1024)
            self.assertTrue(Path(row['raw_response_file']).is_file())
            evidence = json.loads(Path(row['evidence_file']).read_text())
            self.assertEqual(evidence['actual'], {'wrong': True})
            self.assertIn('request', evidence)
            self.assertEqual(json.loads(transport.trace.read_text())['status'], 'FAIL')

    def test_endpoint_failure_still_saves_all_cases_and_final_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            args = argparse.Namespace(unix_socket=directory / 'absent.sock', output=directory / 'report.json',
                                      model=alignment.MODEL, timeout=1)
            suite = alignment.Suite(args)
            self.assertEqual(suite.run(), 1)
            report = json.loads(args.output.read_text())
            self.assertEqual(report['status'], 'FAIL')
            self.assertEqual(len(report['cases']), 37)
            self.assertEqual(report['passed_cases'], 0)
            self.assertIsNotNone(report['finished_at'])
            self.assertNotIn('gpu_executed', report)
            self.assertEqual(report['gpu_validation'], 'NOT_INDEPENDENTLY_CHECKED')

    def test_full_case_workflow_fixture_and_failure_continuation(self):
        class FixtureTransport:
            """Returns explicitly synthetic CPU fixtures, never inference."""
            def __init__(self, fail_case=None):
                self.fail_case = fail_case
                self.bodies = []

            def call(self, protocol, body, name, phase, expected=None, path=None):
                if body is None:
                    return {'status': 'PASS', 'value': {'data': [
                        {'id': 'DeepSeek-Flash', 'max_model_len': 524288}]}, 'fixture_only': True}
                self.bodies.append((name, phase, copy.deepcopy(body)))
                cached = 0 if phase in ('cold-json', 'changed-image-json') else 1024
                result = {'status': 'PASS', 'raw_usage': usage_fixture(protocol, cached),
                          'cached_tokens': cached, 'input_total': 2048, 'actual': expected,
                          'expected': expected, 'fixture_only': True, 'seconds': 0.01}
                if name == self.fail_case:
                    result.update(status='FAIL', error='Deliberate CPU fixture failure')
                return result
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for failing in (False, True):
                args = argparse.Namespace(unix_socket=directory / 'never-used.sock',
                                          output=directory / ('fail.json' if failing else 'pass.json'),
                                          model=alignment.MODEL, timeout=1)
                suite = alignment.Suite(args)
                suite.transport = FixtureTransport('vision-chat-ordered-5-json' if failing else None)
                with redirect_stdout(io.StringIO()):
                    code = suite.run()
                self.assertEqual(code, 1 if failing else 0)
                self.assertEqual(suite.report['passed_cases'], 36 if failing else 37)
                self.assertEqual(suite.report['not_run_cases'], 0)
                self.assertEqual(suite.report['cases'][-1]['status'], 'PASS')
                self.assertNotIn('gpu_executed', suite.report)
                for name, phase, body in suite.transport.bodies:
                    if 'implicit-usage' in name:
                        self.assertNotIn('stream_options', body)
                    if 'dynamic-tail-system' in name:
                        suffix = body['messages'][-1]
                        self.assertEqual(suffix['role'], 'system')
                        self.assertIn('Remaining audit budget:', suffix['content'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
