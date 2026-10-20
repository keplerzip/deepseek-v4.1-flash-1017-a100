"""Controlled Responses probes through the existing engine Unix socket.

No client conversations, credentials, model files or server configuration are
read. This is a leaf-request diagnostic, not a Codex subagent implementation.
"""
import argparse
import datetime
import hashlib
import http.client
import json
from pathlib import Path
import re
import socket
import time


MODEL = 'DeepSeek-Flash'
PROMPT = '请直接用一句简短、友好的中文向用户问好。不使用工具，不输出JSON、代码块、XML或解释。只输出一句问候。'
TASK = {
    'author': '/root', 'recipient': '/root/diagnostic_greeter',
    'content': 'Message Type: NEW_TASK\nTask name: /root/diagnostic_greeter\n'
               'Sender: /root\nPayload:\n' + PROMPT,
}
MAX_EVENT = 2 * 1024 * 1024


def digest(data):
    return hashlib.sha256(data).hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.chmod(0o600)
    temp.replace(path)


class UnixHTTP(http.client.HTTPConnection):
    def __init__(self, path, timeout):
        super().__init__('localhost', timeout=timeout)
        self.path = path

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(str(self.path))


def text_from_item(item):
    if item.get('type') != 'message':
        return ''
    return ''.join(part.get('text', '') for part in item.get('content', [])
                   if isinstance(part, dict) and part.get('type') == 'output_text')


def text_evidence(text):
    raw = text.encode('utf-8')
    return {'text': text, 'repr': repr(text), 'characters': len(text),
            'utf8_hex': raw.hex(), 'sha256': digest(raw)}


def item_metadata(item):
    if not isinstance(item, dict):
        raise ValueError('Response output item must be an object')
    row = {key: item[key] for key in ('type', 'id', 'call_id', 'name', 'namespace',
                                    'role', 'phase', 'status', 'agent') if key in item}
    text = text_from_item(item)
    if item.get('type') == 'message':
        row['output_text'] = text_evidence(text)
    if item.get('type') == 'reasoning':
        # Do not copy chain-of-thought into the user-facing diagnostic.
        summary = json.dumps(item.get('summary', []), ensure_ascii=False).encode()
        row['summary_bytes'] = len(summary)
        row['summary_sha256'] = digest(summary)
    return row


def response_metadata(response):
    if not isinstance(response, dict):
        raise ValueError('Response must be a JSON object')
    row = {key: response[key] for key in ('id', 'model', 'status', 'error',
                                        'incomplete_details', 'usage') if key in response}
    items = response.get('output', [])
    row['output'] = [item_metadata(item) for item in items if isinstance(item, dict)]
    row['output_text'] = text_evidence(''.join(text_from_item(item) for item in items
                                             if isinstance(item, dict)))
    return row


class StreamAudit:
    """Observe real SSE metadata without repairing or hiding inconsistencies."""
    def __init__(self):
        self.events = []
        self.added = {}
        self.done = {}
        self.deltas = []
        self.final = None
        self.final_type = None
        self.errors = []
        self.wire_hash = hashlib.sha256()
        self.wire_bytes = 0
        self.decode_errors = []
        self.text_done = {}

    def feed(self, event):
        kind = event.get('type', '')
        row = {key: event[key] for key in ('type', 'sequence_number', 'output_index',
                                         'content_index', 'item_id', 'call_id', 'agent') if key in event}
        index = event.get('output_index')
        if kind in ('response.output_item.added', 'response.output_item.done'):
            item = event.get('item', {})
            row['item'] = item_metadata(item)
            if kind.endswith('.added'):
                self.added[index] = item
            else:
                self.done[index] = item
        elif kind == 'response.output_text.delta':
            value = event.get('delta', '')
            self.deltas.append((index, event.get('item_id'), value))
            row['delta'] = text_evidence(value)
        elif kind == 'response.output_text.done':
            self.text_done[index] = event.get('text', '')
            row['text'] = text_evidence(event.get('text', ''))
        elif kind in ('response.completed', 'response.incomplete', 'response.failed'):
            self.final = event.get('response', {})
            self.final_type = kind
            row['response'] = response_metadata(self.final)
        elif kind == 'error':
            row['error'] = {key: event.get(key) for key in ('code', 'message', 'param')}
            self.errors.append(row['error'])
        self.events.append(row)

    def read(self, response):
        data_lines = []
        while True:
            raw = response.readline(MAX_EVENT + 1)
            if not raw:
                break
            self.wire_hash.update(raw)
            self.wire_bytes += len(raw)
            if len(raw) > MAX_EVENT:
                raise ValueError('SSE line exceeds the bounded diagnostic limit')
            # Strict decoding makes encoding faults visible instead of replacing
            # malformed bytes with a character that the client never received.
            line = raw.decode('utf-8').rstrip('\r\n')
            if not line:
                self.dispatch(data_lines)
                data_lines = []
            elif line.startswith('data:'):
                data_lines.append(line[5:].lstrip(' '))
        self.dispatch(data_lines)

    def dispatch(self, data_lines):
        if not data_lines:
            return
        value = '\n'.join(data_lines)
        if value == '[DONE]':
            return
        event = json.loads(value)
        if not isinstance(event, dict):
            raise ValueError('SSE data must be a JSON object')
        self.feed(event)

    def contract(self):
        issues = []
        for index, item in self.done.items():
            start = self.added.get(index)
            if start is None:
                issues.append({'kind': 'DONE_WITHOUT_ADDED', 'index': index})
            elif start.get('id') != item.get('id'):
                issues.append({'kind': 'ADDED_DONE_ITEM_ID_MISMATCH', 'index': index,
                               'added': start.get('id'), 'done': item.get('id')})
            pieces = [(ident, text) for idx, ident, text in self.deltas if idx == index]
            for ident, _ in pieces:
                if ident != item.get('id'):
                    issues.append({'kind': 'DELTA_ITEM_ID_MISMATCH', 'index': index,
                                   'delta': ident, 'done': item.get('id')})
            if item.get('type') == 'message' and pieces:
                if ''.join(text for _, text in pieces) != text_from_item(item):
                    issues.append({'kind': 'DELTA_DONE_TEXT_MISMATCH', 'index': index})
        if self.final is None:
            issues.append({'kind': 'NO_TERMINAL_RESPONSE_EVENT'})
        else:
            output = self.final.get('output', [])
            expected_terminal = {'completed': 'response.completed',
                                 'incomplete': 'response.incomplete',
                                 'failed': 'response.failed'}.get(self.final.get('status'))
            if expected_terminal and self.final_type != expected_terminal:
                issues.append({'kind': 'TERMINAL_EVENT_STATUS_MISMATCH',
                               'event': self.final_type, 'response_status': self.final.get('status')})
            if len(output) != len(self.done):
                issues.append({'kind': 'DONE_COMPLETED_ITEM_COUNT_MISMATCH',
                               'done_count': len(self.done), 'completed_count': len(output)})
            for index, item in self.done.items():
                if type(index) is not int or index >= len(output):
                    continue
                final_item = output[index]
                for key in ('id', 'type', 'call_id'):
                    if item.get(key) != final_item.get(key):
                        issues.append({'kind': 'DONE_COMPLETED_' + key.upper() + '_MISMATCH',
                                       'index': index, 'done': item.get(key), 'completed': final_item.get(key)})
                if text_from_item(item) != text_from_item(final_item):
                    issues.append({'kind': 'DONE_COMPLETED_TEXT_MISMATCH', 'index': index})
        if self.errors:
            issues.append({'kind': 'SSE_ERROR_EVENT', 'errors': self.errors})
        return {'status': 'FAIL' if issues else 'PASS', 'issues': issues,
                'event_count': len(self.events), 'terminal_event': self.final_type,
                'stream_text': text_evidence(''.join(value for _, _, value in self.deltas)),
                'wire_bytes': self.wire_bytes, 'wire_sha256': self.wire_hash.hexdigest()}


def answer_check(response):
    text = ''.join(text_from_item(item) for item in response.get('output', [])
                   if isinstance(item, dict))
    stripped = text.strip()
    if response.get('status') == 'incomplete':
        return {'status': 'INCONCLUSIVE_OUTPUT_LIMIT', 'text': text_evidence(text)}
    if not stripped:
        status = 'MISSING_FINAL_TEXT'
    elif stripped in {'"', "'", '“', '”', '「', '」', '``', '```'}:
        status = 'ONLY_PUNCTUATION'
    elif ('"author"' in text and '"recipient"' in text) or 'Message Type: NEW_TASK' in text:
        status = 'TASK_ENVELOPE_ECHO'
    elif not re.search('[\u4e00-\u9fff]', text):
        status = 'NO_CHINESE_TEXT'
    else:
        status = 'PASS'
    return {'status': status, 'text': text_evidence(text)}


def make_request(envelope, stream, effort):
    content = json.dumps(TASK, ensure_ascii=False) if envelope else PROMPT
    return {'model': MODEL, 'input': [{'role': 'user', 'content': content}],
            'stream': stream, 'store': False, 'reasoning': {'effort': effort},
            'max_output_tokens': 1024 if effort != 'none' else 128, 'tools': []}


def json_get(path, socket_path, timeout):
    conn = UnixHTTP(socket_path, timeout)
    try:
        conn.request('GET', path)
        response = conn.getresponse()
        body = response.read(MAX_EVENT)
        if response.status != 200:
            raise ValueError(path + ' returned HTTP ' + str(response.status))
        return json.loads(body) if body else {}
    finally:
        conn.close()


def one_case(socket_path, timeout, envelope, stream, effort):
    body = make_request(envelope, stream, effort)
    raw = json.dumps(body, ensure_ascii=False).encode()
    row = {'name': ('envelope' if envelope else 'plain') + '-' + effort + ('-sse' if stream else '-json'),
           'request': body, 'request_sha256': digest(raw), 'controlled_test': True,
           'status': 'REQUEST_FAILED', 'real_inference_executed': False}
    conn = UnixHTTP(socket_path, timeout)
    started = time.monotonic()
    audit = None
    try:
        conn.request('POST', '/v1/responses', body=raw, headers={'Content-Type': 'application/json'})
        response = conn.getresponse()
        row['http_status'] = response.status
        row['content_type'] = response.getheader('Content-Type')
        if response.status != 200:
            error = response.read(MAX_EVENT)
            row['error_body'] = error.decode('utf-8', errors='replace')[:6000]
            return row
        if stream:
            if 'text/event-stream' not in (row['content_type'] or ''):
                raise ValueError('Streaming request did not return text/event-stream')
            audit = StreamAudit()
            audit.read(response)
            row['events'] = audit.events
            row['stream_contract'] = audit.contract()
            if audit.final is None:
                row['status'] = 'NO_FINAL_RESPONSE'
                return row
            final = audit.final
        else:
            data = response.read(MAX_EVENT + 1)
            if len(data) > MAX_EVENT:
                raise ValueError('Response exceeds the bounded diagnostic limit')
            row['body_sha256'] = digest(data)
            row['body_bytes'] = len(data)
            try:
                final = json.loads(data)
            except ValueError:
                row['invalid_json_preview'] = repr(data[:2048])
                raise
        row['response'] = response_metadata(final)
        row['answer_check'] = answer_check(final)
        usage = final.get('usage') or {}
        row['real_inference_executed'] = type(usage.get('output_tokens')) is int and usage['output_tokens'] > 0
        issues = (row.get('stream_contract', {}).get('status') == 'FAIL'
                  or row['answer_check']['status'] != 'PASS'
                  or bool(final.get('error')) or final.get('model') != MODEL)
        row['status'] = 'ISSUES_DETECTED' if issues else 'PASS'
    except (OSError, ValueError, TypeError, http.client.HTTPException) as error:
        row['error'] = type(error).__name__ + ': ' + str(error)
    finally:
        if audit is not None:
            row['events'] = audit.events
            row['stream_contract'] = audit.contract()
        conn.close()
        row['elapsed_seconds'] = time.monotonic() - started
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--unix-socket', type=Path, default=Path('/state/api.sock'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timeout', type=float, default=120)
    parser.add_argument('--require-pass', action='store_true', help='Exit nonzero when any controlled check fails')
    args = parser.parse_args()
    if not 1 <= args.timeout <= 600:
        parser.error('--timeout must be within 1..600 seconds')
    report = {'schema': 1, 'suite': 'controlled-subagent-leaf-probes', 'status': 'RUNNING',
              'scope': 'Eight direct controlled Responses requests; no actual Codex App agent is spawned or observed.',
              'subagent_end_to_end': 'NOT_RUN', 'network_used': False,
              'service_restarted': False, 'weights_read_by_probe': False,
              'real_inference_executed': False, 'captures_real_client_conversations': False,
              'model': MODEL, 'checks': []}
    atomic_json(args.output, report)
    try:
        if not args.unix_socket.is_socket():
            raise ValueError('Existing engine Unix socket is missing')
        json_get('/health', args.unix_socket, args.timeout)
        identity = json_get('/v1/models', args.unix_socket, args.timeout)
        ids = [row.get('id') for row in identity.get('data', [])]
        if ids != [MODEL]:
            raise ValueError('Expected exactly the deployed DeepSeek-Flash model')
        report['model_ids'] = ids
        for effort in ('none', 'medium'):
            for envelope in (False, True):
                for stream in (False, True):
                    name = ('envelope' if envelope else 'plain') + '-' + effort + ('-sse' if stream else '-json')
                    print('START ' + name, flush=True)
                    row = one_case(args.unix_socket, args.timeout, envelope, stream, effort)
                    report['checks'].append(row)
                    report['real_inference_executed'] |= row['real_inference_executed']
                    atomic_json(args.output, report)
                    print(json.dumps({'name': row['name'], 'status': row['status'],
                                      'http_status': row.get('http_status'),
                                      'answer_status': row.get('answer_check', {}).get('status'),
                                      'stream_status': row.get('stream_contract', {}).get('status')}, ensure_ascii=False), flush=True)
        report['status'] = 'NO_FAILURE_OBSERVED' if all(row['status'] == 'PASS' for row in report['checks']) else 'ISSUES_DETECTED'
    except (OSError, ValueError, TypeError, http.client.HTTPException) as error:
        report['status'] = 'DIAGNOSTIC_COULD_NOT_RUN'
        report['error'] = type(error).__name__ + ': ' + str(error)
    report['finished_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    atomic_json(args.output, report)
    print(json.dumps({'status': report['status'], 'checks': len(report['checks']),
                      'subagent_end_to_end': 'NOT_RUN', 'output': str(args.output)}, ensure_ascii=False))
    if report['status'] == 'DIAGNOSTIC_COULD_NOT_RUN':
        return 2
    return 1 if args.require_pass and report['status'] != 'NO_FAILURE_OBSERVED' else 0


if __name__ == '__main__':
    raise SystemExit(main())
