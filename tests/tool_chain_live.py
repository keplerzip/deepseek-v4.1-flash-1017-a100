"""Fixed, read-only Responses tool round trips against the existing engine.

Only locally simulated tools are used. No files, commands, client conversations
or credentials are exposed to the model. This does not create a Codex App agent.
"""
import argparse
import datetime
import json
from pathlib import Path
import time

from subagent_diagnostic import UnixHTTP, StreamAudit, atomic_json, digest, json_get, text_from_item


MODEL = 'DeepSeek-Flash'
LIMIT = 4 * 1024 * 1024
NAME = 'v7_fixed_protocol_probe'
FUNCTION = {'type': 'function', 'name': NAME,
            'description': 'A fixed protocol test; returns a local acknowledgement.',
            'parameters': {'type': 'object', 'properties': {
                'task': {'type': 'string'}, 'count': {'type': 'integer'}},
                'required': ['task', 'count'], 'additionalProperties': False}}
NAMESPACE = {'type': 'namespace', 'name': 'functions',
             'description': 'Fixed protocol test functions.', 'tools': [FUNCTION]}


class ToolAudit(StreamAudit):
    def __init__(self):
        super().__init__()
        self.argument_deltas = {}
        self.argument_done = {}
        self.response_id = None

    def feed(self, event):
        kind = event.get('type')
        index = event.get('output_index')
        if not self.events and kind != 'response.created':
            raise ValueError('First SSE event must be response.created')
        if self.events and kind == 'response.created':
            raise ValueError('Duplicate response.created')
        if self.final is not None:
            raise ValueError('Unexpected event after terminal response')
        if kind == 'response.output_item.added' and index in self.added:
            raise ValueError('Duplicate output item added')
        if kind == 'response.output_item.done' and index in self.done:
            raise ValueError('Duplicate output item done')
        if kind in ('response.function_call_arguments.delta', 'response.function_call_arguments.done'):
            item = self.added.get(index) or {}
            if item.get('type') != 'function_call' or item.get('id') != event.get('item_id'):
                raise ValueError('Arguments event without matching added function')
        if kind in ('response.created', 'response.in_progress', 'response.completed',
                    'response.incomplete', 'response.failed'):
            identity = (event.get('response') or {}).get('id')
            if type(identity) is not str or not identity:
                raise ValueError('Missing lifecycle response id')
            if self.response_id is not None and identity != self.response_id:
                raise ValueError('Lifecycle response id changed')
            self.response_id = identity
        super().feed(event)
        identity = event.get('item_id')
        if event.get('type') == 'response.function_call_arguments.delta':
            self.argument_deltas.setdefault(identity, []).append(event.get('delta', ''))
        elif event.get('type') == 'response.function_call_arguments.done':
            if identity in self.argument_done:
                raise ValueError('Duplicate arguments.done')
            self.argument_done[identity] = event.get('arguments')

    def read(self, response):
        audit = self
        class BoundedReader:
            def readline(self, size):
                value = response.readline(size)
                if audit.wire_bytes + len(value) > LIMIT:
                    raise ValueError('Bounded SSE total limit exceeded')
                return value
        super().read(BoundedReader())

    def checked_contract(self):
        contract = self.contract()
        if contract['status'] != 'PASS':
            raise ValueError('SSE contract: ' + ','.join(row['kind'] for row in contract['issues']))
        if self.final_type != 'response.completed':
            raise ValueError('Tool round trip requires a genuinely completed response')
        if [event.get('type') for event in self.events[:2]] != ['response.created', 'response.in_progress']:
            raise ValueError('Expected native created/in_progress lifecycle prefix')
        if (set(self.added) != set(self.done)
                or sorted(self.added) != list(range(len(self.added)))):
            raise ValueError('Unclosed or nonconsecutive stream output items')
        sequence = [event.get('sequence_number') for event in self.events]
        if sequence != list(range(len(sequence))):
            raise ValueError('SSE sequence numbers are not consecutive')
        output = self.final.get('output', [])
        if output != [self.done[index] for index in sorted(self.done)]:
            raise ValueError('Final output differs from streamed done items')
        ids = [item.get('id') for item in output]
        if any(type(identity) is not str or not identity for identity in ids) or len(set(ids)) != len(ids):
            raise ValueError('Invalid or duplicate output item id')
        for index, item in enumerate(output):
            if item.get('type') != 'function_call':
                continue
            identity = item.get('id')
            start = self.added[index]
            if any(start.get(key) != item.get(key) for key in ('type', 'id', 'call_id', 'name', 'namespace')):
                raise ValueError('Tool added/done identity changed')
            if self.argument_done.get(identity) != item.get('arguments'):
                raise ValueError('Arguments done/item mismatch')
            if ''.join(self.argument_deltas.get(identity, [])) != item.get('arguments'):
                raise ValueError('Arguments delta/item mismatch')
        # Retain hashes/counts, not arbitrary generated text from the base audit.
        contract['stream_text'] = {key: contract['stream_text'][key]
                                   for key in ('characters', 'sha256')}
        return contract


def request(socket_path, payload, timeout):
    raw = json.dumps(payload, ensure_ascii=False).encode()
    conn = UnixHTTP(socket_path, timeout)
    try:
        conn.request('POST', '/v1/responses', body=raw,
                     headers={'Content-Type': 'application/json'})
        response = conn.getresponse()
        if response.status != 200:
            # Never persist arbitrary server bodies or request headers.
            raise ValueError('HTTP ' + str(response.status))
        audit = None
        if payload['stream']:
            if 'text/event-stream' not in (response.getheader('Content-Type') or ''):
                raise ValueError('SSE content type missing')
            audit = ToolAudit()
            audit.read(response)
            contract = audit.checked_contract()
            final = audit.final
        else:
            body = response.read(LIMIT + 1)
            if len(body) > LIMIT:
                raise ValueError('Bounded response limit exceeded')
            final = json.loads(body)
            contract = None
        if (final.get('model') != MODEL or final.get('status') != 'completed' or final.get('error')
                or type(final.get('id')) is not str or not final['id']):
            raise ValueError('Unexpected model, generation failure or incomplete response')
        usage = final.get('usage') or {}
        if (any(type(usage.get(key)) is not int or usage[key] < 0
                for key in ('input_tokens', 'output_tokens', 'total_tokens'))
                or usage['output_tokens'] <= 0
                or usage['total_tokens'] != usage['input_tokens'] + usage['output_tokens']):
            raise ValueError('No generated-token usage evidence')
        cached = (usage.get('input_tokens_details') or {}).get('cached_tokens')
        if cached is not None and (type(cached) is not int or not 0 <= cached <= usage['input_tokens']):
            raise ValueError('Invalid cached-token usage evidence')
        return final, {'request_sha256': digest(raw), 'http_status': 200,
                       'status': final['status'], 'usage': usage,
                       'response_id': final.get('id'), 'stream_contract': contract}
    finally:
        conn.close()


def validate_calls(output, *, namespaced, parallel):
    calls = [item for item in output if item.get('type') == 'function_call']
    wanted = [{'task': 'V7_FIRST', 'count': 2}]
    if parallel:
        wanted.append({'task': 'V7_SECOND', 'count': 3})
    if len(calls) != len(wanted):
        raise ValueError('Expected ' + str(len(wanted)) + ' fixed tool calls, got ' + str(len(calls)))
    identities = set()
    item_ids = set()
    arguments = []
    for item in calls:
        if (item.get('name') != NAME or item.get('namespace') != ('functions' if namespaced else None)
                or item.get('status') != 'completed'
                or type(item.get('id')) is not str or not item['id']
                or type(item.get('call_id')) is not str or not item['call_id']):
            raise ValueError('Wrong tool identity, namespace or completion status')
        if item['call_id'] in identities or item['id'] in item_ids:
            raise ValueError('Duplicate tool identity')
        identities.add(item['call_id'])
        item_ids.add(item['id'])
        value = json.loads(item['arguments'])
        if (not isinstance(value, dict) or type(value.get('count')) is not int
                or set(value) != {'task', 'count'}):
            raise ValueError('Tool arguments are not the expected typed JSON object')
        arguments.append(value)
    if sorted(arguments, key=lambda row: row['task']) != wanted:
        raise ValueError('Model did not produce the fixed requested tool arguments')
    return calls


def one_case(socket_path, timeout, *, stream, effort, namespaced, parallel):
    name = ('namespace' if namespaced else 'function') + ('-parallel' if parallel else '-single')
    name += '-' + effort + ('-sse' if stream else '-json')
    row = {'name': name, 'status': 'FAIL', 'controlled_test': True,
           'actual_codex_subagent': 'NOT_RUN'}
    started = time.monotonic()
    try:
        user = '请仅调用工具 ' + ('functions.' if namespaced else '') + NAME
        user += '，参数必须是 task="V7_FIRST", count=2（整数）。'
        if parallel:
            user += '同一次回答再调用一次，参数 task="V7_SECOND", count=3（整数）。'
        user += '不要执行其他工具，不要回答问候。'
        history = [{'role': 'user', 'content': user}]
        payload = {'model': MODEL, 'input': history, 'tools': [NAMESPACE if namespaced else FUNCTION],
                   'tool_choice': 'auto', 'parallel_tool_calls': True,
                   'stream': stream, 'store': False, 'reasoning': {'effort': effort},
                   'max_output_tokens': 2048 if effort != 'none' else 512}
        first, first_evidence = request(socket_path, payload, timeout)
        calls = validate_calls(first.get('output', []), namespaced=namespaced, parallel=parallel)
        # Simulate a harmless fixed tool result locally. The model cannot execute
        # anything through this probe. Bind every result to the actual call_id.
        results = []
        expected_acks = []
        for item in reversed(calls):
            value = json.loads(item['arguments'])
            marker = 'ACK_' + value['task']
            expected_acks.append({'call_id': item['call_id'], 'ack': marker, 'count': value['count']})
            results.append({'type': 'function_call_output', 'call_id': item['call_id'],
                            'output': json.dumps({'ack': marker, 'count': value['count']})})
        followup = dict(payload)
        followup['input'] = history + first['output'] + results + [{
            'role': 'user', 'content': '工具均完成。不再调用工具。最终正文仅输出一个JSON数组（无代码块或解释）；'
            '每个元素为 {"call_id":"对应工具调用的call_id","ack":"该工具返回的ack","count":该工具返回的整数count}，不得交换归属。'}]
        followup['tool_choice'] = 'none'
        second, second_evidence = request(socket_path, followup, timeout)
        if any(item.get('type') == 'function_call' for item in second.get('output', [])):
            raise ValueError('Unexpected extra tool call after tool_choice none')
        text = ''.join(text_from_item(item) for item in second.get('output', []))
        acknowledgements = json.loads(text)
        if (not isinstance(acknowledgements, list)
                or any(not isinstance(item, dict) or set(item) != {'call_id', 'ack', 'count'}
                       or type(item['count']) is not int for item in acknowledgements)
                or sorted(acknowledgements, key=lambda item: str(item['call_id']))
                   != sorted(expected_acks, key=lambda item: item['call_id'])):
            raise ValueError('Final acknowledgement call_id, ack or typed count association differs')
        # Persist only fixed task metadata, protocol IDs, token counts and final
        # answer hash. Reasoning text and arbitrary generated output stay in RAM.
        row.update(status='PASS', first=first_evidence, second=second_evidence,
                   calls=[{key: item.get(key) for key in ('id', 'call_id', 'name', 'namespace', 'arguments')}
                          for item in calls], reversed_tool_results=True,
                   final_answer_sha256=digest(text.encode()), final_answer_characters=len(text))
    except Exception as error:
        row['error'] = type(error).__name__ + ': ' + str(error)
    row['elapsed_seconds'] = time.monotonic() - started
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--unix-socket', type=Path, default=Path('/state/api.sock'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timeout', type=float, default=180)
    args = parser.parse_args()
    if not 1 <= args.timeout <= 600:
        parser.error('timeout must be within 1..600 seconds')
    report = {'schema': 1, 'suite': 'v7-fixed-live-tool-round-trips', 'status': 'RUNNING',
              'scope': 'Fixed real model function and namespace calls with typed parameters, reversed results and a second model turn.',
              'model': MODEL, 'new_host_dependencies': [], 'external_network_used': False,
              'actual_codex_app_subagent': 'NOT_RUN', 'newapi_gateway': 'NOT_RUN',
              'captures_real_client_conversations': False, 'checks': []}
    atomic_json(args.output, report)
    try:
        ids = [item.get('id') for item in json_get('/v1/models', args.unix_socket, args.timeout).get('data', [])]
        if ids != [MODEL]:
            raise ValueError('Expected exactly DeepSeek-Flash')
        for effort in ('none', 'medium', 'xhigh'):
            for stream in (False, True):
                # Include ordinary and namespace protocols; parallel cases also
                # return results in reverse order to expose call-ID mixups.
                for namespaced, parallel in ((False, False), (True, True)):
                    row = one_case(args.unix_socket, args.timeout, stream=stream,
                                   effort=effort, namespaced=namespaced, parallel=parallel)
                    report['checks'].append(row)
                    atomic_json(args.output, report)
                    print(json.dumps({'name': row['name'], 'status': row['status'],
                                      'error': row.get('error')}, ensure_ascii=False), flush=True)
        report['status'] = 'PASS' if all(row['status'] == 'PASS' for row in report['checks']) else 'FAIL'
    except Exception as error:
        report.update(status='COULD_NOT_RUN', error=type(error).__name__ + ': ' + str(error))
    report['finished_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    atomic_json(args.output, report)
    print(json.dumps({'status': report['status'], 'checks': len(report['checks']),
                      'actual_codex_app_subagent': 'NOT_RUN', 'report': str(args.output)}, ensure_ascii=False))
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
