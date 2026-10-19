"""Real UDS acceptance for R3.9-compatible cache usage and ordered images.

Only Python's standard library is used. No TCP, network fetch, generated cache
counts, or model-quality claims are involved. PNGs and reference source hashes
are recorded in r39-assets/provenance.json. CPU parser tests are separate.
"""
import argparse
import base64
import copy
import datetime
import hashlib
import http.client
import json
import math
from pathlib import Path
import signal
import socket
import sys
import time
import uuid


MODEL = 'DeepSeek-Flash'
PATHS = {'chat': '/v1/chat/completions', 'responses': '/v1/responses',
         'messages': '/v1/messages'}
ASSETS = Path(__file__).with_name('r39-assets')
UNIT = 'This line is filler for prefix reuse; it contains no answer.\n'
NUMBERS = [1, 3, 2, 1, 2, 3, 3, 1]


def case_specs():
    specs = []
    for protocol in PATHS:
        for content in ('text', 'image'):
            for fmt in ('json', 'sse'):
                specs.append((f'cache-{protocol}-{content}-{fmt}', 'cache', protocol, fmt, content))
    for content in ('text', 'image'):
        specs.append((f'cache-chat-{content}-sse-implicit-usage', 'cache', 'chat', 'sse', content))
    for protocol in PATHS:
        for fmt in ('json', 'sse'):
            specs.append((f'cache-{protocol}-changed-image-{fmt}', 'cache', protocol, fmt, 'changed-image'))
    for fmt in ('json', 'sse'):
        specs.append((f'cache-messages-dynamic-tail-system-{fmt}', 'cache', 'messages', fmt, 'dynamic-tail-system'))
    specs.append(('cache-chat-long-image-json', 'cache', 'chat', 'json', 'long-image'))
    for protocol in PATHS:
        specs.append((f'vision-{protocol}-ordered-5-json', 'vision', protocol, 'json', 'ordered-5'))
        specs.append((f'vision-{protocol}-ordered-8-sse', 'vision', protocol, 'sse', 'ordered-8'))
    for protocol in PATHS:
        for fmt in ('json', 'sse'):
            specs.append((f'vision-{protocol}-reversed-tool-image-id-{fmt}', 'vision', protocol, fmt, 'reversed-tool-image-id'))
    for fmt in ('json', 'sse'):
        specs.append((f'vision-responses-codex-view-image-{fmt}', 'vision', 'responses', fmt, 'codex-view-image'))
    return tuple(specs)


CASE_SPECS = case_specs()
REQUIRED_CASES = tuple(spec[0] for spec in CASE_SPECS)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def strict_equal(actual, expected):
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(strict_equal(actual[k], v) for k, v in expected.items())
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(strict_equal(a, b) for a, b in zip(actual, expected))
    return actual == expected


def parse_sse(raw):
    """Decode complete SSE events; cumulative usage must not be added here."""
    text = raw.decode('utf-8') if isinstance(raw, bytes) else raw
    events, data, event_name = [], [], ''
    # split('\n') preserves the final delimiter, unlike splitlines().
    require(text.endswith('\n\n') or text.endswith('\r\n\r\n'), 'Truncated SSE stream')
    for line in text.split('\n'):
        line = line.rstrip('\r')
        if not line:
            if data:
                encoded = '\n'.join(data)
                if encoded == '[DONE]':
                    require(not event_name, 'Unexpected named [DONE] event')
                    events.append({'type': 'done'})
                else:
                    value = json.loads(encoded)
                    require(isinstance(value, dict), 'SSE data is not an object')
                    require(not event_name or value.get('type') == event_name, 'SSE event/type mismatch')
                    require(not value.get('error') and value.get('type') not in
                            ('error', 'response.failed', 'response.incomplete'), 'SSE error: ' + str(value))
                    events.append(value)
            data, event_name = [], ''
        elif line.startswith('data:'):
            data.append(line[5:].removeprefix(' '))
        elif line.startswith('event:'):
            event_name = line[6:].removeprefix(' ')
    require(not data, 'Truncated SSE event')
    require(events, 'Empty SSE stream')
    return events


def cache_usage(protocol, usage):
    """Keep native usage intact, and compute totals according to each protocol."""
    require(type(usage) is dict, 'Missing final usage')
    if protocol == 'chat':
        total = usage.get('prompt_tokens')
        detail = usage.get('prompt_tokens_details')
        require(type(detail) is dict, 'Missing prompt_tokens_details')
        cached = detail.get('cached_tokens')
    elif protocol == 'responses':
        total = usage.get('input_tokens')
        detail = usage.get('input_tokens_details')
        require(type(detail) is dict, 'Missing input_tokens_details')
        cached = detail.get('cached_tokens')
    else:
        uncached = usage.get('input_tokens')
        cached = usage.get('cache_read_input_tokens')
        created = usage.get('cache_creation_input_tokens', 0)
        require(all(type(v) is int and v >= 0 for v in (uncached, cached, created)),
                'Invalid Messages input/cache usage')
        total = uncached + cached + created
    require(type(total) is int and type(cached) is int and total > 0 and 0 <= cached <= total,
            'Invalid cache usage counts')
    return total, cached


def normalize(protocol, body, raw, model=MODEL):
    """Return native final usage, without inventing any cache creation count."""
    stream = bool(body.get('stream'))
    events = raw if stream else None
    ids = []
    for item in (raw if isinstance(raw, list) else [raw]):
        if isinstance(item, dict):
            for obj in (item, item.get('response', {}), item.get('message', {})):
                if isinstance(obj, dict) and obj.get('id') and obj['id'] not in ids:
                    ids.append(obj['id'])
    if protocol == 'chat':
        if stream:
            require(events[-1].get('type') == 'done', 'Missing Chat [DONE]')
            chunks = [event for event in events if 'choices' in event]
            require(chunks and all(chunk.get('model') == model for chunk in chunks), 'Chat model mismatch')
            require(any(choice.get('finish_reason') == 'stop' for chunk in chunks for choice in chunk['choices']),
                    'Chat did not finish with stop')
            text = ''.join(choice.get('delta', {}).get('content') or ''
                           for chunk in chunks for choice in chunk['choices'])
            usage = next((event['usage'] for event in reversed(events) if event.get('usage')), None)
        else:
            require(raw.get('model') == model, 'Chat model mismatch')
            require(raw.get('choices') and raw['choices'][0].get('finish_reason') == 'stop', 'Chat did not finish with stop')
            text, usage = raw['choices'][0]['message'].get('content'), raw.get('usage')
    elif protocol == 'responses':
        if stream:
            completed = [event.get('response') for event in events if event.get('type') == 'response.completed']
            require(len(completed) == 1, 'Missing or repeated response.completed')
            raw = completed[0]
        require(raw.get('model') == model and raw.get('status') == 'completed', 'Responses model/status mismatch')
        text = ''.join(part.get('text', '') for item in raw.get('output', []) if item.get('type') == 'message'
                       for part in item.get('content', []) if part.get('type') == 'output_text')
        usage = raw.get('usage')
    else:
        if stream:
            require(events[-1].get('type') == 'message_stop', 'Missing message_stop')
            starts = [event.get('message') for event in events if event.get('type') == 'message_start']
            require(len(starts) == 1 and starts[0].get('model') == model, 'Messages model/start mismatch')
            usage = dict(starts[0].get('usage') or {})
            for event in events:
                if event.get('type') == 'message_delta':
                    usage.update(event.get('usage') or {})
            require(any(event.get('delta', {}).get('stop_reason') == 'end_turn' for event in events),
                    'Messages did not finish with end_turn')
            text = ''.join(event.get('delta', {}).get('text', '') for event in events
                           if event.get('type') == 'content_block_delta')
        else:
            require(raw.get('model') == model and raw.get('stop_reason') == 'end_turn', 'Messages model/status mismatch')
            text = ''.join(part.get('text', '') for part in raw.get('content', []) if part.get('type') == 'text')
            usage = raw.get('usage')
    total, cached = cache_usage(protocol, usage)
    require(isinstance(text, str), 'Missing text answer')
    # Usage is available even when the model failed to return strict JSON.
    try:
        actual, answer_error = json.loads(text), None
    except (ValueError, TypeError) as error:
        actual, answer_error = None, 'Answer is not strict JSON: ' + str(error)
    return {'actual': actual, 'text': text, 'answer_error': answer_error,
            'usage': usage, 'input_total': total, 'cached_tokens': cached, 'response_ids': ids}


class UnixConnection(http.client.HTTPConnection):
    def __init__(self, unix_socket, timeout):
        super().__init__('localhost', timeout=timeout)
        self.unix_socket = str(unix_socket)

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self.unix_socket)


class Transport:
    def __init__(self, unix_socket, timeout, output):
        self.unix_socket, self.timeout = unix_socket, timeout
        self.trace = output.with_name(output.stem + '.requests.jsonl')
        self.evidence = output.with_name(output.stem + '.evidence')
        self.evidence.mkdir(parents=True, exist_ok=True)
        self.serial = 0

    def call(self, protocol, body, case_name, phase, expected=None, path=None):
        self.serial += 1
        stem = f'{self.serial:04d}-{case_name}-{phase}'
        request_path = path or PATHS[protocol]
        started = time.monotonic()
        row = {'case': case_name, 'phase': phase, 'protocol': protocol,
               'format': 'sse' if body and body.get('stream') else 'json',
               'path': request_path, 'started_at': now(), 'request': body, 'expected': expected,
               'response_status': None, 'response_headers': {}, 'raw_usage': None, 'status': 'FAIL'}
        raw_file = self.evidence / (stem + '.response.bin')
        connection = UnixConnection(self.unix_socket, self.timeout)
        try:
            encoded = None if body is None else json.dumps(body, ensure_ascii=False).encode()
            connection.request('GET' if body is None else 'POST', request_path, body=encoded,
                               headers={'Content-Type': 'application/json', 'anthropic-version': '2023-06-01'})
            response = connection.getresponse()
            row['response_status'] = response.status
            row['response_headers'] = dict(response.getheaders())
            # Preserve partial raw response even when the peer interrupts a stream.
            with raw_file.open('wb') as stream:
                while True:
                    try:
                        block = response.read(65536)
                    except http.client.IncompleteRead as error:
                        stream.write(error.partial)
                        stream.flush()
                        raise
                    if not block:
                        break
                    stream.write(block)
                    stream.flush()
            raw_bytes = raw_file.read_bytes()
            row['response_sha256'] = hashlib.sha256(raw_bytes).hexdigest()
            row['response_bytes'] = len(raw_bytes)
            require(200 <= response.status < 300, f'HTTP {response.status}: ' + raw_bytes[:4000].decode(errors='replace'))
            is_stream = bool(body and body.get('stream'))
            if is_stream:
                content_type = response.getheader('Content-Type', '')
                require('text/event-stream' in content_type.lower(), 'SSE response has wrong Content-Type')
            parsed = parse_sse(raw_bytes) if is_stream else json.loads(raw_bytes)
            row['parsed_response'] = parsed
            if body is None:
                row['status'] = 'PASS'
                row['value'] = parsed
            else:
                normalized = normalize(protocol, body, parsed, body['model'])
                row.update(normalized)
                row['raw_usage'] = normalized['usage']
                require(not normalized['answer_error'], normalized['answer_error'])
                require(strict_equal(normalized['actual'], expected),
                        'Answer mismatch: expected ' + json.dumps(expected) + ', got ' + json.dumps(normalized['actual']))
                row['status'] = 'PASS'
        except KeyboardInterrupt as error:
            row['error'] = type(error).__name__ + ': ' + str(error)
            raise
        except Exception as error:
            row['error'] = type(error).__name__ + ': ' + str(error)
        finally:
            connection.close()
            row['seconds'] = time.monotonic() - started
            row['finished_at'] = now()
            row['raw_response_file'] = str(raw_file.resolve()) if raw_file.exists() else None
            record_file = self.evidence / (stem + '.json')
            atomic_json(record_file, row)
            row['evidence_file'] = str(record_file.resolve())
            with self.trace.open('a') as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        # Reports keep native usage and evidence references, not duplicated PNGs.
        return {key: value for key, value in row.items() if key not in ('request', 'parsed_response', 'value')} | (
            {'value': row['value']} if 'value' in row else {})


def verify_assets(directory=ASSETS):
    manifest = json.loads((directory / 'provenance.json').read_text())
    require(len(manifest['files']) == 5, 'Unexpected R3.9 fixture set')
    for row in manifest['files']:
        path = directory / row['path']
        require(not path.is_symlink() and path.resolve().is_relative_to(directory.resolve()), 'External asset')
        data = path.read_bytes()
        require(data.startswith(b'\x89PNG\r\n\x1a\n'), 'Invalid PNG fixture: ' + row['path'])
        require(len(data) == row['bytes'] and hashlib.sha256(data).hexdigest() == row['sha256'],
                'R3.9 asset hash mismatch: ' + row['path'])
    return manifest


def image_block(protocol, relative, octet_stream=False):
    data = base64.b64encode((ASSETS / relative).read_bytes()).decode()
    media_type = 'application/octet-stream' if octet_stream else 'image/png'
    if protocol == 'messages':
        return {'type': 'image', 'source': {'type': 'base64', 'media_type': media_type, 'data': data}}
    url = 'data:' + media_type + ';base64,' + data
    if protocol == 'responses':
        # Omitted detail exercises the native default; Codex case uses high.
        result = {'type': 'input_image', 'image_url': url}
        if octet_stream:
            result['detail'] = 'high'
        return result
    return {'type': 'image_url', 'image_url': {'url': url}}


def text_block(protocol, text):
    return {'type': 'input_text' if protocol == 'responses' else 'text', 'text': text}


def base_body(protocol, content, stream=False, max_tokens=512):
    body = {'model': MODEL, 'stream': stream, 'temperature': 0}
    if protocol == 'responses':
        body.update(input=[{'role': 'user', 'content': content}], max_output_tokens=max_tokens,
                    store=False, chat_template_kwargs={'thinking': False})
    else:
        body.update(messages=[{'role': 'user', 'content': content}], max_tokens=max_tokens)
        if protocol == 'messages':
            body['thinking'] = {'type': 'disabled'}
        else:
            body['chat_template_kwargs'] = {'thinking': False}
    return body


def cache_body(protocol, tag, image=None, repeats=512):
    content = [text_block(protocol, 'Independent cache audit ' + tag + '.\n')]
    if image:
        content.append(image_block(protocol, image))
    question = ('Read the chart image. Return only a JSON object with integer field b_value for group B. No Markdown.'
                if image else 'Return only this JSON object: {"check": 73019}. No Markdown.')
    content.append(text_block(protocol, UNIT * repeats + '\nIgnore filler. ' + question))
    return base_body(protocol, content)


def numbered_content(protocol, sequence, octet_stream=False):
    result = []
    for position, number in enumerate(sequence, 1):
        result.append(text_block(protocol, f'Image position {position}:'))
        result.append(image_block(protocol, f'video_frames/{number:02}.png', octet_stream))
    return result


def ordered_body(protocol, count, stream):
    prompt = ('Read the number printed at the top of EACH image in supplied order. Return only a JSON object '
              'with one field numbers, an array of integers. Preserve repetitions. No Markdown.')
    body = base_body(protocol, [text_block(protocol, prompt)] + numbered_content(protocol, NUMBERS[:count]), stream, 1024)
    return body, {'numbers': NUMBERS[:count]}


def reversed_tools_body(protocol, stream=False, codex=False):
    """Request ID-keyed answers so arrival order cannot hide image misbinding."""
    prompt = ('Read image numbers at their tops. Associate images with their tool CALL ID, regardless of tool '
              'result arrival order. Return only JSON with fields call_a and call_b, each an integer array '
              'in that call result\'s image order. No Markdown.')
    sequences = {'call_a': [2, 1], 'call_b': [3, 2]}
    if codex:
        sequences = {'call_a': [2], 'call_b': [1]}
    expected = copy.deepcopy(sequences)
    make_content = lambda ident: numbered_content(protocol, sequences[ident], codex)
    if protocol == 'responses':
        items = [{'role': 'user', 'content': 'Inspect the local images using view_image.'}]
        items += [{'type': 'function_call', 'name': 'view_image', 'call_id': ident,
                   'arguments': json.dumps({'call_label': ident})} for ident in ('call_a', 'call_b')]
        for ident in ('call_b', 'call_a'):
            content = make_content(ident)
            if codex:
                content = [part for part in content if part['type'] == 'input_image']
            items.append({'type': 'function_call_output', 'call_id': ident, 'output': content})
        items.append({'role': 'user', 'content': prompt})
        body = base_body(protocol, [], stream)
        body.update(input=items, tools=[{'type': 'function', 'name': 'view_image',
                    'parameters': {'type': 'object', 'properties': {'call_label': {'type': 'string'}}}}], tool_choice='none')
    elif protocol == 'chat':
        messages = [{'role': 'user', 'content': 'Inspect local images using view_image.'},
                    {'role': 'assistant', 'content': None, 'tool_calls': [
                        {'type': 'function', 'id': ident, 'function': {'name': 'view_image',
                         'arguments': json.dumps({'call_label': ident})}} for ident in ('call_a', 'call_b')]}]
        messages += [{'role': 'tool', 'tool_call_id': ident, 'content': make_content(ident)} for ident in ('call_b', 'call_a')]
        messages.append({'role': 'user', 'content': prompt})
        body = base_body(protocol, [], stream)
        body['messages'] = messages
    else:
        messages = [{'role': 'user', 'content': 'Inspect local images using view_image.'},
                    {'role': 'assistant', 'content': [
                        {'type': 'tool_use', 'id': ident, 'name': 'view_image', 'input': {'call_label': ident}}
                        for ident in ('call_a', 'call_b')]},
                    {'role': 'user', 'content': [
                        {'type': 'tool_result', 'tool_use_id': ident, 'content': make_content(ident)}
                        for ident in ('call_b', 'call_a')] + [text_block(protocol, prompt)]}]
        body = base_body(protocol, [], stream)
        body['messages'] = messages
    return body, expected


def validate_hit(cold, hot, minimum=1):
    require(cold['status'] == hot['status'] == 'PASS', 'Cold or hot request failed')
    require(hot['cached_tokens'] >= minimum, f'Cache read below required {minimum} tokens')
    require(hot['cached_tokens'] > cold['cached_tokens'], 'Repeated exact prefix did not increase actual cache reads')


class Suite:
    def __init__(self, args):
        self.args = args
        args.output.parent.mkdir(parents=True, exist_ok=True)
        self.transport = Transport(args.unix_socket, args.timeout, args.output)
        self.report = {'schema': 1, 'suite': 'r39-alignment', 'model': MODEL, 'context': 524288,
                       'concurrency': 32, 'status': 'RUNNING', 'started_at': now(), 'finished_at': None,
                       'cases': [{'name': name, 'kind': kind, 'protocol': protocol, 'format': fmt,
                                  'status': 'NOT_RUN', 'cold_cached_tokens': None, 'hot_cached_tokens': None,
                                  'requests': [], 'expected': None, 'errors': []}
                                 for name, kind, protocol, fmt, _ in CASE_SPECS],
                       'required_cases': list(REQUIRED_CASES), 'newapi_billing_end_to_end': 'NOT_RUN',
                       'gpu_validation': 'NOT_INDEPENDENTLY_CHECKED', 'target_gpu_required': True,
                       'inference_requests_executed': 0,
                       'scope': 'Real direct UDS model responses and native cache usage; gateway billing and 999-image capacity not certified.',
                       'trace': str(self.transport.trace.resolve()), 'evidence': str(self.transport.evidence.resolve())}
        self.save()

    def save(self):
        atomic_json(self.args.output, self.report)

    def request(self, row, body, expected, phase):
        result = self.transport.call(row['protocol'], body, row['name'], phase, expected)
        row['requests'].append(result)
        self.report['inference_requests_executed'] += 1
        self.save()
        return result

    def cache_case(self, row, content):
        protocol, fmt = row['protocol'], row['format']
        tag = uuid.uuid4().hex
        image = 'chart_before.png' if content in ('image', 'changed-image', 'long-image') else None
        body = cache_body(protocol, tag, image, 6144 if content == 'long-image' else 512)
        expected = {'b_value': 65} if image else {'check': 73019}
        row['expected'] = expected
        if content == 'dynamic-tail-system':
            body = cache_body(protocol, tag, repeats=256)
            body['system'] = 'Follow the user. Return only the requested JSON.'
            body['messages'].append({'role': 'system', 'content': 'Remaining audit budget: 99. Return the requested JSON.'})
        cold = self.request(row, body, expected, 'cold-json')
        row['cold_cached_tokens'] = cold.get('cached_tokens')
        if content == 'changed-image':
            require(cold['status'] == 'PASS', 'Initial image request failed')
            body = cache_body(protocol, tag, 'chart_after.png')
            expected = {'b_value': 60}
            row['expected'] = expected
            changed = self.request(row, body, expected, 'changed-image-json')
            require(changed['status'] == 'PASS', 'Changed-image request failed')
            require(changed['cached_tokens'] < changed['input_total'] // 2,
                    'Unexpected prefix reuse after image changed near prompt start')
            row['original_image_cached_tokens'] = cold.get('cached_tokens')
            row['changed_image_cached_tokens'] = changed['cached_tokens']
            row['changed_image_input_total'] = changed['input_total']
            cold = changed
            row['cold_cached_tokens'] = changed['cached_tokens']
        if content == 'dynamic-tail-system':
            body['messages'].extend([
                {'role': 'assistant', 'content': json.dumps(expected)},
                {'role': 'user', 'content': 'Return the same check JSON again.'},
                {'role': 'system', 'content': 'Remaining audit budget: 98. Return the requested JSON.'}])
            if fmt == 'sse':
                intermediate = self.request(row, body, expected, 'followup-json')
                validate_hit(cold, intermediate, 512)
                body['messages'].extend([
                    {'role': 'assistant', 'content': json.dumps(expected)},
                    {'role': 'user', 'content': 'Return the same check JSON again.'},
                    {'role': 'system', 'content': 'Remaining audit budget: 97. Return the requested JSON.'}])
        body['stream'] = fmt == 'sse'
        if protocol == 'chat' and fmt == 'sse' and 'implicit-usage' not in row['name']:
            body['stream_options'] = {'include_usage': True}
        if 'implicit-usage' in row['name']:
            require('stream_options' not in body, 'Implicit-usage probe must omit stream_options')
        hot = self.request(row, body, expected, 'hot-' + fmt)
        row['hot_cached_tokens'] = hot.get('cached_tokens')
        validate_hit(cold, hot, 512 if content == 'dynamic-tail-system' else 1)
        row.update(cold_usage=cold['raw_usage'], hot_usage=hot['raw_usage'],
                   hot_input_total=hot['input_total'], hot_cache_hit_ratio=hot['cached_tokens'] / hot['input_total'],
                   min_hot_cached_tokens=512 if content == 'dynamic-tail-system' else 1)
        if content != 'dynamic-tail-system':
            # Keep the actual history and verified assistant reply. Never use
            # previous_response_id/server storage to hide missing client history.
            key = 'input' if protocol == 'responses' else 'messages'
            body[key].extend([
                {'role': 'assistant', 'content': json.dumps(expected)},
                {'role': 'user', 'content': 'Using the same original material, return exactly the same requested JSON again. No Markdown.'}])
            followup = self.request(row, body, expected, 'multi-turn-' + fmt)
            require(followup['status'] == 'PASS', followup.get('error', 'Multi-turn request failed'))
            require(followup['cached_tokens'] >= 1, 'Multi-turn continuation did not read cached input')
            row.update(multi_turn_usage=followup['raw_usage'], multi_turn_cached_tokens=followup['cached_tokens'],
                       multi_turn_input_total=followup['input_total'], multi_turn_format=fmt)

    def vision_case(self, row, content):
        stream = row['format'] == 'sse'
        if content.startswith('ordered-'):
            count = int(content.rsplit('-', 1)[1])
            body, expected = ordered_body(row['protocol'], count, stream)
            row['image_count'] = count
        else:
            codex = content == 'codex-view-image'
            body, expected = reversed_tools_body(row['protocol'], stream, codex)
            row.update(tool_call_order=['call_a', 'call_b'], tool_result_arrival_order=['call_b', 'call_a'],
                       image_count=2 if codex else 4, tool_call_id_binding=True)
        if row['protocol'] == 'chat' and stream:
            body['stream_options'] = {'include_usage': True}
        row['expected'] = expected
        result = self.request(row, body, expected, 'vision-' + row['format'])
        row['actual'] = result.get('actual')
        require(result['status'] == 'PASS', result.get('error', 'Vision request failed'))
        row['usage'] = result['raw_usage']

    def run(self):
        try:
            manifest = verify_assets()
            self.report['assets'] = manifest
            identity = self.transport.call('chat', None, 'model-identity', 'json', path='/v1/models')
            self.report['model_identity'] = identity
            require(identity['status'] == 'PASS', identity.get('error', 'Cannot obtain model identity'))
            models = identity['value'].get('data')
            require(isinstance(models, list) and len(models) == 1 and models[0].get('id') == MODEL,
                    'Endpoint must expose only DeepSeek-Flash')
            require(models[0].get('max_model_len') == 524288, 'Endpoint must expose the fixed 524288 context')
            for row, spec in zip(self.report['cases'], CASE_SPECS):
                row['status'], row['started_at'] = 'RUNNING', now()
                self.save()
                try:
                    if row['kind'] == 'cache':
                        self.cache_case(row, spec[4])
                    else:
                        self.vision_case(row, spec[4])
                    row['status'] = 'PASS'
                except Exception as error:
                    row['status'] = 'FAIL'
                    row['errors'].append(type(error).__name__ + ': ' + str(error))
                row['finished_at'] = now()
                self.save()
                print(json.dumps({key: row[key] for key in
                      ('name', 'kind', 'protocol', 'format', 'status', 'cold_cached_tokens', 'hot_cached_tokens', 'errors')},
                      ensure_ascii=False), flush=True)
        except (Exception, KeyboardInterrupt) as error:
            self.report.setdefault('errors', []).append(type(error).__name__ + ': ' + str(error))
        finally:
            passed = sum(row['status'] == 'PASS' for row in self.report['cases'])
            self.report.update(status='PASS' if passed == len(REQUIRED_CASES) and not self.report.get('errors') else 'FAIL',
                               finished_at=now(), passed_cases=passed,
                               failed_cases=sum(row['status'] == 'FAIL' for row in self.report['cases']),
                               not_run_cases=sum(row['status'] not in ('PASS', 'FAIL') for row in self.report['cases']))
            self.save()
            print(json.dumps({'suite': 'r39-alignment', 'status': self.report['status'], 'passed': passed,
                              'required': len(REQUIRED_CASES), 'report': str(self.args.output.resolve()),
                              'newapi_billing_end_to_end': 'NOT_RUN'}, ensure_ascii=False), flush=True)
        return 0 if self.report['status'] == 'PASS' else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--unix-socket', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--model', default=MODEL, choices=[MODEL])
    parser.add_argument('--timeout', type=float, default=300)
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('timeout must be finite and positive')
    # Do not overwrite an earlier run's raw evidence.
    if args.output.exists() or args.output.with_name(args.output.stem + '.requests.jsonl').exists():
        parser.error('output already exists; choose a fresh report path')
    def interrupted(signum, frame):
        raise KeyboardInterrupt('Signal ' + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    return Suite(args).run()


if __name__ == '__main__':
    raise SystemExit(main())
