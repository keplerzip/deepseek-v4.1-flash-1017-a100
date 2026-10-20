"""Six real local inference requests after V4 startup; no speed/quality claim."""
import argparse
import datetime
import hashlib
import http.client
import json
from pathlib import Path
import socket
import time

MODEL = 'DeepSeek-Flash'


class UnixHTTPConnection(http.client.HTTPConnection):
    def __init__(self, path, timeout):
        super().__init__('localhost', timeout=timeout)
        self.path = path

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(str(self.path))


def require(value, message):
    if not value:
        raise ValueError(message)


def payload(protocol, effort):
    message = {'role': 'user', 'content': 'Reply with the word READY.'}
    if protocol == 'chat':
        return '/v1/chat/completions', {'model': MODEL, 'messages': [message], 'reasoning_effort': effort,
                                     'max_tokens': 128, 'stream': False}
    if protocol == 'responses':
        return '/v1/responses', {'model': MODEL, 'input': [message], 'reasoning': {'effort': effort},
                                'max_output_tokens': 128, 'stream': False}
    return '/v1/messages', {'model': MODEL, 'messages': [message], 'thinking': {'type': 'adaptive'},
                            'output_config': {'effort': effort}, 'max_tokens': 128, 'stream': False}


def validate(protocol, status, response):
    require(status == 200, 'Expected HTTP 200, got ' + str(status))
    require(type(response) is dict and not response.get('error'), 'Server returned an error')
    require(response.get('model') == MODEL, 'Model identity differs')
    usage = response.get('usage')
    require(type(usage) is dict, 'Missing usage')
    input_key, output_key = (('prompt_tokens', 'completion_tokens') if protocol == 'chat' else ('input_tokens', 'output_tokens'))
    require(type(usage.get(input_key)) is int and usage[input_key] >= 0, 'Invalid input token usage')
    require(type(usage.get(output_key)) is int and usage[output_key] > 0, 'No real generated tokens in usage')
    return usage


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--unix-socket', type=Path, default=Path('/state/api.sock'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--timeout', type=float, default=300)
    args = p.parse_args()
    report = {'schema': 1, 'suite': 'reasoning-live-after-startup', 'status': 'RUNNING',
              'model': MODEL, 'transport': 'existing engine Unix socket',
              'real_inference_executed': False, 'real_inference_attempted': True, 'scope': 'HTTP 200, model identity and generated token usage for medium/xhigh through all three protocols; not depth quality, TPS or budget enforcement',
              'reasoning_depth_quality': 'NOT_MEASURED', 'budget_gpu_enforcement': 'NOT_PROVEN',
              'checks': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        temp = args.output.with_name(args.output.name + '.tmp')
        temp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        temp.replace(args.output)
    save()
    for protocol in ('chat', 'responses', 'messages'):
        for effort in ('medium', 'xhigh'):
            path, body = payload(protocol, effort)
            raw = json.dumps(body, separators=(',', ':')).encode()
            row = {'name': protocol + '-' + effort, 'status': 'FAIL', 'request': body,
                   'request_sha256': hashlib.sha256(raw).hexdigest()}
            started = time.monotonic()
            conn = UnixHTTPConnection(args.unix_socket, args.timeout)
            try:
                conn.request('POST', path, body=raw, headers={'Content-Type': 'application/json', 'anthropic-version': '2023-06-01'})
                response = conn.getresponse()
                data = response.read(2 * 1024 * 1024)
                row['http_status'] = response.status
                row['response_sha256'] = hashlib.sha256(data).hexdigest()
                row['response'] = json.loads(data)
                row['usage'] = validate(protocol, response.status, row['response'])
                report['real_inference_executed'] = True
                row['status'] = 'PASS'
            except (OSError, ValueError, http.client.HTTPException) as error:
                row['error'] = type(error).__name__ + ': ' + str(error)
            finally:
                conn.close()
            row['elapsed_seconds'] = time.monotonic() - started
            report['checks'].append(row)
            save()
            print(json.dumps({key: value for key, value in row.items() if key not in ('request', 'response')}, ensure_ascii=False), flush=True)
    report['status'] = 'PASS' if all(row['status'] == 'PASS' for row in report['checks']) else 'FAIL'
    report['finished_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    save()
    print(json.dumps({'status': report['status'], 'checks': len(report['checks']), 'budget_gpu_enforcement': 'NOT_PROVEN'}))
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
