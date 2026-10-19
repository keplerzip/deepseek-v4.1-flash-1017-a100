"""Local/Docker-host API check: fixed local destinations, never external hosts."""
import argparse
import json
from pathlib import Path
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--docker-client', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    base = 'http://host.docker.internal:8005' if args.docker_client else 'http://127.0.0.1:8005'
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    report = {'status': 'RUNNING', 'base': base, 'authentication_headers_sent': False,
              'external_internet_requested': False, 'model': 'DeepSeek-Flash'}
    code = 0
    try:
        with opener.open(urllib.request.Request(base + '/v1/models', headers={}), timeout=30) as response:
            payload = json.load(response)
            assert response.status == 200
            assert {item['id'] for item in payload['data']} == {'DeepSeek-Flash'}
            report.update(status='PASS', http_status=200, models=payload)
    except Exception as error:
        report.update(status='FAIL', error=type(error).__name__ + ': ' + str(error))
        code = 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(args.output)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
