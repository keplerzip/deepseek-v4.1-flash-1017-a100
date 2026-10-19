"""Exercise real Docker network=none -> UDS -> loopback TCP transport, without a GPU."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time
import urllib.request
import uuid

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--image', required=True)
args = parser.parse_args()
docker = ['sudo', '-n', 'docker']
names = ['dsv411017-transport-engine-' + uuid.uuid4().hex[:10],
         'dsv411017-transport-api-' + uuid.uuid4().hex[:10]]
probe = socket.socket()
probe.bind(('127.0.0.1', 0))
port = probe.getsockname()[1]
probe.close()
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
with tempfile.TemporaryDirectory(dir=root / 'build-state', prefix='transport-') as directory:
    state = Path(directory)
    (state / 'server.py').write_text('''import http.server,json,socketserver
class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def address_string(self):return 'uds'
    def do_GET(self):
        body=json.dumps({'data':[{'id':'DeepSeek-Flash'}]}).encode() if self.path=='/v1/models' else b'{}'
        self.send_response(200);self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
class Server(socketserver.UnixStreamServer):allow_reuse_address=True
Server('/state/api.sock',Handler).serve_forever()
''')
    user = ['--user', f'{os.getuid()}:{os.getgid()}']
    try:
        subprocess.run([*docker, 'run', '-d', '--pull', 'never', '--name', names[0],
                        '--network', 'none', *user, '-v', f'{state}:/state',
                        '--entrypoint', '/usr/bin/python3', args.image, '/state/server.py'], check=True, capture_output=True)
        for _ in range(100):
            if (state / 'api.sock').is_socket():
                break
            time.sleep(.1)
        assert (state / 'api.sock').is_socket(), 'UDS server did not start'
        subprocess.run([*docker, 'exec', names[0], '/usr/bin/python3', '/opt/dsv411017/scripts/runtime.py',
                        'network-proof', '--output', '/state/network-proof.json'], check=True, capture_output=True)
        subprocess.run([*docker, 'run', '-d', '--pull', 'never', '--name', names[1],
                        '--network', 'host', '--read-only', *user, '-v', f'{state}:/state:ro',
                        '--entrypoint', '/usr/bin/python3', args.image,
                        '/opt/dsv411017/scripts/frontend.py', '--port', str(port)], check=True, capture_output=True)
        for _ in range(100):
            try:
                with opener.open(f'http://127.0.0.1:{port}/v1/models', timeout=1) as response:
                    assert json.load(response)['data'][0]['id'] == 'DeepSeek-Flash'
                break
            except OSError:
                time.sleep(.1)
        else:
            raise RuntimeError('Loopback TCP frontend failed')
        subprocess.run([*docker, 'exec', names[1], '/usr/bin/python3', '/opt/dsv411017/scripts/runtime.py',
                        'frontend-health', '--port', str(port)], check=True, capture_output=True)
        report = {'status': 'PASS', 'scope': 'real Docker offline transport using a deterministic HTTP fixture; no inference',
                  'same_nonroot_uid': os.getuid(), 'engine_network_none': True,
                  'frontend_readonly_socket_mount': True, 'loopback_tcp_to_uds': True,
                  'network_proof': json.loads((state / 'network-proof.json').read_text()), 'gpu_executed': False}
        (root / 'build-state/offline-transport.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report))
    except Exception:
        errors={}
        for name in names:
            errors[name]=subprocess.run([*docker,'logs',name],text=True,capture_output=True).stderr[-8000:]
        (root/'build-state/offline-transport-errors.json').write_text(json.dumps(errors,indent=2)+'\n')
        raise
    finally:
        for name in reversed(names):
            subprocess.run([*docker, 'rm', '-f', name], capture_output=True)
