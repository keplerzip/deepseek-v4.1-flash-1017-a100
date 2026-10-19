"""Run real pinned-image CPU protocol checks before/after read-only overlays."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / '.'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-dir', type=Path, default=ROOT / 'build-state/r39-protocol-v2')
    args = parser.parse_args()
    args.evidence_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = PROJECT / 'manifests/r39-overlay-provenance.json'
    manifest = json.loads(manifest_path.read_text())
    command = ['sudo', '-n', 'docker', 'run', '--rm', '--pull', 'never', '--network', 'none',
               '--read-only', '--tmpfs', '/tmp:rw,size=2g', '-e', 'CUDA_VISIBLE_DEVICES=',
               '-e', 'PYTHONDONTWRITEBYTECODE=1', '-v', str(PROJECT) + ':/package:ro',
               '-v', str(args.evidence_dir.resolve()) + ':/audit-output']
    checks = {}
    for label in ('baseline', 'patched'):
        argv = command.copy()
        if label == 'patched':
            for row in manifest['files']:
                source = PROJECT / row['overlay_path']
                if hashlib.sha256(source.read_bytes()).hexdigest() != row['patched_sha256']:
                    raise RuntimeError('Unreviewed protocol overlay bytes: ' + row['path'])
                argv += ['-v', str(source) + ':' + row['site_packages_path'] + ':ro']
        argv += ['--entrypoint', '/usr/bin/python3', manifest['native_image_tag'],
                 '/package/tests/check_r39_protocols.py', '--project', '/package',
                 '--output', '/audit-output/' + label + '.json']
        print('Running pinned-image CPU protocols:', label, flush=True)
        result = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        (args.evidence_dir / (label + '.log')).write_text(result.stdout)
        report_path = args.evidence_dir / (label + '.json')
        if not report_path.exists():
            raise RuntimeError(label + ' produced no report: ' + result.stdout[-4000:])
        report = json.loads(report_path.read_text())
        checks[label] = {'exit_code': result.returncode, 'status': report['status'],
                         'checks': [{'name': row['name'], 'status': row['status']} for row in report.get('checks', [])],
                         'report_sha256': hashlib.sha256(report_path.read_bytes()).hexdigest(),
                         'log_sha256': hashlib.sha256((args.evidence_dir / (label + '.log')).read_bytes()).hexdigest()}
        if label == 'baseline':
            names = {row['name']: row['status'] for row in report.get('checks', [])}
            if result.returncode != 1 or report['status'] != 'FAIL' or not all(names.get(name) == 'FAIL' for name in (
                    'v41-inline-system-capability-and-fallback',
                    'messages-tail-system-multiturn-prefix-and-count',
                    'messages-tool-result-image-call-id-order',
                    'responses-image-detail-native-pydantic',
                    'codex-octet-stream-full-guard-protocol-image-pipeline')):
                raise RuntimeError('Pinned native baseline did not reproduce the documented gaps: ' + result.stdout[-7000:])
        elif result.returncode != 0 or report['status'] != 'PASS':
            raise RuntimeError('Read-only overlays did not pass actual installed protocol checks: ' + result.stdout[-10000:])
    files = [PROJECT / row['overlay_path'] for row in manifest['files']]
    files += [manifest_path, PROJECT / 'manifests/r39-overlay-files.txt',
              PROJECT / 'tests/check_r39_protocols.py', Path(__file__)]
    result = {'schema': 1, 'status': 'PASS', 'gpu_executed': False, 'weights_loaded': False,
              'network_used': False, 'real_inference_executed': False,
              'scope': 'Actual installed native CPU imports/Pydantic/ASGI/protocol/image decoding/official encoder; expected baseline failures reproduced and patched checks passed. No cache-hit/TPS claims.',
              'image': manifest['native_image_tag'], 'results': checks,
              'finished_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'source_sha256': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}}
    (args.evidence_dir / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': 'PASS', 'baseline': 'KNOWN_FAILURES_REPRODUCED',
                      'patched': 'PASS', 'gpu_executed': False,
                      'verification': str(args.evidence_dir / 'verification.json')}, indent=2))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        print(type(error).__name__ + ': ' + str(error), file=sys.stderr)
        raise SystemExit(1)
