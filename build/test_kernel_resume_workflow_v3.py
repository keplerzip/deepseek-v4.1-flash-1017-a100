"""Run production V3 kernels.sh with explicitly synthetic CPU Docker functions."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / '.'
SCRIPT = PROJECT / 'tests/kernels.sh'
COMMON = PROJECT / 'scripts/common.sh'

MOCK = r'''import json,os,pathlib,sys
kind,*args=sys.argv[1:]
log=pathlib.Path(os.environ['KERNEL_MOCK_LOG'])
if kind=='image_python':
    assert args[0]=='/package/scripts/kernel_checkpoint.py'
    assert 'check' in args
    event='checkpoint-record' if '--record' in args else 'checkpoint-check'
elif '--entrypoint' in args and args[args.index('--entrypoint')+1]=='nvidia-smi':
    assert '--query-gpu=index,name,memory.total,driver_version,mig.mode.current,uuid' in args
    event='gpu-query'
elif '/deploy/tests/check_kernel_dependencies.py' in args: event='dependencies'
elif '/deploy/tests/performance_kernels.py' in args: event='single-gpu'
elif '/deploy/tests/performance_ep8.py' in args: event='ep8'
else: raise AssertionError('Unknown synthetic Docker operation '+repr(args))
with log.open('a') as stream: stream.write(json.dumps({'event':event,'arguments':args,'synthetic_cpu_mock':True})+'\n')
if event=='gpu-query':
    for i in range(8): print(f'{i}, NVIDIA A100-SXM4-80GB, 81920, 580.159.04, Disabled, GPU-SYNTHETIC-{i}')
elif event=='checkpoint-check': sys.exit(int(os.environ.get('KERNEL_CHECKPOINT_EXIT','0')))
elif event=='ep8':
    expected={'--nnodes':'1','--node-rank':'0','--nproc-per-node':'8','--rdzv-backend':'static',
              '--master-addr':'127.0.0.1','--master-port':'29500','--local-addr':'127.0.0.1',
              '--network':'none'}
    for key,value in expected.items():
        assert args.count(key)==1 and args[args.index(key)+1]==value,(key,value,args)
    assert args.count('GLOO_SOCKET_IFNAME=lo')==1
    assert '-m' in args and args[args.index('-m')+1]=='torch.distributed.run'
    print('Synthetic EP8 command assertion only; no GPU or process group executed')
    sys.exit(int(os.environ.get('KERNEL_EP8_EXIT','0')))
else: print('Synthetic CPU mock operation:',event)
'''


class KernelResumeWorkflowV3Tests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='kernel-resume-v3-cpu-')
        self.addCleanup(self.temporary.cleanup)
        self.project = Path(self.temporary.name) / 'project'
        for directory in ('scripts', 'tests', 'manifests', 'runtime'):
            (self.project / directory).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SCRIPT, self.project / 'tests/kernels.sh')
        # Use the actual USER_ARGS initializer, including the production Gloo
        # interface setting. Only Docker and checkpoint execution are mocked.
        match = re.search(r'^USER_ARGS=\(.*?^  -e HTTP_PROXY=.*?\)$', COMMON.read_text(), re.MULTILINE | re.DOTALL)
        self.assertIsNotNone(match, 'Production common USER_ARGS initializer changed')
        user_args = match.group()
        self.assertIn('-e GLOO_SOCKET_IFNAME=lo', user_args)
        helper = self.project / 'scripts/synthetic-docker.py'
        helper.write_text(MOCK)
        common = '''#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
STATE_DIR="$ROOT/runtime"
IMAGE_ID=sha256:a63405a72ce202bb8347f17fc2432be0df484b794fba0224fd210b6daff004f6
resolve_image() { :; }
fail() { printf '%s\\n' "$*" >&2; exit 1; }
__USER_ARGS__
docker() { "__PYTHON__" "$ROOT/scripts/synthetic-docker.py" docker "$@"; }
image_python() { "__PYTHON__" "$ROOT/scripts/synthetic-docker.py" image_python "$@"; }
'''.replace('__USER_ARGS__', user_args).replace('__PYTHON__', sys.executable)
        (self.project / 'scripts/common.sh').write_text(common)
        fixture = self.project / 'tests/fixture-support.txt'
        fixture.write_text('Synthetic CPU fixture; no GPU accepted.\n')
        digest = hashlib.sha256(fixture.read_bytes()).hexdigest()
        (self.project / 'manifests/kernel-test-support.sha256').write_text(digest + '  tests/fixture-support.txt\n')
        self.log = self.project / 'runtime/mock-events.jsonl'
        self.signature = self.project / 'runtime/kernels.signature'
        self.environment = os.environ | {'KERNEL_MOCK_LOG': str(self.log)}

    def run_kernel(self, **environment):
        return subprocess.run(['bash', str(self.project / 'tests/kernels.sh')],
                              env=self.environment | environment, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=15)

    def events(self):
        return [json.loads(line)['event'] for line in self.log.read_text().splitlines()]

    def test_checkpoint_hit_skips_single_but_requires_static_ep8(self):
        command = self.run_kernel(KERNEL_CHECKPOINT_EXIT='0')
        self.assertEqual(command.returncode, 0, command.stdout)
        self.assertEqual(self.events(), ['gpu-query', 'checkpoint-check', 'ep8'])
        self.assertTrue(self.signature.is_file())
        self.assertEqual(len((self.project / 'runtime/current-gpus.csv').read_text().splitlines()), 8)

    def test_checkpoint_miss_runs_dependencies_single_record_then_ep8(self):
        command = self.run_kernel(KERNEL_CHECKPOINT_EXIT='10')
        self.assertEqual(command.returncode, 0, command.stdout)
        self.assertEqual(self.events(), ['gpu-query', 'checkpoint-check', 'dependencies', 'single-gpu', 'checkpoint-record', 'ep8'])
        self.assertTrue(self.signature.is_file())

    def test_ep8_failure_never_seals_complete_signature(self):
        command = self.run_kernel(KERNEL_CHECKPOINT_EXIT='0', KERNEL_EP8_EXIT='7')
        self.assertEqual(command.returncode, 7, command.stdout)
        self.assertEqual(self.events(), ['gpu-query', 'checkpoint-check', 'ep8'])
        self.assertFalse(self.signature.exists())

    def test_complete_signature_reuses_without_repeating_any_kernel(self):
        first = self.run_kernel(KERNEL_CHECKPOINT_EXIT='0')
        self.assertEqual(first.returncode, 0, first.stdout)
        before = self.signature.read_bytes()
        second = self.run_kernel(KERNEL_CHECKPOINT_EXIT='10', KERNEL_EP8_EXIT='7')
        self.assertEqual(second.returncode, 0, second.stdout)
        self.assertIn('Reusing complete checks', second.stdout)
        self.assertEqual(self.events(), ['gpu-query', 'checkpoint-check', 'ep8', 'gpu-query'])
        self.assertEqual(self.signature.read_bytes(), before)


if __name__ == '__main__':
    unittest.main(verbosity=2)
