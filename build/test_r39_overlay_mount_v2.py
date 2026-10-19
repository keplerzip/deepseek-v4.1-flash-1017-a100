"""CPU-only fixed mount-list validation, no Docker or model execution."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / 'scripts/r39_overlay.sh'
PATHS = ['vllm/tokenizers/deepseek_v41.py', 'vllm/entrypoints/anthropic/serving.py',
         'vllm/entrypoints/openai/responses/protocol.py', 'dsv41_guard.py']


class OverlayMountTests(unittest.TestCase):
    def check_list(self, entries, missing=False, symlink=False, parent_symlink=False):
        with tempfile.TemporaryDirectory(prefix='r39 mount ') as directory:
            project = Path(directory)
            (project/'manifests').mkdir()
            (project/'manifests/r39-overlay-files.txt').write_text('\n'.join(entries)+'\n')
            for relative in PATHS:
                path=project/'source/r39-overlay'/relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('# fixture\n')
            path=project/'source/r39-overlay'/PATHS[0]
            if missing or symlink:
                path.unlink()
                if symlink:
                    path.symlink_to(project/'source/r39-overlay'/PATHS[1])
            if parent_symlink:
                source=project/'source/r39-overlay'
                source.rename(project/'relocated-overlay')
                source.symlink_to(project/'relocated-overlay',target_is_directory=True)
            run=subprocess.run(['bash','-c','set -euo pipefail; source "$1"; prepare_r39_overlay "$2"; printf "%s\\0" "${R39_OVERLAY_ARGS[@]}"','bash',str(HELPER),directory],capture_output=True)
            return run

    def test_all_four_mounts_are_read_only_with_space_in_path(self):
        run=self.check_list(PATHS)
        self.assertEqual(run.returncode,0,run.stderr)
        values=run.stdout.decode().strip('\0').split('\0')
        self.assertEqual(values[::2],['-v']*4)
        self.assertTrue(all(value.endswith(':ro') for value in values[1::2]))
        self.assertTrue(all('/usr/local/lib/python3.12/dist-packages/' in value for value in values[1::2]))

    def test_duplicate_and_missing_path_rejected(self):
        self.assertNotEqual(self.check_list([PATHS[0]]*4).returncode,0)
        self.assertNotEqual(self.check_list(PATHS[:-1]).returncode,0)

    def test_unknown_path_and_missing_source_rejected(self):
        self.assertNotEqual(self.check_list(PATHS[:-1]+['../../etc/passwd']).returncode,0)
        self.assertNotEqual(self.check_list(PATHS,missing=True).returncode,0)

    def test_symlink_source_rejected(self):
        self.assertNotEqual(self.check_list(PATHS,symlink=True).returncode,0)
        self.assertNotEqual(self.check_list(PATHS,parent_symlink=True).returncode,0)


if __name__=='__main__':
    unittest.main(verbosity=2)
