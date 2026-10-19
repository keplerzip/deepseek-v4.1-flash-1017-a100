"""CPU-only synthetic evidence fixtures; no GPU/model validation."""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'scripts/kernel_checkpoint.py'
SPEC = importlib.util.spec_from_file_location('kernel_checkpoint_v3', SOURCE)
checkpoint = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checkpoint)


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.project = Path(self.temporary.name) / 'project'
        self.runtime = self.project / 'runtime'
        self.run = self.runtime / 'kernel-check-CPU-FIXTURE'
        self.run.mkdir(parents=True)
        self.output = self.runtime / 'single-gpu-checkpoint.json'
        self.gpu = self.runtime / 'current-gpus.csv'
        self.image = sorted(checkpoint.IMAGE_IDS)[0]
        self.gpu.write_text('\n'.join(f'{i}, NVIDIA A100-SXM4-80GB, 81920, 580.159.04, Disabled, GPU-FIXTURE-{i}' for i in range(8)) + '\n')
        current = checkpoint.gpu_rows(self.gpu)
        hardware = {'status': 'PASS', 'errors': [], 'topology': 'GPU0 GPU1 NV12',
                    'observed_at_epoch': time.time() - 20,
                    'gpus': [{key: value for key, value in row.items() if key != 'uuid'} for row in current]}
        checkpoint.atomic_json(self.runtime / 'hardware.before-v3.json', hardware)
        checkpoint.atomic_json(self.runtime / 'hardware.json', hardware)
        checkpoint.atomic_json(self.runtime / 'hotfix-20261019-v2.json',
            {'status': 'APPLIED', 'combined_manifest_sha256': checkpoint.V2_MANIFEST,
             'image_unchanged': True, 'applied_at_epoch': time.time() - 30})
        report = {'status': 'PASS', 'full_model_tested': False, 'gpu': 'NVIDIA A100-SXM4-80GB',
                  'groups': {name: 1 for name in checkpoint.GROUPS},
                  'cases': [{'name': name, 'status': 'PASS'} for name in checkpoint.CATEGORIES] +
                           [{'tokens': 32, 'carried_pre_mix': value, 'status': 'PASS'} for value in (False, True)]}
        checkpoint.atomic_json(self.run / 'kernels.json', report)
        modules, refs = [], []
        for name in sorted(checkpoint.MODULES):
            path = 'tests/' + name.replace('.', '/') + '.py'
            target = self.project / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text('# CPU fixture only: ' + name + '\n')
            modules.append({'name': name, 'path': path, 'sha256': checkpoint.sha(target)})
        for path in sorted(checkpoint.REFERENCES):
            target = self.project / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text('# CPU fixture only reference\n')
            refs.append({'path': path, 'sha256': checkpoint.sha(target), 'pinned_image_bytes_equal': True,
                         'installed_production_bytes_equal': True})
        dependencies = {'status': 'DEPENDENCIES_CHECKED', 'gpu_executed': False,
                        'cuda_context_initialized': False, 'modules': modules, 'references': refs,
                        'network_isolation': {'status': 'PASS', 'interfaces': ['lo'], 'default_route': False,
                                              'failures': [], 'observation_errors': []}}
        for flag in ('production_interfaces_loaded', 'dspark_production_methods_loaded',
                     'reference_loaders_checked', 'upstream_file_loading_checked'):
            dependencies[flag] = True
        checkpoint.atomic_json(self.run / 'dependencies.json', dependencies)

    def call(self, record=False):
        argv = ['--project', str(self.project), '--image-id', self.image, '--gpu-signature', str(self.gpu),
                '--output', str(self.output), 'check']
        if record:
            argv += ['--record', str(self.run / 'kernels.json')]
        with contextlib.redirect_stdout(io.StringIO()):
            return checkpoint.main(argv)

    def mutate(self, path, action):
        value = json.loads(path.read_text())
        action(value)
        checkpoint.atomic_json(path, value)

    def test_migrate_and_reuse_honestly_binds_current_uuid(self):
        self.assertEqual(self.call(), 0)
        result = json.loads(self.output.read_text())
        self.assertEqual(result['origin'], 'V2_MIGRATED')
        self.assertFalse(result['original_uuid_recorded'])
        self.assertFalse(result['legacy_uuid_equality_proven'])
        self.assertEqual(len(result['numerical_source_sha256']), 18)
        self.assertEqual(result['ep8_validation'], 'NOT_SKIPPED_BY_THIS_CHECKPOINT')
        self.assertEqual(self.call(), 0)

    def test_launcher_change_does_not_invalidate_single_gpu(self):
        self.assertEqual(self.call(), 0)
        (self.project / 'tests/kernels.sh').write_text('Changed EP8 launcher only\n')
        self.assertEqual(self.call(), 0)

    def test_uuid_or_image_change_rejects_without_remigration(self):
        self.assertEqual(self.call(), 0)
        self.gpu.write_text(self.gpu.read_text().replace('GPU-FIXTURE-0', 'GPU-OTHER-0'))
        self.assertEqual(self.call(), 10)
        self.assertEqual(self.call(), 10)

    def test_report_and_dependency_tampering_rejects(self):
        for filename in ('kernels.json', 'dependencies.json'):
            self.assertEqual(self.call(), 0)
            path = self.run / filename
            path.write_text(path.read_text() + '\n')
            self.assertEqual(self.call(), 10)
            self.output.unlink()

    def test_changed_numerical_source_rejects(self):
        self.assertEqual(self.call(), 0)
        path = self.project / 'tests/performance_kernels.py'
        path.write_text(path.read_text() + '# changed\n')
        self.assertEqual(self.call(), 10)

    def test_incomplete_or_failed_case_or_disabled_group_rejects(self):
        original = json.loads((self.run / 'kernels.json').read_text())
        for kind in ('category', 'case', 'group', 'mhc'):
            report = copy.deepcopy(original)
            if kind == 'category':
                report['cases'] = [row for row in report['cases'] if row.get('name') != 'r14-swa-validity-graph']
            elif kind == 'case':
                report['cases'][0]['status'] = 'FAIL'
            elif kind == 'group':
                report['groups']['indexer'] = 0
            else:
                report['cases'] = [row for row in report['cases'] if 'name' in row]
            checkpoint.atomic_json(self.run / 'kernels.json', report)
            self.output.unlink(missing_ok=True)
            self.assertEqual(self.call(), 10, kind)

    def test_dependency_closure_network_and_hardware_must_match(self):
        original = json.loads((self.run / 'dependencies.json').read_text())
        for kind in ('module', 'reference', 'network'):
            value = copy.deepcopy(original)
            if kind == 'module':
                value['modules'].pop()
            elif kind == 'reference':
                value['references'][0]['pinned_image_bytes_equal'] = False
            else:
                value['network_isolation']['interfaces'] = ['eth0', 'lo']
            checkpoint.atomic_json(self.run / 'dependencies.json', value)
            self.output.unlink(missing_ok=True)
            self.assertEqual(self.call(), 10, kind)
        checkpoint.atomic_json(self.run / 'dependencies.json', original)
        self.mutate(self.runtime / 'hardware.before-v3.json', lambda value: value['gpus'][0].update(driver='580.159.05'))
        self.output.unlink(missing_ok=True)
        self.assertEqual(self.call(), 10)

    def test_v3_previous_manifest_can_prove_v2_origin(self):
        (self.runtime / 'hotfix-20261019-v2.json').unlink()
        checkpoint.atomic_json(self.runtime / 'hotfix-20261019-v3.json', {'status': 'APPLIED',
            'previous_manifest_sha256': checkpoint.V2_MANIFEST, 'image_unchanged': True})
        self.assertEqual(self.call(), 0)

    def test_overwritten_hardware_fallback_requires_original_time_proof(self):
        (self.runtime / 'hardware.before-v3.json').unlink()
        self.assertEqual(self.call(), 0)
        self.output.unlink()
        self.mutate(self.runtime / 'hardware.json', lambda value: value.update(observed_at_epoch=time.time() + 10))
        self.assertEqual(self.call(), 10)

    def test_fresh_record_preserves_hardware_snapshot_across_reprobe(self):
        path = self.project / 'manifests/package.sha256'
        path.parent.mkdir(parents=True)
        path.write_text('CPU fixture package\n')
        checkpoint.atomic_json(self.runtime / 'hotfix-20261019-v3.json', {'status': 'APPLIED',
            'combined_manifest_sha256': checkpoint.sha(path), 'image_unchanged': True})
        self.assertEqual(self.call(record=True), 0)
        result = json.loads(self.output.read_text())
        self.assertEqual(result['origin'], 'V3_RECORDED')
        self.mutate(self.runtime / 'hardware.json', lambda value: value.update(observed_at_epoch=time.time() + 10))
        self.mutate(self.runtime / 'hotfix-20261019-v3.json', lambda value: value.update(applied_at_epoch=time.time() + 10))
        self.assertEqual(self.call(), 0)

    def public_source_fixture(self):
        for name in ('scripts/kernel_checkpoint.py', 'scripts/common.sh', 'configs/runtime.json', 'manifests/runtime.env'):
            path = self.project / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('# synthetic public source fixture\n')
        files = sorted(path for path in self.project.rglob('*') if path.is_file() and 'runtime' not in path.relative_to(self.project).parts)
        manifest = self.project / 'PUBLIC_SOURCE_SHA256SUMS'
        manifest.write_text(''.join(checkpoint.sha(path) + '  ' + str(path.relative_to(self.project)) + '\n' for path in files))
        return manifest

    def test_public_source_fresh_record_needs_no_legacy_installer_receipt(self):
        self.public_source_fixture()
        self.assertEqual(self.call(record=True), 0)
        result = json.loads(self.output.read_text())
        self.assertEqual(result['origin'], 'PUBLIC_SOURCE_RECORDED')
        self.assertEqual(self.call(), 0)

    def test_public_source_modified_file_rejects_checkpoint_reuse(self):
        self.public_source_fixture()
        self.assertEqual(self.call(record=True), 0)
        (self.project / 'scripts/common.sh').write_text('# unexpected change\n')
        self.assertEqual(self.call(), 10)

    def test_external_or_symlink_report_rejects(self):
        report = self.run / 'kernels.json'
        outside = Path(self.temporary.name) / 'outside.json'
        report.rename(outside)
        report.symlink_to(outside)
        self.assertEqual(self.call(), 10)


if __name__ == '__main__':
    unittest.main(verbosity=2)
