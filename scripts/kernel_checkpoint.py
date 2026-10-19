"""Verify reusable single-GPU numerical evidence; never skips EP8 or inference.

Legacy V2 reports did not record GPU UUIDs. Migration verifies the recorded
model/index/memory/driver/MIG and explicitly binds today's UUIDs for future
checks; it cannot prove historical UUID equality. Exit 10 means rerun the test.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import time

V2_MANIFEST = '545753a0a889ed1d571d7c3b9e5c08c99eaef44b3441bb2bd2d4ea38e285ccd6'
IMAGE_IDS = {
    'sha256:a63405a72ce202bb8347f17fc2432be0df484b794fba0224fd210b6daff004f6',
    'sha256:b89ad3acbbb21b98993284167bb838f966d393ad158d8d218afdc00b7f6a2f20',
    'sha256:fc91b6843bba21cee966f6e5f42def56cc72ca513a5faf5de001738f40a1dd40',
}
GROUPS = {'mhc', 'indexer', 'moe_align', 'dspark_kv', 'topk_reuse', 'decode_metadata', 'inverse_rope'}
CATEGORIES = {
    'compact-prefill', 'compact-paged-decode', 'stable-moe-align', 'dense-bf16-or-marlin',
    'shared-expert-stream-vs-serial', 'sparse-decode-8-warps-reference',
    'dspark-stacked-kv-native-mxfp8', 'ragged-topk-reuse-reference',
    'r14-decode-metadata', 'r14-swa-validity-graph', 'inverse-rope-reduce-epilogue',
}
MODULES = {
    'performance_kernels', 'performance_r11_kernels', 'performance_r12_kernels',
    'performance_r13_kernels', 'performance_r14_kernels', 'performance_inv_rope_kernels',
    'performance_ep8', 'performance_tp_ar', 'performance_r13_reference',
    'performance_r14_reference', 'performance_r14_cpu', 'performance_inv_rope_reference',
    'upstream.test_shared_experts_multistream', 'upstream.test_rocm_triton_attn_dsv4',
}
REFERENCES = {
    'overrides/performance/baseline/vllm/model_executor/layers/fused_moe/moe_align_block_size.py',
    'overrides/performance/baseline/vllm/models/deepseek_v4_1/amd/rocm.py',
    'overrides/performance/patched/vllm/v1/attention/ops/rocm_aiter_mla_sparse.py',
    'overrides/performance/patched/vllm/models/deepseek_v4_1/amd/rocm.py',
}


def require(value, message):
    if not value:
        raise ValueError(message)


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def sha(path):
    require(path.is_file() and not path.is_symlink(), 'Missing/nonregular evidence: ' + str(path))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe(project, relative):
    relative = PurePosixPath(relative)
    require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe evidence path')
    path = project / str(relative)
    require(not path.is_symlink() and path.resolve().is_relative_to(project), 'External/symlink evidence')
    return path


def relative(project, path):
    path = path.absolute()
    require(not path.is_symlink(), 'Symlink evidence')
    if path.resolve().is_relative_to(project):
        return str(path.relative_to(project))
    # Container /state is a writable bind of the same runtime directory that
    # also appears under the read-only /package mount. Resolve by inode, never
    # by assuming an arbitrary external directory called /state is trusted.
    runtime = project / 'runtime'
    for anchor in path.parents:
        if anchor.is_dir() and runtime.is_dir() and anchor.samefile(runtime):
            return str(PurePosixPath('runtime') / path.relative_to(anchor).as_posix())
    raise ValueError('Evidence outside project runtime bind')


def load(path):
    sha(path)
    result = json.loads(path.read_text())
    require(type(result) is dict, 'Evidence must be an object')
    return result


def gpu_rows(path):
    rows = []
    for fields in csv.reader(path.read_text().splitlines()):
        if not fields:
            continue
        require(len(fields) == 6, 'Current GPU CSV needs six fields including UUID')
        index, name, memory, driver, mig, uuid = [item.strip() for item in fields]
        require(index.isdigit() and 'A100' in name and float(memory) >= 80000, 'Requires indexed A100 80GB GPUs')
        require(mig in {'Disabled', 'N/A', '[N/A]'}, 'MIG must be disabled')
        require(tuple(int(x) for x in driver.split('.')) >= (580, 65, 6), 'Inadequate target driver')
        require(uuid.startswith('GPU-') and len(uuid) > 4, 'Missing GPU UUID')
        rows.append({'index': str(int(index)), 'name': name, 'memory_mib': str(int(float(memory))),
                     'driver': driver, 'mig': mig, 'uuid': uuid})
    require(len(rows) == 8 and {row['index'] for row in rows} == {str(x) for x in range(8)}, 'Exactly eight distinct indices required')
    require(len({row['uuid'] for row in rows}) == 8, 'GPU UUIDs must be distinct')
    return sorted(rows, key=lambda row: int(row['index']))


def hardware_matches(path, current):
    report = load(path)
    require(report.get('status') == 'PASS' and not report.get('errors'), 'Historical hardware probe did not pass')
    old = report.get('gpus')
    require(type(old) is list and len(old) == 8, 'Hardware report must contain eight GPUs')
    require(re.search(r'\bNV\d+\b', report.get('topology', '')), 'Historical topology lacks NVLink evidence')
    expected = {row['index']: row for row in current}
    seen = set()
    for row in old:
        index = str(row.get('index'))
        require(index in expected and index not in seen, 'Historical GPU indices differ')
        seen.add(index)
        actual = expected[index]
        require(row.get('name') == actual['name'] and float(row['memory_mib']) == float(actual['memory_mib'])
                and row.get('driver') == actual['driver'] and row.get('mig') == actual['mig'],
                'Historical GPU model/memory/driver/MIG differs')
        if row.get('uuid'):
            require(row['uuid'] == actual['uuid'], 'Historical GPU UUID differs')
    return report, all(bool(row.get('uuid')) for row in old)


def kernel_report(path):
    report = load(path)
    require(report.get('status') == 'PASS' and report.get('full_model_tested') is False, 'Single-GPU report did not pass')
    groups = report.get('groups', {})
    require(type(groups) is dict and set(groups) == GROUPS and all(
        type(groups[name]) in (int, bool) and groups[name] == 1 for name in GROUPS), 'All single-GPU groups must be enabled')
    cases = report.get('cases')
    require(type(cases) is list and cases and all(type(row) is dict and row.get('status') == 'PASS' for row in cases),
            'Every numerical case must pass')
    require(CATEGORIES <= {row.get('name') for row in cases}, 'Missing required numerical category')
    mhc = [row for row in cases if 'name' not in row and type(row.get('tokens')) is int
           and row['tokens'] > 0 and type(row.get('carried_pre_mix')) is bool]
    require(mhc and {row['carried_pre_mix'] for row in mhc} == {False, True}, 'Missing unnamed mHC carried/noncarried checks')
    require('A100' in report.get('gpu', ''), 'Single-GPU result is not an A100 result')
    return report


def dependencies(project, path):
    report = load(path)
    require(report.get('status') == 'DEPENDENCIES_CHECKED', 'Kernel dependencies were not checked')
    require(report.get('gpu_executed') is False and report.get('cuda_context_initialized') is False,
            'Dependency report has invalid CPU scope')
    for flag in ('production_interfaces_loaded', 'dspark_production_methods_loaded',
                 'reference_loaders_checked', 'upstream_file_loading_checked'):
        require(report.get(flag) is True, 'Dependency closure incomplete: ' + flag)
    net = report.get('network_isolation', {})
    require(net.get('status') == 'PASS' and net.get('interfaces') == ['lo'] and net.get('default_route') is False
            and not net.get('failures') and not net.get('observation_errors'), 'No passed offline network evidence')
    modules, refs = report.get('modules'), report.get('references')
    require(type(modules) is list and len(modules) == 14 and {row.get('name') for row in modules} == MODULES,
            'Expected all fourteen dependency modules exactly once')
    require(type(refs) is list and len(refs) == 4 and {row.get('path') for row in refs} == REFERENCES,
            'Expected all four native numerical references')
    sources = {}
    for row in modules + refs:
        name = row.get('path', '')
        expected_path = ('tests/' + row['name'].replace('.', '/') + '.py') if row in modules else name
        require(name == expected_path, 'Unexpected numerical module path')
        observed = sha(safe(project, name))
        require(observed == row.get('sha256'), 'Numerical source changed: ' + name)
        if row in refs:
            require(row.get('pinned_image_bytes_equal') is True, 'Reference did not match pinned image')
            if '/patched/' in name:
                require(row.get('installed_production_bytes_equal') is True, 'Patched reference was not installed production code')
        sources[name] = observed
    return sources


def public_source_receipt(project):
    manifest = project / 'PUBLIC_SOURCE_SHA256SUMS'
    entries = {}
    for line in manifest.read_text().splitlines():
        checksum, name = line.split('  ', 1)
        require(re.fullmatch(r'[0-9a-f]{64}', checksum) and name not in entries,
                'Invalid public source checksum entry')
        require(sha(safe(project, name)) == checksum, 'Public source changed: ' + name)
        entries[name] = checksum
    require({'scripts/kernel_checkpoint.py', 'scripts/common.sh', 'configs/runtime.json',
             'manifests/runtime.env'} <= set(entries), 'Incomplete public source inventory')
    return manifest, {'status': 'PUBLIC_SOURCE_VERIFIED',
                      'public_source_manifest_sha256': sha(manifest),
                      'image_unchanged': True}


def receipt(project, legacy):
    if not legacy and (project / 'PUBLIC_SOURCE_SHA256SUMS').is_file():
        return public_source_receipt(project)
    v2 = project / 'runtime/hotfix-20261019-v2.json'
    v3 = project / 'runtime/hotfix-20261019-v3.json'
    if legacy and v2.is_file():
        report = load(v2)
        require(report.get('status') == 'APPLIED' and report.get('combined_manifest_sha256') == V2_MANIFEST
                and report.get('image_unchanged') is True, 'Not a valid original V2 installation receipt')
        return v2, report
    report = load(v3)
    require(report.get('status') == 'APPLIED' and report.get('image_unchanged') is True, 'Missing V3 installation provenance')
    if legacy:
        require(report.get('previous_manifest_sha256') == V2_MANIFEST, 'V3 was not applied over the exact V2 manifest')
    else:
        require(report.get('combined_manifest_sha256') == sha(project / 'manifests/package.sha256'),
                'V3 installation receipt does not match current package')
    return v3, report


def bind(project, paths):
    return {relative(project, path): sha(path) for path in paths}


def create(args, report_path, legacy):
    project = args.project
    current = gpu_rows(args.gpu_signature)
    report_path = safe(project, relative(project, report_path))
    require(report_path.parent.parent == project / 'runtime' and report_path.parent.name.startswith('kernel-check-')
            and report_path.name == 'kernels.json', 'Only runtime/kernel-check-*/kernels.json can be reused')
    kernel_report(report_path)
    dep = report_path.with_name('dependencies.json')
    sources = dependencies(project, dep)
    install_path, install = receipt(project, legacy)
    hardware = project / 'runtime/hardware.before-v3.json' if legacy else project / 'runtime/hardware.json'
    if legacy and not hardware.exists():
        hardware = project / 'runtime/hardware.json'
        old_hardware = load(hardware)
        observed = old_hardware.get('observed_at_epoch')
        applied = install.get('applied_at_epoch')
        require(type(observed) in (int, float) and type(applied) in (int, float)
                and applied <= observed <= report_path.stat().st_mtime + 1,
                'Unpreserved hardware probe is newer than the legacy test or lacks V2 time proof')
    _, original_uuid_recorded = hardware_matches(hardware, current)
    if not legacy:
        # start.sh refreshes hardware.json each time; preserve this test's exact
        # hardware evidence so an unrelated probe timestamp cannot invalidate it.
        snapshot = args.output.with_name(args.output.stem + '.hardware.json')
        atomic_json(snapshot, load(hardware))
        hardware = snapshot
        receipt_snapshot = args.output.with_name(args.output.stem + '.installation.json')
        atomic_json(receipt_snapshot, install)
        install_path = receipt_snapshot
    result = {'schema': 1, 'status': 'PASS', 'scope': 'single-GPU numerical checks only; EP8 is never skipped here',
              'origin': ('V2_MIGRATED' if legacy else 'PUBLIC_SOURCE_RECORDED' if install.get('status') == 'PUBLIC_SOURCE_VERIFIED' else 'V3_RECORDED'), 'image_id': args.image_id,
              'gpu_signature_sha256': sha(args.gpu_signature), 'gpus': current,
              'original_uuid_recorded': original_uuid_recorded,
              'legacy_uuid_equality_proven': original_uuid_recorded if legacy else None,
              'current_uuids_bound_at_checkpoint_creation': True,
              'historical_gpu_identity_scope': 'exact UUIDs' if original_uuid_recorded else 'model/index/memory/driver/MIG only; historical UUIDs unavailable',
              'numerical_source_sha256': sources,
              'evidence_sha256': bind(project, [report_path, dep, hardware, install_path]),
              'kernel_report': relative(project, report_path), 'dependencies_report': relative(project, dep),
              'hardware_report': relative(project, hardware), 'installation_receipt': relative(project, install_path),
              'created_at_epoch': time.time(), 'ep8_validation': 'NOT_SKIPPED_BY_THIS_CHECKPOINT',
              'full_model_validation': 'NOT_TESTED_BY_THIS_CHECKPOINT', 'reused': legacy}
    atomic_json(args.output, result)
    return result


def reuse(args):
    saved = load(args.output)
    require(saved.get('schema') == 1 and saved.get('status') == 'PASS', 'Existing checkpoint was not a verified PASS')
    require(saved.get('image_id') == args.image_id, 'Native image identity changed')
    require(saved.get('gpu_signature_sha256') == sha(args.gpu_signature) and saved.get('gpus') == gpu_rows(args.gpu_signature),
            'Current GPU UUID/driver/index signature changed')
    bindings = saved.get('evidence_sha256')
    require(type(bindings) is dict and len(bindings) == 4, 'Incomplete checkpoint evidence bindings')
    for name, checksum in bindings.items():
        require(sha(safe(args.project, name)) == checksum, 'Original report/dependency/hardware/receipt changed: ' + name)
    installation = load(safe(args.project, saved['installation_receipt']))
    if installation.get('status') == 'PUBLIC_SOURCE_VERIFIED':
        _, current_source = public_source_receipt(args.project)
        require(current_source['public_source_manifest_sha256'] == installation['public_source_manifest_sha256'],
                'Public source release changed')
    report_path = safe(args.project, saved['kernel_report'])
    kernel_report(report_path)
    observed = dependencies(args.project, safe(args.project, saved['dependencies_report']))
    require(observed == saved.get('numerical_source_sha256'), 'Numerical test/reference sources changed')
    hardware_matches(safe(args.project, saved['hardware_report']), gpu_rows(args.gpu_signature))
    saved.update(reused=True, last_verified_at_epoch=time.time())
    atomic_json(args.output, saved)
    return saved


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--image-id', required=True)
    parser.add_argument('--gpu-signature', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('command', choices=['check'])
    parser.add_argument('--record', type=Path)
    args = parser.parse_args(argv)
    args.project = args.project.resolve()
    args.output = args.output.absolute()
    output_valid = False
    try:
        require(args.image_id in IMAGE_IDS, 'Checkpoint requires the exact reviewed native image')
        relative(args.project, args.output)
        output_valid = True
        sha(args.gpu_signature)
        if args.record is not None:
            result = create(args, args.record.absolute(), False)
        elif args.output.exists():
            # Never re-migrate a stale checkpoint onto a different UUID set.
            result = reuse(args)
        else:
            failures = []
            result = None
            candidates = sorted((args.project / 'runtime').glob('kernel-check-*/kernels.json'),
                                key=lambda path: path.stat().st_mtime, reverse=True)
            for path in candidates:
                try:
                    result = create(args, path, True)
                    break
                except (OSError, ValueError, TypeError, KeyError) as error:
                    failures.append(str(path.name) + ': ' + str(error))
            require(result is not None, 'No valid prior single-GPU checkpoint: ' + '; '.join(failures[:5]))
        print(json.dumps({'status': 'PASS', 'origin': result['origin'], 'reused': result['reused'],
                          'original_uuid_recorded': result['original_uuid_recorded'],
                          'output': str(args.output), 'ep8_validation': 'NOT_SKIPPED_BY_THIS_CHECKPOINT'}))
        return 0
    except (OSError, ValueError, TypeError, KeyError) as error:
        if output_valid:
            atomic_json(args.output, {'schema': 1, 'status': 'MISS', 'reason': str(error),
                    'checked_at_epoch': time.time(), 'single_gpu_rerun_required': True,
                    'ep8_validation': 'NOT_SKIPPED_BY_THIS_CHECKPOINT'})
        print(json.dumps({'status': 'MISS', 'reason': str(error), 'exit_code': 10}))
        return 10


if __name__ == '__main__':
    raise SystemExit(main())
