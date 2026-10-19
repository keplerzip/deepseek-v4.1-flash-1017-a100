"""Extend the original R3 summary with strict R3.9 cache/multi-image evidence."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import shutil


spec = importlib.util.spec_from_file_location('original_r3_summary', Path(__file__).with_name('oneclick-summary.py'))
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)
original.STAGES = original.STAGES + ('r39-alignment',)
original_render = original.render


def render_with_extended_scope(result):
    return original_render(result).replace(
        '图像验收限单图/三图OCR和图片工具关联；不代表全部视觉任务质量已证明。',
        '基础图像验收与扩展5/8图顺序、工具图片归属、view_image状态分别记录；质量结论限这些任务。')


original.render = render_with_extended_scope

REQUIRED = {f'cache-{p}-{kind}-{fmt}' for p in ('chat', 'responses', 'messages')
            for kind in ('text', 'image') for fmt in ('json', 'sse')}
REQUIRED |= {f'cache-chat-{kind}-sse-implicit-usage' for kind in ('text', 'image')}
REQUIRED |= {f'cache-{p}-changed-image-{fmt}' for p in ('chat', 'responses', 'messages') for fmt in ('json', 'sse')}
REQUIRED |= {f'cache-messages-dynamic-tail-system-{fmt}' for fmt in ('json', 'sse')}
REQUIRED |= {'cache-chat-long-image-json'}
REQUIRED |= {f'vision-{p}-ordered-{count}-{fmt}' for p in ('chat', 'responses', 'messages')
             for count, fmt in ((5, 'json'), (8, 'sse'))}
REQUIRED |= {f'vision-{p}-reversed-tool-image-id-{fmt}' for p in ('chat', 'responses', 'messages') for fmt in ('json', 'sse')}
REQUIRED |= {f'vision-responses-codex-view-image-{fmt}' for fmt in ('json', 'sse')}
assert len(REQUIRED) == 37


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(16 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def extend(base, run, context, result):
    relative = (context.get('report_paths') or {}).get('r39-alignment')
    if not isinstance(relative, str) or not relative:
        raise ValueError('Missing exact R3.9 alignment report path')
    rel = PurePosixPath(relative)
    if rel.is_absolute() or '..' in rel.parts:
        raise ValueError('Unsafe R3.9 alignment report path')
    project = (base / original.PROJECT).resolve()
    source = project / relative
    if source.is_symlink() or not source.resolve().is_relative_to(project):
        raise ValueError('R3.9 report escaped this deployment')
    report = json.loads(source.read_text())
    if (report.get('schema') != 1 or report.get('suite') != 'r39-alignment' or
            report.get('model') != 'DeepSeek-Flash' or report.get('context') != 524288 or
            report.get('concurrency') != 32 or not report.get('finished_at')):
        raise ValueError('R3.9 report identity/scope/completion mismatch')
    cases = report.get('cases')
    if not isinstance(cases, list) or len(cases) != len(REQUIRED):
        raise ValueError('R3.9 cache/multi-image report requires all 37 cases')
    names = [row.get('name') for row in cases if isinstance(row, dict)]
    if len(names) != len(REQUIRED) or len(set(names)) != len(names) or set(names) != REQUIRED:
        raise ValueError('R3.9 case set missing, duplicated or unexpected')
    destination = run / 'reports/r39-alignment.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    record = {'status': report.get('status'), 'source_relative_path': relative,
              'evidence_relative_path': 'reports/r39-alignment.json', 'sha256': sha(destination),
              'bytes': destination.stat().st_size}
    result['reports']['r39-alignment'] = record
    evidence = []
    trace = source.with_suffix('.requests.jsonl')
    paths = [trace] if trace.exists() else []
    raw_dir = source.with_suffix('.evidence')
    if raw_dir.exists():
        paths.extend(p for p in raw_dir.rglob('*') if p.is_file())
    for path in paths:
        if path.is_symlink() or not path.resolve().is_relative_to(source.parent):
            raise ValueError('Unsafe raw R3.9 evidence path')
        name = path.relative_to(source.parent)
        target = run / 'reports/r39-alignment-raw' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        evidence.append({'path': str(target.relative_to(run)), 'sha256': sha(target), 'bytes': target.stat().st_size})
    if not evidence:
        raise ValueError('R3.9 raw request/response evidence missing')
    failed = [row['name'] for row in cases if row.get('status') != 'PASS']
    for row in cases:
        kind = 'cache' if row['name'].startswith('cache-') else 'vision'
        protocol = row['name'].split('-')[1]
        fmt = 'sse' if '-sse' in row['name'] else 'json'
        if row.get('kind') != kind or row.get('protocol') != protocol or row.get('format') != fmt:
            failed.append(row['name'] + ': kind/protocol/format mismatch')
        if not row.get('requests'):
            failed.append(row['name'] + ': no request evidence')
        if kind == 'cache':
            if any(type(row.get(field)) is not int or row[field] < 0
                   for field in ('cold_cached_tokens', 'hot_cached_tokens')):
                failed.append(row['name'] + ': missing real cached-token counts')
            elif row['hot_cached_tokens'] < (512 if 'dynamic-tail-system' in row['name'] else 1):
                failed.append(row['name'] + ': hot cache read below required token threshold')
            if 'dynamic-tail-system' not in row['name']:
                cached, total = row.get('multi_turn_cached_tokens'), row.get('multi_turn_input_total')
                if type(cached) is not int or cached < 1:
                    failed.append(row['name'] + ': missing positive retained-history cache read')
                if type(total) is not int or total < 1 or (type(cached) is int and cached > total):
                    failed.append(row['name'] + ': invalid retained-history input/cache token counts')
                if not isinstance(row.get('multi_turn_usage'), dict) or not row['multi_turn_usage']:
                    failed.append(row['name'] + ': missing native retained-history usage')
                if row.get('multi_turn_format') != fmt:
                    failed.append(row['name'] + ': retained-history response format mismatch')
    result['r39_alignment'] = {'status': 'PASS' if report.get('status') == 'PASS' and not failed else 'FAIL',
                               'cases': cases, 'required_cases': 37, 'raw_evidence': evidence,
                               'newapi_billing_end_to_end': 'NOT_RUN',
                               'scope': 'Direct backend native cache fields, retained-history JSON/SSE and representative 5/8 ordered/tool-related images; visual quality conclusions limited to these tasks'}
    if report.get('status') != 'PASS' or failed:
        raise ValueError('R3.9 cache/multi-image cases failed: ' + ', '.join(failed[:20]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--model-directory', type=Path)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--exit-code', type=int, default=0)
    parser.add_argument('--interrupted', default='')
    parser.add_argument('--check-config', action='store_true')
    args = parser.parse_args()
    if args.check_config:
        forwarded = ['--base', str(args.base), '--check-config']
        if args.model_directory:
            forwarded += ['--model-directory', str(args.model_directory)]
        return original.main(forwarded)
    if args.run is None:
        parser.error('--run required')
    base, run = args.base.resolve(), args.run.resolve()
    result = original.build_summary(base, run, args.exit_code, args.interrupted, args.model_directory)
    try:
        context = json.loads((run / 'context.json').read_text())
        extend(base, run, context, result)
    except Exception as error:
        original.issue(result, 'FAIL', 'R3.9缓存/多图验收未签收: ' + str(error))
    if result['issues']:
        result['status'] = 'FAIL' if any(row['severity'] == 'FAIL' for row in result['issues']) else 'INCOMPLETE'
        result['exit_code'] = args.exit_code or 1
        for row in result.get('benchmarks', {}).values():
            row['signoff'] = False
        for row in result.get('diagnostics', {}).values():
            row['signoff'] = False
    original.write_output(run, result)
    alignment = result.get('r39_alignment', {})
    lines = ['\nR3.9缓存与多图扩展验收: ' + alignment.get('status', '未完成'),
             '端口8005；本机与Docker默认网关入口；模型服务免API Key。',
             '项目 | 状态 | 冷缓存token | 热缓存token | 多轮缓存token']
    for row in alignment.get('cases', []):
        lines.append('%s | %s | %s | %s | %s' % (row['name'], row.get('status'),
                     row.get('cold_cached_tokens'), row.get('hot_cached_tokens'), row.get('multi_turn_cached_tokens')))
    lines.append('New API网关账单/倍率实链路: NOT_RUN；后端原始协议用量证据另存。')
    with (run / 'summary.txt').open('a') as stream:
        stream.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
