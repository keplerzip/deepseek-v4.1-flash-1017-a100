"""Offline runtime operations; executed by the bundled container Python."""
import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
import urllib.request

ROOT = Path('/opt/dsv411017')
MODEL = 'DeepSeek-Flash'
WEIGHT_MODEL = 'DeepSeek-V4.1-Flash-1017'
MODEL_PATH = Path('/models') / WEIGHT_MODEL


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(16 * 1024**2), b''):
            result.update(block)
    return result.hexdigest()


def load_config(path):
    return validate_config(json.loads(Path(path).read_text()))


def exact_value(actual, expected):
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(exact_value(actual[k], v) for k, v in expected.items())
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(exact_value(a, b) for a, b in zip(actual, expected))
    return actual == expected


def validate_config(config):
    protected = {
        'model': str(MODEL_PATH), 'served_model_name': MODEL,
        'pipeline_parallel_size': 1, 'enable_expert_parallel': True,
        'max_model_len': 524288,
        'kv_cache_dtype': 'fp8_ds_mla', 'engram_config': {'cpu_offload': True},
        'enable_prefix_caching': True, 'disable_chunked_mm_input': True,
        'enable_chunked_prefill': True, 'tokenizer_mode': 'deepseek_v41',
        'tool_call_parser': 'deepseek_v41', 'reasoning_parser': 'deepseek_v41',
        'all2all_backend': 'allgather_reducescatter', 'enable_eplb': True,
        'enable_prompt_tokens_details': True, 'enable_auto_tool_choice': True,
        'generation_config': 'vllm', 'gpu_memory_utilization': 0.92,
        'limit_mm_per_prompt': {'image': 999},
    }
    variable = {'tensor_parallel_size', 'data_parallel_size', 'data_parallel_size_local',
                'max_num_seqs', 'max_num_batched_tokens', 'disable_custom_all_reduce',
                'eplb_config', 'speculative_config', 'compilation_config'}
    if type(config) is not dict or set(config) - (set(protected) | variable):
        raise ValueError('Unexpected runtime option; reviewed profiles preserve native vision and protocol behavior')
    for key, expected in protected.items():
        if not exact_value(config.get(key), expected):
            raise ValueError('Deployment constraint changed: ' + key)
    tp, dp, local = (config.get(k) for k in ('tensor_parallel_size', 'data_parallel_size', 'max_num_seqs'))
    if any(type(x) is not int for x in (tp, dp, local)) or (tp, dp) not in {(8, 1), (4, 2)}:
        raise ValueError('Only TP8/DP1 or TP4/DP2 on all eight GPUs are reviewed')
    if local * dp != 32:
        raise ValueError('Global scheduler concurrency must remain exactly C32')
    if dp == 2 and not exact_value(config.get('data_parallel_size_local'), 2):
        raise ValueError('Both DP pools must run on this eight-GPU node')
    if dp == 1 and 'data_parallel_size_local' in config and not exact_value(config['data_parallel_size_local'], 1):
        raise ValueError('Unexpected local DP count')
    expected_spec = {'method': 'dspark', 'num_speculative_tokens': 5,
                     'draft_sample_method': 'greedy', 'rejection_sample_method': 'standard',
                     'use_local_argmax_reduction': True, 'enable_adaptive_verification': False}
    if not exact_value(config.get('speculative_config'), expected_spec):
        raise ValueError('Only the reviewed DSpark k5 sampling is supported; adaptive verification needs a separate SM80 backend port')
    eplb = config.get('eplb_config', {})
    expected_eplb = {'window_size': 1000, 'step_interval': 3000, 'num_redundant_experts': 0,
                     'use_async': True, 'log_balancedness': True, 'log_balancedness_interval': 500}
    if type(eplb.get('log_balancedness')) is not bool:
        raise ValueError('Invalid EPLB logging switch')
    expected_eplb['log_balancedness'] = eplb['log_balancedness']
    if not exact_value(eplb, expected_eplb) or type(config.get('disable_custom_all_reduce')) is not bool:
        raise ValueError('Unreviewed communication or EPLB configuration')
    if type(config.get('max_num_batched_tokens')) is not int or not 1028 <= config['max_num_batched_tokens'] <= 32768:
        raise ValueError('Scheduling budget must hold one complete image plus DSpark slots')
    if config['max_num_seqs'] * 6 > config['max_num_batched_tokens']:
        raise ValueError('Scheduling budget cannot hold C32 DSpark verification')
    sizes = [5, 6, 10, 12, 20, 24, 40, 48, 80, 96] + ([160, 192] if dp == 1 else [])
    expected_graph = {'mode': 0, 'cudagraph_mode': 'FULL_DECODE_ONLY',
                      'cudagraph_capture_sizes': sizes, 'max_cudagraph_capture_size': sizes[-1]}
    if not exact_value(config.get('compilation_config'), expected_graph):
        raise ValueError('Decode CUDA Graph configuration changed')
    return config


def argv(config):
    args = [config['model'], '--uds', '/state/api.sock',
            '--middleware', 'dsv41_guard.OfflineGuard']
    for key, value in config.items():
        if key == 'model':
            continue
        flag = '--' + key.replace('_', '-')
        if isinstance(value, bool):
            args.append(flag if value else '--no-' + key.replace('_', '-'))
        else:
            args += [flag, json.dumps(value, separators=(',', ':'))
                     if isinstance(value, (dict, list)) else str(value)]
    return args


def fingerprint(model, manifest):
    rows = []
    for item in manifest['files']:
        path = model / item['path']
        if path.is_symlink() or not path.resolve().is_relative_to(model.resolve()):
            raise ValueError('Symlink or external model file: ' + item['path'])
        stat = path.stat()
        if stat.st_size != item['size']:
            raise ValueError('Wrong model file size: ' + item['path'])
        rows.append((item['path'], stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns,
                     stat.st_ino, stat.st_dev))
    return hashlib.sha256(json.dumps(rows, separators=(',', ':')).encode()).hexdigest()


def verify_model(model, manifest_path, receipt_path):
    manifest = json.loads(manifest_path.read_text())
    metadata = {'source': 'ModelScope', 'model': manifest['modelscope_model_id'],
                'revision': manifest['modelscope_revision'],
                'manifest_sha256': digest(manifest_path), 'status': 'RUNNING',
                'gpu_executed': False, 'files': []}
    try:
        fingerprint(model, manifest)  # Check all paths/sizes before hashing 475 GiB.
        for i, item in enumerate(manifest['files'], 1):
            actual = digest(model / item['path'])
            if actual != item['sha256']:
                raise ValueError('ModelScope SHA256 mismatch: ' + item['path'])
            metadata['files'].append({'path': item['path'], 'sha256': actual})
            print(f'VERIFIED {i}/{len(manifest["files"])} {item["path"]}', flush=True)
        config = json.loads((model / 'config.json').read_text())
        if config['text_config']['dspark_block_size'] != 5:
            raise ValueError('Unexpected model DSpark block size')
        if config['vision_config']['max_image_tokens'] != 1024:
            raise ValueError('Unexpected original vision configuration')
        metadata.update(status='PASS', fingerprint=fingerprint(model, manifest),
                        verified_at_epoch=time.time(), model_directory=str(model))
    except Exception as exc:
        metadata.update(status='FAIL', error=str(exc))
        raise
    finally:
        write_json(receipt_path, metadata)
    return metadata


def check_receipt(model, manifest_path, receipt_path):
    manifest = json.loads(manifest_path.read_text())
    receipt = json.loads(receipt_path.read_text())
    if receipt.get('status') != 'PASS' or receipt.get('manifest_sha256') != digest(manifest_path):
        raise ValueError('Full snapshot verification required: run deploy.sh first')
    if receipt.get('fingerprint') != fingerprint(model, manifest):
        raise ValueError('Model files changed or moved: run deploy.sh to reverify')


def probe(output):
    report = {'status': 'RUNNING', 'scope': 'current target hardware and free resources',
              'observed_at_epoch': time.time(), 'errors': []}
    try:
        command = ['nvidia-smi', '--query-gpu=index,name,memory.total,driver_version,mig.mode.current',
                   '--format=csv,noheader,nounits']
        rows = subprocess.check_output(command, text=True).strip().splitlines()
        gpus = [dict(zip(['index', 'name', 'memory_mib', 'driver', 'mig'],
                        [part.strip() for part in row.split(',')])) for row in rows]
        report['gpus'] = gpus
        if len(gpus) != 8:
            report['errors'].append('Exactly eight visible A100 GPUs are required')
        for gpu in gpus:
            if 'A100' not in gpu['name'] or float(gpu['memory_mib']) < 80000:
                report['errors'].append('Requires A100 80GB: GPU ' + gpu['index'])
            if gpu['mig'] not in {'Disabled', '[N/A]', 'N/A'}:
                report['errors'].append('MIG must be disabled: GPU ' + gpu['index'])
            version = tuple(int(x) for x in gpu['driver'].split('.'))
            if version < (580, 65, 6):
                report['errors'].append('CUDA 13 runtime requires an adequate NVIDIA driver')
        report['topology'] = subprocess.check_output(['nvidia-smi', 'topo', '-m'], text=True)
        if not re.search(r'\bNV\d+\b', report['topology']):
            report['errors'].append('Expected NVLink/NVSwitch links were not found')
        processes = subprocess.check_output([
            'nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
            '--format=csv,noheader,nounits'], text=True).strip()
        report['compute_processes'] = processes
        if processes:
            report['errors'].append('GPUs are occupied; existing services were left running')
        memory = {}
        for line in Path('/proc/meminfo').read_text().splitlines():
            fields = line.split()
            if len(fields) >= 2 and fields[0] in {'MemTotal:', 'MemAvailable:'}:
                memory[fields[0].rstrip(':')] = int(fields[1]) * 1024
        report['host_memory_bytes'] = memory
        if memory['MemAvailable'] < 512 * 1024**3:
            report['errors'].append('At least 512 GiB available host RAM is required for Engram and loading')
        report['cpu_count'] = os.cpu_count()
    except Exception as exc:
        report['errors'].append(type(exc).__name__ + ': ' + str(exc))
    report['status'] = 'FAIL' if report['errors'] else 'PASS'
    write_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report['errors']:
        raise SystemExit(1)


def capacity(directory, config, output, require_replay=False):
    records = [json.loads(p.read_text()) for p in directory.glob('*.json')]
    rows = {}
    for row in records:
        if row.get('kind') == 'capacity':
            key = (row['pid'], row['data_parallel_rank'])
            if row['time_ns'] > rows.get(key, {}).get('time_ns', -1):
                rows[key] = row
    report = {'status': 'RUNNING', 'configured_context': config['max_model_len'],
              'configured_scheduler_concurrency': 32,
              'scope': 'measured KV capacity and loaded DSpark; not a full-window C32 load test',
              'gpu_executed': True, 'errors': [], 'capacity_records': list(rows.values())}
    if not rows:
        report['errors'].append('Missing measured capacity records')
    for row in rows.values():
        if type(row.get('data_parallel_rank')) is not int:
            report['errors'].append('DP rank must be an integer, excluding booleans')
        for key, wanted in [('max_model_len', 524288), ('configured_max_num_seqs', config['max_num_seqs']),
                            ('tensor_parallel_size', config['tensor_parallel_size']),
                            ('data_parallel_size', config['data_parallel_size'])]:
            if row.get(key) != wanted:
                report['errors'].append('Wrong observed runtime: ' + key)
        blocks = sum(group['full_window_blocks'] for group in row['groups'])
        if blocks <= 0 or blocks != row['blocks_per_full_window']:
            report['errors'].append('Invalid measured block accounting')
        elif row['full_windows'] != row['pool_blocks'] // blocks:
            report['errors'].append('Measured full-window capacity mismatch')
        if row['full_windows'] < 1:
            report['errors'].append('KV pool cannot hold even one requested full window')
    pools = {}
    for row in rows.values():
        pools.setdefault(row['data_parallel_rank'], []).append(row['full_windows'])
    if set(pools) != set(range(config['data_parallel_size'])):
        report['errors'].append('Missing or unexpected DP pool capacity records')
    if rows:
        report['per_dp_full_window_capacity'] = {str(rank): min(windows) for rank, windows in pools.items()}
        report['measured_full_window_capacity'] = sum(min(windows) for windows in pools.values())
        report['full_window_C32_proven'] = False
        report['capacity_sufficient_for_32_full_windows'] = (set(pools) == set(range(config['data_parallel_size']))
            and all(min(windows) >= config['max_num_seqs'] for windows in pools.values()))
    drafts = [r for r in records if r.get('kind') == 'dspark']
    report['dspark_workers'] = len({r['pid'] for r in drafts})
    if report['dspark_workers'] != 8:
        report['errors'].append('Expected DSpark observations from all eight workers')
    if any(r.get('num_speculative_tokens') != 5 for r in drafts):
        report['errors'].append('DSpark k=5 was not loaded')
    captures = [r for r in records if r.get('kind') == 'graph-capture']
    replays = [r for r in records if r.get('kind') == 'graph-replay']
    report.update(graph_capture_records=len(captures), graph_replay_records=len(replays))
    if not captures:
        report['errors'].append('Missing CUDA Graph capture telemetry')
    if require_replay and not replays:
        report['errors'].append('Missing CUDA Graph replay telemetry after smoke requests')
    if os.environ.get('VLLM_DSV41_CAPTURE_CONTEXT_COMBINE', '0') == '1':
        # Bind an actual FULL draft replay to its captured manager. A target
        # graph or a draft silently downgraded to eager cannot satisfy this.
        actual_captures = {(r['pid'], r.get('manager_id')) for r in captures
            if r.get('manager') == 'DFlashCudaGraphManager'
            and r.get('draft_context_combine_captured') is True
            and r.get('memory_estimation_probe') is False
            and any(d.get('mode') == 'FULL' for d in r.get('captured_full_graphs', []))}
        actual_replays = {r['pid'] for r in replays
            if (r['pid'], r.get('manager_id')) in actual_captures
            and r.get('manager') == 'DFlashCudaGraphManager'
            and r.get('draft_context_combine_captured') is True
            and r.get('descriptor', {}).get('mode') == 'FULL'}
        report['context_combine_full_draft_replay_workers'] = len(actual_replays)
        if require_replay and actual_replays != {r['pid'] for r in drafts}:
            report['errors'].append('Context-combine profile needs actual FULL draft graph capture/replay on every worker')
    if report['errors']:
        report['capacity_sufficient_for_32_full_windows'] = False
    report['status'] = 'FAIL' if report['errors'] else 'PASS'
    write_json(output, report)
    print(json.dumps({k: v for k, v in report.items() if k != 'capacity_records'}, ensure_ascii=False, indent=2))
    if report['errors']:
        raise SystemExit(1)


def cli_test(config):
    import vllm.platforms
    from vllm.platforms.cpu import CpuPlatform
    vllm.platforms._current_platform = CpuPlatform()
    from vllm.entrypoints.launchers.cli_args import make_arg_parser
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    parsed = make_arg_parser(FlexibleArgumentParser()).parse_args(argv(config))
    assert parsed.max_model_len == 524288 and parsed.max_num_seqs * parsed.data_parallel_size == 32
    assert (parsed.tensor_parallel_size, parsed.data_parallel_size) in {(8, 1), (4, 2)}
    assert parsed.served_model_name == [MODEL]
    assert parsed.disable_chunked_mm_input
    assert not parsed.hf_overrides
    assert parsed.uds == '/state/api.sock'
    print(json.dumps({'status': 'PASS', 'scope': 'installed backend CLI parse; no GPU inference',
                      'argv': argv(config)}))


def health(path):
    connection = http.client.HTTPConnection('localhost', timeout=5)
    connection.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.sock.settimeout(5)
    connection.sock.connect(str(path))
    try:
        connection.request('GET', '/health')
        response = connection.getresponse()
        response.read()
        if response.status != 200:
            raise ValueError('Engine health HTTP ' + str(response.status))
    finally:
        connection.close()


def network_proof(output):
    routes = Path('/proc/net/route').read_text()
    interfaces = list(Path('/sys/class/net').iterdir())
    assert {p.name for p in interfaces} == {'lo'}, 'Engine has a non-loopback network interface'
    assert not any(r.split()[1] == '00000000' for r in routes.splitlines()[1:]), 'Default route exists'
    write_json(output, {'status': 'PASS', 'interfaces': ['lo'], 'routes': routes,
                        'scope': 'engine network=none; external TCP frontend reaches only a Unix socket'})
    print('ENGINE_NETWORK_NONE=PASS')


def frontend_health(port):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(f'http://127.0.0.1:{port}/health', timeout=3) as response:
        if response.status != 200:
            raise ValueError('Frontend health failed')
    with opener.open(f'http://127.0.0.1:{port}/v1/models', timeout=3) as response:
        models = json.load(response)
    if MODEL not in {row['id'] for row in models['data']}:
        raise ValueError('Frontend serves an unexpected model')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=['serve', 'probe', 'verify', 'check', 'capacity', 'cli-test', 'argv', 'health', 'network-proof', 'frontend-health'])
    parser.add_argument('--model', type=Path, default=MODEL_PATH)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/runtime.json')
    parser.add_argument('--manifest', type=Path, default=ROOT / 'manifests/model-files.json')
    parser.add_argument('--receipt', type=Path, default=Path('/state/model-verification.json'))
    parser.add_argument('--observations', type=Path, default=Path('/state/observations'))
    parser.add_argument('--output', type=Path, default=Path('/state/report.json'))
    parser.add_argument('--require-replay', action='store_true')
    parser.add_argument('--port', type=int, default=8007)
    args = parser.parse_args()
    if args.operation == 'probe':
        return probe(args.output)
    if args.operation == 'health':
        return health('/state/api.sock')
    if args.operation == 'network-proof':
        return network_proof(args.output)
    if args.operation == 'frontend-health':
        return frontend_health(args.port)
    if args.operation == 'verify':
        verify_model(args.model, args.manifest, args.receipt)
    elif args.operation == 'check':
        check_receipt(args.model, args.manifest, args.receipt)
        print('MODEL_RECEIPT=PASS')
    else:
        config = load_config(args.config)
        if args.operation == 'cli-test':
            cli_test(config)
        elif args.operation == 'argv':
            print(json.dumps(argv(config)))
        elif args.operation == 'capacity':
            capacity(args.observations, config, args.output, args.require_replay)
        elif args.operation == 'serve':
            check_receipt(args.model, args.manifest, args.receipt)
            os.environ.pop('NCCL_ALGO', None)
            os.environ.pop('NCCL_PROTO', None)
            os.environ['DSV41_DP_SIZE'] = str(config['data_parallel_size'])
            write_json('/state/runtime.resolved.json', config)
            command = ['vllm', 'serve', *argv(config)]
            print('Starting', MODEL, 'TP', config['tensor_parallel_size'], 'DP', config['data_parallel_size'],
                  '/ DSpark5 / global C32 / 524288 tokens', flush=True)
            os.execvp(command[0], command)


if __name__ == '__main__':
    main()
