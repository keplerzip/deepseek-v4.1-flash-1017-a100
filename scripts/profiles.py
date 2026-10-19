"""Resolve reviewed A/B profiles without relaxing the user's fixed capabilities."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import runtime

ENVIRONMENT_VALUES = {
    'VLLM_DETERMINISTIC_MOE_ALIGN': {'0', '1'},
    'VLLM_AMPERE_DENSE_BF16_MIN_TOKENS': {'32'},
    'VLLM_DSV41_CAPTURE_CONTEXT_COMBINE': {'0', '1'},
}


def merge(base, changes):
    result = copy.deepcopy(base)
    for key, value in changes.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else copy.deepcopy(value)
    return result


def resolve(project, name):
    catalog_path = project / 'configs/profiles.json'
    catalog = json.loads(catalog_path.read_text())
    if name not in catalog['profiles']:
        raise ValueError('Unknown speed profile: ' + name)
    profile = catalog['profiles'][name]
    config = merge(json.loads((project / 'configs/runtime.json').read_text()), profile['config'])
    runtime.validate_config(config)
    environment = {**catalog['base_environment'], **profile['environment']}
    if set(environment) != set(ENVIRONMENT_VALUES):
        raise ValueError('Unexpected profile environment keys')
    for key, value in environment.items():
        if value not in ENVIRONMENT_VALUES[key]:
            raise ValueError('Unsupported profile value: ' + key)
    gates = profile.get('gates', [])
    if set(gates) - {'tp-ar', 'context-graph'}:
        raise ValueError('Unknown GPU correctness gate')
    if config['disable_custom_all_reduce'] is False and 'tp-ar' not in gates:
        raise ValueError('TP custom AllReduce requires its actual GPU/graph gate')
    if environment['VLLM_DSV41_CAPTURE_CONTEXT_COMBINE'] == '1' and 'context-graph' not in gates:
        raise ValueError('Context projection capture requires its actual GPU gate')
    record = {'name': name, 'description': profile['description'], 'configuration': config,
              'environment': environment, 'gates': gates,
              'catalog_sha256': hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
              'global_concurrency': config['max_num_seqs'] * config['data_parallel_size'],
              'max_model_len': config['max_model_len'], 'vision': 'original configuration, no processor override',
              'gpu_acceptance': 'NOT_RUN_BY_SELECTION', 'tps_gain': 'NOT_MEASURED'}
    return record


def validate_record(record, directory):
    runtime.validate_config(record['configuration'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=['list', 'select', 'prepare', 'env', 'requires'])
    parser.add_argument('name', nargs='?')
    parser.add_argument('--project', type=Path, default=Path('/package'))
    parser.add_argument('--state', type=Path, default=Path('/state'))
    parser.add_argument('--gate', choices=['tp-ar', 'context-graph'])
    args = parser.parse_args()
    catalog = json.loads((args.project / 'configs/profiles.json').read_text())
    selection = args.state / 'profile-selection.json'
    name = args.name or (json.loads(selection.read_text())['name'] if selection.exists() else catalog['default'])
    if args.operation == 'list':
        for key, value in catalog['profiles'].items():
            print(key + ': ' + value['description'])
        return
    args.state.mkdir(parents=True, exist_ok=True)
    if args.operation in {'select', 'prepare'}:
        record = resolve(args.project, name)
        validate_record(record, args.state)
        if args.operation == 'select':
            runtime.write_json(selection, {'name': name})
        runtime.write_json(args.state / 'runtime.active.json', record['configuration'])
        runtime.write_json(args.state / 'speed-profile.json', record)
        print(json.dumps(record, ensure_ascii=False, indent=2))
    elif args.operation == 'env':
        record = json.loads((args.state / 'speed-profile.json').read_text())
        for key, value in record['environment'].items():
            sys.stdout.buffer.write(b'-e\0' + (key + '=' + value).encode() + b'\0')
    elif args.operation == 'requires':
        record = json.loads((args.state / 'speed-profile.json').read_text())
        print('required' if args.gate in record['gates'] else 'skip')


if __name__ == '__main__':
    main()
