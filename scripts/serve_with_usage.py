"""Retain the pinned runtime contract, with R3.9-compatible native usage flags."""
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, '/opt/dsv411017/scripts')
import runtime

original_argv = runtime.argv


def usage_argv(config):
    runtime.validate_config(config)
    args = original_argv(config)
    assert '--api-key' not in args
    return args + ['--enable-force-include-usage', '--enable-per-request-metrics',
                   '--prefix-caching-hash-algo', 'sha256']


runtime.argv = usage_argv
original_cli_test = runtime.cli_test


def cli_test(config):
    original_cli_test(config)
    from vllm.entrypoints.launchers.cli_args import make_arg_parser
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    parsed = make_arg_parser(FlexibleArgumentParser()).parse_args(usage_argv(config))
    assert parsed.enable_force_include_usage is True
    assert parsed.enable_per_request_metrics is True
    assert parsed.enable_prompt_tokens_details is True
    assert parsed.enable_prefix_caching is True
    assert parsed.prefix_caching_hash_algo == 'sha256'
    assert not parsed.api_key
    print(json.dumps({'status': 'CLI_CHECKED', 'gpu_executed': False,
                      'prefix_caching': True, 'prompt_tokens_details': True,
                      'force_include_usage': True, 'per_request_metrics': True,
                      'prefix_hash': 'sha256', 'authentication': 'disabled'}), flush=True)


runtime.cli_test = cli_test
os.environ.pop('VLLM_API_KEY', None)
os.environ.pop('R3_API_KEY', None)

if __name__ == '__main__':
    runtime.main()
