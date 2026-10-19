"""Weight-free TP8 custom all-reduce correctness and latency probe.

Run with the pinned image's ``torchrun --standalone --nproc-per-node=8``.
``--import-check`` only checks installed Python interfaces and never reports a
GPU PASS. This operator probe cannot establish model TPS or DSpark acceptance.
"""
import argparse
from datetime import timedelta
import gc
import inspect
import json
import os
from pathlib import Path
import statistics
import time
import traceback


def write_json(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(row, indent=2) + '\n')
    temporary.replace(path)


def installed_api():
    import torch
    from vllm.config import VllmConfig, ParallelConfig, set_current_vllm_config
    from vllm.distributed import parallel_state as ps
    from vllm.distributed.communication_op import tensor_model_parallel_all_reduce
    required = {
        ps.init_distributed_environment: ('world_size', 'rank', 'local_rank', 'backend', 'timeout'),
        ps.initialize_model_parallel: ('tensor_model_parallel_size', 'pipeline_model_parallel_size', 'backend'),
        ps.graph_capture: ('device',),
        ParallelConfig: ('tensor_parallel_size', 'data_parallel_size', 'disable_custom_all_reduce'),
    }
    for function, names in required.items():
        params = inspect.signature(function).parameters
        assert all(name in params for name in names), function.__name__
    return torch, VllmConfig, ParallelConfig, set_current_vllm_config, ps, tensor_model_parallel_all_reduce


def main(args):
    # Test a single explicit implementation pair. Other optional all-reduce
    # backends must not intercept the dispatcher and produce a false comparison.
    for name in ('VLLM_ALLREDUCE_USE_SYMM_MEM', 'VLLM_ALLREDUCE_USE_FLASHINFER',
                 'VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC', 'VLLM_USE_NCCL_SYMM_MEM'):
        os.environ[name] = '0'
    os.environ.pop('VLLM_HIER_ALL_REDUCE', None)
    os.environ.pop('VLLM_MAX_SIZE_MB_CUSTOM_ALL_REDUCE', None)
    torch, VllmConfig, ParallelConfig, set_config, ps, tp_all_reduce = installed_api()
    if args.import_check:
        print(json.dumps({'status': 'IMPORT_CHECKED', 'gpu_executed': False,
                          'gpu_validation': 'NOT_RUN', 'torch': torch.__version__,
                          'scope': 'installed API signatures only; no collectives executed'}), flush=True)
        return
    if not torch.cuda.is_available() or torch.cuda.device_count() != 8:
        raise RuntimeError('GPU validation requires eight visible A100 GPUs; no CPU PASS is allowed')
    import torch.distributed as dist
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    run_id = os.environ.get('TORCHELASTIC_RUN_ID', 'port-' + os.environ.get('MASTER_PORT', 'unknown'))
    assert int(os.environ['WORLD_SIZE']) == 8
    device = torch.device('cuda', local_rank)
    torch.cuda.set_device(device)
    assert torch.cuda.get_device_capability(device) == (8, 0)
    assert 'A100' in torch.cuda.get_device_name(device)
    config = VllmConfig(parallel_config=ParallelConfig(
        tensor_parallel_size=8, data_parallel_size=1,
        enable_expert_parallel=True, all2all_backend='allgather_reducescatter',
        disable_custom_all_reduce=False))
    cases = []
    with set_config(config):
        ps.set_custom_all_reduce(True)
        ps.init_distributed_environment(world_size=8, rank=rank, local_rank=local_rank,
                                        backend='nccl', timeout=timedelta(seconds=180))
        ps.initialize_model_parallel(tensor_model_parallel_size=8,
                                     pipeline_model_parallel_size=1, backend='nccl')
        group = ps.get_tp_group()
        comm = group.device_communicator
        assert group.world_size == 8 and comm is not None
        ca = comm.ca_comm
        assert ca is not None and not ca.disabled and ca.fully_connected
        assert ca.max_size == 8 * 1024**2, ca.max_size
        nccl = comm.pynccl_comm
        assert nccl is not None and not nccl.disabled
        counts = {'custom_calls': 0, 'custom_outputs': 0, 'pynccl_calls': 0}
        original_custom, original_nccl = ca.custom_all_reduce, nccl.all_reduce

        def custom_counted(x):
            counts['custom_calls'] += 1
            result = original_custom(x)
            counts['custom_outputs'] += int(result is not None)
            return result

        def nccl_counted(*positional, **kwargs):
            counts['pynccl_calls'] += 1
            return original_nccl(*positional, **kwargs)

        ca.custom_all_reduce, nccl.all_reduce = custom_counted, nccl_counted

        def reference(x):
            expected = x.float().clone()
            dist.all_reduce(expected, group=group.device_group)
            return expected

        def check(actual, expected, dtype):
            assert actual is not None
            # BF16 summation order differs from NCCL/FP32. Check an explicit
            # small absolute + relative error budget, not bitwise equality.
            atol, rtol = (0.0078125, 0.03125) if dtype == torch.bfloat16 else (3e-6, 3e-6)
            torch.testing.assert_close(actual.float(), expected, atol=atol, rtol=rtol)
            return float((actual.float() - expected).abs().max().item())

        def capture(fn, x):
            # Use vLLM's real TP/PP capture contexts so custom_AR registers the
            # graph's input pointers, exactly as the model graph manager does.
            dist.barrier(group=group.cpu_group)
            with ps.graph_capture(device=device) as context:
                for _ in range(args.warmup):
                    fn(x)
                torch.cuda.synchronize(device)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=context.stream):
                    output = fn(x)
            torch.cuda.synchronize(device)
            return graph, output

        def latency(graph):
            for _ in range(args.warmup):
                graph.replay()
            torch.cuda.synchronize(device)
            dist.barrier(group=group.cpu_group)
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            for _ in range(args.iterations):
                graph.replay()
            end.record()
            end.synchronize()
            return begin.elapsed_time(end) * 1000 / args.iterations

        # wo_b returns [tokens, hidden=5120]. HC=4 is mixed locally; the real
        # attention AR does NOT send [tokens, 4, 5120]. These are its actual
        # DSpark/target single and C32 decode widths.
        for rows in (5, 6, 160, 192):
            for dtype in (torch.bfloat16, torch.float32):
                x = torch.full((rows, 5120), rank + 1, device=device, dtype=dtype)
                assert ca.should_custom_ar(x), (rows, dtype, x.nbytes)
                before = counts.copy()
                actual = tp_all_reduce(x)
                torch.cuda.synchronize(device)
                torch.testing.assert_close(actual, torch.full_like(x, 36), atol=0, rtol=0)
                assert counts['custom_outputs'] > before['custom_outputs']
                assert counts['pynccl_calls'] == before['pynccl_calls']
                torch.manual_seed(11017 + rank)
                x.copy_(torch.randn_like(x).mul_(0.0625))
                eager_error = check(tp_all_reduce(x), reference(x), dtype)
                graph_custom, out_custom = capture(tp_all_reduce, x)
                x.mul_(0.5).add_((rank + 1) / 512)
                out_custom.fill_(float('nan'))
                graph_custom.replay()
                torch.cuda.synchronize(device)
                graph_error = check(out_custom, reference(x), dtype)
                graph_nccl, out_nccl = capture(nccl.all_reduce, x)
                x.mul_(0.75).sub_((rank + 1) / 1024)
                out_nccl.fill_(float('nan'))
                graph_nccl.replay()
                graph_custom.replay()
                torch.cuda.synchronize(device)
                expected = reference(x)
                nccl_error = check(out_nccl, expected, dtype)
                second_error = check(out_custom, expected, dtype)
                # Alternate measurement order across cases to reduce a fixed
                # always-first bias. Report all ranks, not just the fastest.
                order = ((graph_nccl, 'nccl'), (graph_custom, 'custom')) if rows in (6, 192) else ((graph_custom, 'custom'), (graph_nccl, 'nccl'))
                times = {name: latency(graph) for graph, name in order}
                cases.append({'rows': rows, 'hidden': 5120, 'dtype': str(dtype),
                              'bytes': x.nbytes, 'known_constant_exact': True,
                              'max_abs_error_vs_fp32': max(eager_error, graph_error, second_error),
                              'nccl_max_abs_error_vs_fp32': nccl_error,
                              'graph_input_mutations': 2,
                              'custom_graph_us': times['custom'], 'nccl_graph_us': times['nccl'],
                              'operator_speedup': times['nccl'] / times['custom'],
                              'custom_calls_during_eager_capture': counts['custom_calls'] - before['custom_calls']})
                graph_custom.reset()
                graph_nccl.reset()
                del graph_custom, graph_nccl, out_custom, out_nccl, x, actual, expected
                gc.collect()
                torch.cuda.synchronize(device)

        fallback = []
        # The selection rule is strict '< max_size': equality and larger
        # payloads must go through PyNCCL without entering custom_AR.
        for nbytes in (ca.max_size, ca.max_size + 32):
            x = torch.full((nbytes // 2,), rank + 1, dtype=torch.bfloat16, device=device)
            assert not ca.should_custom_ar(x)
            before = counts.copy()
            actual = tp_all_reduce(x)
            torch.cuda.synchronize(device)
            torch.testing.assert_close(actual, torch.full_like(x, 36), atol=0, rtol=0)
            assert counts['custom_calls'] == before['custom_calls']
            assert counts['pynccl_calls'] == before['pynccl_calls'] + 1
            graph, output = capture(tp_all_reduce, x)
            x.add_(0.5)
            output.fill_(float('nan'))
            graph.replay()
            torch.cuda.synchronize(device)
            torch.testing.assert_close(output, torch.full_like(x, 40), atol=0, rtol=0)
            assert counts['custom_calls'] == before['custom_calls']
            graph.reset()
            del graph, output, actual, x
            gc.collect()
            fallback.append({'bytes': nbytes, 'custom_bypassed': True,
                             'pynccl_selected': True, 'graph_mutation_exact': True})

        row = {'status': 'PASS', 'gpu_executed': True, 'rank': rank, 'run_id': run_id,
               'world_size': group.world_size, 'gpu': torch.cuda.get_device_name(device),
               'torch': torch.__version__, 'cuda': torch.version.cuda,
               'nccl': torch.cuda.nccl.version(), 'custom_enabled': True,
               'fully_connected': ca.fully_connected, 'custom_max_bytes': ca.max_size,
               'counts': counts, 'cases': cases, 'fallback': fallback,
               'iterations': args.iterations, 'warmup': args.warmup,
               'scope': 'operator correctness and latency only; no weights/model TPS/DSpark acceptance',
               'limitations': ['BF16 reduction is not associative; tolerances do not imply identical generated tokens.',
                               'mHC HC=4 state is not the attention output all-reduce payload.']}
        all_rows = [None] * 8
        dist.all_gather_object(all_rows, row, group=group.cpu_group)
        if rank == 0:
            summary_cases = []
            for i, case in enumerate(cases):
                custom = [r['cases'][i]['custom_graph_us'] for r in all_rows]
                nccl_times = [r['cases'][i]['nccl_graph_us'] for r in all_rows]
                summary_cases.append({'rows': case['rows'], 'dtype': case['dtype'], 'bytes': case['bytes'],
                                      'custom_median_rank_us': statistics.median(custom),
                                      'custom_max_rank_us': max(custom),
                                      'nccl_median_rank_us': statistics.median(nccl_times),
                                      'nccl_max_rank_us': max(nccl_times),
                                      'max_rank_operator_speedup': max(nccl_times) / max(custom)})
            summary = {
                'status': 'PASS', 'gpu_executed': True, 'run_id': run_id, 'passed_ranks': len(all_rows),
                'cases': summary_cases, 'scope': row['scope'], 'limitations': row['limitations']}
        dist.barrier(group=group.cpu_group)
        ca.custom_all_reduce, nccl.all_reduce = original_custom, original_nccl
        del comm, ca, nccl, group
        gc.collect()
        torch.cuda.synchronize(device)
        ps.destroy_model_parallel()
        ps.destroy_distributed_environment()
        write_json(args.output / f'rank-{rank}.json', row)
        if rank == 0:
            deadline = time.monotonic() + 30
            while True:
                paths = [args.output / f'rank-{r}.json' for r in range(8)]
                if all(p.exists() for p in paths):
                    reports = [json.loads(p.read_text()) for p in paths]
                    if all(r.get('run_id') == run_id for r in reports):
                        assert all(r['status'] == 'PASS' and r['gpu_executed'] for r in reports)
                        break
                if time.monotonic() >= deadline:
                    raise RuntimeError('Not all eight ranks completed communicator teardown')
                time.sleep(0.1)
            write_json(args.output / 'summary.json', summary)
        print(json.dumps({'rank': rank, 'status': 'PASS', 'gpu_executed': True}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('/state/tp-ar'))
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--iterations', type=int, default=50)
    parser.add_argument('--import-check', action='store_true')
    options = parser.parse_args()
    if options.warmup < 1 or options.iterations < 1:
        parser.error('warmup and iterations must be positive')
    try:
        main(options)
    except BaseException as exc:
        write_json(options.output / ('rank-' + os.environ.get('RANK', 'unstarted') + '.json'), {
            'status': 'FAIL', 'gpu_executed': False, 'error': repr(exc),
            'traceback': traceback.format_exc(), 'scope': 'GPU test did not complete'})
        raise
