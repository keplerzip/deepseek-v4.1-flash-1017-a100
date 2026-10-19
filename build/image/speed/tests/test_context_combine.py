"""Run with runtime Python/torch; CPU logic checks do not load model weights.

--gpu requires a real SM80 device and the installed speed overlay. It uses the
production ReplicatedLinear/ModelOpt MXFP8/Marlin dispatcher, 5120x15360 synthetic
weights, and RMSNorm. It checks projection dispatch and context graph replay,
not complete DSpark inference, attention/KV kernels, acceptance, or model TPS.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import ExitStack
from enum import Enum
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import patch

try:
    import torch
except ModuleNotFoundError:
    torch = None

ROOT = Path(__file__).resolve().parents[1]
RELATIVE = "vllm/v1/worker/gpu/spec_decode/dflash/speculator.py"


class Mode(Enum):
    NONE = 0
    FULL = 1


def load_installer():
    spec = importlib.util.spec_from_file_location("speed_installer", ROOT / "install.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def context_class():
    source = ast.parse((ROOT / "patched" / RELATIVE).read_text())
    original = next(n for n in source.body if isinstance(n, ast.ClassDef)
                    and n.name == "DFlashSpeculator")
    wanted = {"_num_graph_context_tokens", "_precompute_context_kv",
              "_prepare_graph_context", "_store_hidden_states",
              "_prepare_context_kv_for_execution"}
    methods = [n for n in original.body if isinstance(n, ast.FunctionDef)
               and n.name in wanted]
    for node in methods:
        node.decorator_list = []
    tree = ast.fix_missing_locations(ast.Module(body=[ast.ClassDef(
        name="Context", bases=[], keywords=[], body=methods, decorator_list=[])],
        type_ignores=[]))
    scope = {"torch": torch, "Any": object, "CUDAGraphMode": Mode, "PAD_SLOT_ID": -1}
    exec(compile(tree, str(ROOT / "patched" / RELATIVE), "exec"), scope)
    return scope["Context"]


def harness(device="cpu", enabled=True):
    cls = context_class()
    obj = cls()
    obj.max_num_tokens = 24
    obj.num_speculative_steps = 5
    obj.hidden_states = torch.zeros((24, 4), device=device)
    obj.context_positions = torch.arange(24, device=device)
    obj._aux_staging = torch.zeros((12, 12), device=device) if enabled else None
    obj._context_slot_mappings = torch.full((2, 24), -1, dtype=torch.int64, device=device)
    obj._layer_group_idx = [1, 0, 1]
    weights = torch.arange(48, device=device, dtype=torch.float32).reshape(12, 4) / 48
    obj.calls = []
    obj.caches = [torch.full((40, 4), -99.0, device=device) for _ in range(3)]

    def combine(aux):
        obj.calls.append(tuple(aux.shape))
        return aux @ weights

    def precompute(hidden, positions, slots):
        if slots is None:
            return
        for cache, mapping in zip(obj.caches, slots):
            # No host-dependent branch or dynamic output shape in CUDA graphs.
            # PAD writes go to a disposable last slot and cannot alter real KV.
            real = mapping >= 0
            safe = torch.where(real, mapping, cache.shape[0] - 1)
            cache.index_copy_(0, safe, hidden)

    obj.model = SimpleNamespace(combine_hidden_states=combine,
                                precompute_and_store_context_kv=precompute)
    obj.reference = lambda aux: torch.cat(aux, dim=-1) @ weights
    return obj


class InstallerTests(unittest.TestCase):
    def test_install_idempotent_and_rollback(self):
        installer = load_installer()
        manifest = json.loads((ROOT / "manifest.json").read_text())
        with tempfile.TemporaryDirectory() as tmp:
            for row in manifest["files"]:
                target = Path(tmp) / row["path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / row["baseline_source"], target)
            installer.install(Path(tmp), ROOT)
            for row in manifest["files"]:
                self.assertEqual(installer.digest(Path(tmp) / row["path"]), row["patched_sha256"])
            installer.install(Path(tmp), ROOT)
            installer.install(Path(tmp), ROOT, rollback=True)
            for row in manifest["files"]:
                self.assertEqual(installer.digest(Path(tmp) / row["path"]), row["base_sha256"])

    def test_unexpected_bytes_rejected_without_mutation(self):
        manifest = json.loads((ROOT / "manifest.json").read_text())
        with tempfile.TemporaryDirectory() as tmp:
            for row in manifest["files"]:
                target = Path(tmp) / row["path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / row["baseline_source"], target)
            target = Path(tmp) / RELATIVE
            target.write_text("unexpected\n")
            with self.assertRaisesRegex(RuntimeError, "Unexpected"):
                load_installer().install(Path(tmp), ROOT)
            self.assertEqual(target.read_text(), "unexpected\n")
            for row in manifest["files"]:
                if row["path"] != RELATIVE:
                    self.assertEqual(load_installer().digest(Path(tmp) / row["path"]), row["base_sha256"])

    def test_payloads_and_upstream_are_pinned(self):
        manifest = json.loads((ROOT / "manifest.json").read_text())
        for row in manifest["files"]:
            self.assertEqual(hashlib.sha256((ROOT / "patched" / row["path"]).read_bytes()).hexdigest(),
                             row["patched_sha256"])
        for row in manifest["upstream_sources"]:
            self.assertEqual(hashlib.sha256((ROOT / row["file"]).read_bytes()).hexdigest(), row["sha256"])


@unittest.skipIf(torch is None, "run in the offline runtime image with torch")
class ContextTests(unittest.TestCase):
    def aux(self, rows, device="cpu", offset=0):
        return [torch.arange(rows * 4, device=device, dtype=torch.float32).reshape(rows, 4)
                + offset + i * 7 for i in range(3)]

    def test_enabled_projects_inside_graph_callback(self):
        obj = harness()
        aux = self.aux(6)
        obj._store_hidden_states(6, torch.empty(0), aux, 6)
        self.assertEqual(obj.calls, [])
        obj._prepare_graph_context(1)
        torch.testing.assert_close(obj.hidden_states[:6], obj.reference(aux))
        self.assertEqual(obj.calls, [(6, 12)])

    def test_disabled_eager_path_and_missing_aux_fallback(self):
        obj = harness(enabled=False)
        aux = self.aux(8)
        obj._store_hidden_states(6, torch.empty(0), aux, 6)
        torch.testing.assert_close(obj.hidden_states[:6], obj.reference(aux)[:6])
        self.assertEqual(obj.calls, [(8, 12)])
        expected = torch.randn(5, 4)
        obj._store_hidden_states(5, expected, None)
        torch.testing.assert_close(obj.hidden_states[:5], expected)

    def test_prefill_tail_and_per_layer_kv_groups(self):
        obj = harness()
        aux = self.aux(17)
        obj._context_slot_mappings[0, :17] = torch.arange(17)
        obj._context_slot_mappings[1, :17] = torch.arange(17) + 17
        obj._store_hidden_states(17, torch.empty(0), aux, 12)
        self.assertEqual(obj.calls, [(5, 12)])
        obj._prepare_context_kv_for_execution(SimpleNamespace(cg_mode=Mode.FULL, num_reqs=2), 17, False)
        obj._prepare_graph_context(2)
        expected = obj.reference(aux)
        for cache, start in zip(obj.caches, (17, 0, 17)):
            torch.testing.assert_close(cache[start:start + 17], expected)

    def test_short_and_dummy_batches_clear_stale_writes(self):
        obj = harness()
        obj._context_slot_mappings.fill_(3)
        desc = SimpleNamespace(cg_mode=Mode.FULL, num_reqs=2)
        obj._prepare_context_kv_for_execution(desc, 4, False)
        self.assertTrue((obj._context_slot_mappings[:, 4:12] == -1).all())
        obj._prepare_context_kv_for_execution(desc, 4, True)
        self.assertTrue((obj._context_slot_mappings[:, :12] == -1).all())

    def test_enabled_rejects_missing_aux(self):
        with self.assertRaisesRegex(ValueError, "auxiliary"):
            harness()._store_hidden_states(6, torch.empty(0), None, 6)

    def test_eager_execution_projects_and_writes_all_rows(self):
        obj = harness()
        aux = self.aux(9)
        obj._context_slot_mappings[:, :9] = torch.arange(9)
        obj._store_hidden_states(9, torch.empty(0), aux)
        obj._prepare_context_kv_for_execution(SimpleNamespace(cg_mode=Mode.NONE), 9, False)
        for cache in obj.caches:
            torch.testing.assert_close(cache[:9], obj.reference(aux))


def gpu_check():
    if torch is None or not torch.cuda.is_available():
        raise RuntimeError("GPU validation NOT_RUN: --gpu requires a real CUDA device")
    if torch.cuda.get_device_capability() != (8, 0):
        raise RuntimeError("GPU validation requires SM80; other devices cannot validate this A100 gate")
    from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config
    config = VllmConfig(compilation_config=CompilationConfig(mode=0, custom_ops=['all']))
    # CustomOp (including RMSNorm) resolves its dispatch from the current
    # compilation config during construction AND execution. Keep a real config
    # active for every module import/construction, eager call and graph capture.
    with torch.inference_mode(), set_current_vllm_config(config):
        return production_gpu_check()


def production_gpu_check():
    # Imports intentionally happen only after detecting a real GPU. The pinned
    # image disables Triton on a CPU-only host; importing GPU model modules there
    # does not establish (and may not even permit) a production-kernel check.
    import inspect
    from vllm.config.compilation import CUDAGraphMode
    from vllm.model_executor.kernels.linear.mxfp8.marlin import MarlinMxfp8LinearKernel
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.model_executor.layers.linear import ReplicatedLinear
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptLinearMethod, QuantSpec, CkptCtx, kMxfp8Static, kMxfp8Dynamic,
    )
    from vllm.model_executor.layers.quantization.utils import marlin_utils_fp8
    from vllm.models.deepseek_v4_1.nvidia.dspark import DSparkDeepseekV4Model
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

    installed = Path(inspect.getsourcefile(DFlashSpeculator))
    manifest = json.loads((ROOT / "manifest.json").read_text())
    spec_manifest = next(row for row in manifest["files"] if row["path"] == RELATIVE)
    assert hashlib.sha256(installed.read_bytes()).hexdigest() == spec_manifest["patched_sha256"], (
        "Install the pinned speed overlay before running its production GPU gate")
    # Test the real inherited methods, not only AST-extracted helper copies.
    for name in ("capture", "propose", "_prepare_graph_context", "_store_hidden_states"):
        assert getattr(DSparkSpeculator, name) is getattr(DFlashSpeculator, name), name

    hidden, width, max_tokens, max_reqs, eps = 5120, 15360, 256, 32, 1e-20
    device = torch.device('cuda', torch.cuda.current_device())
    quant = SimpleNamespace(online_quantization_config=None, weight_block_size=(32, 32))
    quant.get_quant_method = lambda *a, **kw: ModelOptLinearMethod(
        QuantSpec(weight=kMxfp8Static, activation=kMxfp8Dynamic),
        CkptCtx(scale_block_size=(32, 32)))
    previous_dtype = torch.get_default_dtype()
    previous_threshold = os.environ.get('VLLM_AMPERE_DENSE_BF16_MIN_TOKENS')
    os.environ['VLLM_AMPERE_DENSE_BF16_MIN_TOKENS'] = '32'
    owner = torch.nn.Module()
    try:
        torch.set_default_dtype(torch.bfloat16)
        with ExitStack() as stack:
            # ReplicatedLinear never shards this projection. Only provide the
            # rank identity that its parameter constructors normally get from
            # an initialized TP group; no compute/backend function is stubbed.
            for module in ('vllm.model_executor.layers.linear', 'vllm.model_executor.parameter'):
                stack.enter_context(patch(module + '.get_tensor_model_parallel_rank', return_value=0))
                stack.enter_context(patch(module + '.get_tensor_model_parallel_world_size', return_value=1))
            stack.enter_context(torch.device(device))
            owner.main_proj = ReplicatedLinear(
                width, hidden, bias=False, return_bias=False, quant_config=quant,
                prefix='model.main_proj')
            owner.main_norm = RMSNorm(hidden, eps=eps, dtype=torch.bfloat16)
        owner.combine_hidden_states = MethodType(DSparkDeepseekV4Model.combine_hidden_states, owner)
        owner.requires_grad_(False)
        layer = owner.main_proj
        assert isinstance(layer.quant_method.kernel, MarlinMxfp8LinearKernel), (
            'Gate requires the production SM80 MXFP8 Marlin dispatcher')
        torch.manual_seed(59511)
        # Real checkpoint geometry: e4m3 weight [5120,15360], exponent scale
        # [160,480] expanded by the production 32x32 scale loader.
        weight = (torch.randn((hidden, width), device=device, dtype=torch.float32)
                  * 0.03125).to(torch.float8_e4m3fn)
        scales = torch.randint(125, 129, (hidden // 32, width // 32),
                               dtype=torch.uint8, device=device)
        layer.weight.weight_loader(layer.weight, weight)
        layer.weight_scale.weight_loader(layer.weight_scale, scales.view(torch.float8_e8m0fnu))
        expanded = scales.repeat_interleave(32, dim=0)
        torch.testing.assert_close(layer.weight_scale.view(torch.uint8), expanded, atol=0, rtol=0)
        # Independent reference dequantization, before Marlin repacks weights.
        reference_weight = (weight.float() * torch.exp2(expanded.float() - 127)
                            .repeat_interleave(32, dim=1)).bfloat16()
        norm_weight = torch.empty(hidden, dtype=torch.bfloat16, device=device).uniform_(0.75, 1.25)
        owner.main_norm.weight.copy_(norm_weight)
        layer.quant_method.process_weights_after_loading(layer)
        assert layer.ampere_bf16_min_tokens == 32
        assert layer.ampere_bf16_weight.shape == (hidden, width)
        torch.testing.assert_close(layer.ampere_bf16_weight, reference_weight, atol=0, rtol=0)
        del weight, scales, expanded

        calls = []
        original_linear = torch.nn.functional.linear
        original_marlin = marlin_utils_fp8.apply_mxfp8_marlin_linear

        def counted_linear(x, w, bias=None):
            if tuple(w.shape) == (hidden, width) and x.shape[-1] == width:
                calls.append((int(x.numel() // x.shape[-1]), 'bf16'))
            return original_linear(x, w, bias)

        def counted_marlin(*args, **kwargs):
            x = kwargs.get('input', args[0] if args else None)
            if x is not None and x.shape[-1] == width:
                calls.append((int(x.numel() // x.shape[-1]), 'marlin'))
            return original_marlin(*args, **kwargs)

        def path(rows):
            return 'bf16' if rows >= 32 else 'marlin'

        def aux(rows, seed):
            generator = torch.Generator(device=device).manual_seed(seed)
            return [torch.randn((rows, hidden), device=device, dtype=torch.bfloat16,
                                generator=generator).mul_(0.0625) for _ in range(3)]

        def check(got, expected):
            torch.testing.assert_close(got, expected, atol=3e-2, rtol=2e-2)
            error = (got.float() - expected.float()).abs()
            return {'max_abs_error': float(error.max().item()),
                    'rms_error': float(error.square().mean().sqrt().item())}

        def mathematical_reference(states):
            # BF16 dequantized GEMM followed by an independent FP32 RMSNorm.
            projected = torch.mm(states, reference_weight.t())
            normalized = projected.float() * torch.rsqrt(projected.float().square().mean(-1, keepdim=True) + eps)
            return (normalized.to(torch.bfloat16) * norm_weight).to(torch.bfloat16)

        def fixture(enabled=True):
            obj = DSparkSpeculator.__new__(DSparkSpeculator)
            obj.max_num_tokens, obj.max_num_reqs, obj.num_speculative_steps = max_tokens, max_reqs, 5
            obj.hidden_states = torch.empty((max_tokens, hidden), dtype=torch.bfloat16, device=device)
            obj.context_positions = torch.arange(max_tokens, dtype=torch.int64, device=device)
            obj._aux_staging = torch.zeros((max_reqs * 6, width), dtype=torch.bfloat16, device=device) if enabled else None
            obj._context_slot_mappings = torch.full((2, max_tokens), -1, dtype=torch.int64, device=device)
            obj._layer_group_idx = [1, 0, 1]
            obj.caches = [torch.full((2 * max_tokens + 1, hidden), -99.0,
                                     dtype=torch.bfloat16, device=device) for _ in range(3)]

            def store(hidden_states, positions, slots):
                if slots is None:
                    return
                for cache, mapping in zip(obj.caches, slots):
                    # A synthetic consumer isolates projection routing. PAD
                    # goes to a disposable final row; real KV/RoPE/quant/cache
                    # kernels are explicitly outside this gate's scope.
                    safe = torch.where(mapping >= 0, mapping, cache.shape[0] - 1)
                    cache.index_copy_(0, safe, hidden_states)
            obj.model = SimpleNamespace(combine_hidden_states=owner.combine_hidden_states,
                                        precompute_and_store_context_kv=store)
            return obj

        independent, eager_shape_checks, graph_checks = [], [], []
        with (patch.object(torch.nn.functional, 'linear', new=counted_linear),
              patch.object(marlin_utils_fp8, 'apply_mxfp8_marlin_linear', new=counted_marlin)):
            for rows in (1, 5, 6, 31, 32, 33, 160, 192):
                states = torch.cat(aux(rows, 1000 + rows), dim=-1)
                calls.clear()
                actual = owner.combine_hidden_states(states)
                assert calls == [(rows, path(rows))], calls
                independent.append({'rows': rows, 'actual_kernel': path(rows),
                                    **check(actual, mathematical_reference(states))})

            # The baseline combines every padded auxiliary row before slicing.
            # Both disabled and eager feature paths MUST preserve that GEMM
            # row count; comparing only active rows could miss a dispatch change.
            for enabled in (False, True):
                obj = fixture(enabled)
                for padded_rows in (31, 32, 33, 40):
                    states = aux(padded_rows, 2000 + padded_rows)
                    expected = owner.combine_hidden_states(torch.cat(states, dim=-1))[:6].clone()
                    calls.clear()
                    obj._store_hidden_states(6, torch.empty(0, device=device), states,
                                             6 if not enabled else 0)
                    assert calls == [(padded_rows, path(padded_rows))], calls
                    eager_shape_checks.append({'enabled': enabled, 'active_rows': 6,
                                               'original_auxiliary_rows': padded_rows,
                                               'actual_projection_rows': calls[0][0],
                                               'actual_kernel': calls[0][1],
                                               **check(obj.hidden_states[:6], expected)})
                del obj
                gc.collect()

            # C1 and C32 contexts use k+1 rows even though the DSpark query has
            # only k rows. Context48 additionally exercises a captured BF16
            # prefix. Prefill tails hit 31/32/33 to straddle the dispatcher.
            schedules = {
                1: [(6, 6, False), (1, 16, False), (31, 32, False),
                    (32, 32, False), (33, 40, False), (37, 40, False),
                    (38, 40, False), (39, 40, False), (4, 16, False), (34, 40, True)],
                8: [(48, 48, False), (31, 32, False), (32, 32, False),
                    (33, 40, False), (79, 96, False), (80, 96, False), (81, 96, False)],
                32: [(192, 192, False), (160, 192, False), (223, 224, False),
                     (224, 224, False), (225, 256, False), (6, 16, False)],
            }
            for num_graph_reqs, schedule in schedules.items():
                obj = fixture()
                context_rows = obj._num_graph_context_tokens(num_graph_reqs)
                calls.clear()
                for _ in range(3):
                    obj._prepare_graph_context(num_graph_reqs)
                torch.cuda.synchronize()
                calls.clear()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    obj._prepare_graph_context(num_graph_reqs)
                assert calls == [(context_rows, path(context_rows))], calls
                addresses = [obj._aux_staging.data_ptr(), obj.hidden_states.data_ptr(),
                             obj._context_slot_mappings.data_ptr()]
                for index, (rows, padded_rows, dummy) in enumerate(schedule):
                    states = aux(padded_rows, 3000 + 100 * num_graph_reqs + index)
                    calls.clear()
                    expected = owner.combine_hidden_states(torch.cat(states, dim=-1))[:rows].clone()
                    assert calls == [(padded_rows, path(padded_rows))], calls
                    for cache in obj.caches:
                        cache.fill_(-99)
                    obj.hidden_states.fill_(float('nan'))
                    obj._aux_staging.fill_(9 + index)
                    # Poison stale slots, then replace just the actual target
                    # rows as prepare_dflash_inputs does before PAD cleanup.
                    obj._context_slot_mappings.fill_(7)
                    permutation = torch.arange(rows, device=device).roll(index + 1)
                    obj._context_slot_mappings[0, :rows] = permutation
                    obj._context_slot_mappings[1, :rows] = permutation + max_tokens
                    calls.clear()
                    obj._store_hidden_states(rows, torch.empty(0, device=device), states, context_rows)
                    tail_rows = max(0, rows - context_rows)
                    assert calls == ([(tail_rows, path(tail_rows))] if tail_rows else []), calls
                    obj._prepare_context_kv_for_execution(
                        SimpleNamespace(cg_mode=CUDAGraphMode.FULL, num_reqs=num_graph_reqs), rows, dummy)
                    graph.replay()
                    torch.cuda.synchronize()
                    numerical = check(obj.hidden_states[:rows], expected)
                    if rows < context_rows:
                        assert bool((obj._context_slot_mappings[:, rows:context_rows] == -1).all())
                    for cache, group in zip(obj.caches, obj._layer_group_idx):
                        expected_cache = torch.full_like(cache[:-1], -99)
                        touched = torch.zeros(2 * max_tokens, dtype=torch.bool, device=device)
                        if not dummy:
                            expected_cache.index_copy_(0, permutation + group * max_tokens, expected)
                            touched[permutation + group * max_tokens] = True
                        check(cache[:-1], expected_cache)
                        torch.testing.assert_close(cache[:-1][~touched], expected_cache[~touched], atol=0, rtol=0)
                    assert addresses == [obj._aux_staging.data_ptr(), obj.hidden_states.data_ptr(),
                                         obj._context_slot_mappings.data_ptr()]
                    graph_checks.append({'graph_requests': num_graph_reqs,
                                         'graph_context_rows': context_rows, 'active_target_rows': rows,
                                         'original_auxiliary_rows': padded_rows,
                                         'original_eager_kernel': path(padded_rows),
                                         'captured_prefix_kernel': path(context_rows),
                                         'eager_tail_rows': tail_rows,
                                         'eager_tail_kernel': path(tail_rows) if tail_rows else None,
                                         'dummy_no_real_cache_writes': dummy,
                                         'fixed_addresses': True, 'per_layer_group_mapping': True,
                                         **numerical})
                graph.reset()
                del graph, obj
                gc.collect()
                torch.cuda.synchronize()
        return {
            'status': 'PASS', 'gpu_executed': True, 'gpu': torch.cuda.get_device_name(),
            'torch': torch.__version__, 'cuda': torch.version.cuda,
            'installed_overlay_sha256': spec_manifest['patched_sha256'],
            'actual_dspark_inheritance_checked': True,
            'projection': {'weight_shape': [hidden, width], 'checkpoint_scale_shape': [hidden // 32, width // 32],
                           'weight_dtype': 'float8_e4m3fn', 'scale_dtype': 'float8_e8m0fnu',
                           'activation_dtype': 'bfloat16', 'bf16_min_rows': 32,
                           'norm': 'vllm.RMSNorm', 'norm_eps': eps,
                           'compilation_mode': 0, 'custom_ops': ['all'],
                           'weights': 'synthetic; no model snapshot downloaded'},
            'atol': 3e-2, 'rtol': 2e-2,
            'independent_reference_checks': independent,
            'disabled_and_eager_original_shape_checks': eager_shape_checks,
            'context_graph_checks': graph_checks,
            'scope': 'production replicated projection/quantization/Marlin/BF16/norm and inherited context routing only',
            'limitations': ['Synthetic weights do not prove accuracy of loaded model weights.',
                           'Cache consumer is a fixture; real KV/RoPE/attention/Markov/acceptance and TP collectives are not tested.',
                           'No model-level TPS, 512K residency, multimodal quality or full-model draft graph claim.',
                           'Changing projection row count can change floating-point results and kernel dispatch.'],
        }
    finally:
        torch.set_default_dtype(previous_dtype)
        if previous_threshold is None:
            os.environ.pop('VLLM_AMPERE_DENSE_BF16_MIN_TOKENS', None)
        else:
            os.environ['VLLM_AMPERE_DENSE_BF16_MIN_TOKENS'] = previous_threshold


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--output", type=Path)
    args, rest = parser.parse_known_args()
    report = {'status': 'RUNNING', 'gpu_executed': False, 'gpu_validation': 'NOT_RUN',
              'scope': 'CPU context-routing checks; optional GPU production-size projection/Marlin/BF16/norm and context replay gate; no model-level validation'}
    try:
        result = unittest.main(argv=[__file__, *rest], exit=False)
        report['unit_tests'] = result.result.testsRun
        if not result.result.wasSuccessful():
            raise RuntimeError('Context combine unit checks failed')
        if args.gpu:
            if torch is not None and torch.cuda.is_available():
                report['gpu_validation'] = 'RUNNING'
            report['production_projection_gpu_gate'] = gpu_check()
            report['gpu_executed'] = True
            report['gpu_validation'] = 'PASS'
        report['status'] = 'PASS'
    except BaseException as exc:
        if report['gpu_validation'] == 'RUNNING':
            report['gpu_validation'] = 'FAIL'
        report.update(status='FAIL', error=repr(exc))
        raise
    finally:
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + '\n')
