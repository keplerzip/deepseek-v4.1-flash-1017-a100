"""Check the complete kernel-test import/reference closure without GPU execution.

Run with the pinned runtime Python, --network none, and the repaired project
mounted read-only at /deploy. This does not execute CUDA kernels, graphs,
collectives or model inference and cannot satisfy a GPU acceptance gate.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys
import sysconfig
import traceback


# This checker is a separate process; never create CUDA contexts even on A100.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True

REFERENCE_FILES = (
    ("baseline", "vllm/model_executor/layers/fused_moe/moe_align_block_size.py"),
    ("baseline", "vllm/models/deepseek_v4_1/amd/rocm.py"),
    ("patched", "vllm/v1/attention/ops/rocm_aiter_mla_sparse.py"),
    ("patched", "vllm/models/deepseek_v4_1/amd/rocm.py"),
)
MODULES = {
    "performance_kernels": ("check_mhc", "main"),
    "performance_r11_kernels": ("record", "check_moe", "check_indexer"),
    "performance_r12_kernels": ("check_dense", "check_shared", "check_sparse"),
    "performance_r13_kernels": ("check_dspark_kv", "check_topk_reuse"),
    "performance_r14_kernels": ("check_decode_metadata",),
    "performance_inv_rope_kernels": ("check_inverse_rope",),
    "performance_ep8": ("uniform_case", "nonuniform_case", "main"),
    "performance_tp_ar": ("installed_api", "main"),
    "performance_r13_reference": (
        "production_method", "draft_fixture", "assert_loaded", "context_method",
        "attention_fixture", "ragged_reference", "assert_ragged",
    ),
    "performance_r14_reference": ("load_pack", "ragged_inputs", "mutate_inputs", "metadata_prefix"),
    "performance_r14_cpu": ("prefix_function",),
    "performance_inv_rope_reference": ("load_ops", "fixture", "reduce", "inverse_reference"),
    "upstream.test_shared_experts_multistream": (
        "test_decode_width_runs_on_aux_stream_and_matches_serial",
        "test_above_threshold_stays_serial_on_main_stream",
    ),
    "upstream.test_rocm_triton_attn_dsv4": (
        "_pack_fp8_ds_mla_cache", "_ragged_from_rows", "_ref_sparse_decode_ragged",
    ),
}


def digest(path):
    path = Path(path)
    assert path.is_file() and not path.is_symlink(), "Missing/nonregular support file: " + str(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def functions_exist(module, names):
    for name in names:
        assert callable(getattr(module, name, None)), "Missing callable: " + module.__name__ + "." + name


def load_reference(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def check(project, report):
    assert project == Path("/deploy"), "Mount repaired project at /deploy; references use that absolute path"
    sys.path.insert(0, str(project / 'scripts'))
    from network_proof import collect_network_evidence, require_network_none
    evidence = collect_network_evidence()
    report['network_isolation'] = evidence
    require_network_none(evidence)
    tests = project / "tests"
    sys.path.insert(0, str(tests))
    site = Path(sysconfig.get_paths()["purelib"])
    image_references = Path("/opt/dsv411017/image/performance")

    for kind, relative in REFERENCE_FILES:
        payload = project / "overrides/performance" / kind / relative
        image_source = image_references / kind / relative
        observed = digest(payload)
        assert observed == digest(image_source), "Reference differs from pinned image: " + str(payload)
        row = {"path": str(payload.relative_to(project)), "sha256": observed,
               "pinned_image_source": str(image_source), "pinned_image_bytes_equal": True}
        if kind == "patched":
            installed = site / relative
            assert observed == digest(installed), "Reference differs from installed production source: " + str(payload)
            row.update(installed_source=str(installed), installed_production_bytes_equal=True)
        report["references"].append(row)

    loaded = {}
    for name, required in MODULES.items():
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve(strict=True)
        assert path.is_relative_to(tests), "Kernel dependency imported outside repaired tests: " + name
        functions_exist(module, required)
        loaded[name] = module
        report["modules"].append({"name": name, "path": str(path.relative_to(project)),
                                  "sha256": digest(path), "required_callables": list(required)})

    # These are imported lazily by the real CUDA tests; check their interfaces
    # without running any kernels or allocating model-sized fixture tensors.
    from vllm.model_executor.layers.linear import MergedColumnParallelLinear
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptLinearMethod, QuantSpec, CkptCtx, kMxfp8Static, kMxfp8Dynamic,
    )
    from vllm.model_executor.kernels.linear.mxfp8.marlin import MarlinMxfp8LinearKernel
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import dequant_mxfp8_to_bf16
    from vllm.model_executor.layers.fused_moe.stable_moe_align import stable_moe_align
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import _moe_align_block_size_deterministic
    from vllm.v1.attention.ops import mqa_logits_candidates_triton
    from vllm.v1.attention.ops.mqa_logits_triton import fp8_mqa_logits_triton, fp8_paged_mqa_logits_triton
    from vllm.model_executor.kernels.attention.dsa.candidate_blocks import apply_candidate_mask
    from vllm.model_executor.kernels.mhc import dsv41_sm80
    from vllm.model_executor.kernels.mhc.tilelang import mhc_post_tilelang, mhc_pre_delayed_tilelang
    from vllm.model_executor.kernels.mhc.torch import mhc_pre_delayed_torch
    from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import _rocm_sparse_attn_decode_ragged_triton
    for value in (
        MergedColumnParallelLinear, ModelOptLinearMethod, QuantSpec, CkptCtx,
        MarlinMxfp8LinearKernel, dequant_mxfp8_to_bf16, stable_moe_align,
        _moe_align_block_size_deterministic, fp8_mqa_logits_triton,
        fp8_paged_mqa_logits_triton, apply_candidate_mask, mhc_post_tilelang,
        mhc_pre_delayed_tilelang, mhc_pre_delayed_torch, CudaCommunicator,
        _rocm_sparse_attn_decode_ragged_triton,
    ):
        assert callable(value), "Production interface is not callable: " + str(value)
    assert kMxfp8Static is not None and kMxfp8Dynamic is not None
    functions_exist(mqa_logits_candidates_triton, ("candidate_topk_prefill", "candidate_topk_decode"))
    functions_exist(dsv41_sm80, ("mhc_fused_post_pre_delayed_tilelang", "mhc_fused_post_pre_split_config"))
    loaded["performance_tp_ar"].installed_api()
    report["production_interfaces_loaded"] = True

    r13 = loaded["performance_r13_reference"]
    r14 = loaded["performance_r14_reference"]
    r14_cpu = loaded["performance_r14_cpu"]
    inverse = loaded["performance_inv_rope_reference"]
    import torch
    assert not torch.cuda.is_initialized(), "CPU checker unexpectedly initialized CUDA"
    assert callable(r13.context_method())
    for name in ("_remap_dspark_name", "load_weights"):
        assert callable(r13.production_method(r13.DSPARK, "DSparkDeepseekV4ForCausalLM", name,
                                             {"torch": torch}))
    assert callable(r13.production_method(r13.ATTENTION, "DeepseekV41ROCMAiterMLAAttention",
                                         "_decode_topk_ragged", {"torch": torch}))
    report["dspark_production_methods_loaded"] = True

    baseline_moe = project / "overrides/performance/baseline/vllm/model_executor/layers/fused_moe/moe_align_block_size.py"
    functions_exist(load_reference(baseline_moe, "hotfix_baseline_moe"),
                    ("_moe_align_block_size_deterministic",))
    for name, path in (("installed", site / r14.ROCM),
                       ("baseline", project / "overrides/performance/baseline" / r14.ROCM)):
        functions_exist(r14.load_pack(path, "hotfix_" + name + "_pack"),
                        ("compute_global_topk_ragged_indices_and_indptr",))
    assert callable(r14_cpu.prefix_function(site / "vllm/v1/attention/backends/mla/sparse_swa.py",
                                            "DeepseekSparseSWAMetadataBuilder", "swa", (1, 0, 6, 0)))
    functions_exist(inverse.load_ops(), ("_sparse_attn_decode_reduce_kernel", "_inverse_rope_gptj_kernel",
                                       "rocm_inverse_rope_rows_", "rocm_inv_rope_einsum"))
    assert inverse.OPS == project / "overrides/performance/patched/vllm/v1/attention/ops/rocm_aiter_mla_sparse.py"
    assert inverse.ATTN == project / "overrides/performance/patched/vllm/models/deepseek_v4_1/amd/rocm.py"
    report["reference_loaders_checked"] = True

    # Match file-based loading as well as imports; no pytest test is executed.
    for relative, names in (
        ("upstream/test_shared_experts_multistream.py", MODULES["upstream.test_shared_experts_multistream"]),
        ("upstream/test_rocm_triton_attn_dsv4.py", MODULES["upstream.test_rocm_triton_attn_dsv4"]),
    ):
        namespace = runpy.run_path(str(tests / relative), run_name="__hotfix_dependency_check__")
        assert all(callable(namespace.get(name)) for name in names), "runpy dependency missing: " + relative
    assert not torch.cuda.is_initialized(), "Dependency checks unexpectedly initialized CUDA"
    report["upstream_file_loading_checked"] = True
    report["cuda_context_initialized"] = False
    report["torch"] = torch.__version__


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path("/deploy"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {"status": "RUNNING", "scope": "Kernel-test Python imports, reference-source identity and loader closure only",
              "gpu_executed": False, "gpu_validation": "NOT_RUN", "cuda_graph_validation": "NOT_RUN",
              "ep8_validation": "NOT_RUN", "model_inference_executed": False,
              "model_speed": "NOT_MEASURED", "modules": [], "references": []}
    code = 0
    try:
        check(args.project.resolve(strict=True), report)
        report["status"] = "DEPENDENCIES_CHECKED"
    except Exception as exc:
        report.update(status="FAIL", error=type(exc).__name__ + ": " + str(exc), traceback=traceback.format_exc())
        code = 1
    report["finished_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(serialized, encoding="utf-8")
        temporary.replace(args.output)
    print(serialized, end="")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
