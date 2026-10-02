"""Required public-only conformance prelude inside an authorized EP21 job."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import zipfile

from public_runtime_harness import (
    ARCHITECTURE_SHA256, BASE, FASTL2LIR_SHA256, OUTCOME, PUBLIC_MEMBERS,
    RESOURCES, SCOPE, SOURCE_AUTH, read_allocation_authorization,
    require, run_in_allocation, scratch_usage, stage_timeout, validate_allocation_authorization,
    validate_source_authorization,
)

def inert_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run_worker(allocation_authorization):
    os.umask(0o077)
    job = os.environ.get("SLURM_JOB_ID")
    require(job is not None, "existing_Slurm_allocation_required")
    validate_source_authorization(json.loads(SOURCE_AUTH.read_text()))
    validate_allocation_authorization(read_allocation_authorization(allocation_authorization),
                                      os.environ, public_step=True)
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and all(
        os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")),
        "CPU_only_single_thread_stage_required")
    out = BASE / "outputs/pipeline_conformance/attempt-007"
    out.mkdir(mode=0o700, exist_ok=False)
    scratch = Path(os.environ["SCRATCH"]) / "br_autoresearch" / BASE.name / f"public-runtime-{job}"
    scratch.mkdir(mode=0o700, parents=True, exist_ok=False)
    for name in ("cache", "tmp"):
        (scratch / name).mkdir(mode=0o700)
    require(os.environ.get("XDG_CACHE_HOME") == str(scratch / "cache")
            and os.environ.get("TORCH_HOME") == str(scratch / "cache/torch")
            and os.environ.get("TMPDIR") == str(scratch / "tmp"), "source_owned_cache_required")
    seconds = min(28 * 60, stage_timeout(os.environ), float(os.environ["EP21_PUBLIC_STAGE_SECONDS"]))
    require(seconds > 0, "public_stage_deadline_reached")
    deadline = time.monotonic() + seconds
    archive = Path(os.environ["SCRATCH"]) / "br_autoresearch" / BASE.name / "public_model_assets/VGG_ILSVRC_19_layers.zip"
    with zipfile.ZipFile(archive) as z:
        require(sum(z.getinfo("VGG_ILSVRC_19_layers/" + name).file_size for name in PUBLIC_MEMBERS)
                <= RESOURCES["scratch_bytes"], "scratch_cap_exceeded")
        written = 0
        for name, digest in PUBLIC_MEMBERS.items():
            h = hashlib.sha256()
            with z.open("VGG_ILSVRC_19_layers/" + name) as src, (scratch / name).open("xb") as dst:
                while block := src.read(1 << 20):
                    require(time.monotonic() < deadline, "deadline_reached")
                    written += len(block)
                    require(written <= RESOURCES["scratch_bytes"], "scratch_cap_exceeded")
                    h.update(block); dst.write(block)
            require(h.hexdigest() == digest, "qualified_public_member_mismatch")
    import numpy as np
    import torch
    torch.set_num_threads(1)
    architecture = BASE / "outputs/source_pipeline_binding/bdpy_0_25/plaintext/bdpy/dl/torch/models.py"
    require(hashlib.sha256(architecture.read_bytes()).hexdigest() == ARCHITECTURE_SHA256, "architecture_mismatch")
    module = inert_module("ep21_pinned_public_architecture", architecture)
    model = module.VGG19()
    state = torch.load(scratch / "VGG_ILSVRC_19_layers.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    del state
    model.eval()
    require(all(torch.isfinite(p).all().item() for p in model.parameters()), "nonfinite_public_checkpoint_parameter")
    require(all(not m.inplace for m in model.modules() if isinstance(m, torch.nn.ReLU)), "pre_relu_architecture_mismatch")
    parameter_count = sum(p.numel() for p in model.parameters())
    del model
    mean = np.load(scratch / "ilsvrc_2012_mean.npy", allow_pickle=False)
    require(mean.ndim == 3 and mean.shape[0] == 3 and np.isfinite(mean).all(), "public_mean_shape_finiteness_failed")
    mean_shape = list(mean.shape)
    del mean
    # Analytic source checks are explicitly required pipeline conformance,
    # not scientific observations. Six zero-input rows identify the penalized
    # bias exactly: b = n/(n+alpha) times the constant target.
    package = BASE / "outputs/source_pipeline_binding/fastl2lir_0_9"
    wheel = package / "fastl2lir-0.9-py2.py3-none-any.whl"
    require(hashlib.sha256(wheel.read_bytes()).hexdigest() == FASTL2LIR_SHA256, "decoder_package_identity_failed")
    decoder_source = package / "plaintext/fastl2lir/fastl2lir.py"
    with zipfile.ZipFile(wheel) as z:
        require(decoder_source.read_bytes() == z.read("fastl2lir/fastl2lir.py"), "decoder_plaintext_identity_failed")
    decoder = inert_module("ep21_exact_fastl2lir", decoder_source)
    x = np.zeros((6, 2), dtype=np.float32)
    target = np.broadcast_to(np.arange(1, 7, dtype=np.float32).reshape(1, 2, 3), (6, 2, 3)).copy()
    fitted = decoder.FastL2LiR().fit(x, target, alpha=1, n_feat=0, dtype=np.float32)
    predicted = fitted.predict(x)
    require(fitted.W.shape == (2, 2, 3) and fitted.b.shape == (1, 2, 3), "decoder_tensor_storage_shape_failed")
    require(fitted.W.dtype == np.float32 and predicted.dtype == np.float64, "decoder_source_dtype_failed")
    require(np.allclose(predicted, target.astype(np.float64) * (6 / 7), rtol=1e-6, atol=1e-6) and np.all(fitted.W == 0), "penalized_bias_structured_prediction_failed")
    report = {"schema_version": "ep21.public_runtime_conformance.v3", "astra_analysis_id": "pipeline_conformance", "attempt": 7,
              "job_id": job, "step_id": os.environ["SLURM_STEP_ID"], "outcome": OUTCOME,
              "scope": SCOPE, "resources": RESOURCES,
              "allocation_authorization": str(Path(allocation_authorization).resolve()),
              "VGG_parameter_count": parameter_count,
              "public_mean_shape": mean_shape, "strict_load": True, "pre_relu_non_inplace": True,
              "decoder_penalized_bias_structured_prediction": True, "runtime": {"python": sys.version.split()[0], "numpy": np.__version__, "torch": torch.__version__},
              "module_label_torch": "py-pytorch/2.4.1_py312", "module_label_is_not_distribution_version": True,
              "LAION_images_or_neural_values_read": False, "OOD_access": False, "empirical_fit_score_run": False,
              "historical_Caffe_feature_parity": "unverified", "GPU_runtime_verified": False, "protocol_lock_emitted": False,
              "complete_pipeline_conformance": False, "unchanged_guard_reports_reused": [5, 6]}
    require(scratch_usage(scratch) <= RESOURCES["scratch_bytes"], "scratch_cap_exceeded")
    encoded = (json.dumps(report, indent=2) + "\n").encode()
    require(len(encoded) <= RESOURCES["durable_bytes"] and time.monotonic() < deadline, "output_or_deadline_cap_exceeded")
    import shutil
    shutil.rmtree(scratch)
    # Publish completion last, only after scratch has been released for the
    # substantive stage. Partial/failed attempts cannot be reused.
    pending = out / "public_runtime_conformance_report.json.pending"
    with pending.open("xb") as dst:
        dst.write(encoded); dst.flush(); os.fsync(dst.fileno())
    pending.chmod(0o400)
    pending.rename(out / "public_runtime_conformance_report.json")
    print(json.dumps({"status": "passed", "report": str(out / "public_runtime_conformance_report.json"), "empirical_fitting_scoring": False}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allocation-authorization", type=Path, required=True,
                        help="Approved substantive EP21 job metadata; source-only approval is preserved separately.")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        run_worker(args.allocation_authorization)
    else:
        report = run_in_allocation(args.allocation_authorization)
        print(json.dumps({"status": "public_conformance_available", "original_job_id": report["job_id"],
                          "protocol_lock_emitted": False, "empirical_fitting_scoring": False}))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"status": "failed", "type": type(exc).__name__, "code": str(exc) if type(exc) is RuntimeError else "no_values_logged"}), flush=True)
        raise SystemExit(1)
