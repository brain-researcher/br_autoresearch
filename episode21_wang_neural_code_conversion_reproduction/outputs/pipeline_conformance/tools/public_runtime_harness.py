"""Public-only prelude inside an independently authorized EP21 allocation.

This module imports no array/model packages, opens no scientific payload, and
submits no jobs. A passing public report supplies conformance evidence only.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time


BASE = Path(__file__).resolve().parents[3]
SOURCE_AUTH = BASE / "outputs/public_runtime_conformance_authorization_2026-09-30.json"
WORKER = Path(__file__).with_name("run_public_runtime_conformance.py")
REPORT = BASE / "outputs/pipeline_conformance/attempt-007/public_runtime_conformance_report.json"
RESOURCES = {"jobs": 1, "cpus": 1, "memory_gib": 4, "minutes": 30,
             "gpus": 0, "durable_bytes": 10_000_000, "scratch_bytes": 2_000_000_000}
PUBLIC_MEMBERS = {
    "VGG_ILSVRC_19_layers.pt": "db5a4dbdbbfa55046d57a39663fa4a16d669ef6548051a2968b8df1c4fa057e8",
    "ilsvrc_2012_mean.npy": "97eefba7e046ee097121ad18564329636d2f9c153b748c7313653cae8594a149",
}
ARCHITECTURE_SHA256 = "d1f10981e3dd95c9196db3bb97b9431158192605a65cf0644c19eb8a43ab8d5d"
FASTL2LIR_SHA256 = "2abd8f46c8faa69aaea4e56568c8c4048178c24cc1ea88ad7c33f661151a0412"
OUTCOME = "passed_public_VGG_strict_load_and_focused_decoder_analytic_fixture"
SCOPE = "public_assets_and_contract_required_synthetic_analytic_fixture_only"
BOUNDARY_FIELDS = ("LAION_images_or_neural_values_read", "OOD_access",
                   "empirical_fit_score_run", "protocol_lock_emitted",
                   "complete_pipeline_conformance", "GPU_runtime_verified")
PARENT_PURPOSES = {"pipeline_conformance", "regular_vgg19_derivative_export"}


def require(condition, code):
    if not condition:
        raise RuntimeError(code)


def validate_source_authorization(auth):
    require(auth.get("approved") is True and auth.get("purpose") == "public_runtime_conformance",
            "source_only_approval_required")
    require(auth.get("resources") == RESOURCES, "source_resource_approval_mismatch")
    require(all(auth.get(k) is False for k in ("allow_LAION_image_values", "allow_neural_values",
            "allow_OOD", "allow_empirical_fitting_scoring")), "source_boundary_mismatch")


def read_allocation_authorization(path):
    # Only explicitly named, top-level authorization metadata can be read.
    path = Path(path).resolve()
    require(path.parent == BASE / "outputs" and "authorization" in path.name
            and path.suffix == ".json", "allocation_authorization_path_required")
    return json.loads(path.read_text())


def validate_allocation_authorization(auth, env, *, public_step=False):
    require(auth.get("approved") is True and auth.get("episode_id") == "ep21"
            and auth.get("purpose") in PARENT_PURPOSES,
            "substantive_EP21_allocation_authority_required")
    require(auth.get("includes_public_runtime_conformance") is True,
            "public_stage_missing_from_allocation_plan")
    resources = auth.get("resources", {})
    for name in ("jobs", "cpus", "memory_gib", "minutes", "gpus", "durable_bytes", "scratch_bytes"):
        value = resources.get(name)
        require(type(value) is int and value >= RESOURCES[name], "parent_resource_capacity_required")
    require(resources["jobs"] == 1, "single_parent_job_required")
    require(env.get("SLURM_JOB_ID"), "existing_Slurm_allocation_required")
    start = int(env.get("SLURM_JOB_START_TIME", "0"))
    end = int(env.get("SLURM_JOB_END_TIME", "0"))
    require(start > 0 and 0 < end - start <= resources["minutes"] * 60,
            "parent_allocation_time_approval_mismatch")
    if public_step:
        require(env.get("SLURM_STEP_ID") not in (None, "batch", "extern")
                and int(env.get("SLURM_CPUS_PER_TASK", "0")) == 1,
                "bounded_public_step_required")
        # Memory is requested by srun --exact --mem=4G, not established by
        # rewriting or trusting inherited parent SLURM_MEM_* values.
    else:
        require(int(env.get("SLURM_CPUS_PER_TASK", "0")) == resources["cpus"]
                and int(env.get("SLURM_MEM_PER_NODE", "0")) == resources["memory_gib"] * 1024,
                "parent_allocation_approval_mismatch")
    # The caller remains responsible for the substantive stage's own gates.
    # Source-only approval does not authorize this parent allocation.


def stage_timeout(env, now=None):
    now = time.time() if now is None else now
    require("SLURM_JOB_END_TIME" in env, "allocation_deadline_required")
    seconds = min(RESOURCES["minutes"] * 60, int(env["SLURM_JOB_END_TIME"]) - now - 30)
    require(seconds > 0, "allocation_deadline_reached")
    return seconds


def validate_passed_report(report):
    # v2 reports retain their original provenance; the old binding inventory
    # is neither recomputed nor treated as a current-runtime qualification.
    require(report.get("schema_version") in ("ep21.public_runtime_conformance.v2",
                                            "ep21.public_runtime_conformance.v3")
            and report.get("outcome") == OUTCOME and report.get("scope") == SCOPE
            and report.get("resources") == RESOURCES,
            "public_report_contract_mismatch")
    require(all(report.get(k) is False for k in BOUNDARY_FIELDS), "public_report_boundary_mismatch")
    require(all(report.get(k) is True for k in ("strict_load", "pre_relu_non_inplace",
            "decoder_penalized_bias_structured_prediction")), "public_report_not_passed")
    require(report.get("historical_Caffe_feature_parity") == "unverified",
            "public_report_changed_scientific_disposition")
    return report


def reuse_report(path=None, *, relevant_change=None):
    path = REPORT if path is None else path
    if not path.exists():
        require(not path.parent.exists(), "incomplete_public_attempt_requires_diagnosis")
        return None
    require(not relevant_change, "known_public_conformance_change_requires_diagnosis")
    require(path.stat().st_size <= RESOURCES["durable_bytes"], "public_report_cap_exceeded")
    # A declared relevant change or failed/incomplete result stops; no rerun.
    return validate_passed_report(json.loads(path.read_text()))


def step_command(authorization_path, *, job_id, timeout):
    # --jobid ensures srun can only create a step in this existing job.
    # Official semantics: https://slurm.schedmd.com/srun.html
    scratch = Path(os.environ["SCRATCH"]) / "br_autoresearch" / BASE.name / ("public-runtime-" + job_id)
    return ["srun", "--jobid=" + job_id, "--exact", "--nodes=1", "--ntasks=1",
            "--cpus-per-task=1", "--mem=4G", "--gres=none", "--cpu-bind=cores",
            "--time=00:30:00", "--kill-on-bad-exit=1", "env", "CUDA_VISIBLE_DEVICES=",
            "OMP_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1", "MKL_NUM_THREADS=1",
            "XDG_CACHE_HOME=" + str(scratch / "cache"), "TORCH_HOME=" + str(scratch / "cache/torch"),
            "TMPDIR=" + str(scratch / "tmp"),
            "PYTHONDONTWRITEBYTECODE=1", "EP21_PUBLIC_STAGE_SECONDS=" + str(timeout),
            sys.executable, "-B", str(WORKER), "--worker",
            "--allocation-authorization", str(authorization_path)]


def run_in_allocation(authorization_path, *, relevant_change=None):
    """Reuse or run the prelude; never emit a lock or call a data stage.

    The caller records known code/input/runtime/protocol changes in its existing
    ledger and passes their reason here, or keeps
    EP21_PUBLIC_CONFORMANCE_RELEVANT_CHANGE in the launch configuration across
    resumes until diagnosis and a new authorized conformance attempt. An unset
    variable is not proof that the runtime is unchanged.
    """
    env = os.environ
    validate_allocation_authorization(read_allocation_authorization(authorization_path), env)
    passed = reuse_report(relevant_change=(relevant_change or
                          env.get("EP21_PUBLIC_CONFORMANCE_RELEVANT_CHANGE")))
    if passed is not None:
        return passed  # Preserve the original result and job provenance.
    timeout = stage_timeout(env)
    try:
        subprocess.run(step_command(authorization_path, job_id=env["SLURM_JOB_ID"], timeout=timeout),
                       check=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("public_stage_timeout_no_continuation") from exc
    return validate_passed_report(json.loads(REPORT.read_text()))


def scratch_usage(path):
    """Count only the source-owned stage, including its caches/temporary files."""
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())
