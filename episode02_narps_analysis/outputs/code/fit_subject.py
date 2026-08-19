#!/usr/bin/env python3
"""Fit one subject for the two frozen smoothing endpoints."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.glm.contrasts import compute_contrast
from nilearn.glm.first_level import mean_scaling, run_glm
from nilearn.masking import apply_mask


WORKSPACE = Path(__file__).resolve().parents[2]
FMRIPREP = WORKSPACE / "inputs" / "fmriprep"
FITLINS = WORKSPACE / "inputs" / "fitlins" / "task-MGT" / "node-runLevel"
PARTICIPANTS = WORKSPACE / "inputs" / "raw" / "participants.tsv"
MASK = WORKSPACE / "outputs" / "common_analysis_mask.nii.gz"
CONTRACT = WORKSPACE / "outputs" / "frozen_successor_contract.json"
SCRATCH = Path("/scratch/users/zijiao/episode02_narps_analysis")
CHECKPOINTS = SCRATCH / "checkpoints"

MOTION = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]
TASK = [
    "trial_type.decision",
    "trial_type.missed",
    "gain_demean",
    "loss_demean",
    "rt_reg.rt",
]
CONTRASTS = ("gain_demean", "loss_demean")
SMOOTHINGS = {"S0": None, "S8": 8.0}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def subject_list() -> list[str]:
    with PARTICIPANTS.open("r", encoding="utf-8", newline="") as stream:
        return sorted(row["participant_id"] for row in csv.DictReader(stream, delimiter="\t"))


def resolve_subject(value: str) -> str:
    subjects = subject_list()
    if value.startswith("sub-"):
        if value not in subjects:
            raise ValueError(f"Unknown subject: {value}")
        return value
    index = int(value)
    if index < 0 or index >= len(subjects):
        raise ValueError(f"Subject index out of range: {index}")
    return subjects[index]


def paths_for(subject: str, run: int) -> tuple[Path, Path, Path]:
    stem = f"{subject}_task-MGT_run-{run}"
    bold = FMRIPREP / subject / "func" / (
        stem + "_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz"
    )
    confounds = FMRIPREP / subject / "func" / (stem + "_desc-confounds_timeseries.tsv")
    design = FITLINS / subject / (stem + "_design.tsv")
    for path in (bold, confounds, design):
        if not path.exists():
            raise FileNotFoundError(path)
    return bold, confounds, design


def build_design(confounds_path: Path, design_path: Path) -> pd.DataFrame:
    frozen = pd.read_csv(design_path, sep="\t")
    confounds = pd.read_csv(confounds_path, sep="\t")
    if len(frozen) != len(confounds):
        raise ValueError(f"Row mismatch: {design_path} vs {confounds_path}")
    required_task = ["gain_demean", "loss_demean", "intercept"]
    missing = [name for name in required_task if name not in frozen.columns]
    missing += [name for name in MOTION if name not in confounds.columns]
    if missing:
        raise ValueError(f"Missing design/confound columns: {missing}")

    motion = confounds[MOTION].astype(float).copy()
    if not np.isfinite(motion.to_numpy()).all():
        raise ValueError(f"Nonfinite native motion values: {confounds_path}")
    derivatives = motion.diff().fillna(0.0)

    columns: dict[str, np.ndarray] = {}
    for name in TASK:
        if name in frozen.columns:
            columns[name] = frozen[name].to_numpy(dtype=float)
    for name in MOTION:
        columns[name] = motion[name].to_numpy(dtype=float)
        columns[f"{name}_derivative1"] = derivatives[name].to_numpy(dtype=float)
        columns[f"{name}_power2"] = np.square(columns[name])
        columns[f"{name}_derivative1_power2"] = np.square(columns[f"{name}_derivative1"])
    for name in frozen.columns:
        if name.startswith("cosine"):
            columns[name] = frozen[name].to_numpy(dtype=float)
    columns["intercept"] = frozen["intercept"].to_numpy(dtype=float)
    design = pd.DataFrame(columns)
    if not np.isfinite(design.to_numpy()).all():
        raise ValueError(f"Nonfinite frozen design: {design_path}")
    return design


def fit_run(bold: Path, design: pd.DataFrame, mask_img: nib.spatialimages.SpatialImage, fwhm: float | None) -> dict:
    series = apply_mask(bold, mask_img, dtype="f", smoothing_fwhm=fwhm, ensure_finite=True)
    if series.shape[0] != len(design):
        raise ValueError(f"BOLD/design rows differ: {bold}: {series.shape[0]} vs {len(design)}")
    scaled, mean = mean_scaling(series, axis=0)
    if not np.isfinite(scaled).all():
        raise ValueError(f"Nonfinite scaled BOLD: {bold}")
    x = design.to_numpy(dtype=float)
    labels, results = run_glm(
        scaled,
        x,
        noise_model="ar1",
        bins=100,
        n_jobs=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
        random_state=0,
    )
    outputs = {
        "design_rank": int(np.linalg.matrix_rank(x)),
        "design_columns": list(design.columns),
        "mean_scaling_lt1": int(np.sum(mean <= 1.0)),
    }
    for contrast in CONTRASTS:
        vector = np.zeros(x.shape[1], dtype=float)
        vector[design.columns.get_loc(contrast)] = 1.0
        estimate = compute_contrast(labels, results, vector, stat_type="t")
        effect = np.asarray(estimate.effect_size(), dtype=np.float64)
        variance = np.asarray(estimate.effect_variance(), dtype=np.float64)
        if effect.ndim != 1 or variance.shape != effect.shape:
            raise ValueError(f"Unexpected contrast shape for {bold}: {effect.shape}, {variance.shape}")
        if not np.isfinite(effect).all():
            raise ValueError(f"Nonfinite {contrast} effect: {bold}")
        if not np.isfinite(variance).all() or np.any(variance <= 0):
            raise ValueError(f"Invalid {contrast} variance: {bold}")
        outputs[contrast] = {"effect": effect, "variance": variance}
    return outputs


def aggregate(effects: np.ndarray, variances: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    weights = 1.0 / variances
    ivw = np.sum(effects * weights, axis=0) / np.sum(weights, axis=0)
    equal = np.mean(effects, axis=0)
    floors = np.quantile(variances, 0.001, axis=1)
    stabilized_variances = np.maximum(variances, floors[:, None])
    stabilized_weights = 1.0 / stabilized_variances
    stabilized = np.sum(effects * stabilized_weights, axis=0) / np.sum(stabilized_weights, axis=0)
    return ivw, equal, stabilized, floors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("subject", help="Subject ID or zero-based sorted subject index")
    args = parser.parse_args()
    subject = resolve_subject(args.subject)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    checkpoint = CHECKPOINTS / f"{subject}.npz"
    sidecar = CHECKPOINTS / f"{subject}.json"
    contract_sha = sha256(CONTRACT)
    if checkpoint.exists() and sidecar.exists():
        prior = json.loads(sidecar.read_text())
        if prior.get("contract_sha256") == contract_sha and prior.get("status") == "complete":
            print(json.dumps({"subject": subject, "status": "already_complete"}))
            return
        raise RuntimeError(f"Refusing to overwrite incompatible checkpoint: {checkpoint}")

    started = time.time()
    mask_img = nib.load(str(MASK))
    arrays: dict[str, np.ndarray] = {}
    run_metadata = []
    for smoothing, fwhm in SMOOTHINGS.items():
        per_contrast = {contrast: {"effect": [], "variance": []} for contrast in CONTRASTS}
        for run in range(1, 5):
            bold, confounds, design_path = paths_for(subject, run)
            design = build_design(confounds, design_path)
            fitted = fit_run(bold, design, mask_img, fwhm)
            run_metadata.append({
                "smoothing": smoothing,
                "run": run,
                "bold": str(bold),
                "confounds": str(confounds),
                "design": str(design_path),
                "design_rank": fitted["design_rank"],
                "design_column_count": len(fitted["design_columns"]),
                "mean_scaling_lt1": fitted["mean_scaling_lt1"],
            })
            for contrast in CONTRASTS:
                per_contrast[contrast]["effect"].append(fitted[contrast]["effect"])
                per_contrast[contrast]["variance"].append(fitted[contrast]["variance"])

        for contrast in CONTRASTS:
            effects = np.stack(per_contrast[contrast]["effect"])
            variances = np.stack(per_contrast[contrast]["variance"])
            ivw, equal, stabilized, floors = aggregate(effects, variances)
            prefix = f"{contrast}__{smoothing}"
            arrays[f"{prefix}__run_effects"] = effects.astype(np.float32)
            arrays[f"{prefix}__run_variances"] = variances.astype(np.float32)
            arrays[f"{prefix}__ivw"] = ivw.astype(np.float32)
            arrays[f"{prefix}__equal"] = equal.astype(np.float32)
            arrays[f"{prefix}__stabilized"] = stabilized.astype(np.float32)
            arrays[f"{prefix}__variance_floors"] = floors.astype(np.float64)

    arrays["mask_voxels"] = np.array([int(np.asanyarray(mask_img.dataobj).sum())], dtype=np.int64)
    with tempfile.NamedTemporaryFile(dir=CHECKPOINTS, prefix=f".{subject}.", suffix=".npz", delete=False) as stream:
        temp_path = Path(stream.name)
    try:
        np.savez_compressed(temp_path, **arrays)
        os.replace(temp_path, checkpoint)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    metadata = {
        "schema_version": "narps.episode02.subject_checkpoint.v1",
        "status": "complete",
        "subject": subject,
        "contract_sha256": contract_sha,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "elapsed_seconds": time.time() - started,
        "hostname": os.uname().nodename,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "runs": run_metadata,
    }
    sidecar.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
