#!/usr/bin/env python
"""Fit every frozen C x S first-level model for one NARPS subject.

The program reads each of the subject's four BOLD runs exactly once, processes
the three smoothing levels in memory, fits all three confound models, and
precision-combines run effects/variances into a compact subject checkpoint.
QC does not enter this stage; it is a group-level selection axis.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path

import nibabel as nib
import nilearn
import numpy as np
import pandas as pd
from nilearn.glm.contrasts import compute_contrast
from nilearn.glm.first_level import mean_scaling, run_glm
from nilearn.image.image import smooth_array

from narps_common import (
    CONFOUNDS,
    CONTRASTS,
    OUTPUTS,
    RUNS,
    SMOOTHINGS,
    SMOOTHING_FWHM,
    assert_frozen_contract,
    contrast_vector,
    load_frozen_design,
    load_mask_vector,
    participants,
    run_paths,
    save_json_atomic,
    sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    identity = parser.add_mutually_exclusive_group(required=True)
    identity.add_argument("--subject")
    identity.add_argument("--subject-index", type=int)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--chunk-voxels", type=int, default=12000)
    parser.add_argument(
        "--pilot-voxels",
        type=int,
        default=0,
        help="Run a non-production one-run C0/S0 pilot on the first N mask voxels.",
    )
    return parser.parse_args()


def resolve_subject(args: argparse.Namespace) -> str:
    people = participants()
    if args.subject_index is not None:
        if args.subject_index < 0 or args.subject_index >= len(people):
            raise ValueError(f"subject-index out of range: {args.subject_index}")
        return str(people.iloc[args.subject_index].participant_id)
    if args.subject not in set(people.participant_id):
        raise ValueError(f"Unknown subject: {args.subject}")
    return str(args.subject)


def current_rss_gib() -> float:
    # Linux reports ru_maxrss in KiB.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)


def atomic_save_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing subject result: {path}")
    temporary.replace(path)


def validate_existing(
    path: Path, subject: str, n_voxels: int, expected_mask_sha256: str
) -> bool:
    if not path.is_file():
        return False
    with np.load(path, allow_pickle=False) as payload:
        required = {"effect", "variance", "confounds", "smoothings", "contrasts",
                    "subject", "mask_sha256"}
        if not required.issubset(payload.files):
            raise RuntimeError(f"Existing {path} is incomplete: {payload.files}")
        expected_shape = (len(CONFOUNDS), len(SMOOTHINGS), len(CONTRASTS), n_voxels)
        if payload["effect"].shape != expected_shape:
            raise RuntimeError(
                f"Existing {path} shape {payload['effect'].shape} != {expected_shape}"
            )
        if payload["variance"].shape != expected_shape:
            raise RuntimeError(
                f"Existing {path} variance shape {payload['variance'].shape} "
                f"!= {expected_shape}"
            )
        if str(payload["subject"].item()) != subject:
            raise RuntimeError(f"Existing {path} subject mismatch")
        if payload["confounds"].tolist() != list(CONFOUNDS):
            raise RuntimeError(f"Existing {path} confound-axis mismatch")
        if payload["smoothings"].tolist() != list(SMOOTHINGS):
            raise RuntimeError(f"Existing {path} smoothing-axis mismatch")
        if payload["contrasts"].tolist() != list(CONTRASTS):
            raise RuntimeError(f"Existing {path} contrast-axis mismatch")
        if str(payload["mask_sha256"].item()) != expected_mask_sha256:
            raise RuntimeError(f"Existing {path} frozen-mask fingerprint mismatch")
        if not np.isfinite(payload["effect"]).all():
            raise RuntimeError(f"Existing {path} has nonfinite effects")
        variance = payload["variance"]
        if not np.isfinite(variance).all() or not (variance > 0).all():
            raise RuntimeError(f"Existing {path} has invalid variances")
    sidecar = path.with_suffix(".json")
    if not sidecar.is_file():
        raise RuntimeError(
            f"Existing {path} has no JSON sidecar; refusing silent partial recovery"
        )
    print(f"Validated existing immutable result: {path}", flush=True)
    return True


def fit_one_design(
    y: np.ndarray,
    design: pd.DataFrame,
    chunk_voxels: int,
    n_jobs: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_voxels = y.shape[1]
    effects = np.empty((len(CONTRASTS), n_voxels), dtype=np.float32)
    variances = np.empty_like(effects)
    matrix = design.to_numpy(dtype=np.float64)
    vectors = [contrast_vector(design, contrast) for contrast in CONTRASTS]
    for start in range(0, n_voxels, chunk_voxels):
        stop = min(start + chunk_voxels, n_voxels)
        labels, results = run_glm(
            y[:, start:stop],
            matrix,
            noise_model="ar1",
            bins=100,
            n_jobs=n_jobs,
            verbose=0,
            random_state=0,
        )
        for contrast_index, vector in enumerate(vectors):
            result = compute_contrast(labels, results, vector, stat_type="t")
            effects[contrast_index, start:stop] = np.asarray(
                result.effect_size(), dtype=np.float32
            )
            variances[contrast_index, start:stop] = np.asarray(
                result.effect_variance(), dtype=np.float32
            )
    if not np.isfinite(effects).all():
        raise RuntimeError("Nonfinite run contrast effect")
    if not np.isfinite(variances).all() or not (variances > 0).all():
        invalid = int((~np.isfinite(variances) | (variances <= 0)).sum())
        raise RuntimeError(f"Invalid run contrast variances: {invalid} values")
    return effects, variances


def main() -> int:
    args = parse_args()
    if args.n_jobs < 1 or args.chunk_voxels < 100:
        raise ValueError("n-jobs must be >=1 and chunk-voxels must be >=100")
    assert_frozen_contract()
    subject = resolve_subject(args)
    mask_image, full_mask = load_mask_vector()
    mask_sha = sha256(OUTPUTS / "common_analysis_mask.nii.gz")
    mask_indices = np.flatnonzero(full_mask.ravel(order="C"))
    if args.pilot_voxels:
        if args.pilot_voxels < 100 or args.pilot_voxels > len(mask_indices):
            raise ValueError("pilot-voxels must be between 100 and the mask size")
        selected = mask_indices[: args.pilot_voxels]
        mask = np.zeros(full_mask.size, dtype=bool)
        mask[selected] = True
        mask = mask.reshape(full_mask.shape, order="C")
        models = ("C0",)
        smoothings = ("S0",)
        runs = (1,)
    else:
        mask = full_mask
        models = CONFOUNDS
        smoothings = SMOOTHINGS
        runs = RUNS

    n_voxels = int(mask.sum())
    output_path = OUTPUTS / "subject_maps" / f"{subject}_fixed_effects.npz"
    if not args.pilot_voxels and validate_existing(
        output_path, subject, n_voxels, mask_sha
    ):
        return 0

    start_time = time.time()
    numerator = np.zeros(
        (len(models), len(smoothings), len(CONTRASTS), n_voxels),
        dtype=np.float64,
    )
    precision = np.zeros_like(numerator)
    run_records = []

    for run in runs:
        run_start = time.time()
        paths = run_paths(subject, run)
        image = nib.load(paths["bold"])
        if image.shape[:3] != mask.shape:
            raise RuntimeError(f"{paths['bold']}: unexpected spatial shape {image.shape}")
        if not np.allclose(image.affine, mask_image.affine):
            raise RuntimeError(f"{paths['bold']}: affine differs from frozen mask")
        # get_fdata applies NIfTI slope/intercept for the 13 scaled-int16 files.
        raw = image.get_fdata(dtype=np.float32, caching="unchanged")
        if raw.shape[3] != 453:
            raise RuntimeError(f"{paths['bold']}: expected 453 frames, got {raw.shape[3]}")
        print(
            f"{subject} run={run}: loaded {raw.shape}, maxrss={current_rss_gib():.2f} GiB",
            flush=True,
        )

        designs = {
            model: load_frozen_design(subject, run, model) for model in models
        }
        for smoothing_index, smoothing in enumerate(smoothings):
            fwhm = SMOOTHING_FWHM[smoothing]
            smoothed = raw if fwhm == 0 else smooth_array(
                raw,
                image.affine,
                fwhm=fwhm,
                ensure_finite=True,
                copy=True,
            )
            y_native = np.asarray(smoothed[mask, :].T, dtype=np.float32)
            if fwhm != 0:
                del smoothed
            temporal_mean = np.mean(y_native, axis=0, dtype=np.float64)
            if not np.isfinite(temporal_mean).all():
                raise RuntimeError(
                    f"{subject} run {run} {smoothing}: "
                    "nonfinite common-mask temporal mean"
                )
            # This intentionally follows Nilearn's signal_scaling=0 implementation
            # exactly: mean_scaling clamps every temporal mean below 1 to 1.  The
            # frozen 95%-coverage mask permits a small number of zero-valued edge
            # voxels in individual run images, so rejecting those voxels here would
            # impose an unfrozen 100%-run-coverage rule.
            clamped_mean_voxels = int(np.count_nonzero(temporal_mean < 1.0))
            y, _ = mean_scaling(y_native, axis=0)
            del y_native
            y = np.asarray(y, dtype=np.float32)
            if not np.isfinite(y).all():
                raise RuntimeError(f"{subject} run {run} {smoothing}: nonfinite scaled BOLD")
            if clamped_mean_voxels:
                print(
                    f"{subject} run={run} {smoothing}: Nilearn mean_scaling "
                    f"clamped {clamped_mean_voxels} voxel means below 1",
                    flush=True,
                )

            for model_index, model in enumerate(models):
                fit_start = time.time()
                effect, variance = fit_one_design(
                    y,
                    designs[model],
                    chunk_voxels=args.chunk_voxels,
                    n_jobs=args.n_jobs,
                )
                weight = 1.0 / variance.astype(np.float64)
                numerator[model_index, smoothing_index] += (
                    effect.astype(np.float64) * weight
                )
                precision[model_index, smoothing_index] += weight
                run_records.append({
                    "run": run,
                    "confound": model,
                    "smoothing": smoothing,
                    "mean_scaling_clamped_voxels": clamped_mean_voxels,
                    "fit_seconds": time.time() - fit_start,
                    "maxrss_gib": current_rss_gib(),
                })
                print(
                    f"{subject} run={run} {model}/{smoothing}: "
                    f"fit {time.time()-fit_start:.1f}s, maxrss={current_rss_gib():.2f} GiB",
                    flush=True,
                )
                del effect, variance, weight
            del y
        del raw
        print(
            f"{subject} run={run}: complete in {time.time()-run_start:.1f}s",
            flush=True,
        )

    if not np.isfinite(precision).all() or not (precision > 0).all():
        raise RuntimeError("Invalid accumulated precision")
    subject_effect = (numerator / precision).astype(np.float32)
    subject_variance = (1.0 / precision).astype(np.float32)
    if not np.isfinite(subject_effect).all():
        raise RuntimeError("Nonfinite subject fixed effects")
    if not np.isfinite(subject_variance).all() or not (subject_variance > 0).all():
        raise RuntimeError("Invalid subject fixed-effect variances")

    metadata = {
        "schema_version": "narps.subject_fixed_effects.v1",
        "subject": subject,
        "pilot": bool(args.pilot_voxels),
        "pilot_voxels": int(args.pilot_voxels),
        "runs": list(runs),
        "confounds": list(models),
        "smoothings": list(smoothings),
        "contrasts": list(CONTRASTS),
        "n_voxels": n_voxels,
        "mask_sha256": mask_sha,
        "nilearn_ar1_bins": 100,
        "n_jobs": args.n_jobs,
        "chunk_voxels": args.chunk_voxels,
        "elapsed_seconds": time.time() - start_time,
        "maxrss_gib": current_rss_gib(),
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "nibabel": nib.__version__,
            "nilearn": nilearn.__version__,
        },
        "run_records": run_records,
    }
    if args.pilot_voxels:
        pilot_dir = OUTPUTS / "pilots"
        pilot_dir.mkdir(parents=True, exist_ok=True)
        pilot_name = f"{subject}_pilot_{args.pilot_voxels}vox.json"
        save_json_atomic(pilot_dir / pilot_name, {
            **metadata,
            "effect_summary": {
                "min": float(subject_effect.min()),
                "max": float(subject_effect.max()),
                "mean": float(subject_effect.mean()),
            },
            "variance_summary": {
                "min": float(subject_variance.min()),
                "max": float(subject_variance.max()),
                "mean": float(subject_variance.mean()),
            },
        })
        print(json.dumps(metadata, indent=2), flush=True)
        return 0

    atomic_save_npz(
        output_path,
        effect=subject_effect,
        variance=subject_variance,
        confounds=np.asarray(models),
        smoothings=np.asarray(smoothings),
        contrasts=np.asarray(CONTRASTS),
        subject=np.asarray(subject),
        mask_sha256=np.asarray(mask_sha),
    )
    save_json_atomic(
        OUTPUTS / "subject_maps" / f"{subject}_fixed_effects.json", metadata
    )
    print(f"Wrote immutable subject result {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
