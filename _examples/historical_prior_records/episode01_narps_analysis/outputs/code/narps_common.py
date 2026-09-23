"""Shared, frozen-contract helpers for the NARPS multiverse analysis."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


WORKSPACE = Path(__file__).resolve().parents[2]
OUTPUTS = WORKSPACE / "outputs"
RAW_ROOT = Path(
    "/oak/stanford/groups/russpold/data/OpenNeuro_analyses/"
    "openneuro_fitlins/input/ds001734"
)
FMRIPREP_ROOT = Path(
    "/oak/stanford/groups/russpold/data/OpenNeuro_analyses/"
    "openneuro_fitlins/fmriprep/ds001734/derivatives"
)
DESIGN_ROOT = Path(
    "/oak/stanford/groups/russpold/data/OpenNeuro_analyses/"
    "openneuro_fitlins/analyses/ds001734/task-MGT/node-runLevel"
)

CONTRACT_HASHES = {
    "frozen_analysis_contract.json":
        "edae18d760897767058f31f4313cc732ab6bdba2acbc4e36a7af7a1e3d9fc0a8",
    "frozen_analysis_contract.md":
        "1bad0c7f0ac03cb4da0d8fddaa69e4f403af83bc00808303107dc06b7cdabd34",
}

MOTION6 = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]
TASK_COLUMNS = [
    "trial_type.decision",
    "trial_type.missed",
    "gain_demean",
    "loss_demean",
    "rt_reg.rt",
]
ACOMPCOR6 = [f"a_comp_cor_{index:02d}" for index in range(6)]
CONFOUNDS = ("C0", "C1", "C2")
SMOOTHINGS = ("S0", "S1", "S2")
SMOOTHING_FWHM = {"S0": 0.0, "S1": 5.0, "S2": 8.0}
QC_SETS = ("Q0_all", "Q1_lenient", "Q2_strict")
CONTRASTS = ("gain_demean", "loss_demean")
RUNS = (1, 2, 3, 4)
EXPECTED_EXCLUSIONS_Q1 = {"sub-030", "sub-100"}
EXPECTED_EXCLUSIONS_Q2 = {
    "sub-016", "sub-018", "sub-030", "sub-088", "sub-100"
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_frozen_contract() -> None:
    for name, expected in CONTRACT_HASHES.items():
        path = OUTPUTS / name
        observed = sha256(path)
        if observed != expected:
            raise RuntimeError(
                f"Frozen contract integrity failure for {path}: "
                f"expected {expected}, observed {observed}"
            )


def participants() -> pd.DataFrame:
    table = pd.read_csv(RAW_ROOT / "participants.tsv", sep="\t")
    required = {"participant_id", "group"}
    if not required.issubset(table.columns):
        raise RuntimeError(f"participants.tsv lacks {sorted(required - set(table.columns))}")
    table = table.sort_values(
        "participant_id", key=lambda col: col.str.removeprefix("sub-").astype(int)
    ).reset_index(drop=True)
    if table["participant_id"].duplicated().any():
        raise RuntimeError("Duplicate participant IDs")
    return table


def run_paths(subject: str, run: int) -> dict[str, Path]:
    stem = f"{subject}_task-MGT_run-{run}"
    func = FMRIPREP_ROOT / subject / "func"
    return {
        "bold": func / (
            f"{stem}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz"
        ),
        "mask": func / (
            f"{stem}_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz"
        ),
        "confounds_tsv": func / f"{stem}_desc-confounds_timeseries.tsv",
        "confounds_json": func / f"{stem}_desc-confounds_timeseries.json",
        "design": DESIGN_ROOT / subject / f"{stem}_design.tsv",
    }


def validate_acompcor_metadata(metadata: dict, path: Path) -> None:
    for column in ACOMPCOR6:
        if column not in metadata:
            raise RuntimeError(f"{path}: missing {column}")
        item = metadata[column]
        mask = str(item.get("Mask", "")).lower()
        retained = item.get("Retained")
        if mask != "combined" or retained is not True:
            raise RuntimeError(
                f"{path}: {column} requires Mask=combined and Retained=true; "
                f"observed Mask={item.get('Mask')!r}, Retained={retained!r}"
            )


def friston24(confounds: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in MOTION6 if column not in confounds]
    if missing:
        raise RuntimeError(f"Missing motion columns: {missing}")
    originals = confounds[MOTION6].astype(float).copy()
    derivatives = originals.diff()
    derivatives.iloc[0, :] = 0.0
    if derivatives.iloc[1:, :].isna().any().any():
        raise RuntimeError("Non-definition-driven NaN in motion derivatives")
    pieces: dict[str, pd.Series] = {}
    for column in MOTION6:
        derivative_name = f"{column}_derivative1"
        pieces[column] = originals[column]
        pieces[derivative_name] = derivatives[column]
        pieces[f"{column}_power2"] = originals[column] ** 2
        pieces[f"{derivative_name}_power2"] = derivatives[column] ** 2
    result = pd.DataFrame(pieces, index=confounds.index)
    if result.shape[1] != 24 or not np.isfinite(result.to_numpy()).all():
        raise RuntimeError("Constructed Friston24 is not a finite 24-column matrix")
    return result


def load_frozen_design(subject: str, run: int, confound_model: str) -> pd.DataFrame:
    paths = run_paths(subject, run)
    fitlins = pd.read_csv(paths["design"], sep="\t")
    confounds = pd.read_csv(paths["confounds_tsv"], sep="\t")
    if len(fitlins) != len(confounds):
        raise RuntimeError(
            f"{subject} run {run}: design rows {len(fitlins)} != confounds rows "
            f"{len(confounds)}"
        )
    cosine = [column for column in fitlins.columns if column.startswith("cosine")]
    task_columns = [column for column in TASK_COLUMNS if column in fitlins.columns]
    required_task = [
        "trial_type.decision", "gain_demean", "loss_demean", "rt_reg.rt"
    ]
    required_fitlins = required_task + cosine + ["intercept"]
    missing = [column for column in required_fitlins if column not in fitlins]
    if missing:
        raise RuntimeError(f"{paths['design']}: missing frozen columns {missing}")
    nuisance: pd.DataFrame
    if confound_model == "C0":
        nuisance = confounds[MOTION6].astype(float).copy()
    elif confound_model in {"C1", "C2"}:
        nuisance = friston24(confounds)
        if confound_model == "C2":
            with paths["confounds_json"].open() as stream:
                metadata = json.load(stream)
            validate_acompcor_metadata(metadata, paths["confounds_json"])
            missing_compcor = [column for column in ACOMPCOR6 if column not in confounds]
            if missing_compcor:
                raise RuntimeError(
                    f"{paths['confounds_tsv']}: missing {missing_compcor}"
                )
            nuisance = pd.concat(
                [nuisance, confounds[ACOMPCOR6].astype(float)], axis=1
            )
    else:
        raise ValueError(f"Unknown confound model {confound_model}")
    design = pd.concat(
        [fitlins[task_columns], nuisance, fitlins[cosine], fitlins[["intercept"]]],
        axis=1,
    )
    values = design.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        locations = np.argwhere(~np.isfinite(values))[:10].tolist()
        raise RuntimeError(
            f"{subject} run {run} {confound_model}: nonfinite design at {locations}"
        )
    return design


def contrast_vector(design: pd.DataFrame, contrast: str) -> np.ndarray:
    if contrast not in CONTRASTS or contrast not in design.columns:
        raise ValueError(f"Unavailable frozen contrast {contrast}")
    vector = np.zeros(design.shape[1], dtype=np.float64)
    vector[design.columns.get_loc(contrast)] = 1.0
    return vector


def estimability(design: pd.DataFrame, contrast: str) -> tuple[bool, int, float]:
    matrix = design.to_numpy(dtype=np.float64)
    vector = contrast_vector(design, contrast)
    _, singular, vt = np.linalg.svd(matrix, full_matrices=False)
    tolerance = max(matrix.shape) * np.finfo(float).eps * singular[0]
    rank = int(np.sum(singular > tolerance))
    row_basis = vt[:rank, :]
    projected = row_basis.T @ (row_basis @ vector)
    error = float(np.linalg.norm(vector - projected))
    scale = max(1.0, float(np.linalg.norm(vector)))
    return error <= 1e-8 * scale, rank, error


def q_membership(subject: str) -> dict[str, bool]:
    return {
        "Q0_all": True,
        "Q1_lenient": subject not in EXPECTED_EXCLUSIONS_Q1,
        "Q2_strict": subject not in EXPECTED_EXCLUSIONS_Q2,
    }


def save_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)


def save_tsv_atomic(path: Path, table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    table.to_csv(temporary, sep="\t", index=False, lineterminator="\n")
    temporary.replace(path)


def load_mask_vector() -> tuple[nib.Nifti1Image, np.ndarray]:
    image = nib.load(OUTPUTS / "common_analysis_mask.nii.gz")
    mask = np.asarray(image.dataobj, dtype=np.uint8) > 0
    if not mask.any():
        raise RuntimeError("Frozen common mask is empty")
    return image, mask


def ceil_fraction(fraction: float, count: int) -> int:
    return int(math.ceil(fraction * count - 1e-12))
