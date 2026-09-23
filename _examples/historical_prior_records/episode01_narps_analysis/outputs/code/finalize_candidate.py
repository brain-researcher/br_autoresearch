#!/usr/bin/env python3
"""Create the immutable terminal CandidateBundleV1 for the NARPS goal.

This utility is intentionally small and result-aware.  It validates the
completed frozen analysis, applies the frozen numerical decision independently
to gain_demean and loss_demean, expands figures/ into uploadable file refs, and
writes outputs/candidate_bundle.json without overwriting an existing bundle.
It never submits the bundle or invokes Society.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


SCRIPT = Path(__file__).resolve()
WORKSPACE = SCRIPT.parents[2]
MAX_UPLOAD_BYTES = 500_000
CONTRASTS = ("gain_demean", "loss_demean")
CONFOUNDS = ("C0", "C1", "C2")
QC_SETS = ("Q0_all", "Q1_lenient", "Q2_strict")
SMOOTHINGS = ("S0", "S1", "S2")
Q_SHORT = {"Q0_all": "Q0", "Q1_lenient": "Q1", "Q2_strict": "Q2"}
GRID_CELLS = tuple(
    (confound, qc, smoothing)
    for confound in CONFOUNDS for smoothing in SMOOTHINGS for qc in QC_SETS
)
ANOVA_TERMS = ("C", "Q", "S", "C:Q", "C:S", "Q:S", "C:Q:S")
SHAPLEY_FACTORS = ("C", "Q", "S")
METRICS = ("normalized_rms_difference", "one_minus_pearson_r")
TRANSITIONS = ("Q0_to_Q1", "Q1_to_Q2")
EXPECTED_EXCLUSIONS_Q1 = {"sub-030", "sub-100"}
EXPECTED_EXCLUSIONS_Q2 = {
    "sub-016", "sub-018", "sub-030", "sub-088", "sub-100"
}
CONTRACT_HASHES = {
    "frozen_analysis_contract.json":
        "edae18d760897767058f31f4313cc732ab6bdba2acbc4e36a7af7a1e3d9fc0a8",
    "frozen_analysis_contract.md":
        "1bad0c7f0ac03cb4da0d8fddaa69e4f403af83bc00808303107dc06b7cdabd34",
}
COMMON_MASK_SHA256 = (
    "3f52df0149a6cfd55c7bda38de8fd63d7af9148047945320facd107ccf360252"
)
EXPECTED_FIGURES = {
    "continuous_pearson_heatmaps.png",
    "exploratory_er_loo_influence.png",
    "fdr_stability.png",
    "matched_n_calibration.png",
    "variance_attribution.png",
}
REQUIRED_FILE_REFS = (
    "outputs/frozen_analysis_contract.json",
    "outputs/frozen_analysis_contract.md",
    "outputs/qc_subject_sets.tsv",
    "outputs/multiverse_manifest.tsv",
    "outputs/variance_attribution.tsv",
    "outputs/conclusion_stability.tsv",
    "outputs/matched_n_calibration.tsv",
)
SUPPORTING_FILE_REFS = (
    "outputs/common_analysis_mask.nii.gz",
    "outputs/matched_n_random_sets.tsv",
    "outputs/exploratory_er_loo_influence.tsv",
    "outputs/aggregation_provenance.json",
    "outputs/edge_scaling_diagnostic.tsv",
    "outputs/edge_scaling_diagnostic_summary.json",
    "outputs/source_audit.json",
    "outputs/run_design_qc_audit.tsv",
    "outputs/code/audit_and_mask.py",
    "outputs/code/fit_subject.py",
    "outputs/code/fit_subjects.sbatch",
    "outputs/code/narps_common.py",
    "outputs/code/audit_edge_scaling.py",
    "outputs/code/aggregate_results.py",
    "outputs/code/aggregate_results.sbatch",
    "outputs/code/finalize_candidate.py",
    "outputs/tests/test_math_review.py",
    "outputs/tests/test_aggregate_results_math.py",
    "outputs/tests/math_review.md",
)
RESULT_REF = "outputs/RESULT.md"
BUNDLE_REF = "outputs/candidate_bundle.json"

SCHEMA_VERSION = "br.autoresearch_goal_candidate_bundle.v1"
ALL_BUNDLE_FIELDS = {
    "schema_version", "terminal_status", "research_question", "hypothesis",
    "prediction", "falsifier", "primary_test", "controls",
    "development_summary", "reproduction", "output_artifacts", "limitations",
    "deviations", "failed_attempts", "isolation_assurance",
    "scientific_status", "confirmation_requirement", "confirmation_eligible",
    "scientific_acceptance",
}
REQUIRED_BUNDLE_FIELDS = {
    "schema_version", "terminal_status", "research_question",
    "output_artifacts", "limitations", "deviations", "failed_attempts",
    "isolation_assurance", "scientific_status", "confirmation_requirement",
    "confirmation_eligible", "scientific_acceptance",
}
ARTIFACT_RE = re.compile(
    r"^outputs/(?!\.{1,2}(?:/|$))(?!.*(?:/\.{1,2})(?:/|$))"
    r"(?!.*//)(?!.*\\).+$"
)


class ValidationError(RuntimeError):
    """A frozen completion, schema, or upload-bound invariant failed."""


@dataclass(frozen=True)
class ContrastDecision:
    contrast: str
    c_over_s: float
    c_lower: float
    q_over_s: float
    q_lower: float
    supported: bool


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def nonblank(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{label} must be a nonblank string")
    return value.strip()


def parse_bool(value: str, label: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValidationError(f"{label} is not a boolean: {value!r}")


def parse_float(value: str, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(number):
        raise ValidationError(f"{label} is nonfinite: {value!r}")
    return number


def parse_int(value: str, label: str) -> int:
    number = parse_float(value, label)
    if not number.is_integer():
        raise ValidationError(f"{label} is not an integer: {value!r}")
    return int(number)


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.is_symlink():
        raise ValidationError(f"Required TSV is missing or not a regular file: {path}")
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if not reader.fieldnames:
            raise ValidationError(f"TSV has no header: {path}")
        rows = list(reader)
    if not rows:
        raise ValidationError(f"TSV has no data rows: {path}")
    return rows


def require_columns(rows: list[dict[str, str]], columns: Iterable[str], label: str) -> None:
    missing = set(columns) - set(rows[0])
    if missing:
        raise ValidationError(f"{label} lacks columns: {sorted(missing)}")


def validate_workspace(workspace: Path) -> Path:
    workspace = workspace.resolve()
    for name in ("GOAL.md", "DATASETS.md"):
        if not (workspace / name).is_file():
            raise ValidationError(f"workspace_preparation_required: missing {name}")
    for name in ("inputs", "outputs"):
        if not (workspace / name).is_dir():
            raise ValidationError(f"workspace_preparation_required: missing {name}/")
    return workspace / "outputs"


def validate_contract(
    output_root: Path,
    expected_hashes: dict[str, str] = CONTRACT_HASHES,
) -> dict[str, Any]:
    for name, expected in expected_hashes.items():
        path = output_root / name
        if not path.is_file() or path.is_symlink():
            raise ValidationError(f"Frozen contract is missing: {path}")
        observed = sha256(path)
        if observed != expected:
            raise ValidationError(
                f"Frozen contract changed: {name} {observed} != {expected}"
            )
    try:
        contract = json.loads(
            (output_root / "frozen_analysis_contract.json").read_text("utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError("Frozen JSON contract is unreadable") from exc
    expected_boundary = {
        "scientific_status": "exploratory_only",
        "confirmation_eligible": False,
        "scientific_acceptance": False,
        "confirmation_requirement": "independent_fresh_confirmation_required",
    }
    for key, expected in expected_boundary.items():
        if contract.get(key) != expected:
            raise ValidationError(f"Frozen claim boundary mismatch for {key}")
    nonblank(contract.get("research_question"), "contract research_question")
    return contract


def validate_common_mask(output_root: Path, expected_hash: str) -> None:
    path = output_root / "common_analysis_mask.nii.gz"
    if not path.is_file() or path.is_symlink():
        raise ValidationError(f"Frozen common mask is missing: {path}")
    observed = sha256(path)
    if observed != expected_hash:
        raise ValidationError(
            f"Frozen common mask changed: {observed} != {expected_hash}"
        )


def validate_qc(output_root: Path) -> list[dict[str, str]]:
    rows = read_tsv(output_root / "qc_subject_sets.tsv")
    require_columns(
        rows,
        ("participant_id", "group", "Q0_all", "Q1_lenient", "Q2_strict"),
        "qc_subject_sets.tsv",
    )
    if len(rows) != 108:
        raise ValidationError(f"QC table must contain 108 subjects, found {len(rows)}")
    ids = [row["participant_id"] for row in rows]
    if len(set(ids)) != 108:
        raise ValidationError("QC participant IDs are not unique")
    groups = {"equalIndifference": 0, "equalRange": 0}
    memberships: dict[str, set[str]] = {name: set() for name in QC_SETS}
    group_by_id: dict[str, str] = {}
    for row in rows:
        subject = nonblank(row["participant_id"], "participant_id")
        group = row["group"]
        if group not in groups:
            raise ValidationError(f"Unexpected task-version group for {subject}: {group}")
        groups[group] += 1
        group_by_id[subject] = group
        flags = [parse_bool(row[q], f"{subject} {q}") for q in QC_SETS]
        if not (flags[2] <= flags[1] <= flags[0]):
            raise ValidationError(f"QC nesting failed for {subject}")
        for q, included in zip(QC_SETS, flags):
            if included:
                memberships[q].add(subject)
    if groups != {"equalIndifference": 54, "equalRange": 54}:
        raise ValidationError(f"Task-version counts differ from frozen counts: {groups}")
    if [len(memberships[q]) for q in QC_SETS] != [108, 106, 103]:
        raise ValidationError("QC membership counts differ from 108/106/103")
    q0 = memberships["Q0_all"]
    q1_excluded = q0 - memberships["Q1_lenient"]
    q2_excluded = q0 - memberships["Q2_strict"]
    if q1_excluded != EXPECTED_EXCLUSIONS_Q1:
        raise ValidationError(f"Q1 exclusions differ: {sorted(q1_excluded)}")
    if q2_excluded != EXPECTED_EXCLUSIONS_Q2:
        raise ValidationError(f"Q2 exclusions differ: {sorted(q2_excluded)}")
    if any(group_by_id[s] != "equalRange" for s in q2_excluded):
        raise ValidationError("Not all five frozen exclusions are ER subjects")
    return rows


def artifact_path(
    workspace: Path,
    ref: str,
    *,
    require_exists: bool = True,
    require_uploadable: bool = True,
) -> Path:
    if not isinstance(ref, str) or not ARTIFACT_RE.fullmatch(ref):
        raise ValidationError(f"Invalid outputs-relative artifact ref: {ref!r}")
    pure = PurePosixPath(ref)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValidationError(f"Unsafe artifact ref: {ref!r}")
    path = workspace.joinpath(*pure.parts)
    output_root = (workspace / "outputs").resolve()
    try:
        path.resolve(strict=False).relative_to(output_root)
    except ValueError as exc:
        raise ValidationError(f"Artifact escapes outputs/: {ref}") from exc
    if require_exists:
        if not path.is_file() or path.is_symlink():
            raise ValidationError(f"Artifact is missing or not a regular file: {ref}")
        size = path.stat().st_size
        if size <= 0:
            raise ValidationError(f"Artifact is empty: {ref}")
        if require_uploadable and size > MAX_UPLOAD_BYTES:
            raise ValidationError(
                f"Artifact exceeds MCP raw-byte limit ({size}>{MAX_UPLOAD_BYTES}): {ref}"
            )
    return path


def validate_artifact_refs(workspace: Path, refs: list[str]) -> None:
    if len(refs) != len(set(refs)):
        raise ValidationError("output_artifacts contains duplicate refs")
    for ref in refs:
        artifact_path(workspace, ref)


def validate_manifest(workspace: Path, output_root: Path) -> None:
    rows = read_tsv(output_root / "multiverse_manifest.tsv")
    require_columns(
        rows,
        ("contrast", "confound", "qc", "smoothing", "fwhm_mm", "expected_subjects",
         "status", "effect_map", "t_map", "fdr_map", "error"),
        "multiverse_manifest.tsv",
    )
    expected = {
        (contrast, confound, qc, smoothing)
        for contrast in CONTRASTS for confound in CONFOUNDS
        for qc in QC_SETS for smoothing in SMOOTHINGS
    }
    observed: set[tuple[str, str, str, str]] = set()
    expected_n = {"Q0_all": 108, "Q1_lenient": 106, "Q2_strict": 103}
    expected_fwhm = {"S0": 0.0, "S1": 5.0, "S2": 8.0}
    for row in rows:
        key = (row["contrast"], row["confound"], row["qc"], row["smoothing"])
        if key in observed:
            raise ValidationError(f"Duplicate manifest cell: {key}")
        observed.add(key)
        if row["status"] != "complete" or row["error"].strip():
            raise ValidationError(f"Manifest cell is not cleanly complete: {key}")
        if parse_int(row["expected_subjects"], f"{key} expected_subjects") != expected_n[key[2]]:
            raise ValidationError(f"Manifest subject count differs for {key}")
        if parse_float(row["fwhm_mm"], f"{key} fwhm_mm") != expected_fwhm[key[3]]:
            raise ValidationError(f"Manifest smoothing FWHM differs for {key}")
        for column in ("effect_map", "t_map", "fdr_map"):
            # Group maps are validated as local results, but are intentionally not
            # declared for MCP upload because they exceed the bounded packet scope.
            artifact_path(workspace, row[column], require_uploadable=False)
    if observed != expected or len(rows) != 54:
        raise ValidationError("Manifest does not contain exactly the frozen 54 cells")


def validate_conclusion(output_root: Path) -> None:
    rows = read_tsv(output_root / "conclusion_stability.tsv")
    require_columns(
        rows,
        (
            "record_type", "contrast", "confound", "qc", "smoothing",
            "n_subjects", "df", "fdr_rejected_voxels",
            "fdr_rejected_proportion", "fdr_positive_voxels",
            "fdr_negative_voxels", "fdr_any_rejection", "cell_a", "cell_b",
            "differing_axes", "n_differing_axes", "pearson_r",
            "pearson_r_ci_lower", "pearson_r_ci_upper", "cosine_similarity",
            "cosine_similarity_ci_lower", "cosine_similarity_ci_upper",
            "normalized_rms_difference",
            "normalized_rms_difference_ci_lower",
            "normalized_rms_difference_ci_upper", "sign_agreement",
            "fdr_jaccard", "bootstrap_replicates", "bootstrap_seed",
        ),
        "conclusion_stability.tsv",
    )
    cell_rows = [row for row in rows if row["record_type"] == "cell_bh_fdr_summary"]
    expected_cells = {
        (contrast, confound, qc, smoothing)
        for contrast in CONTRASTS for confound in CONFOUNDS
        for qc in QC_SETS for smoothing in SMOOTHINGS
    }
    observed_cells = {
        (row["contrast"], row["confound"], row["qc"], row["smoothing"])
        for row in cell_rows
    }
    if len(cell_rows) != 54 or observed_cells != expected_cells:
        raise ValidationError("Conclusion table lacks the 54 BH-FDR cell summaries")
    expected_n = {"Q0_all": 108, "Q1_lenient": 106, "Q2_strict": 103}
    for row in cell_rows:
        key = (row["contrast"], row["confound"], row["qc"], row["smoothing"])
        n_subjects = parse_int(row["n_subjects"], f"{key} n_subjects")
        if n_subjects != expected_n[row["qc"]]:
            raise ValidationError(f"Conclusion subject count differs for {key}")
        if parse_int(row["df"], f"{key} df") != n_subjects - 2:
            raise ValidationError(f"Conclusion degrees of freedom differ for {key}")
        rejected = parse_int(
            row["fdr_rejected_voxels"], f"{key} fdr_rejected_voxels"
        )
        positive = parse_int(
            row["fdr_positive_voxels"], f"{key} fdr_positive_voxels"
        )
        negative = parse_int(
            row["fdr_negative_voxels"], f"{key} fdr_negative_voxels"
        )
        if min(rejected, positive, negative) < 0 or positive + negative != rejected:
            raise ValidationError(f"Conclusion FDR counts are inconsistent for {key}")
        proportion = parse_float(
            row["fdr_rejected_proportion"], f"{key} fdr_rejected_proportion"
        )
        if not 0.0 <= proportion <= 1.0:
            raise ValidationError(f"Conclusion FDR proportion is invalid for {key}")
        if parse_bool(row["fdr_any_rejection"], f"{key} fdr_any_rejection") != (
            rejected > 0
        ):
            raise ValidationError(f"Conclusion FDR decision is inconsistent for {key}")

    pair_rows = [
        row for row in rows if row["record_type"] == "pairwise_cell_stability"
    ]
    if len(rows) != 756 or len(pair_rows) != 702:
        raise ValidationError(
            "Conclusion table must contain exactly 54 cell and 702 pairwise rows"
        )
    cell_lookup = {
        f"{confound}/{Q_SHORT[qc]}/{smoothing}": (confound, qc, smoothing)
        for confound, qc, smoothing in GRID_CELLS
    }
    labels = tuple(cell_lookup)
    expected_pairs = {
        frozenset((labels[i], labels[j]))
        for i in range(len(labels)) for j in range(i + 1, len(labels))
    }
    for contrast in CONTRASTS:
        contrast_rows = [row for row in pair_rows if row["contrast"] == contrast]
        observed_pairs: set[frozenset[str]] = set()
        for row in contrast_rows:
            first, second = row["cell_a"], row["cell_b"]
            if first not in cell_lookup or second not in cell_lookup or first == second:
                raise ValidationError(
                    f"Invalid pairwise cell labels for {contrast}: {first!r}, {second!r}"
                )
            pair = frozenset((first, second))
            if pair in observed_pairs:
                raise ValidationError(f"Duplicate pairwise row for {contrast}: {first}, {second}")
            observed_pairs.add(pair)
            axes = tuple(
                name for name, left, right in zip(
                    ("C", "Q", "S"), cell_lookup[first], cell_lookup[second]
                ) if left != right
            )
            if row["differing_axes"] != ":".join(axes) or parse_int(
                row["n_differing_axes"], f"{contrast} {first}/{second} n_differing_axes"
            ) != len(axes):
                raise ValidationError(
                    f"Pairwise differing-axis label is inconsistent: {contrast} {first}/{second}"
                )
            for metric in (
                "pearson_r", "cosine_similarity", "normalized_rms_difference",
                "sign_agreement", "fdr_jaccard",
            ):
                value = parse_float(row[metric], f"{contrast} {first}/{second} {metric}")
                if metric in {"pearson_r", "cosine_similarity"} and not -1.0 <= value <= 1.0:
                    raise ValidationError(f"Pairwise {metric} is out of range")
                if metric in {"sign_agreement", "fdr_jaccard"} and not 0.0 <= value <= 1.0:
                    raise ValidationError(f"Pairwise {metric} is out of range")
                if metric == "normalized_rms_difference" and value < 0.0:
                    raise ValidationError("Pairwise normalized RMS difference is negative")
            for metric in (
                "pearson_r", "cosine_similarity", "normalized_rms_difference",
            ):
                lower = parse_float(
                    row[f"{metric}_ci_lower"],
                    f"{contrast} {first}/{second} {metric}_ci_lower",
                )
                upper = parse_float(
                    row[f"{metric}_ci_upper"],
                    f"{contrast} {first}/{second} {metric}_ci_upper",
                )
                if lower > upper:
                    raise ValidationError(f"Pairwise {metric} interval is reversed")
            if (
                parse_int(
                    row["bootstrap_replicates"],
                    f"{contrast} {first}/{second} bootstrap_replicates",
                ) != 2000
                or parse_int(
                    row["bootstrap_seed"],
                    f"{contrast} {first}/{second} bootstrap_seed",
                ) != 20260815
            ):
                raise ValidationError(f"Pairwise bootstrap specification differs for {contrast}")
        if len(contrast_rows) != 351 or observed_pairs != expected_pairs:
            raise ValidationError(
                f"Conclusion table lacks complete 27-cell pair coverage for {contrast}"
            )


def validate_matched_calibration(output_root: Path) -> None:
    rows = read_tsv(output_root / "matched_n_calibration.tsv")
    require_columns(
        rows,
        ("transition", "contrast", "confound", "smoothing", "metric",
         "replicates", "rng", "seed", "removed_ei", "removed_er",
         "observed", "null_median", "null_percentile_2p5",
         "null_percentile_97p5", "observed_to_null_median_ratio",
         "upper_tail_empirical_p"),
        "matched_n_calibration.tsv",
    )
    expected = {
        (transition, contrast, confound, smoothing, metric)
        for transition in TRANSITIONS for contrast in CONTRASTS
        for confound in CONFOUNDS for smoothing in SMOOTHINGS
        for metric in METRICS
    }
    observed_keys: set[tuple[str, str, str, str, str]] = set()
    for row in rows:
        key = (
            row["transition"], row["contrast"], row["confound"],
            row["smoothing"], row["metric"],
        )
        if key in observed_keys:
            raise ValidationError(f"Duplicate matched-calibration row: {key}")
        observed_keys.add(key)
        if parse_int(row["replicates"], f"{key} replicates") != 1000:
            raise ValidationError(f"Matched calibration replicate mismatch: {key}")
        if row["rng"] != "PCG64" or parse_int(row["seed"], f"{key} seed") != 20260814:
            raise ValidationError(f"Matched calibration RNG mismatch: {key}")
        expected_removed = 2 if key[0] == "Q0_to_Q1" else 3
        if (
            parse_int(row["removed_ei"], f"{key} removed_ei") != 0
            or parse_int(row["removed_er"], f"{key} removed_er") != expected_removed
        ):
            raise ValidationError(f"Matched deletion count mismatch: {key}")
        for column in (
            "observed", "null_median", "null_percentile_2p5",
            "null_percentile_97p5", "observed_to_null_median_ratio",
            "upper_tail_empirical_p",
        ):
            parse_float(row[column], f"{key} {column}")
    if len(rows) != 72 or observed_keys != expected:
        raise ValidationError("Matched calibration does not contain exactly 72 summaries")


def validate_matched_random_sets(
    output_root: Path,
    qc_rows: list[dict[str, str]],
) -> None:
    rows = read_tsv(output_root / "matched_n_random_sets.tsv")
    require_columns(
        rows,
        (
            "replicate", "transition", "base_qc", "removed_ei",
            "removed_er", "removed_subjects", "rng", "seed", "draw_order",
        ),
        "matched_n_random_sets.tsv",
    )
    subject_group = {row["participant_id"]: row["group"] for row in qc_rows}
    memberships = {
        qc: {
            row["participant_id"]
            for row in qc_rows if parse_bool(row[qc], f"{row['participant_id']} {qc}")
        }
        for qc in QC_SETS
    }
    transition_spec = {
        "Q0_to_Q1": ("Q0_all", 2),
        "Q1_to_Q2": ("Q1_lenient", 3),
    }
    expected_keys = {
        (replicate, transition)
        for replicate in range(1, 1001) for transition in TRANSITIONS
    }
    observed_keys: set[tuple[int, str]] = set()
    for row in rows:
        replicate = parse_int(row["replicate"], "matched deletion replicate")
        transition = row["transition"]
        key = (replicate, transition)
        if key in observed_keys:
            raise ValidationError(f"Duplicate matched deletion sample row: {key}")
        observed_keys.add(key)
        if transition not in transition_spec:
            raise ValidationError(f"Unexpected matched deletion transition: {transition}")
        base_qc, n_remove = transition_spec[transition]
        if row["base_qc"] != base_qc:
            raise ValidationError(f"Wrong base QC set for matched deletion row: {key}")
        if (
            parse_int(row["removed_ei"], f"{key} removed_ei") != 0
            or parse_int(row["removed_er"], f"{key} removed_er") != n_remove
        ):
            raise ValidationError(f"Wrong matched deletion counts for {key}")
        removed = row["removed_subjects"].split(";")
        if len(removed) != n_remove or len(set(removed)) != n_remove:
            raise ValidationError(f"Wrong or duplicate removed subjects for {key}")
        for subject in removed:
            if subject not in subject_group:
                raise ValidationError(f"Unknown matched deletion subject for {key}: {subject}")
            if subject_group[subject] != "equalRange" or subject not in memberships[base_qc]:
                raise ValidationError(
                    f"Matched deletion subject is outside the frozen ER base pool: {key} {subject}"
                )
        if (
            row["rng"] != "PCG64"
            or parse_int(row["seed"], f"{key} seed") != 20260814
            or row["draw_order"] != "replicate_major_Q0_to_Q1_then_Q1_to_Q2"
        ):
            raise ValidationError(f"Matched deletion generator specification differs for {key}")
    if len(rows) != 2000 or observed_keys != expected_keys:
        raise ValidationError(
            "Matched deletion sample table must contain one row per transition in each of 1000 replicates"
        )


def validate_exploratory_diagnostic(output_root: Path, qc_rows: list[dict[str, str]]) -> None:
    rows = read_tsv(output_root / "exploratory_er_loo_influence.tsv")
    require_columns(
        rows,
        ("label", "contrast", "confound", "smoothing", "participant_id",
         "frozen_status", "normalized_rms_loo_influence", "interpretation"),
        "exploratory_er_loo_influence.tsv",
    )
    er_ids = {row["participant_id"] for row in qc_rows if row["group"] == "equalRange"}
    expected = {
        (contrast, confound, smoothing, subject)
        for contrast in CONTRASTS for confound in CONFOUNDS
        for smoothing in SMOOTHINGS for subject in er_ids
    }
    observed: set[tuple[str, str, str, str]] = set()
    for row in rows:
        key = (
            row["contrast"], row["confound"], row["smoothing"],
            row["participant_id"],
        )
        observed.add(key)
        expected_status = (
            "frozen_excluded_ER"
            if row["participant_id"] in EXPECTED_EXCLUSIONS_Q2
            else "retained_ER"
        )
        if row["label"] != "exploratory_post_primary":
            raise ValidationError(f"Exploratory diagnostic label mismatch: {key}")
        if row["frozen_status"] != expected_status:
            raise ValidationError(f"Exploratory diagnostic status mismatch: {key}")
        if row["interpretation"] != "exploratory_descriptive_not_primary":
            raise ValidationError(f"Exploratory diagnostic label mismatch: {key}")
        parse_float(
            row["normalized_rms_loo_influence"], f"{key} LOO influence"
        )
    if len(rows) != 972 or len(observed) != 972 or observed != expected:
        raise ValidationError("Exploratory ER LOO diagnostic is incomplete")


def validate_provenance(
    output_root: Path,
    expected_contract_hashes: dict[str, str] = CONTRACT_HASHES,
) -> tuple[list[str], list[str]]:
    path = output_root / "aggregation_provenance.json"
    if not path.is_file() or path.is_symlink():
        raise ValidationError("aggregation_provenance.json is missing or not a regular file")
    try:
        payload = json.loads(path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError("aggregation_provenance.json is missing or invalid") from exc
    expected = {
        "schema_version": "narps.aggregation_provenance.v1",
        "status": "complete",
        "scientific_status": "exploratory_only",
        "confirmation_eligible": False,
        "scientific_acceptance": False,
        "confirmation_requirement": "independent_fresh_confirmation_required",
        "completed_grid_cells": 54,
        "group_map_triplets": 54,
        "subject_checkpoints": 108,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValidationError("Aggregation provenance is not terminal and complete")
    if payload.get("contract_hashes") != expected_contract_hashes:
        raise ValidationError("Aggregation provenance contract hashes differ")
    failures = string_list(payload.get("failed_attempts", []), "provenance failed_attempts")
    deviations = string_list(
        payload.get("deviations_from_frozen_analysis", []),
        "provenance deviations_from_frozen_analysis",
    )
    return failures, deviations


def evaluate_variance(output_root: Path) -> list[ContrastDecision]:
    rows = read_tsv(output_root / "variance_attribution.tsv")
    require_columns(
        rows,
        ("contrast", "record_type", "component", "point_estimate", "ci_coverage",
         "ci_lower", "ci_upper", "interval_type", "bootstrap_replicates",
         "rng", "seed"),
        "variance_attribution.tsv",
    )
    expected_inventory = {
        (contrast, "anova_total", "total_across_cell")
        for contrast in CONTRASTS
    }
    expected_inventory.update(
        (contrast, "anova_term", component)
        for contrast in CONTRASTS for component in ANOVA_TERMS
    )
    expected_inventory.update(
        (contrast, "shapley_factor", component)
        for contrast in CONTRASTS for component in SHAPLEY_FACTORS
    )
    expected_inventory.update(
        (contrast, "primary_ratio", component)
        for contrast in CONTRASTS
        for component in ("C_over_S", "Q_over_S", "C_plus_Q_over_S")
    )
    observed_inventory: set[tuple[str, str, str]] = set()
    for row in rows:
        key = (row["contrast"], row["record_type"], row["component"])
        if key in observed_inventory:
            raise ValidationError(f"Duplicate variance-attribution row: {key}")
        observed_inventory.add(key)
        if (
            parse_int(row["bootstrap_replicates"], f"{key} bootstrap_replicates") != 2000
            or row["rng"] != "PCG64"
            or parse_int(row["seed"], f"{key} seed") != 20260815
        ):
            raise ValidationError(f"Wrong variance bootstrap specification: {key}")
    if len(rows) != 28 or observed_inventory != expected_inventory:
        missing = sorted(expected_inventory - observed_inventory)
        unexpected = sorted(observed_inventory - expected_inventory)
        raise ValidationError(
            "Variance-attribution table lacks the exact 28-row frozen inventory; "
            f"missing={missing}, unexpected={unexpected}"
        )
    primary = [row for row in rows if row["record_type"] == "primary_ratio"]
    expected_components = {"C_over_S", "Q_over_S", "C_plus_Q_over_S"}
    decisions: list[ContrastDecision] = []
    for contrast in CONTRASTS:
        contrast_rows = [row for row in primary if row["contrast"] == contrast]
        by_component: dict[str, dict[str, str]] = {}
        for row in contrast_rows:
            component = row["component"]
            if component in by_component:
                raise ValidationError(f"Duplicate primary ratio: {contrast} {component}")
            by_component[component] = row
        if set(by_component) != expected_components:
            raise ValidationError(
                f"Primary ratios differ for {contrast}: {sorted(by_component)}"
            )
        values: dict[str, tuple[float, float]] = {}
        for component, row in by_component.items():
            point = parse_float(
                row["point_estimate"], f"{contrast} {component} point_estimate"
            )
            lower = parse_float(row["ci_lower"], f"{contrast} {component} ci_lower")
            upper = parse_float(row["ci_upper"], f"{contrast} {component} ci_upper")
            if lower > upper:
                raise ValidationError(f"Reversed interval: {contrast} {component}")
            expected_coverage = 0.975 if component in {"C_over_S", "Q_over_S"} else 0.95
            coverage = parse_float(
                row["ci_coverage"], f"{contrast} {component} ci_coverage"
            )
            if not math.isclose(coverage, expected_coverage, abs_tol=1e-12):
                raise ValidationError(f"Wrong interval coverage: {contrast} {component}")
            expected_interval = (
                "Bonferroni_simultaneous_Bayesian_bootstrap_percentile"
                if expected_coverage == 0.975
                else "paired_cohort_stratified_Bayesian_bootstrap_percentile"
            )
            if row["interval_type"] != expected_interval:
                raise ValidationError(f"Wrong interval type: {contrast} {component}")
            if (
                parse_int(row["bootstrap_replicates"], f"{contrast} bootstrap") != 2000
                or row["rng"] != "PCG64"
                or parse_int(row["seed"], f"{contrast} seed") != 20260815
            ):
                raise ValidationError(f"Wrong bootstrap specification: {contrast}")
            values[component] = (point, lower)
        c_point, c_lower = values["C_over_S"]
        q_point, q_lower = values["Q_over_S"]
        decisions.append(ContrastDecision(
            contrast=contrast,
            c_over_s=c_point,
            c_lower=c_lower,
            q_over_s=q_point,
            q_lower=q_lower,
            supported=(
                c_point > 1.0 and q_point > 1.0
                and c_lower > 1.0 and q_lower > 1.0
            ),
        ))
    if {row["contrast"] for row in primary} != set(CONTRASTS) or len(primary) != 6:
        raise ValidationError("Primary-ratio rows include unexpected contrasts or counts")
    return decisions


def markdown_section(text: str, heading: str) -> str:
    lines = text.splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == heading) + 1
    except StopIteration as exc:
        raise ValidationError(f"RESULT.md lacks section {heading}") from exc
    selected: list[str] = []
    for line in lines[start:]:
        if line.startswith("## "):
            break
        if line.strip():
            selected.append(line.strip())
    value = " ".join(selected)
    return nonblank(value, f"RESULT.md {heading} section")


def validate_result(output_root: Path, decisions: list[ContrastDecision]) -> str:
    path = output_root / "RESULT.md"
    try:
        text = path.read_text("utf-8")
    except OSError as exc:
        raise ValidationError("RESULT.md is missing or unreadable") from exc
    lowered = text.lower()
    required_phrases = (
        "exploratory-only analysis-choice attribution study",
        "not independent confirmation",
        "independent fresh confirmation remains required",
        "gain_demean",
        "loss_demean",
    )
    for phrase in required_phrases:
        if phrase not in lowered:
            raise ValidationError(f"RESULT.md lacks claim-boundary phrase: {phrase}")
    section = markdown_section(text, "## Frozen decision rule")
    supported = {item.contrast for item in decisions if item.supported}
    reports_supported = bool(re.search(
        r"frozen numerical full-objective rule is met for:",
        section,
        flags=re.IGNORECASE,
    ))
    reports_closed = "no contrast met the complete frozen rule" in section.lower()
    if reports_supported == reports_closed:
        raise ValidationError(
            "RESULT.md frozen-decision section is missing a unique supported/closed decision"
        )
    if supported:
        match = re.search(
            r"frozen numerical full-objective rule is met for:\s*([^\.\n]+)",
            section,
            flags=re.IGNORECASE,
        )
        if not match:
            raise ValidationError("RESULT.md does not report the supported contrast(s)")
        reported = {item.strip() for item in match.group(1).split(",")}
        if reported != supported:
            raise ValidationError(
                f"RESULT/variance decision mismatch: {sorted(reported)} != {sorted(supported)}"
            )
    elif not reports_closed:
        raise ValidationError("RESULT.md does not report the frozen no-candidate decision")
    return section


def string_list(value: Any, label: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValidationError(f"{label} must be a list")
    return [nonblank(item, f"{label} item") for item in value]


def record_text(record: Any, label: str) -> str:
    if isinstance(record, str):
        return nonblank(record, label)
    if not isinstance(record, dict):
        raise ValidationError(f"{label} must be a string or object")
    for key in ("frozen_contract_unchanged", "contract_unchanged"):
        if key in record and record[key] is not True:
            raise ValidationError(f"{label} reports a changed frozen contract")
    if record.get("analysis_contract_changed") is True:
        raise ValidationError(f"{label} reports a changed frozen contract")
    repair_id = record.get("repair_id", "")
    failure = record.get("failure", record.get("failed_attempt", ""))
    reason = record.get("reason", "")
    repair = record.get("repair", record.get("change", ""))
    parts: list[str] = []
    if isinstance(repair_id, str) and repair_id.strip():
        parts.append(f"Engineering repair {repair_id.strip()}.")
    if isinstance(failure, str) and failure.strip():
        parts.append(f"Engineering failure: {failure.strip()}")
    if isinstance(reason, str) and reason.strip():
        parts.append(f"Reason: {reason.strip()}")
    if isinstance(repair, str) and repair.strip():
        parts.append(f"Repair: {repair.strip()}")
    if not parts:
        summary = record.get("summary", record.get("description", ""))
        if isinstance(summary, str) and summary.strip():
            parts.append(summary.strip())
    if not parts:
        raise ValidationError(f"{label} has no failure/repair text")
    return " ".join(parts)


def load_engineering_repairs(output_root: Path) -> tuple[list[str], list[str], bool]:
    path = output_root / "engineering_repairs.json"
    if not path.exists():
        return [], [], False
    if not path.is_file() or path.is_symlink():
        raise ValidationError("engineering_repairs.json is not a regular file")
    try:
        payload = json.loads(path.read_text("utf-8"))
    except json.JSONDecodeError as exc:
        raise ValidationError("engineering_repairs.json is invalid JSON") from exc
    failures: list[str] = []
    deviations: list[str] = []
    if isinstance(payload, list):
        failures.extend(record_text(item, "engineering repair record") for item in payload)
    elif isinstance(payload, dict):
        if payload.get("schema_version") not in {None, "narps.engineering_repairs.v1"}:
            raise ValidationError("engineering_repairs.json schema_version differs")
        for key in ("frozen_contract_unchanged", "contract_unchanged"):
            if key in payload and payload[key] is not True:
                raise ValidationError("Engineering repair file reports a changed contract")
        if payload.get("analysis_contract_changed") is True:
            raise ValidationError("Engineering repair file reports a changed contract")
        reported_hashes = payload.get("frozen_contract_sha256")
        if reported_hashes is not None:
            expected_hashes = {
                f"outputs/{name}": digest for name, digest in CONTRACT_HASHES.items()
            }
            if reported_hashes != expected_hashes:
                raise ValidationError("Engineering repair contract hashes differ")
        failures.extend(string_list(payload.get("failed_attempts", []), "engineering failed_attempts"))
        records = payload.get("records", payload.get("repairs", []))
        if records is None:
            records = []
        if not isinstance(records, list):
            raise ValidationError("engineering repair records must be a list")
        failures.extend(record_text(item, "engineering repair record") for item in records)
        if any(key in payload for key in ("failure", "failed_attempt", "repair")):
            failures.append(record_text(payload, "engineering repair"))
        deviations.extend(string_list(payload.get("deviations", []), "engineering deviations"))
    else:
        raise ValidationError("engineering_repairs.json must contain an object or list")
    if not failures and not deviations:
        raise ValidationError("engineering_repairs.json contains no recordable entries")
    return failures, deviations, True


def deduplicate(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def figure_refs(workspace: Path, output_root: Path) -> list[str]:
    root = output_root / "figures"
    if not root.is_dir() or root.is_symlink():
        raise ValidationError("outputs/figures/ is missing or not a real directory")
    paths = sorted(path for path in root.rglob("*") if path.is_file())
    observed_names = {
        path.relative_to(root).as_posix() for path in paths
        if len(path.relative_to(root).parts) == 1
    }
    if not EXPECTED_FIGURES.issubset(observed_names):
        raise ValidationError(
            f"Required figures are missing: {sorted(EXPECTED_FIGURES - observed_names)}"
        )
    refs: list[str] = []
    for path in paths:
        if path.is_symlink():
            raise ValidationError(f"Figure may not be a symlink: {path}")
        refs.append(path.relative_to(workspace).as_posix())
    return refs


def validate_bundle_schema(bundle: dict[str, Any]) -> None:
    keys = set(bundle)
    if keys - ALL_BUNDLE_FIELDS:
        raise ValidationError(f"CandidateBundle has additional fields: {sorted(keys - ALL_BUNDLE_FIELDS)}")
    if REQUIRED_BUNDLE_FIELDS - keys:
        raise ValidationError(f"CandidateBundle lacks fields: {sorted(REQUIRED_BUNDLE_FIELDS - keys)}")
    constants = {
        "schema_version": SCHEMA_VERSION,
        "isolation_assurance": "client_unattested",
        "scientific_status": "exploratory_only",
        "confirmation_requirement": "independent_fresh_confirmation_required",
        "confirmation_eligible": False,
        "scientific_acceptance": False,
    }
    for key, expected in constants.items():
        if bundle.get(key) != expected:
            raise ValidationError(f"CandidateBundle constant mismatch: {key}")
    if bundle.get("terminal_status") not in {
        "candidate_ready", "closed_no_candidate", "technical_failure"
    }:
        raise ValidationError("CandidateBundle terminal_status is invalid")
    nonblank(bundle.get("research_question"), "CandidateBundle research_question")
    for key in ("limitations", "deviations", "failed_attempts"):
        string_list(bundle.get(key), f"CandidateBundle {key}")
    refs = bundle.get("output_artifacts")
    if not isinstance(refs, list) or any(not isinstance(item, str) for item in refs):
        raise ValidationError("CandidateBundle output_artifacts must be a string list")
    if len(refs) != len(set(refs)):
        raise ValidationError("CandidateBundle output_artifacts are not unique")
    for ref in refs:
        if not ARTIFACT_RE.fullmatch(ref):
            raise ValidationError(f"CandidateBundle artifact ref is invalid: {ref!r}")
    if bundle["terminal_status"] == "candidate_ready":
        for key in (
            "hypothesis", "prediction", "falsifier", "primary_test",
            "development_summary", "reproduction",
        ):
            nonblank(bundle.get(key), f"candidate_ready {key}")
        controls = bundle.get("controls")
        if not isinstance(controls, list) or not controls:
            raise ValidationError("candidate_ready controls must be a nonempty list")
        for item in controls:
            nonblank(item, "candidate_ready control")


def decision_summary(decisions: list[ContrastDecision]) -> str:
    return "; ".join(
        f"{item.contrast}: C/S={item.c_over_s:.6g} "
        f"(simultaneous lower={item.c_lower:.6g}), Q/S={item.q_over_s:.6g} "
        f"(simultaneous lower={item.q_lower:.6g}), frozen rule="
        f"{'met' if item.supported else 'not met'}"
        for item in decisions
    )


def build_bundle(
    contract: dict[str, Any],
    decisions: list[ContrastDecision],
    result_decision_section: str,
    refs: list[str],
    failures: list[str],
    deviations: list[str],
) -> dict[str, Any]:
    supported = [item.contrast for item in decisions if item.supported]
    terminal_status = "candidate_ready" if supported else "closed_no_candidate"
    ratio_summary = decision_summary(decisions)
    scope = ", ".join(supported)
    if supported:
        hypothesis: str | None = (
            f"For {scope}, under the frozen ds001734 analysis-choice estimand, "
            "confound handling and outcome-blind analysis-set/sample-composition "
            "exclusion each contribute more continuous effect-map variance than "
            "spatial smoothing. This is an exploratory methodological direction, "
            "not a biological-effect claim."
        )
        prediction: str | None = (
            f"In independent fresh confirmation for {scope}, both frozen "
            "Shapley-attributed C/S and Q/S ratios will exceed 1 and both "
            "prespecified Bonferroni-simultaneous 97.5% Bayesian-bootstrap lower "
            "bounds will exceed 1."
        )
        falsifier: str | None = (
            "For any proposed confirming contrast, the direction is falsified if "
            "either C/S or Q/S is at most 1, or if either prespecified simultaneous "
            "lower bound is at most 1."
        )
        primary_test: str | None = (
            f"For {scope}, repeat the frozen 3x3x3 C x Q x S grid in an independent "
            "fresh confirmation sample, construct the same continuous group-intercept "
            "maps on one frozen common mask, apply the exact balanced functional ANOVA "
            "and Shapley allocation, and test both prespecified ratios and simultaneous "
            "lower bounds without threshold tuning."
        )
        controls: list[str] | None = [
            "Keep gain_demean and loss_demean separate; do not use pooled gain-minus-loss as primary.",
            "Use the complete frozen C0/C1/C2 x Q0/Q1/Q2 x S0/S1/S2 grid and one outcome-blind common mask.",
            "Recompute true Friston24 explicitly and verify all six combined-mask retained aCompCor components.",
            "Preserve task-version adjustment and interpret Q only as analysis-set/sample-composition sensitivity.",
            "Reuse cohort-stratified subject weights across cells and use the frozen matched-size deletion calibration and seeds.",
            "Treat BH-FDR q<=0.05 only as the frozen secondary decision-stability readout.",
        ]
    else:
        hypothesis = prediction = falsifier = primary_test = None
        controls = None
    development_summary = (
        "The complete exploratory ds001734 grid, primary continuous-map attribution, "
        "paired 2,000-replicate uncertainty analysis, 1,000-replicate matched-size "
        "deletion calibration, pre-aggregation edge-scaling diagnostic, secondary "
        "BH-FDR stability, and exploratory ER leave-one-out diagnostic completed. "
        f"{ratio_summary}. RESULT decision: "
        f"{result_decision_section}"
    )
    reproduction = (
        "From the workspace root, preserve the frozen contracts byte-for-byte and run "
        "`module load python/3.12.1`; submit `sbatch outputs/code/fit_subjects.sbatch` "
        "for all 108 subject indices; after every immutable checkpoint validates, run "
        "`/home/users/zijiao/pcvenv/bin/python outputs/code/audit_edge_scaling.py` "
        "before `/home/users/zijiao/pcvenv/bin/python "
        "outputs/code/aggregate_results.py --gram-voxel-chunk 8192`; finally run "
        "`/home/users/zijiao/pcvenv/bin/python outputs/code/finalize_candidate.py`. "
        "The frozen bootstrap seed is 20260815 and matched-deletion seed is 20260814."
    )
    bundle = {
        "schema_version": SCHEMA_VERSION,
        "terminal_status": terminal_status,
        "research_question": nonblank(
            contract.get("research_question"), "contract research_question"
        ),
        "hypothesis": hypothesis,
        "prediction": prediction,
        "falsifier": falsifier,
        "primary_test": primary_test,
        "controls": controls,
        "development_summary": development_summary,
        "reproduction": reproduction,
        "output_artifacts": refs,
        "limitations": [
            "Every result is exploratory_only in ds001734; this is not independent confirmation and does not establish a biological effect.",
            "All five frozen exclusions are ER participants, so Q measures analysis-set/sample-composition sensitivity and is not a causal effect of QC.",
            "EI and ER are task versions with different gain support; any direct task-version comparison is exploratory and not a primary biological contrast.",
            "The 27 multiverse cells overlap in subjects and are not independent observations; attribution is conditional on this frozen grid and estimand.",
            "BH-FDR q<=0.05 is a secondary stability readout only and was not tuned or treated as another variance factor.",
            "At mask-edge voxels, Nilearn mean_scaling clamps temporal means below 1, so coefficients there are not literal own-mean percent-signal estimates; very small fixed-effect variances can receive large inverse-variance weight, and smoothing changes the prevalence of this edge condition, so it may contribute to S attribution. Separately computed float64 below-1 counts are diagnostic rather than exact internal clamp counts, and no post-hoc trimming or variance floor was applied.",
        ],
        "deviations": deduplicate(deviations),
        "failed_attempts": deduplicate(failures),
        "isolation_assurance": "client_unattested",
        "scientific_status": "exploratory_only",
        "confirmation_requirement": "independent_fresh_confirmation_required",
        "confirmation_eligible": False,
        "scientific_acceptance": False,
    }
    validate_bundle_schema(bundle)
    return bundle


def write_once(path: Path, payload: bytes, tmp_root: Path) -> bool:
    """Atomically create path. Return False only for an identical existing file."""
    if path.exists():
        if not path.is_file() or path.is_symlink():
            raise ValidationError(f"Refusing to replace non-regular bundle path: {path}")
        if path.read_bytes() == payload:
            return False
        raise ValidationError(f"Refusing to overwrite existing nonidentical bundle: {path}")
    tmp_root.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=tmp_root, prefix=".candidate_bundle.",
            suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise ValidationError(f"Candidate bundle appeared concurrently: {path}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return True


def finalize(
    workspace: Path,
    *,
    expected_contract_hashes: dict[str, str] = CONTRACT_HASHES,
    expected_mask_hash: str = COMMON_MASK_SHA256,
) -> dict[str, Any]:
    output_root = validate_workspace(workspace)
    workspace = output_root.parent
    contract = validate_contract(output_root, expected_contract_hashes)
    validate_common_mask(output_root, expected_mask_hash)
    qc_rows = validate_qc(output_root)
    validate_manifest(workspace, output_root)
    validate_conclusion(output_root)
    validate_matched_calibration(output_root)
    validate_matched_random_sets(output_root, qc_rows)
    validate_exploratory_diagnostic(output_root, qc_rows)
    provenance_failures, provenance_deviations = validate_provenance(
        output_root, expected_contract_hashes
    )
    decisions = evaluate_variance(output_root)
    result_section = validate_result(output_root, decisions)
    repair_failures, repair_deviations, repairs_present = load_engineering_repairs(
        output_root
    )

    refs = list(REQUIRED_FILE_REFS)
    refs.extend(SUPPORTING_FILE_REFS)
    refs.extend(figure_refs(workspace, output_root))
    refs.append(RESULT_REF)
    if repairs_present:
        refs.append("outputs/engineering_repairs.json")
    refs.append(BUNDLE_REF)
    if len(refs) != len(set(refs)):
        raise ValidationError("Expanded required output refs are not unique")

    # Validate every already-existing declared upload before constructing the bundle.
    validate_artifact_refs(workspace, [ref for ref in refs if ref != BUNDLE_REF])
    bundle = build_bundle(
        contract,
        decisions,
        result_section,
        refs,
        provenance_failures + repair_failures,
        provenance_deviations + repair_deviations,
    )
    raw = (json.dumps(bundle, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise ValidationError(
            f"Candidate bundle exceeds MCP raw-byte limit: {len(raw)}>{MAX_UPLOAD_BYTES}"
        )
    created = write_once(output_root / "candidate_bundle.json", raw, output_root / "tmp")
    validate_artifact_refs(workspace, refs)
    status = "created" if created else "already_present_identical"
    print(
        json.dumps({
            "candidate_bundle": BUNDLE_REF,
            "write_status": status,
            "terminal_status": bundle["terminal_status"],
            "supported_contrasts": [
                item.contrast for item in decisions if item.supported
            ],
            "artifact_count": len(refs),
        }, sort_keys=True)
    )
    return bundle


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def synthetic_workspace(root: Path, *, supported: bool) -> tuple[Path, dict[str, str]]:
    workspace = root
    outputs = workspace / "outputs"
    (workspace / "inputs").mkdir(parents=True)
    (outputs / "tmp").mkdir(parents=True)
    (workspace / "GOAL.md").write_text("synthetic goal\n", encoding="utf-8")
    (workspace / "DATASETS.md").write_text("synthetic data contract\n", encoding="utf-8")
    contract = {
        "research_question": "Synthetic NARPS analysis-choice question?",
        "scientific_status": "exploratory_only",
        "confirmation_eligible": False,
        "scientific_acceptance": False,
        "confirmation_requirement": "independent_fresh_confirmation_required",
    }
    (outputs / "frozen_analysis_contract.json").write_text(
        json.dumps(contract, sort_keys=True) + "\n", encoding="utf-8"
    )
    (outputs / "frozen_analysis_contract.md").write_text(
        "# Synthetic frozen contract\n", encoding="utf-8"
    )
    hashes = {
        name: sha256(outputs / name)
        for name in ("frozen_analysis_contract.json", "frozen_analysis_contract.md")
    }

    excluded = EXPECTED_EXCLUSIONS_Q2
    all_ids = [f"sub-{index:03d}" for index in range(1, 109)]
    ei_ids = set([subject for subject in all_ids if subject not in excluded][:54])
    qc_rows: list[dict[str, Any]] = []
    for subject in all_ids:
        qc_rows.append({
            "participant_id": subject,
            "group": "equalIndifference" if subject in ei_ids else "equalRange",
            "Q0_all": True,
            "Q1_lenient": subject not in EXPECTED_EXCLUSIONS_Q1,
            "Q2_strict": subject not in EXPECTED_EXCLUSIONS_Q2,
        })
    write_tsv(
        outputs / "qc_subject_sets.tsv",
        ["participant_id", "group", "Q0_all", "Q1_lenient", "Q2_strict"],
        qc_rows,
    )

    manifest_rows: list[dict[str, Any]] = []
    expected_n = {"Q0_all": 108, "Q1_lenient": 106, "Q2_strict": 103}
    for contrast in CONTRASTS:
        for confound in CONFOUNDS:
            for qc in QC_SETS:
                for smoothing in SMOOTHINGS:
                    stem = f"{contrast}_{confound}_{qc}_{smoothing}"
                    map_refs = [
                        f"outputs/group_maps/{stem}_{suffix}.nii.gz"
                        for suffix in ("effect", "t", "fdr")
                    ]
                    for ref in map_refs:
                        path = workspace.joinpath(*PurePosixPath(ref).parts)
                        path.parent.mkdir(parents=True, exist_ok=True)
                        oversized_local_map = (
                            contrast == "gain_demean" and confound == "C0"
                            and qc == "Q0_all" and smoothing == "S0"
                            and ref.endswith("_effect.nii.gz")
                        )
                        path.write_bytes(
                            b"x" * (MAX_UPLOAD_BYTES + 1)
                            if oversized_local_map else b"synthetic-map"
                        )
                    manifest_rows.append({
                        "contrast": contrast, "confound": confound, "qc": qc,
                        "smoothing": smoothing,
                        "fwhm_mm": {"S0": 0, "S1": 5, "S2": 8}[smoothing],
                        "expected_subjects": expected_n[qc], "status": "complete",
                        "effect_map": map_refs[0], "t_map": map_refs[1],
                        "fdr_map": map_refs[2], "error": "",
                    })
    write_tsv(
        outputs / "multiverse_manifest.tsv",
        ["contrast", "confound", "qc", "smoothing", "fwhm_mm", "expected_subjects",
         "status", "effect_map", "t_map", "fdr_map", "error"],
        manifest_rows,
    )

    conclusion_rows: list[dict[str, Any]] = []
    conclusion_columns = [
        "record_type", "contrast", "confound", "qc", "smoothing",
        "n_subjects", "df", "fdr_rejected_voxels",
        "fdr_rejected_proportion", "fdr_positive_voxels",
        "fdr_negative_voxels", "fdr_any_rejection", "cell_a", "cell_b",
        "differing_axes", "n_differing_axes", "pearson_r",
        "pearson_r_ci_lower", "pearson_r_ci_upper", "cosine_similarity",
        "cosine_similarity_ci_lower", "cosine_similarity_ci_upper",
        "normalized_rms_difference", "normalized_rms_difference_ci_lower",
        "normalized_rms_difference_ci_upper", "sign_agreement", "fdr_jaccard",
        "bootstrap_replicates", "bootstrap_seed", "notes",
    ]
    expected_n = {"Q0_all": 108, "Q1_lenient": 106, "Q2_strict": 103}
    for contrast in CONTRASTS:
        for confound, qc, smoothing in GRID_CELLS:
            n_subjects = expected_n[qc]
            conclusion_rows.append({
                "record_type": "cell_bh_fdr_summary", "contrast": contrast,
                "confound": confound, "qc": qc, "smoothing": smoothing,
                "n_subjects": n_subjects, "df": n_subjects - 2,
                "fdr_rejected_voxels": 0, "fdr_rejected_proportion": 0.0,
                "fdr_positive_voxels": 0, "fdr_negative_voxels": 0,
                "fdr_any_rejection": False, "cell_a": "", "cell_b": "",
                "differing_axes": "", "n_differing_axes": "",
                "pearson_r": "", "pearson_r_ci_lower": "",
                "pearson_r_ci_upper": "", "cosine_similarity": "",
                "cosine_similarity_ci_lower": "",
                "cosine_similarity_ci_upper": "",
                "normalized_rms_difference": "",
                "normalized_rms_difference_ci_lower": "",
                "normalized_rms_difference_ci_upper": "",
                "sign_agreement": "", "fdr_jaccard": "",
                "bootstrap_replicates": "", "bootstrap_seed": "",
                "notes": "BH-FDR q<=0.05 is secondary; threshold is not a factor",
            })
        for first_index, first in enumerate(GRID_CELLS):
            for second in GRID_CELLS[first_index + 1:]:
                first_label = f"{first[0]}/{Q_SHORT[first[1]]}/{first[2]}"
                second_label = f"{second[0]}/{Q_SHORT[second[1]]}/{second[2]}"
                axes = [
                    name for name, left, right in zip(("C", "Q", "S"), first, second)
                    if left != right
                ]
                conclusion_rows.append({
                    "record_type": "pairwise_cell_stability", "contrast": contrast,
                    "confound": "", "qc": "", "smoothing": "",
                    "n_subjects": "", "df": "", "fdr_rejected_voxels": "",
                    "fdr_rejected_proportion": "", "fdr_positive_voxels": "",
                    "fdr_negative_voxels": "", "fdr_any_rejection": "",
                    "cell_a": first_label, "cell_b": second_label,
                    "differing_axes": ":".join(axes),
                    "n_differing_axes": len(axes), "pearson_r": 0.9,
                    "pearson_r_ci_lower": 0.8, "pearson_r_ci_upper": 0.95,
                    "cosine_similarity": 0.9,
                    "cosine_similarity_ci_lower": 0.8,
                    "cosine_similarity_ci_upper": 0.95,
                    "normalized_rms_difference": 0.1,
                    "normalized_rms_difference_ci_lower": 0.05,
                    "normalized_rms_difference_ci_upper": 0.15,
                    "sign_agreement": 0.9, "fdr_jaccard": 1.0,
                    "bootstrap_replicates": 2000, "bootstrap_seed": 20260815,
                    "notes": "sign agreement and FDR Jaccard are descriptive",
                })
    write_tsv(
        outputs / "conclusion_stability.tsv",
        conclusion_columns,
        conclusion_rows,
    )

    matched_rows: list[dict[str, Any]] = []
    for transition in TRANSITIONS:
        for contrast in CONTRASTS:
            for confound in CONFOUNDS:
                for smoothing in SMOOTHINGS:
                    for metric in METRICS:
                        matched_rows.append({
                            "transition": transition, "contrast": contrast,
                            "confound": confound, "smoothing": smoothing,
                            "metric": metric, "replicates": 1000, "rng": "PCG64",
                            "seed": 20260814, "removed_ei": 0,
                            "removed_er": 2 if transition == "Q0_to_Q1" else 3,
                            "observed": 0.1, "null_median": 0.08,
                            "null_percentile_2p5": 0.01,
                            "null_percentile_97p5": 0.2,
                            "observed_to_null_median_ratio": 1.25,
                            "upper_tail_empirical_p": 0.2,
                        })
    write_tsv(
        outputs / "matched_n_calibration.tsv", list(matched_rows[0]), matched_rows
    )
    group_by_id = {row["participant_id"]: row["group"] for row in qc_rows}
    qc_by_id = {row["participant_id"]: row for row in qc_rows}
    random_set_rows: list[dict[str, Any]] = []
    transition_spec = {
        "Q0_to_Q1": ("Q0_all", 2),
        "Q1_to_Q2": ("Q1_lenient", 3),
    }
    for replicate in range(1, 1001):
        for transition in TRANSITIONS:
            base_qc, n_remove = transition_spec[transition]
            er_pool = sorted(
                subject for subject, group in group_by_id.items()
                if group == "equalRange" and bool(qc_by_id[subject][base_qc])
            )
            removed = er_pool[:n_remove]
            random_set_rows.append({
                "replicate": replicate, "transition": transition,
                "base_qc": base_qc, "removed_ei": 0, "removed_er": n_remove,
                "removed_subjects": ";".join(removed), "rng": "PCG64",
                "seed": 20260814,
                "draw_order": "replicate_major_Q0_to_Q1_then_Q1_to_Q2",
            })
    write_tsv(
        outputs / "matched_n_random_sets.tsv",
        list(random_set_rows[0]), random_set_rows,
    )

    variance_rows: list[dict[str, Any]] = []
    for contrast in CONTRASTS:
        if contrast == "gain_demean" and supported:
            values = {"C_over_S": (1.4, 1.1), "Q_over_S": (1.3, 1.05),
                      "C_plus_Q_over_S": (2.7, 2.1)}
        else:
            values = {"C_over_S": (0.9, 0.7), "Q_over_S": (1.3, 1.1),
                      "C_plus_Q_over_S": (2.2, 1.7)}
        variance_rows.append({
            "contrast": contrast, "record_type": "anova_total",
            "component": "total_across_cell", "point_estimate": 1.0,
            "ci_coverage": "", "ci_lower": "", "ci_upper": "",
            "interval_type": "not_applicable", "bootstrap_replicates": 2000,
            "rng": "PCG64", "seed": 20260815,
        })
        for component in ANOVA_TERMS:
            variance_rows.append({
                "contrast": contrast, "record_type": "anova_term",
                "component": component, "point_estimate": 1.0 / len(ANOVA_TERMS),
                "ci_coverage": 0.95, "ci_lower": 0.05, "ci_upper": 0.25,
                "interval_type": "paired_cohort_stratified_Bayesian_bootstrap_percentile",
                "bootstrap_replicates": 2000, "rng": "PCG64", "seed": 20260815,
            })
        c_ratio = values["C_over_S"][0]
        q_ratio = values["Q_over_S"][0]
        s_share = 1.0 / (1.0 + c_ratio + q_ratio)
        for component, point in zip(
            SHAPLEY_FACTORS,
            (c_ratio * s_share, q_ratio * s_share, s_share),
        ):
            variance_rows.append({
                "contrast": contrast, "record_type": "shapley_factor",
                "component": component, "point_estimate": point,
                "ci_coverage": 0.95, "ci_lower": max(0.0, point - 0.1),
                "ci_upper": min(1.0, point + 0.1),
                "interval_type": "paired_cohort_stratified_Bayesian_bootstrap_percentile",
                "bootstrap_replicates": 2000, "rng": "PCG64", "seed": 20260815,
            })
        for component, (point, lower) in values.items():
            simultaneous = component in {"C_over_S", "Q_over_S"}
            variance_rows.append({
                "contrast": contrast, "record_type": "primary_ratio",
                "component": component, "point_estimate": point,
                "ci_coverage": 0.975 if simultaneous else 0.95,
                "ci_lower": lower, "ci_upper": point + 0.5,
                "interval_type": (
                    "Bonferroni_simultaneous_Bayesian_bootstrap_percentile"
                    if simultaneous else
                    "paired_cohort_stratified_Bayesian_bootstrap_percentile"
                ),
                "bootstrap_replicates": 2000, "rng": "PCG64", "seed": 20260815,
            })
    write_tsv(
        outputs / "variance_attribution.tsv", list(variance_rows[0]), variance_rows
    )

    er_ids = [row["participant_id"] for row in qc_rows if row["group"] == "equalRange"]
    diagnostic_rows = [
        {"label": "exploratory_post_primary", "contrast": contrast,
         "confound": confound, "smoothing": smoothing,
         "participant_id": subject,
         "frozen_status": (
             "frozen_excluded_ER" if subject in excluded else "retained_ER"
         ),
         "normalized_rms_loo_influence": 0.01,
         "interpretation": "exploratory_descriptive_not_primary"}
        for contrast in CONTRASTS for confound in CONFOUNDS
        for smoothing in SMOOTHINGS for subject in er_ids
    ]
    write_tsv(
        outputs / "exploratory_er_loo_influence.tsv",
        list(diagnostic_rows[0]), diagnostic_rows,
    )
    (outputs / "aggregation_provenance.json").write_text(json.dumps({
        "schema_version": "narps.aggregation_provenance.v1",
        "status": "complete", "scientific_status": "exploratory_only",
        "confirmation_eligible": False, "scientific_acceptance": False,
        "confirmation_requirement": "independent_fresh_confirmation_required",
        "contract_hashes": hashes, "completed_grid_cells": 54,
        "group_map_triplets": 54, "subject_checkpoints": 108,
        "failed_attempts": [],
        "deviations_from_frozen_analysis": [],
    }) + "\n", encoding="utf-8")
    (outputs / "edge_scaling_diagnostic.tsv").write_text(
        "subject\tstatus\nsub-001\tpass\n", encoding="utf-8"
    )
    (outputs / "edge_scaling_diagnostic_summary.json").write_text(
        json.dumps({"status": "complete", "subject_count": 108}) + "\n",
        encoding="utf-8",
    )
    (outputs / "source_audit.json").write_text(
        json.dumps({"status": "pass", "subjects": 108, "runs": 432}) + "\n",
        encoding="utf-8",
    )
    (outputs / "run_design_qc_audit.tsv").write_text(
        "participant_id\trun\tstatus\nsub-001\t1\tpass\n", encoding="utf-8"
    )
    (outputs / "common_analysis_mask.nii.gz").write_bytes(b"synthetic-mask")
    code_root = outputs / "code"
    code_root.mkdir()
    for name in (
        "audit_and_mask.py", "fit_subject.py", "fit_subjects.sbatch",
        "narps_common.py", "audit_edge_scaling.py", "aggregate_results.py",
        "aggregate_results.sbatch", "finalize_candidate.py",
    ):
        (code_root / name).write_text(
            f"# synthetic reproducibility fixture: {name}\n", encoding="utf-8"
        )
    tests_root = outputs / "tests"
    tests_root.mkdir()
    for name in (
        "test_math_review.py", "test_aggregate_results_math.py", "math_review.md",
    ):
        (tests_root / name).write_text(
            f"synthetic independent review fixture: {name}\n", encoding="utf-8"
        )
    figures = outputs / "figures"
    figures.mkdir()
    for name in EXPECTED_FIGURES:
        (figures / name).write_bytes(b"synthetic-figure")
    (outputs / "engineering_repairs.json").write_text(json.dumps({
        "schema_version": "narps.engineering_repairs.v1",
        "failed_attempts": ["synthetic transient scheduler interruption"],
        "repairs": [{
            "repair_id": "synthetic_resume_unchanged_contract",
            "reason": "a transient scheduler interruption stopped one task",
            "change": "resumed the unchanged frozen computation",
        }],
        "deviations": [],
    }) + "\n", encoding="utf-8")
    decision_text = (
        "The frozen numerical full-objective rule is met for: gain_demean."
        if supported else
        "No contrast met the complete frozen rule requiring both ratios."
    )
    (outputs / "RESULT.md").write_text(
        "# Synthetic result\n\n"
        "This is an exploratory-only analysis-choice attribution study. It is not "
        "independent confirmation. Independent fresh confirmation remains required. "
        "gain_demean and loss_demean were separate.\n\n"
        f"## Frozen decision rule\n\n{decision_text}\n\n## Next section\n",
        encoding="utf-8",
    )
    return workspace, hashes


def run_self_test() -> int:
    real_tmp = WORKSPACE / "outputs" / "tmp"
    real_tmp.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="finalize_candidate_selftest_", dir=real_tmp
    ) as temporary:
        root = Path(temporary)
        ready_workspace, ready_hashes = synthetic_workspace(
            root / "candidate_ready", supported=True
        )
        ready = finalize(
            ready_workspace,
            expected_contract_hashes=ready_hashes,
            expected_mask_hash=sha256(
                ready_workspace / "outputs" / "common_analysis_mask.nii.gz"
            ),
        )
        assert ready["terminal_status"] == "candidate_ready"
        assert ready["hypothesis"] and ready["controls"]
        assert any(
            "scheduler interruption" in item for item in ready["failed_attempts"]
        )
        assert any(
            "synthetic_resume_unchanged_contract" in item
            and "resumed the unchanged frozen computation" in item
            for item in ready["failed_attempts"]
        )
        assert all(ref != "outputs/figures/" for ref in ready["output_artifacts"])
        assert EXPECTED_FIGURES.issubset({Path(ref).name for ref in ready["output_artifacts"]})
        assert set(SUPPORTING_FILE_REFS).issubset(ready["output_artifacts"])
        assert "module load python/3.12.1" in ready["reproduction"]
        assert ready["reproduction"].index("audit_edge_scaling.py") < ready["reproduction"].index(
            "aggregate_results.py"
        )
        assert any("not literal own-mean percent-signal" in item for item in ready["limitations"])
        local_map = (
            ready_workspace / "outputs" / "group_maps"
            / "gain_demean_C0_Q0_all_S0_effect.nii.gz"
        )
        assert local_map.stat().st_size > MAX_UPLOAD_BYTES
        assert local_map.relative_to(ready_workspace).as_posix() not in ready["output_artifacts"]
        ready_bundle_path = ready_workspace / "outputs" / "candidate_bundle.json"
        ready_bundle_bytes = ready_bundle_path.read_bytes()
        ready_again = finalize(
            ready_workspace,
            expected_contract_hashes=ready_hashes,
            expected_mask_hash=sha256(
                ready_workspace / "outputs" / "common_analysis_mask.nii.gz"
            ),
        )
        assert ready_again == ready
        assert ready_bundle_path.read_bytes() == ready_bundle_bytes
        try:
            write_once(
                ready_bundle_path,
                b'{"different":true}\n',
                ready_workspace / "outputs" / "tmp",
            )
        except ValidationError:
            pass
        else:
            raise AssertionError("Nonidentical candidate bundle overwrite was accepted")
        assert ready_bundle_path.read_bytes() == ready_bundle_bytes

        closed_workspace, closed_hashes = synthetic_workspace(
            root / "closed", supported=False
        )
        closed = finalize(
            closed_workspace,
            expected_contract_hashes=closed_hashes,
            expected_mask_hash=sha256(
                closed_workspace / "outputs" / "common_analysis_mask.nii.gz"
            ),
        )
        assert closed["terminal_status"] == "closed_no_candidate"
        assert closed["hypothesis"] is None and closed["controls"] is None

        def assert_removed_row_rejected(
            path: Path,
            validator: Any,
            label: str,
        ) -> None:
            original = path.read_bytes()
            rows = read_tsv(path)
            write_tsv(path, list(rows[0]), rows[:-1])
            try:
                try:
                    validator()
                except ValidationError:
                    pass
                else:
                    raise AssertionError(f"Incomplete {label} was accepted")
            finally:
                path.write_bytes(original)

        closed_outputs = closed_workspace / "outputs"
        assert_removed_row_rejected(
            closed_outputs / "variance_attribution.tsv",
            lambda: evaluate_variance(closed_outputs),
            "variance-attribution inventory",
        )
        assert_removed_row_rejected(
            closed_outputs / "conclusion_stability.tsv",
            lambda: validate_conclusion(closed_outputs),
            "conclusion pair coverage",
        )
        closed_qc_rows = read_tsv(closed_outputs / "qc_subject_sets.tsv")
        assert_removed_row_rejected(
            closed_outputs / "matched_n_random_sets.tsv",
            lambda: validate_matched_random_sets(closed_outputs, closed_qc_rows),
            "matched deletion sample inventory",
        )

        oversized = closed_workspace / "outputs" / "oversized.bin"
        oversized.write_bytes(b"x" * (MAX_UPLOAD_BYTES + 1))
        try:
            validate_artifact_refs(closed_workspace, ["outputs/oversized.bin"])
        except ValidationError:
            pass
        else:
            raise AssertionError("Oversized artifact was accepted")
        try:
            validate_artifact_refs(
                closed_workspace, [RESULT_REF, RESULT_REF]
            )
        except ValidationError:
            pass
        else:
            raise AssertionError("Duplicate artifact refs were accepted")
    print("finalize_candidate.py synthetic self-test: PASS")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--self-test", action="store_true",
        help="run only synthetic tests under outputs/tmp; do not inspect real results",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        return run_self_test()
    finalize(WORKSPACE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
