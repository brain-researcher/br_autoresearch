#!/usr/bin/env python3
"""Apply the fixed M1 publication comparison to sixteen completed references.

Reads selected reference artifacts, never the provider payload; makes no fits
to source trials. Numerical reconstruction is not an automatic primary pass.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.stats import f

from source_adapter import M1_SESSIONS, compute_only, write_json
from run_publication_reproduction import ANALYSIS_REVISION, TRIALDATA_REVISION


def association(x, y):
    x, y = np.asarray(x, dtype=float).ravel(), np.asarray(y, dtype=float).ravel()
    if x.shape != y.shape or len(x) < 3 or not np.isfinite([x, y]).all():
        raise ValueError("association requires matching finite vectors")
    xc, yc = x - x.mean(), y - y.mean()
    xx, yy = xc @ xc, yc @ yc
    if xx <= 0 or yy <= 0:
        raise ValueError("association cannot be defined for a constant vector")
    slope = (xc @ yc) / xx
    residual = yc - slope * xc
    sse = residual @ residual
    r_squared = 1 - sse / yy
    p = 0.0 if sse == 0 else float(f.sf((yy - sse) / (sse / (len(x) - 2)), 1, len(x) - 2))
    return {"points": len(x), "intercept": float(y.mean() - slope * x.mean()),
            "slope": float(slope), "r_squared": float(r_squared), "F_test_p": p,
            "positive_slope": bool(slope > 0), "paper_p_lt_001": bool(p < .001),
            "published_R_squared": .78,
            "rounds_to_published_value": bool(.775 <= r_squared < .785),
            "difference_from_printed_value": float(r_squared - .78)}


def split_monte_carlo(cca_means, split_xy, observed):
    # Averaging X/Y before resampling is algebraically identical to carrying
    # each entire pair through its shared split index and averaging afterwards.
    scores = split_xy[:, :9].mean(axis=-1)
    xc = cca_means.ravel() - cca_means.mean()
    xx = xc @ xc
    rng = np.random.Generator(np.random.PCG64(20261003))
    differences = np.empty(10000)
    for k in range(10000):
        values = []
        for _ in range(2):
            indices = rng.integers(0, 50, size=scores.shape)
            y = np.take_along_axis(scores, indices, axis=2).mean(axis=2).ravel()
            yc = y - y.mean()
            if yc @ yc <= 0:
                raise ValueError("undefined bootstrap association; no replicate dropped")
            values.append(float((xc @ yc) ** 2 / (xx * (yc @ yc))))
        differences[k] = values[0] - values[1]
    limits = np.quantile(differences, [.025, .975], method="linear")
    shifted = observed + limits
    return {"replicates": 10000, "generator": "PCG64", "seed": 20261003,
            "difference_quantiles_025_975": limits.tolist(),
            "shifted_interval": shifted.tolist(),
            "overlaps_published_rounding_interval": bool(shifted[1] >= .775 and shifted[0] < .785),
            "interpretation": "approximate conditional split-Monte-Carlo compatibility only",
            "equivalence_or_primary_pass_rule": False}


def read_session(directory, expected, index):
    with (directory / "reproduction_report.json").open() as stream:
        report = json.load(stream)
    identity = f"{expected[0]}_{expected[2]}"
    if (report["status"] != "reference_session_completed" or report["session_id"] != identity
            or report["analysis_revision"] != ANALYSIS_REVISION
            or report["trialdata_compatibility_revision"] != TRIALDATA_REVISION
            or report["reference_functions_modified"]
            or report["seed"] != 20261003 + index
            or not report["reference_result"]["fixtures_passed"]):
        raise ValueError(f"incomplete or changed reference identity/recipe: {identity}")
    source = report["selected_source"]
    if source["animal"] != expected[0] or source["implant"] != expected[1]:
        raise ValueError("animal/implant mapping disagrees with the fixed session list")
    bands = np.asarray(source["upper_band_hz"], dtype=float)
    if bands.shape != (9,) or not np.isfinite(bands).all() or np.any(np.diff(bands) <= 0):
        raise ValueError("nine ordered guide-defined frequency bands are required")
    cca = np.asarray(loadmat(directory / "reference_cca.mat", variable_names=["cca_coef"])["cca_coef"])
    if cca.shape != (len(source["matched_electrodes"]), 9) or not np.isfinite(cca).all():
        raise ValueError("CCA support or numerical validity differs from the reference")
    # MATLAB stores predictions together with scores in this published result
    # struct; loadmat deserializes that selected artifact, not provider arrays.
    velocity = loadmat(directory / "reference_velocity.mat", variable_names=["velocity"],
                       squeeze_me=True, struct_as_record=False)["velocity"]
    velocity = np.asarray(velocity).ravel(order="F")
    if len(velocity) != 500:
        raise ValueError("all fifty splits for all ten representations are required")
    scores = np.empty((10, 50, 2))
    for band in range(1, 11):
        selected = [np.asarray(v.r2, dtype=float).ravel() for v in velocity if int(v.band) == band]
        values = np.asarray(selected)
        if values.shape != (50, 2) or not np.isfinite(values).all():
            raise ValueError("missing, nonfinite, or incorrectly paired XY split scores")
        scores[band - 1] = values
    return report, bands, cca, scores


def session_directory(index, expected, first_session, remaining_root, chewie_20150313_recovery=None):
    """Replace only the explicitly recovered session, never a favorable subset."""
    identity = f"{expected[0]}_{expected[2]}"
    if identity == "Chewie_20150313" and chewie_20150313_recovery is not None:
        return chewie_20150313_recovery
    return first_session if index == 0 else remaining_root / identity


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--first-session", type=Path, required=True)
    p.add_argument("--remaining-root", type=Path, required=True)
    p.add_argument("--chewie-20150313-recovery", type=Path,
                   help="Use the fresh completion directory only for Chewie_20150313; all identity/seed/revision checks remain")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--split-bootstrap", action="store_true")
    a = p.parse_args()
    compute_only()
    a.out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    result = {"analysis_id": "publication_reproduction", "status": "started",
              "scope": "sixteen-session M1 movement-execution benchmark",
              "profile_comparison": "descriptive; no invented bandwise pass cutoffs",
              "contract_decision": "requires evidence interpretation recorded in experiment_log",
              "automatic_primary_search_authorization": False,
              "scientific_trial_count_increment": 0, "source_payload_read": False}
    try:
        all_cca, all_scores, session_rows = [], [], []
        common_bands = None
        for index, expected in enumerate(M1_SESSIONS):
            sid = f"{expected[0]}_{expected[2]}"
            directory = session_directory(index, expected, a.first_session, a.remaining_root,
                                          a.chewie_20150313_recovery)
            report, bands, cca, scores = read_session(directory, expected, index)
            if common_bands is None:
                common_bands = bands
            elif not np.array_equal(common_bands, bands):
                raise ValueError("guide-defined band identity differs across sessions")
            all_cca.append(cca)
            all_scores.append(scores)
            session_rows.append({"session_id": sid, "animal": expected[0], "implant": expected[1],
                                 "input_directory": str(directory), "seed": report["seed"],
                                 "matched_electrodes": len(cca),
                                 "channel_mean_CCA": cca.mean(axis=0).tolist(),
                                 "mean_XY_split_R_squared": scores.mean(axis=(1, 2)).tolist()})
        pooled_cca = np.concatenate(all_cca, axis=0)
        split_xy = np.stack(all_scores)
        pooled_scores = split_xy.transpose(1, 0, 2, 3).reshape(10, -1)
        cca_means = np.stack([c.mean(axis=0) for c in all_cca])
        decoder_means = split_xy.mean(axis=(2, 3))
        counts = {"sessions": len(all_cca), "CCA_values_per_band": len(pooled_cca),
                  "decoder_values_per_representation": pooled_scores.shape[1],
                  "association_points": cca_means.size}
        counts["all_published_support_counts_match"] = bool(
            len(all_cca) == 16 and len(pooled_cca) == 606
            and pooled_scores.shape[1] == 1600 and cca_means.size == 144)
        result.update(support=counts, upper_band_hz=common_bands.tolist(), sessions=session_rows,
                      CCA_caption_medians=np.median(pooled_cca, axis=0).tolist(),
                      CCA_code_means=pooled_cca.mean(axis=0).tolist(),
                      decoder_caption_medians=np.median(pooled_scores, axis=1).tolist(),
                      decoder_code_means=pooled_scores.mean(axis=1).tolist(),
                      association=association(cca_means, decoder_means[:, :9]))
        result["group_profiles"] = {}
        for field in ("animal", "implant"):
            for name in sorted({row[field] for row in session_rows}):
                indices = [i for i, row in enumerate(session_rows) if row[field] == name]
                result["group_profiles"][f"{field}:{name}"] = {
                    "sessions": len(indices),
                    "session_mean_CCA": cca_means[indices].mean(axis=0).tolist(),
                    "session_mean_decoder_R_squared": decoder_means[indices].mean(axis=0).tolist()}
        if a.split_bootstrap:
            result["split_monte_carlo"] = split_monte_carlo(
                cca_means, split_xy, result["association"]["r_squared"])
        result["status"] = ("complete_scoped_benchmark_reconstruction" if
                            counts["all_published_support_counts_match"] else "published_support_count_discrepancy")
    except Exception as error:
        result["status"] = "technical_failure"
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        result["elapsed_seconds"] = time.monotonic() - start
        write_json(a.out / "benchmark_comparison.json", result)
    print(json.dumps({"status": result["status"], "support": counts,
                      "automatic_primary_search_authorization": False}))


if __name__ == "__main__":
    main()
