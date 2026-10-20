"""Fixed adjusted matched-image lag curves from released numeric records only.

Forty post-event, forty pre-event and forty same-person translated contrasts
share one 120-endpoint, two-sided 95% Bonferroni participant-t interval family.
No EEG, model refit, new preprocessing, peak selection or primary pass rule.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket

import numpy as np
from scipy.stats import t


EPISODE = Path(__file__).resolve().parents[2]
INPUT_ROOT = EPISODE / "outputs/runs/repeated_signal_detrend/detrend001"
PEOPLE = tuple("sub-" + suffix for suffix in (
    "07", "08", "09", "10", "11", "12", "13", "14", "20", "21", "22", "24", "25", "26", "27",
    "28", "29", "30", "31", "32", "34", "35", "38", "43", "44", "45", "46", "47", "48", "49"))
FAMILY_SIZE, ALPHA = 120, .05
CURVES = ("post", "pre", "post_minus_pre")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _read(path):
    require(path.stat().st_size <= 2 * 1024**2, "bounded complete participant numeric record required")
    with path.open() as source:
        return json.load(source)


def _numeric(values, *, nonnegative=False):
    result = np.asarray(values, dtype=float)
    require(result.shape == (40,) and np.isfinite(result).all()
            and (not nonnegative or np.all(result >= 0)), "forty finite complete lag values required")
    return result


def _curve(structure, person, *, negative):
    require(structure.get("schema") == "ep18_repeated_signal_structure_v1"
        and structure.get("status") == "complete_repeated_signal_structure"
        and structure.get("participant_id") == person
        and structure.get("physical_sequence_count") == 12
        and structure.get("image_identity_count") == 200
        and structure.get("recorded_channel_count") == 62
        and structure.get("sensor_contrast_count") == 61
        and structure.get("raw_EEG_detrending") is True
        and structure.get("audit_access") is False
        and structure.get("concept_or_false_scores_computed") is False
        and structure.get("primary_rescue") is False,
        "complete original-person adjusted template structure required")
    definition = structure["template_definition"]
    edges = list(range(-800, 1, 20)) if negative else list(range(0, 801, 20))
    require(definition.get("lag_bin_edges_ms") == edges
        and definition.get("lag_window_relative_to_original_event_ms") == ([-800, 0] if negative else [0, 800])
        and definition.get("diagnostic_onset_shift_ms") == (-800 if negative else 0)
        and definition.get("final_bin_includes_0ms" if negative else "final_bin_includes_800ms") is True
        and (not negative or definition.get("final_bin_includes_800ms") is False)
        and definition.get("centering") == "across200_images_separately_within_each_sequence_lag_sensor_contrast"
        and definition.get("halves") == "six_chronological_human_odd_and_six_even_sequences"
        and definition.get("interpolation") is False and definition.get("event_snapping") is False
        and definition.get("prestimulus_correction") is False,
        "unchanged adjusted fixed 40-bin acquisition-lag definition required")
    sequences = structure["sequence_metrics"]
    require(len(sequences) == 12 and [row["ordinal"] for row in sequences] == list(range(12))
        and len({row["sequence_id"] for row in sequences}) == 12,
        "all twelve original sequence identities required")
    lag = structure["template_metrics"]["by_lag"]
    cross = _numeric(lag["matched_cross_covariance_V2"])
    odd, even = (_numeric(lag[key], nonnegative=True) for key in ("odd_energy_V2", "even_energy_V2"))
    denominator = np.sqrt(odd) * np.sqrt(even)
    values = lag["matched_cosine"]
    require(len(values) == 40, "all forty saved matched cosines required")
    result = []
    for index, value in enumerate(values):
        if denominator[index] == 0:
            require(value is None and cross[index] == 0, "zero-energy cosine must remain explicitly undefined")
            result.append(None)
        else:
            require(value is not None and type(value) in (int, float)
                and np.isfinite(value) and abs(value) <= 1 + 1e-10
                and np.isclose(value, cross[index] / denominator[index], rtol=1e-10, atol=1e-12),
                "saved per-lag cosine must use its own two half-template energies")
            result.append(float(value))
    support = [(row["sequence_id"], row["ordinal"], row["retained_samples"]) for row in sequences]
    return result, support


def _participant(row, person):
    require(row.get("schema") == "ep18_repeated_signal_detrend_v1"
        and row.get("status") == "complete_repeated_signal_detrend"
        and row.get("participant_id") == person and row.get("original_R_changed") is False
        and row.get("original_control_threshold_changed") is False
        and row.get("out_of_sample_prediction") is False and row.get("primary_rescue") is False
        and row.get("audit_access") is False and row.get("concept_or_false_scores_computed") is False,
        "same complete adjusted person without primary or data-role change required")
    recipe = row["projection_recipe"]
    require(recipe.get("choice_count") == 1 and recipe.get("degree") == 3
        and recipe.get("basis") == "degree_0_to_3_time_Legendre_QR_on_original_retained_samples"
        and recipe.get("scope") == "independent_physical_sequence_and_each_of_62_original_channels"
        and recipe.get("lag_windows_relative_to_original_event_ms") == [[0, 800], [-800, 0]]
        and recipe.get("whole_sequence_data_used") is True
        and recipe.get("preprocessing_search") is False,
        "one fixed degree-three adjusted source required")
    post, post_support = _curve(row["structure"], person, negative=False)
    pre, pre_support = _curve(row["negative_lag_structure"], person, negative=True)
    require(post_support == pre_support
        and row["structure"]["image_order"] == row["negative_lag_structure"]["image_order"],
        "paired lag windows must retain identical persons, sequences and image order")
    return {"post": post, "pre": pre,
        "post_minus_pre": [None if first is None or second is None else first - second
                           for first, second in zip(post, pre)]}


def _interval(values, critical):
    missing = [person for person, value in zip(PEOPLE, values) if value is None]
    if missing:
        return {"status": "unavailable_all30", "undefined_participants": missing,
            "mean": None, "lower": None, "upper": None}
    values = np.asarray(values, dtype=float)
    require(values.shape == (30,) and np.isfinite(values).all(), "thirty finite participant values required")
    mean = float(values.mean())
    standard_error = float(values.std(ddof=1) / np.sqrt(30))
    return {"status": "complete_all30", "mean": mean, "standard_error": standard_error,
        "lower": mean - critical * standard_error, "upper": mean + critical * standard_error}


def _figure(estimates, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, 1, figsize=(10, 9), layout="constrained")
    titles = ("Adjusted post-event matched-image cosine", "Adjusted pre-event matched-image cosine",
              "Same-person post minus pre (800-ms translated bins)")
    for axis, key, title in zip(axes, CURVES, titles):
        x = np.arange(10, 800, 20) - (800 if key == "pre" else 0)
        rows = estimates[key]
        vectors = [np.array([np.nan if row[field] is None else row[field] for row in rows])
                   for field in ("mean", "lower", "upper")]
        axis.axhline(0, color=".5", linewidth=.8)
        axis.fill_between(x, vectors[1], vectors[2], color="#4b7195", alpha=.23)
        axis.plot(x, vectors[0], "o-", color="#244b72", linewidth=1, markersize=3)
        axis.set(title=title, ylabel="Cosine difference" if key == "post_minus_pre" else "Cosine",
            xlabel="Post-event acquisition-lag midpoint (ms)" if key == "post_minus_pre" else "Acquisition lag from original image onset (ms)")
        axis.set_xlim((-800, 0) if key == "pre" else (0, 800))
        unavailable = sum(row["status"] != "complete_all30" for row in rows)
        if unavailable:
            axis.text(.02, .95, f"{unavailable}/40 endpoints unavailable; no participants dropped",
                transform=axis.transAxes, va="top", fontsize=8)
    figure.suptitle("EP18: complete fixed lag curves, all 30 development participants", fontsize=13)
    figure.supxlabel("One simultaneous two-sided 95% participant-t family across all 120 endpoints (df=29).\n"
        "Acquisition lags retain the causal filter's 64-ms delay; these are not corrected neural latencies.\n"
        "Whole-sequence adjustment is descriptive; the failed original R control is unchanged.", fontsize=9)
    for extension in ("png", "pdf"):
        figure.savefig(output / ("repeated_signal_lags." + extension), dpi=160)
    plt.close(figure)


def report(*, input_root, output_dir, figure_dir):
    require(os.environ.get("SLURM_JOB_ID") and "-ln" not in socket.gethostname(),
            "saved-array reporting requires the coordinator's compute allocation")
    root, output, figures = Path(input_root), Path(output_dir), Path(figure_dir)
    require(root.resolve() == INPUT_ROOT.resolve(), "only the selected completed adjusted records may be read")
    require(output.resolve() != figures.resolve() and not output.exists() and not figures.exists(),
            "fresh distinct report and figure directories required")
    require(all((root / person / "complete.json").is_file() for person in PEOPLE),
            "all thirty original complete participant records required before reporting")
    people = {}
    recipe, configuration = None, None
    for person in PEOPLE:
        row = _read(root / person / "complete.json")
        people[person] = _participant(row, person)
        if recipe is None:
            recipe, configuration = row["projection_recipe"], row["configuration"]
        require(row["projection_recipe"] == recipe and row["configuration"] == configuration,
                "one unchanged source recipe/configuration across all thirty people required")
    critical = float(t.ppf(1 - ALPHA / (2 * FAMILY_SIZE), df=29))
    estimates = {key: [_interval([people[person][key][index] for person in PEOPLE], critical)
                       for index in range(40)] for key in CURVES}
    above = {key: [row["status"] == "complete_all30" and row["lower"] > 0 for row in estimates[key]]
             for key in CURVES}
    joint_count = sum(first and second for first, second in zip(above["post"], above["post_minus_pre"]))
    unavailable = sum(row["status"] != "complete_all30" for rows in estimates.values() for row in rows)
    record = {"status": "complete_30_person_fixed_adjusted_lag_report", "participants": list(PEOPLE),
        "participant_count": 30, "inference_unit": "participant", "degrees_of_freedom": 29,
        "source_root": str(root), "source_choice": "fixed_degree3_adjusted_matched_cosine_only",
        "post_bin_edges_ms": list(range(0, 801, 20)), "pre_bin_edges_ms": list(range(-800, 1, 20)),
        "pairing": "post[j] minus pre[j] within person; bins are translated by800ms, not reflected about zero",
        "lag_time_axis": "unchanged acquisition timestamps with64ms causal FIR delay, not corrected neural latency",
        "boundary_overlap": "an observed0ms sample belongs to both windows; final bins include their right endpoint",
        "participant_curves": people, "estimates": estimates,
        "interval_method": "two_sided_95_percent_Bonferroni_participant_t_across120_fixed_endpoints",
        "family_size": FAMILY_SIZE, "familywise_alpha": ALPHA, "t_critical": critical,
        "unavailable_endpoint_count": unavailable,
        "positive_lower_bound_count": {key: sum(value) for key, value in above.items()},
        "joint_positive_post_and_paired_bin_count": joint_count,
        "interval_role": "descriptive_simultaneous_intervals_after_pooled_diagnostic_not_confirmation",
        "normalization": "Each saved lag cosine uses its own half-template energies. The earlier pooled cosine uses lag-averaged covariance/energies and is not the mean of these cosines.",
        "interpretation": "Joint positive post and paired lower bounds support time-localized repeated-image structure descriptively; unresolved endpoints do not establish signal absence. Neighboring RSVP responses and whole-sequence adjustment prevent uniquely causal attribution. No peak, window, channel or participant is selected.",
        "EEG_amplitudes_read": False, "model_refit": False, "preprocessing_changed": False,
        "original_R_changed": False, "original_control_threshold_changed": False,
        "primary_result_rescued": False, "concept_signal_absence_inferred": False,
        "audit_access": False, "concept_or_false_scores_computed": False, "new_pass_rule": None,
        "job_id": os.environ["SLURM_JOB_ID"]}
    summary = ("All30 development participants: fixed degree-0–3 adjusted matched-image cosines,40 post bins,40 pre bins and40 within-person800ms-translated contrasts; one simultaneous two-sided95% Bonferroni participant-t family (120 endpoints,df29).\n"
        f"Lower bounds above zero: post {sum(above['post'])}/40, pre {sum(above['pre'])}/40, paired {sum(above['post_minus_pre'])}/40; joint positive post and paired {joint_count}/40; unavailable endpoints {unavailable}/120, with no participant dropping.\n"
        "All fixed acquisition-lag curves retain the causal filter's64ms delay; no peak/window selection. This descriptive saved-array diagnostic does not change original R/G/T, clear their controls, establish signal absence or identify a causal mechanism; no raw EEG or closed concept/false/audit outcomes were opened.\n")
    output.mkdir(parents=True, mode=0o700)
    figures.mkdir(parents=True, mode=0o700)
    _figure(estimates, figures)
    (output / "summary.md").write_text(summary)
    (output / "repeated_signal_lags.complete.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--figure-dir", required=True)
    args = parser.parse_args()
    report(input_root=args.input_root, output_dir=args.output_dir, figure_dir=args.figure_dir)


if __name__ == "__main__":
    main()
