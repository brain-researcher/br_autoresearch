"""All-person report of one frozen repeated-image selector sensitivity.

Average the two directions within each person, then describe original R,
alternative R and their paired difference in one three-endpoint interval
family. No model fitting, selection, EEG access or replacement control gate.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket

import numpy as np
from scipy.stats import t

from report_repeated_selector_sensitivity import (
    PEOPLE, DIRECTIONS, RIDGE_GRID, RECIPE_PATH, INPUT_ROOT, _read, require)
from repeated_selector_refit import (
    MATCH_RTOL, MATCH_ATOL, validate_selection_report, validate_completed_person)


EPISODE = Path(__file__).resolve().parents[2]
SELECTOR_REPORT = EPISODE / "outputs/runs/repeated_selector_job_47586225_step_2/report/repeated_selector_sensitivity.complete.json"
ORIGINAL_REPORT = EPISODE / "outputs/runs/repeated_image_signal_recovery/report001/repeated_image_control.complete.json"
REFIT_ROOT = EPISODE / "outputs/runs/repeated_selector_refit/pooled001"
KEYS = ("original_R", "alternative_R", "paired_alternative_minus_original")


def _numeric(value, shape):
    result = np.asarray(value)
    require(result.shape == shape and result.dtype.kind in "iuf" and np.isfinite(result).all(),
            "complete finite numeric report values required")
    return result.astype(float)


def _same_gain(first, second):
    return bool(np.isclose(1 - first, 1 - second, rtol=MATCH_RTOL, atol=MATCH_ATOL))


def _direction(row, binding, choice, parity, *, alternative):
    direction = DIRECTIONS[parity]
    support = binding["sequence_support"]
    train = [item["sequence_id"] for item in support if item["ordinal"] % 2 == parity]
    heldout = [item["sequence_id"] for item in support if item["ordinal"] % 2 != parity]
    prefix = "alternative" if alternative else "original"
    index = choice[prefix + "_selected_penalty_index"]
    require(row.get("status") == "complete_direction" and row.get("binding") == binding
        and row.get("direction") == direction
        and row.get("training_sequence_ids") == train
        and row.get("heldout_sequence_ids") == heldout
        and row.get("inner_validations_completed") == 6
        and row.get("inner_mean_normalized_identity_losses") == choice["original_mean_normalized_identity_losses"]
        and row.get("selected_penalty_index") == index
        and row.get("selected_ridge_penalty") == RIDGE_GRID[index],
        "complete original split and frozen selected penalty required")
    whitening = row["training_whitening"]
    require(whitening.get("sequence_ids") == train and whitening.get("shrinkage_strength") == .1
        and float(_numeric(whitening["regularized_covariance_trace"], ())) > 0,
        "unchanged six-training-sequence residual whitening required")
    identity = _numeric(row["identity_sequence_losses"], (6,))
    nuisance = _numeric(row["nuisance_sequence_losses"], (6,))
    denominator = float(nuisance.mean())
    require(np.all(identity >= 0) and np.all(nuisance >= 0) and denominator > 0,
        "all six paired sequence losses and positive mean nuisance denominator required")
    gain = float(1 - identity.mean() / denominator)
    require(np.isfinite(gain) and _same_gain(gain, float(_numeric(row["gain"], ()))),
        "direction gain must preserve original ratio of equal sequence means")
    return gain


def _original(row, person, selection):
    require(row.get("schema") == "ep18_repeated_image_control_v1"
        and row.get("status") == "complete_repeated_image_control"
        and row.get("participant_id") == person and row.get("data_role") == "development"
        and row.get("binding") == selection["binding"]
        and row.get("image_identity_count") == 200 and row.get("physical_sequence_count") == 12
        and row.get("inference_unit") == "participant"
        and row.get("cohort_inference_performed") is False
        and row.get("audit_access") is False and row.get("concept_or_false_models_computed") is False,
        "same original complete development participant required")
    directions = row["directions"]
    require(len(directions) == 2 and tuple(item["direction"] for item in directions) == DIRECTIONS,
        "both chronological directions required")
    gains = [_direction(item, row["binding"], choice, parity, alternative=False)
             for parity, (item, choice) in enumerate(zip(directions, selection["directions"]))]
    gain = float(np.mean(gains))
    require(_same_gain(gain, float(_numeric(row["gain"], ()))),
        "original person gain must equally average its two directions")
    return gains


def _alternative(row, original, person, selection):
    validate_completed_person(row, selection_record=selection, original_control=original)
    require(row["participant_id"] == person, "matching refitted participant required")
    return [_direction(item, original["binding"], choice, parity, alternative=True)
            for parity, (item, choice) in enumerate(zip(row["directions"], selection["directions"]))]


def _evidence(values):
    values = _numeric(values, (30, 3))
    require(np.array_equal(values[:, 2], values[:, 1] - values[:, 0]),
        "paired differences must be formed within the same thirty people")
    critical = float(t.ppf(1 - .05 / (2 * 3), df=29))
    mean = values.mean(axis=0)
    se = values.std(axis=0, ddof=1) / np.sqrt(30)
    require(np.isfinite(mean).all() and np.isfinite(se).all(), "finite participant estimates required")
    return {"method": "descriptive_two_sided_95_percent_Bonferroni_participant_t",
        "family_size": 3, "familywise_alpha": .05, "tail_alpha": .05 / 6,
        "participant_count": 30, "degrees_of_freedom": 29, "critical_t": critical,
        "endpoints": {key: {"mean": float(mean[index]), "standard_error": float(se[index]),
            "lower": float(mean[index] - critical * se[index]),
            "upper": float(mean[index] + critical * se[index])} for index, key in enumerate(KEYS)}}


def _figure(values, evidence, root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, ScalarFormatter

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), layout="constrained")
    titles = ("Original repeated-image R", "R under the frozen alternative selector",
              "Paired alternative minus original R")
    for index, (axis, key, title) in enumerate(zip(axes, KEYS, titles)):
        endpoint = evidence["endpoints"][key]
        axis.axvline(0, color=".5", linewidth=.8)
        axis.scatter(values[:, index], np.linspace(-.12, .12, 30), s=17, alpha=.7, color="#4b7195")
        axis.errorbar(endpoint["mean"], .35,
            xerr=[[endpoint["mean"] - endpoint["lower"]], [endpoint["upper"] - endpoint["mean"]]],
            fmt="o", color="#172b4d", capsize=4)
        axis.set(title=title, yticks=[], ylim=(-.2, .65),
            xlabel="Paired gain difference" if index == 2 else "Normalized prediction gain")
        axis.xaxis.set_major_locator(MaxNLocator(5))
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-2, 2))
        axis.xaxis.set_major_formatter(formatter)
    fig.suptitle("EP18: one frozen selector sensitivity, all 30 development participants", fontsize=13)
    fig.supxlabel("Dots: people after averaging their two directions. Bars: one simultaneous descriptive 95% family (3 endpoints, df=29).\n"
        "Original R/G/T and their failed controls remain unchanged; no concept or audit release.", fontsize=9)
    for extension in ("png", "pdf"):
        fig.savefig(root / ("repeated_selector_refit." + extension), dpi=160)
    plt.close(fig)


def report(*, selector_report, original_root, refit_root, original_report, output_root, figure_root):
    require(os.environ.get("SLURM_JOB_ID") and "-ln" not in socket.gethostname(),
        "reporting requires the coordinator's compute allocation")
    selector_path, originals, refits, prior_path = map(Path, (selector_report, original_root, refit_root, original_report))
    require(selector_path.resolve() == SELECTOR_REPORT.resolve()
        and originals.resolve() == INPUT_ROOT.resolve() and refits.resolve() == REFIT_ROOT.resolve()
        and prior_path.resolve() == ORIGINAL_REPORT.resolve(),
        "only the fixed selector, original records and selected refit may be reported")
    # Validate the complete frozen vector before opening any original or new outer outcomes.
    frozen = _read(selector_path)
    selections = validate_selection_report(frozen, recipe=_read(RECIPE_PATH))
    require(tuple(selections) == PEOPLE, "complete ordered original thirty-person selection required")
    changed_people = [person for person in PEOPLE if any(row["selection_changed"] for row in selections[person]["directions"])]
    require(all((originals / person / "complete.json").is_file() for person in PEOPLE)
        and all((refits / person / "complete.json").is_file() for person in changed_people),
        "all thirty original people and every changed-person completion required")
    prior = _read(prior_path)
    require(prior.get("status") == "complete_30_person_repeated_image_control_report"
        and tuple(prior["participants"]) == PEOPLE and prior.get("participant_count") == 30
        and set(prior["participant_G_T_R"]) == set(PEOPLE)
        and prior.get("audit_access") is False and prior.get("primary_result_rescued") is False,
        "same released complete original R report required")
    points, direction_details = [], {}
    for person in PEOPLE:
        selection = selections[person]
        original = _read(originals / person / "complete.json")
        old_gains = _original(original, person, selection)
        old_mean = float(np.mean(old_gains))
        released = _numeric(prior["participant_G_T_R"][person], (3,))
        require(_same_gain(old_mean, float(released[2])), "original R must match its released participant estimate")
        alternative = _read(refits / person / "complete.json") if person in changed_people else None
        new_gains = _alternative(alternative, original, person, selection) if alternative is not None else old_gains
        new_mean = float(np.mean(new_gains))
        points.append([old_mean, new_mean, new_mean - old_mean])
        direction_details[person] = [{"direction": direction, "original_gain": old_gains[index],
            "alternative_gain": new_gains[index],
            "source": "refitted_frozen_alternative" if selection["directions"][index]["selection_changed"] else "reused_original"}
            for index, direction in enumerate(DIRECTIONS)]
    values = _numeric(points, (30, 3))
    evidence = _evidence(values)
    changed = sum(row["source"] == "refitted_frozen_alternative" for rows in direction_details.values() for row in rows)
    require(changed == frozen["changed_direction_count"] and len(changed_people) == frozen["changed_participant_count"],
        "complete frozen changed/reused direction counts required")
    record = {"status": "complete_30_person_repeated_selector_refit_report", "participants": list(PEOPLE),
        "participant_count": 30, "inference_unit": "participant", "directions_per_participant": 2,
        "changed_direction_count": changed, "reused_direction_count": 60 - changed,
        "changed_participant_count": len(changed_people), "participant_endpoints": dict(zip(PEOPLE, points)),
        "endpoint_order": list(KEYS), "direction_details": direction_details, "simultaneous_family": evidence,
        "selector_report": str(selector_path), "original_report": str(prior_path), "refit_root": str(refits),
        "frozen_selection_vector": frozen["selection_vector"],
        "scientific_role": "exploratory_development_sensitivity_to_one_frozen_selector_after_original_control_failure",
        "interpretation": "The paired interval describes sensitivity to the selector. Positive alternative R is exploratory repeated-image recovery only; it cannot rescue original R/G/T, qualify concept comparisons or open audit. An unresolved interval establishes neither equivalence nor signal absence.",
        "whitening_caveat": frozen["whitening_caveat"],
        "original_R_changed": False, "original_G_T_changed": False,
        "original_control_threshold_changed": False, "primary_result_rescued": False,
        "concept_signal_absence_inferred": False, "inner_retuning_performed": False,
        "EEG_read_by_reporter": False, "audit_access": False, "concept_or_false_scores_computed": False,
        "new_pass_rule": None, "evaluation_separation": "procedural_same_account_not_technically_enforced",
        "job_id": os.environ["SLURM_JOB_ID"]}
    def describe(key):
        row = evidence["endpoints"][key]
        return f"{row['mean']:.9g} [{row['lower']:.9g}, {row['upper']:.9g}]"
    summary = (f"All 30 original development participants, averaging both directions per person: original R {describe('original_R')}; alternative R {describe('alternative_R')}.\n"
        f"Paired alternative minus original {describe('paired_alternative_minus_original')}; one descriptive two-sided 95% Bonferroni participant-t family of 3 endpoints (df29), using {changed} frozen-choice refits and {60-changed} reused directions.\n"
        "This is exploratory selector sensitivity, not a corrected primary result: original R/G/T and thresholds remain unchanged; no concept/false/audit release, equivalence or signal-absence claim follows.\n")
    output, figures = Path(output_root), Path(figure_root)
    require(output.resolve() != figures.resolve() and not output.exists() and not figures.exists(),
        "fresh distinct report and figure directories required")
    output.mkdir(parents=True, mode=0o700)
    figures.mkdir(parents=True, mode=0o700)
    _figure(values, evidence, figures)
    for directory in (output, figures):
        (directory / "3lines.md").write_text(summary)
    (output / "summary.md").write_text(summary)
    temporary = output / "repeated_selector_refit.complete.json.tmp"
    with temporary.open("x") as target:
        json.dump(record, target, indent=2, allow_nan=False)
        target.write("\n")
        target.flush()
        os.fsync(target.fileno())
    temporary.replace(output / "repeated_selector_refit.complete.json")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("selector-report", "original-root", "refit-root", "original-report", "output-root", "figure-root"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    report(**{key: value for key, value in vars(args).items()})


if __name__ == "__main__":
    main()
