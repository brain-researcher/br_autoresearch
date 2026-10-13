#!/usr/bin/env python3
"""Cached EP14 site/age attribution; no fitting, filtering, or new inference.

Run only in the coordinator's Slurm allocation after the fitted diagnostic.
All external participants, appearances, domains and ages remain visible.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket

import numpy as np
from threadpoolctl import threadpool_limits

OUTPUTS = Path(__file__).resolve().parents[1]
SOURCE = OUTPUTS / "runs/discarded_direction_diagnostics/attempt001"
ORIGINAL = OUTPUTS / "runs/fixed_development_comparison/attempt001"
DESTINATION = OUTPUTS / "runs/projection_heterogeneity/attempt001"
ATLASES = ("desikan", "destrieux")
MODELS = ("raw", "retained", "rescaled", "restored")
PAIRS = (("retained", "raw"), ("rescaled", "retained"),
         ("rescaled", "raw"), ("restored", "raw"))
AGE_BINS = tuple(f"{start}-{start + 10}" for start in range(0, 100, 10)) + ("outside_0_100",)


def read_json(path):
    with path.open() as stream:
        return json.load(stream)


def close(actual, expected, label):
    if not np.allclose(actual, expected, atol=1e-9, rtol=1e-9):
        raise ValueError(f"{label}: {actual} != {expected}")


def age_bin(age):
    if age < 0 or age > 100:
        return "outside_0_100"
    index = 9 if age == 100 else int(age // 10)
    return AGE_BINS[index]


def participant_weights(rows):
    counts = Counter(row["id"] for row in rows)
    return np.array([1.0 / (len(counts) * counts[row["id"]]) for row in rows])


def metrics(rows, weights=None):
    if not rows:
        return None
    if weights is None:
        weights = participant_weights(rows)
    else:
        weights = np.asarray(weights, dtype=float)
        weights = weights / weights.sum()
    close(weights.sum(), 1.0, "normalized appearance weights")
    if len(weights) != len(rows) or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Invalid appearance weights")
    ages = np.array([row["age"] for row in rows])
    predicted = np.array([[row["predictions"][name] for name in MODELS] for row in rows])
    residual = predicted - ages[:, None]
    means = weights @ residual
    centered = residual - means
    model_metrics = {}
    for i, name in enumerate(MODELS):
        model_metrics[name] = {"mae": float(weights @ np.abs(residual[:, i])),
                               "mse": float(weights @ residual[:, i] ** 2),
                               "centered_mae": float(weights @ np.abs(centered[:, i])),
                               "centered_mse": float(weights @ centered[:, i] ** 2),
                               "mean_residual": float(means[i])}
        close(model_metrics[name]["mse"], means[i] ** 2 + model_metrics[name]["centered_mse"], "offset/variance identity")
    contrasts = {}
    for other, base in PAIRS:
        i, j = MODELS.index(other), MODELS.index(base)
        correction = predicted[:, i] - predicted[:, j]
        centered_correction = correction - weights @ correction
        item = {f"delta_{key}": model_metrics[other][key] - model_metrics[base][key]
                for key in ("mae", "mse", "centered_mae", "centered_mse")}
        item.update({"mean_correction": float(weights @ correction),
                     "error_correction_alignment": float(2 * weights @ (residual[:, j] * correction)),
                     "correction_energy": float(weights @ correction ** 2),
                     "centered_error_correction_alignment": float(2 * weights @ (centered[:, j] * centered_correction)),
                     "centered_correction_energy": float(weights @ centered_correction ** 2)})
        close(item["delta_mse"], item["error_correction_alignment"] + item["correction_energy"], "MSE correction identity")
        close(item["delta_centered_mse"], item["centered_error_correction_alignment"] + item["centered_correction_energy"], "centered MSE correction identity")
        contrasts[f"{other}_minus_{base}"] = item
    mean_age = float(weights @ ages)
    return {"participants": len({row["id"] for row in rows}), "appearances": len(rows),
            "age_min": float(ages.min()), "age_max": float(ages.max()),
            "age_mean": mean_age, "age_sd": float(np.sqrt(weights @ (ages - mean_age) ** 2)),
            "models": model_metrics, "contrasts": contrasts}


def macro(domain_metrics):
    if not domain_metrics:
        return None
    values = list(domain_metrics.values())
    return {"domain_count": len(values), "domains": list(domain_metrics),
            "models": {name: {key: float(np.mean([v["models"][name][key] for v in values]))
                              for key in values[0]["models"][name]} for name in MODELS},
            "contrasts": {pair: {key: float(np.mean([v["contrasts"][pair][key] for v in values]))
                                  for key in values[0]["contrasts"][pair]}
                          for pair in values[0]["contrasts"]}}


def common_support_analysis(domain_rows, full_domain_metrics):
    domains, supported, conditional, covered_full = {}, {}, {}, {}
    for domain, rows in domain_rows.items():
        full_weights = participant_weights(rows)
        mask = np.array([row["on_common_support"] for row in rows], dtype=bool)
        selected = [row for row, keep in zip(rows, mask) if keep]
        complement = [row for row, keep in zip(rows, mask) if not keep]
        mass = float(full_weights[mask].sum())
        supported_balanced = metrics(selected)
        supported_original = metrics(selected, full_weights[mask]) if selected else None
        complement_original = metrics(complement, full_weights[~mask]) if complement else None
        item = {"supported_original_weight_mass": mass,
                "all_rows": full_domain_metrics[domain],
                "supported_rebalanced_participants": supported_balanced,
                "supported_conditional_original_weights": supported_original,
                "complement_conditional_original_weights": complement_original}
        for model in MODELS:
            for key in ("mae", "mse"):
                reconstructed = (mass * supported_original["models"][model][key] if selected else 0.0)
                reconstructed += ((1 - mass) * complement_original["models"][model][key] if complement else 0.0)
                close(reconstructed, full_domain_metrics[domain]["models"][model][key], "supported/complement additive-loss reconstruction")
        domains[domain] = item
        if selected:
            supported[domain] = supported_balanced
            conditional[domain] = supported_original
            covered_full[domain] = full_domain_metrics[domain]
    full_macro = macro(full_domain_metrics)
    matched_full = macro(covered_full)
    common_macro = macro(supported)
    conditional_macro = macro(conditional)
    decomposition = {}
    if common_macro:
        for pair in common_macro["contrasts"]:
            decomposition[pair] = {}
            for key in ("delta_mae", "delta_mse"):
                full = full_macro["contrasts"][pair][key]
                matched = matched_full["contrasts"][pair][key]
                selected_original = conditional_macro["contrasts"][pair][key]
                selected_rebalanced = common_macro["contrasts"][pair][key]
                terms = {"common_minus_all": selected_rebalanced - full,
                         "domain_reweighting": matched - full,
                         "within_domain_restriction_original_weights": selected_original - matched,
                         "appearance_reweighting": selected_rebalanced - selected_original}
                close(terms["common_minus_all"], terms["domain_reweighting"] + terms["within_domain_restriction_original_weights"] + terms["appearance_reweighting"], "common/all exact additive-loss contrast difference")
                decomposition[pair][key] = terms
    return {"status": "descriptive" if common_macro else "no_supported_appearances",
            "covered_domains": list(covered_full), "uncovered_domains": [d for d in domain_rows if d not in covered_full],
            "all_rows_all_domains": full_macro, "all_rows_covered_domains": matched_full,
            "supported_rebalanced_covered_domains": common_macro,
            "supported_original_weights_covered_domains": conditional_macro,
            "domains": domains, "additive_loss_contrast_difference_decomposition": decomposition,
            "centered_metric_note": "Each population is recentered independently; no additive subgroup decomposition is claimed for centered losses.",
            "no_eligibility_thresholds": True}


def analyze_rows(rows):
    by_domain = defaultdict(list)
    for row in rows:
        by_domain[row["domain"]].append(row)
    by_domain = dict(sorted(by_domain.items()))
    domain_metrics = {domain: metrics(appearances) for domain, appearances in by_domain.items()}
    full_macro = macro(domain_metrics)
    lodo = {}
    for omitted in by_domain:
        # Recompute every retained domain's weighted center from its rows;
        # another domain's exclusion must not change those within-domain fits.
        remaining = {domain: metrics(appearances) for domain, appearances in by_domain.items() if domain != omitted}
        recomputed = macro(remaining)
        influences = {}
        for pair, value in full_macro["contrasts"].items():
            influences[pair] = {}
            for key in ("delta_mae", "delta_mse", "delta_centered_mae", "delta_centered_mse"):
                removed_value = recomputed["contrasts"][pair][key] if recomputed else None
                if removed_value is not None:
                    expected = (len(by_domain) * value[key] - domain_metrics[omitted]["contrasts"][pair][key]) / (len(by_domain) - 1)
                    close(removed_value, expected, "leave-one-domain-out macro reconstruction")
                influences[pair][key] = {"omitted_domain_effect": domain_metrics[omitted]["contrasts"][pair][key],
                                         "contribution_to_full_macro": domain_metrics[omitted]["contrasts"][pair][key] / len(by_domain),
                                         "leave_one_domain_out": removed_value,
                                         "shift_after_omission": removed_value - value[key] if removed_value is not None else None}
        lodo[omitted] = {"recomputed_macro": recomputed, "influence": influences}
    age_panels = {}
    for label in AGE_BINS:
        cells = {}
        for domain, appearances in by_domain.items():
            selected = [row for row in appearances if age_bin(row["age"]) == label]
            cell = metrics(selected)
            cells[domain] = {"status": "descriptive" if cell else "empty",
                             "participants": cell["participants"] if cell else 0,
                             "appearances": len(selected), "metrics": cell,
                             "fraction_of_domain_participants": (cell["participants"] / domain_metrics[domain]["participants"]) if cell else 0.0}
        age_panels[label] = {"domains": cells,
                             "equal_domain_macro_over_nonempty_cells": macro({d: c["metrics"] for d, c in cells.items() if c["metrics"]}),
                             "warning": "Age-bin macros can involve different domain sets; do not interpret their differences as longitudinal aging effects."}
    for domain, full in domain_metrics.items():
        for model in MODELS:
            for key in ("mae", "mse"):
                reconstructed = sum(panel["domains"][domain]["fraction_of_domain_participants"] * panel["domains"][domain]["metrics"]["models"][model][key]
                                    for panel in age_panels.values() if panel["domains"][domain]["metrics"])
                close(full["models"][model][key], reconstructed, "fixed-age-bin additive loss reconstruction")
    return {"participants": len({row["id"] for row in rows}), "appearances": len(rows),
            "domains": domain_metrics, "domain_macro": full_macro, "leave_one_domain_out": lodo,
            "age_bins": age_panels, "common_support": common_support_analysis(by_domain, domain_metrics)}


def load_rows(atlas):
    rows, seen, identities = [], set(), {}
    for view in range(3):
        record = read_json(SOURCE / "outer" / f"{atlas}_view{view}.json")
        if (record["atlas"], record["view"]) != (atlas, view):
            raise ValueError("Unexpected cached identity")
        selected = [r for r in record["appearances"] if r["role"] == "external"]
        if len(selected) != (316, 318, 320)[view]:
            raise ValueError("Incomplete cached external role")
        for row in selected:
            key = (row["id"], row["view"])
            if key in seen or row["view"] != view:
                raise ValueError("Duplicate or mismatched appearance")
            seen.add(key)
            identity = (row["domain"], row["age"])
            if identities.setdefault(row["id"], identity) != identity:
                raise ValueError("Participant age/domain changes between appearances")
            predictions = {name: row["predictions"][f"geometry_{name}_alpha1000"] for name in MODELS}
            if not all(np.isfinite(value) for value in (row["age"], *predictions.values())):
                raise ValueError("Nonfinite cached value")
            rows.append({"id": row["id"], "domain": row["domain"], "view": view,
                         "age": row["age"], "on_common_support": row["on_common_support"],
                         "predictions": predictions})
    if len({row["domain"] for row in rows}) != 13:
        raise ValueError("Expected all13 original external domains")
    return rows


def original_context(atlas, current):
    old = (read_json(ORIGINAL / "development_report.json")["results"] if atlas == "desikan"
           else read_json(ORIGINAL / "controls_report.json")["destrieux_sensitivity"])
    pairs = (("R0", "raw"), ("R3", "rescaled"))
    return {"note": "Original frozen summaries are context, not replacement predictions. A prior strict replay failure and its numerical diagnosis remain preserved in the fitted diagnostic run.",
            "domain_macro_mae": {old_arm: {"original": old[old_arm]["external_age"]["domain_macro"],
                                            "current_alpha1000": current["domain_macro"]["models"][new]["mae"],
                                            "difference": current["domain_macro"]["models"][new]["mae"] - old[old_arm]["external_age"]["domain_macro"]}
                                  for old_arm, new in pairs},
            "per_domain_original_mae": {old_arm: old[old_arm]["external_age"]["domains"] for old_arm, _ in pairs}}


def markdown(report):
    lines = ["# EP14 cached projection heterogeneity", "", "External development predictions only; alpha1000 fixed. All13 domains, all ages and every appearance are retained. No new model, support threshold, significance test or causal claim.", "",
             "Each participant has equal total weight within its domain, divided among appearances; domain means are then equally weighted. Centering is recomputed per model/domain for each population. Predictions are never averaged across views.", "",
             "For correction c = prediction_other − prediction_base and baseline residual e = prediction_base − age, the exact identity is ΔMSE = 2 E[e c] + E[c²]. The first term describes error/correction alignment; the second is correction energy. Centered MSE uses centered e and c. MAE has no such decomposition.", ""]
    for atlas in ATLASES:
        result = report["atlases"][atlas]
        lines += [f"## {atlas.capitalize()}", "", "| Comparison | Δ MAE, years | Δ centered MAE, years | Δ MSE, years² | Alignment, years² | Correction energy, years² |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for pair, item in result["domain_macro"]["contrasts"].items():
            lines.append(f"| {pair} | {item['delta_mae']:.6f} | {item['delta_centered_mae']:.6f} | {item['delta_mse']:.6f} | {item['error_correction_alignment']:.6f} | {item['correction_energy']:.6f} |")
        lines += ["", "| Omitted domain | Participants | Domain Δ MAE | Remaining-domain Δ MAE | Domain Δ centered MAE | Remaining-domain Δ centered MAE |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for domain, item in result["leave_one_domain_out"].items():
            influence = item["influence"]["rescaled_minus_raw"]
            lines.append(f"| {domain} | {result['domains'][domain]['participants']} | {influence['delta_mae']['omitted_domain_effect']:.6f} | {influence['delta_mae']['leave_one_domain_out']:.6f} | {influence['delta_centered_mae']['omitted_domain_effect']:.6f} | {influence['delta_centered_mae']['leave_one_domain_out']:.6f} |")
        support = result["common_support"]
        lines += ["", "### Common support and changed weights", "",
                  f"Covered domains: {', '.join(support['covered_domains']) or 'none'}. Uncovered domains remain visible: {', '.join(support['uncovered_domains']) or 'none'}.", "",
                  "Supported/complement mixtures use conditional original appearance weights. Actual common-support scores rebalance participants and their eligible appearances. The exact difference below retains a separate appearance-reweighting term; it does not equate these two weighting schemes.", "",
                  "| Contrast/metric | Common − all | Domain reweighting | Within-domain restriction | Appearance reweighting |", "| --- | ---: | ---: | ---: | ---: |"]
        for pair, values in support["additive_loss_contrast_difference_decomposition"].items():
            for key, item in values.items():
                lines.append(f"| {pair}/{key} | {item['common_minus_all']:.6f} | {item['domain_reweighting']:.6f} | {item['within_domain_restriction_original_weights']:.6f} | {item['appearance_reweighting']:.6f} |")
        lines += ["", "The all-row and supported summaries on identical covered domains are included in JSON for absolute and independently recentered metrics. No additive support-mixture interpretation is assigned to recentered losses. No new common-support eligibility threshold is imposed.", ""]
    lines += ["Fixed age bins are [0,10), …, [90,100], including100 in the last bin. Out-of-range ages receive a separate visible category. Every domain/bin cell, unique-participant count and appearance count is retained in JSON; empty cells are marked. Age-bin macros can have different contributing domain sets, and sparse cells are descriptive only.", "",
              "The original frozen R0/R3 summaries are retained as context. Previously diagnosed cross-run numerical drift is not relabeled as a passed strict replay. The final holdout remains closed, and this report does not change candidate selection or identify a causal mechanism.", ""]
    return "\n".join(lines)


def plot(report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    domains = list(report["atlases"]["desikan"]["domains"])
    labels = list(AGE_BINS[:10])
    if any(report["atlases"][atlas]["age_bins"][AGE_BINS[-1]]["equal_domain_macro_over_nonempty_cells"] for atlas in ATLASES):
        labels.append(AGE_BINS[-1])
    matrices, counts = {}, {}
    for atlas in ATLASES:
        matrices[atlas] = np.full((len(domains), len(labels)), np.nan)
        counts[atlas] = np.zeros((len(domains), len(labels)), dtype=int)
        for i, domain in enumerate(domains):
            for j, label in enumerate(labels):
                cell = report["atlases"][atlas]["age_bins"][label]["domains"][domain]
                counts[atlas][i, j] = cell["participants"]
                if cell["metrics"]:
                    matrices[atlas][i, j] = cell["metrics"]["contrasts"]["rescaled_minus_raw"]["delta_mae"]
    maximum = max(float(np.nanmax(np.abs(matrix))) for matrix in matrices.values())
    maximum = max(maximum, 1e-9)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(13, 7), sharey=True)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#ECEFF1")
    norm = TwoSlopeNorm(vmin=-maximum, vcenter=0, vmax=maximum)
    for ax, atlas in zip(axes, ATLASES):
        displayed = ax.imshow(matrices[atlas], cmap=cmap, norm=norm, aspect="auto")
        ax.set_title(atlas.capitalize(), loc="left", fontweight="bold")
        ax.set_xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(domains)), domains)
        ax.set_xlabel("Chronological age bin (years)")
        ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(domains), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.6)
        ax.tick_params(which="both", length=0)
        for i, j in np.argwhere(counts[atlas] > 0):
            color = "white" if abs(matrices[atlas][i, j]) > 0.55 * maximum else "#18212B"
            ax.text(j, i, str(counts[atlas][i, j]), ha="center", va="center", fontsize=8, color=color)
    axes[0].set_ylabel("Acquisition domain")
    fig.suptitle("Where does site-direction removal change age-prediction error?", fontsize=15, fontweight="bold", y=0.985)
    fig.subplots_adjust(left=0.07, right=0.88, top=0.90, bottom=0.23, wspace=0.12)
    color_ax = fig.add_axes([0.91, 0.30, 0.018, 0.48])
    fig.colorbar(displayed, cax=color_ax, label="Rescaled retained − raw MAE (years)")
    fig.text(0.07, 0.045, "Color: paired change in age error at fixed α=1000; red = worse, blue = better. Numbers: unique participants. Gray: empty cell.\n"
             "All external domains/ages retained; participant-balanced appearance losses. Small cells are descriptive, with no uncertainty claim.\n"
             "[90,100] includes age100. Atlases use the same participants; these are not independent replications or causal effects.", fontsize=9, linespacing=1.5)
    directory = OUTPUTS / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for extension in ("png", "svg"):
        path = directory / f"projection_heterogeneity.{extension}"
        fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
        paths.append(str(path))
    plt.close(fig)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DESTINATION)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise RuntimeError("Run only in the coordinator's actual Slurm allocation")
    out = args.out.resolve()
    if not out.is_relative_to(OUTPUTS / "runs/projection_heterogeneity") or out.exists():
        raise ValueError("Use a fresh attempt under outputs/runs/projection_heterogeneity")
    if not (SOURCE / "diagnostics_report.json").exists():
        raise ValueError("Require the completed diagnostic report")
    report = {"analysis_id": "projection_heterogeneity", "evidence": "exploratory_cached_development_description",
              "created_utc": datetime.now(timezone.utc).isoformat(), "job_id": os.environ["SLURM_JOB_ID"],
              "hostname": socket.gethostname(), "source_run": str(SOURCE), "original_run": str(ORIGINAL),
              "fitted_models": False, "holdout_access": False, "new_eligibility_thresholds": False,
              "intervals_or_significance_tests": False, "atlases": {}}
    signatures = []
    with threadpool_limits(limits=1):
        for atlas in ATLASES:
            rows = load_rows(atlas)
            signatures.append({(r["id"], r["view"]): (r["age"], r["domain"], r["on_common_support"]) for r in rows})
            if signatures[-1] != signatures[0]:
                raise ValueError("Atlas participant/domain/age/support mismatch")
            result = analyze_rows(rows)
            result["original_fixed_context"] = original_context(atlas, result)
            result["per_view"] = {str(view): analyze_rows([r for r in rows if r["view"] == view]) for view in range(3)}
            result["raw_restored_max_prediction_difference_years"] = max(abs(r["predictions"]["raw"] - r["predictions"]["restored"]) for r in rows)
            report["atlases"][atlas] = result
    serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
    body = markdown(report)
    out.mkdir(parents=True, exist_ok=False)
    (out / "heterogeneity_report.json").write_text(serialized)
    (out / "heterogeneity_report.md").write_text(body)
    figures = [] if args.no_plot else plot(report)
    print(json.dumps({"complete": True, "output": str(out), "figures": figures,
                      "checks": "paired support; correction identities; leave-one-domain-out reconstruction; age/support loss mixtures and appearance reweighting"}), flush=True)


if __name__ == "__main__":
    main()
