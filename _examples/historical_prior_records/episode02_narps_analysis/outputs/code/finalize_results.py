#!/usr/bin/env python3
"""Write RESULT.md and CandidateBundleV1 from frozen decision outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[2]
OUT = WORKSPACE / "outputs"
CONTRASTS = ("gain_demean", "loss_demean")


def read_tsv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def f(value: str) -> float:
    return float(value)


def main() -> None:
    decision = json.loads((OUT / "decision.json").read_text())
    mechanism = read_tsv(OUT / "mechanism_results.tsv")
    robustness = read_tsv(OUT / "robustness_results.tsv")
    threshold = read_tsv(OUT / "threshold_stability.tsv")
    point = {
        (row["contrast"], row["scope"], row["strategy"], row["region"]): row
        for row in mechanism
    }
    boot = {
        (row["contrast"], row["strategy"], row["metric"]): row
        for row in robustness if row["analysis"] == "paired_cohort_stratified_bootstrap"
    }
    thresh = {(row["contrast"], row["scope"]): row for row in threshold}

    table_rows = []
    for contrast in CONTRASTS:
        primary = point[(contrast, "full", "ivw", "common")]
        scale_ci = boot[(contrast, "ivw", "scale_attenuation")]
        r_ci = boot[(contrast, "ivw", "pearson_r")]
        edge_ci = boot[(contrast, "ivw", "edge_enrichment")]
        core_ci = boot[(contrast, "ivw", "core_ratio")]
        equal = point[(contrast, "full", "equal", "common")]
        stabilized = point[(contrast, "full", "stabilized", "common")]
        table_rows.append(
            "| {contrast} | {d:.4f} | {att:.4f} [{attlo:.4f}, {atthi:.4f}] | "
            "{r:.4f} [{rlo:.4f}, {rhi:.4f}] | {edge:.3f} [{edgelo:.3f}, {edgehi:.3f}] | "
            "{core:.3f} [{corelo:.3f}, {corehi:.3f}] | {equal:.3f} | {stab:.3f} |".format(
                contrast=contrast,
                d=f(primary["d_raw"]),
                att=f(primary["scale_attenuation"]),
                attlo=f(scale_ci["lower_95"]),
                atthi=f(scale_ci["upper_95"]),
                r=f(primary["pearson_r"]),
                rlo=f(r_ci["lower_95"]),
                rhi=f(r_ci["upper_95"]),
                edge=f(primary["edge_enrichment"]),
                edgelo=f(edge_ci["lower_95"]),
                edgehi=f(edge_ci["upper_95"]),
                core=f(primary["core_ratio"]),
                corelo=f(core_ci["lower_95"]),
                corehi=f(core_ci["upper_95"]),
                equal=f(equal["aggregation_attenuation"]),
                stab=f(stabilized["aggregation_attenuation"]),
            )
        )

    threshold_rows = []
    for contrast in CONTRASTS:
        row = thresh[(contrast, "full")]
        threshold_rows.append(
            f"| {contrast} | {row['s0_rejected']} | {row['s8_rejected']} | "
            f"{f(row['fdr_jaccard']):.4f} | {row['matched_k']} | "
            f"{f(row['top_abs_t_jaccard']):.4f} |"
        )

    status = decision["terminal_status"]
    candidate = decision["selected_candidate"]
    if status == "candidate_ready":
        conclusion = (
            f"The frozen hierarchy selected `{candidate}` as one reproducible exploratory "
            "methods candidate. Here, substantive means a material spatial sensitivity to "
            "the analysis choice, not a biological mechanism. This is an internal ds001734 "
            "result, not independent confirmation."
        )
    else:
        conclusion = (
            "No mechanism met its complete frozen support and robustness rule. The bounded "
            "episode therefore closes without a candidate; exact partial patterns remain "
            "methodologically informative but cannot be promoted by widening the search."
        )

    gain = point[("gain_demean", "full", "ivw", "common")]
    loss = point[("loss_demean", "full", "ivw", "common")]
    gain_equal = point[("gain_demean", "full", "equal", "common")]
    loss_equal = point[("loss_demean", "full", "equal", "common")]
    gain_threshold = thresh[("gain_demean", "full")]
    loss_threshold = thresh[("loss_demean", "full")]
    mechanistic_interpretation = (
        "This successor did not rerun or re-estimate the v1 89.5% variance share; it "
        "tested why smoothing could dominate it. The fixed S0-versus-S8 normalized "
        f"differences were {f(gain['d_raw']):.3f} for gain and {f(loss['d_raw']):.3f} "
        f"for loss, while affine scale/offset alignment removed only "
        f"{100*f(gain['scale_attenuation']):.2f}% and "
        f"{100*f(loss['scale_attenuation']):.2f}% of those differences; spatial "
        f"correlations were only {f(gain['pearson_r']):.3f} and "
        f"{f(loss['pearson_r']):.3f}. Periphery change was enriched by about "
        f"{f(gain['edge_enrichment']):.2f}x/{f(loss['edge_enrichment']):.2f}x, but "
        f"core/full ratios were {f(gain['core_ratio']):.3f}/"
        f"{f(loss['core_ratio']):.3f}, and equal-run aggregation attenuated the "
        f"difference by {100*f(gain_equal['aggregation_attenuation']):.2f}%/"
        f"{100*f(loss_equal['aggregation_attenuation']):.2f}% (negative means slightly "
        "larger). Thus neither the mask edge nor inverse-variance weighting explains "
        "the dominant change. Finally, matched top-|t| Jaccards were only "
        f"{f(gain_threshold['top_abs_t_jaccard']):.3f}/"
        f"{f(loss_threshold['top_abs_t_jaccard']):.3f}, so the very low FDR-set "
        "overlap was not a stable-rank threshold-crossing effect. The bounded "
        "explanation is therefore a broad analysis-induced spatial-pattern change "
        "that persists in the well-covered core, under non-extreme weighting, and "
        "within both task-version strata."
    )

    rules = "\n".join(
        f"- `{name}`: `{str(value).lower()}`" for name, value in decision["rules"].items()
    )
    result = f"""# NARPS smoothing-mechanism successor result

## Outcome

Terminal status: **`{status}`**  
Selected candidate: **`{candidate}`**

{conclusion}

## Mechanistic interpretation of the v1 smoothing dominance

{mechanistic_interpretation}

This is exploratory-only analysis in ds001734. It is not scientific acceptance,
does not establish a biological mechanism, and does not provide independent
confirmation. Gain and loss were analyzed separately.

## Frozen primary readouts

| Contrast | IVW full D_raw | Scale attenuation (95% CI) | Pearson r (95% CI) | Edge enrichment (95% CI) | Core/full D ratio (95% CI) | Equal-run attenuation | Stabilized-IVW attenuation |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(table_rows)}

Amplitude dominance required scale attenuation and its lower bound at least 0.50,
Pearson r and its lower bound at least 0.80, and both point thresholds in EI and
ER for both contrasts. Edge/weighting support required edge enrichment/lower
bound at least 2.0, core ratio/upper bound no greater than 0.75, at least 25%
attenuation under a matched aggregation control, and both EI/ER stratum rules.

## Fixed threshold/rank control

| Contrast | S0 BH-FDR voxels | S8 BH-FDR voxels | FDR Jaccard | matched k | top-|t| Jaccard |
|---|---:|---:|---:|---:|---:|
{chr(10).join(threshold_rows)}

BH-FDR q=0.05 was fixed and secondary. The matched-cardinality top-|t| control
distinguishes threshold crossing from wholesale rank relocation; neither was
tuned after outcome access.

## Frozen decision audit

{rules}

The deterministic EI/ER checks, split halves, 1,000 paired cohort-stratified
bootstraps (PCG64 seed 20260817), leave-one-subject-out threshold guard, and
amplitude-alignment negative control are recorded across
`outputs/mechanism_results.tsv`, `outputs/threshold_stability.tsv`, and
`outputs/robustness_results.tsv`. Within-ds001734 robustness is not independent
confirmation.

## Deviations and failed attempts

- Before neural outcome access, the first outcome-blind 100%-coverage core
  failed its own feasibility guard. The contract was transparently re-frozen as
  v2 using one erosion of the fixed common mask; no endpoint or SESOI changed.
- The first pilot wrote no checkpoint because a run lacked the optional
  `trial_type.missed` column. The implementation was repaired to preserve all
  task columns present in the unchanged FitLins design while requiring both
  target contrasts and the intercept.
- `pytest` was absent, so the six test functions were invoked directly;
  all passed, and no package was installed.
- The first completed aggregation attempt stopped at the frozen affine negative
  control because the algebraic `1-r^2` residual lost precision near `r=1`.
  Direct OLS-residual evaluation fixed the numerical implementation; the
  unchanged negative control then passed at residual-RMS ratios below `6e-16`.

## Reproduction

From the workspace root on Sherlock:

```bash
module load python/3.12.1
/home/users/zijiao/pcvenv/bin/python outputs/code/inventory_sources.py
/home/users/zijiao/pcvenv/bin/python outputs/code/build_masks.py
mkdir -p /scratch/users/zijiao/episode02_narps_analysis/{{logs,checkpoints}}
sbatch outputs/code/fit_subjects.sbatch
# After exactly 108 valid checkpoints:
sbatch outputs/code/aggregate_results.sbatch
/home/users/zijiao/pcvenv/bin/python outputs/code/finalize_results.py
```

The frozen v2 contract SHA-256 is
`27a58a09ee043e71b7d1fca93de0ef727b79fc6ecbd9fef2eaf800208317829f`.
High-frequency checkpoints remain on scratch; durable summaries and code are in
`outputs/`.
"""
    (OUT / "RESULT.md").write_text(result, encoding="utf-8")

    artifacts = [
        "outputs/source_inventory.md",
        "outputs/source_inventory.json",
        "outputs/source_inventory_runs.tsv",
        "outputs/mechanism_selection.md",
        "outputs/frozen_successor_contract.md",
        "outputs/frozen_successor_contract.json",
        "outputs/mask_summary.json",
        "outputs/common_analysis_mask.nii.gz",
        "outputs/core_analysis_mask.nii.gz",
        "outputs/periphery_analysis_mask.nii.gz",
        "outputs/analysis_manifest.tsv",
        "outputs/mechanism_results.tsv",
        "outputs/threshold_stability.tsv",
        "outputs/robustness_results.tsv",
        "outputs/decision.json",
        "outputs/figures/mechanism_summary.png",
        "outputs/RESULT.md",
        "outputs/code/inventory_sources.py",
        "outputs/code/build_masks.py",
        "outputs/code/fit_subject.py",
        "outputs/code/fit_subjects.sbatch",
        "outputs/code/aggregate_results.py",
        "outputs/code/aggregate_results.sbatch",
        "outputs/code/finalize_results.py",
        "outputs/tests/test_pipeline_math.py",
        "outputs/experiment_log.md"
    ]
    hypotheses = {
        "edge_weighting_mechanism": (
            "The large smoothing contribution is materially amplified by mask-periphery change and extremely small run variances under inverse-variance fixed effects."
        ),
        "amplitude_offset_mechanism": (
            "The large smoothing contribution is dominated by removable map amplitude or offset rather than spatial pattern change."
        ),
        "residual_substantive_spatial_sensitivity": (
            "The large smoothing contribution persists as spatial-pattern sensitivity after scale alignment, core restriction, and non-extreme aggregation."
        ),
        "continuous_threshold_disconnect": (
            "Continuous map ranks remain stable across smoothing while fixed BH-FDR conclusions change sharply because of threshold crossing."
        ),
    }
    if status == "candidate_ready":
        hypothesis = hypotheses[candidate]
        prediction = f"The complete frozen rule for {candidate} passes for gain and loss, EI and ER controls, split halves, and leave-one-subject-out guard."
        falsifier = f"Any required contrast, task-version, bootstrap interval, split-half direction, negative control, or leave-one-subject-out condition in the frozen {candidate} rule fails."
        primary_test = "The two-test C1 S0-versus-S8 frozen mechanism decomposition in outputs/frozen_successor_contract.json."
        controls = [
            "EI and ER task-version strata analyzed separately as internal controls.",
            "Equal-run and 0.1st-percentile-stabilized inverse-variance aggregation.",
            "Outcome-blind core/periphery masks, deterministic split halves, leave-one-subject-out guard, and fixed BH-FDR/top-|t| comparison."
        ]
    else:
        hypothesis = prediction = falsifier = primary_test = controls = None
    bundle = {
        "schema_version": "br.autoresearch_goal_candidate_bundle.v1",
        "terminal_status": status,
        "research_question": "Why did v1 attribute approximately 89.5% of continuous gain- and loss-map variation to smoothing: amplitude/shape, mask-edge/weighting, task-version/sample composition, substantive spatial sensitivity, or a continuous-threshold disconnect?",
        "hypothesis": hypothesis,
        "prediction": prediction,
        "falsifier": falsifier,
        "primary_test": primary_test,
        "controls": controls,
        "development_summary": conclusion + " All frozen tables, controls, and decision booleans were generated from 108 source-derived subject checkpoints.",
        "reproduction": "On Sherlock, load python/3.12.1, build masks, run outputs/code/fit_subjects.sbatch, then outputs/code/aggregate_results.sbatch and outputs/code/finalize_results.py. See outputs/RESULT.md for exact commands.",
        "output_artifacts": artifacts,
        "limitations": [
            "Every result is exploratory-only within ds001734 and is not independent confirmation.",
            "EI and ER are task versions with different gain support; stratum results are associative internal controls.",
            "The analysis isolates one C1 model and two smoothing endpoints; it does not estimate a universal smoothing effect or rerun the v1 multiverse.",
            "Mask-core, 0.1st-percentile variance stabilization, and SESOI thresholds are method choices whose transportability requires fresh data.",
            "BH-FDR q=0.05 is a secondary fixed stability readout, not a primary inferential family across tests."
        ],
        "deviations": [
            "Before neural outcome access, the initial 100%-coverage core failed its feasibility guard; the contract was transparently versioned and re-frozen using one erosion of the fixed common mask without changing endpoints or SESOIs.",
            "The first aggregation submission requested 96 GB and was canceled at zero runtime when partition scheduling translated it into a 13-CPU allocation; it was resubmitted at a measured-safe 24 GB and 4 CPUs without changing analysis code or endpoints."
        ],
        "failed_attempts": [
            "pytest was unavailable; six test functions were invoked directly and passed without dependency changes.",
            "Pilot Slurm job 39400893 stopped before checkpoint creation because one run omitted optional trial_type.missed; the design-preservation implementation was repaired and the same subject retried.",
            "Aggregation job 39502592 stopped at the prespecified affine negative control because an algebraically equivalent 1-r-squared residual suffered catastrophic cancellation; direct OLS-residual evaluation passed the unchanged control on retry 39502945."
        ],
        "isolation_assurance": "client_unattested",
        "scientific_status": "exploratory_only",
        "confirmation_requirement": "independent_fresh_confirmation_required",
        "confirmation_eligible": False,
        "scientific_acceptance": False
    }
    (OUT / "candidate_bundle.json").write_text(
        json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"terminal_status": status, "selected_candidate": candidate}, indent=2))


if __name__ == "__main__":
    main()
