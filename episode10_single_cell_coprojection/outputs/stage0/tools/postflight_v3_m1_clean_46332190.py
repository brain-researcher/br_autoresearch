#!/usr/bin/env python3
"""Independent artifact/numerical/seal postflight; no model fitting or M0 reader.

Checks the frozen current producer only. Uses the existing selective membership
parser solely to preserve its field-access boundary; all numerical checks below
are independently implemented and do not invoke a fitted model or optimizer.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.util
import itertools
import csv
import json
import math
import os
from pathlib import Path
import socket
import sys

import numpy as np
import yaml


JOB = "46332190"
GROUPS = {"212074": 6, "220187": 35, "220188": 25, "220190": 16,
          "221091": 33, "221225": 39, "221418": 12, "221514": 5,
          "221539": 1, "221588": 7, "221751": 13, "233281": 2}
LAMBDAS = [1000., 100., 10., 1., .1, .01, .001, .0001, .00001, .000001, .0000001, .00000001]
HISTORY = "true_job_45760290_post_M1_mechanical_comparison_only"
GATE = "pass_job_46332190_artifact_numerical_and_access_seal_audits_194_predictions_12_groups_91_checkpoints_949_fits_boundary_0_M0_SWC_final_unassigned_sealed"
STAGE0 = Path(__file__).resolve().parent.parent


def require(condition, reason):
    if not condition:
        raise RuntimeError(reason)


def close(a, b, reason, tolerance=1e-12):
    require(math.isfinite(float(a)) and math.isfinite(float(b)) and
            math.isclose(float(a), float(b), rel_tol=0., abs_tol=tolerance), reason)


def load(path):
    require(path.is_file() and not path.is_symlink(), f"missing/nonordinary {path.name}")
    return json.loads(path.read_text())


def one_se(table, chosen):
    grid = table["grid"]
    require([x["lambda"] for x in grid] == LAMBDAS, "one-SE lambda grid changed")
    require(all(math.isfinite(x["mean_group_nll_nats"]) and
                math.isfinite(x["group_standard_error_nats"]) and
                x["group_standard_error_nats"] >= 0 for x in grid), "invalid one-SE table")
    best = min(grid, key=lambda x: (x["mean_group_nll_nats"], -x["lambda"]))
    threshold = best["mean_group_nll_nats"] + best["group_standard_error_nats"]
    selected = max(x["lambda"] for x in grid if x["mean_group_nll_nats"] <= threshold)
    require(selected == chosen == table["selected_largest_qualifying_lambda"], "one-SE decision mismatch")
    require(best["lambda"] == table["minimum_mean_lambda"], "one-SE minimum mismatch")
    close(threshold, table["one_se_threshold_nats"], "one-SE threshold mismatch")
    close(best["mean_group_nll_nats"], table["minimum_mean_nll_nats"], "one-SE minimum score mismatch")
    close(best["group_standard_error_nats"], table["standard_error_at_minimum_nats"], "one-SE SE mismatch")


def witness(objective, gradient, relative, telemetry, lam):
    require(math.isfinite(objective) and math.isfinite(gradient) and 0 <= gradient <= 1e-7, "failed objective/gradient witness")
    require(telemetry["status"] == "pass" and telemetry["ridge_lambda"] == lam, "telemetry identity mismatch")
    require(telemetry["post_witness_confirmation_attempt_limit"] == 1, "confirmation cap changed")
    require(0 <= telemetry["post_witness_confirmation_accepted_steps"] <= 1, "confirmation count changed")
    kind = telemetry["convergence_witness"]
    if kind in ("gradient_and_relative_objective_change", "gradient_and_repeated_unchanged_initial_objective"):
        require(math.isfinite(relative) and relative <= 1e-10, "relative objective witness failed")
    elif kind == "strong_convexity_gap_after_zoom_stall":
        certificate = telemetry["certificate"]
        require(certificate["failure_message"] in (
            "strong-Wolfe zoom iteration ceiling reached",
            "strong-Wolfe zoom bracket reached floating-point resolution"), "ineligible stall")
        for flag in ("eligible_failure", "line_search_all_values_finite", "certificate_inputs_finite",
                     "all_values_finite", "gradient_predicate_pass", "relative_gap_predicate_pass", "certificate_pass"):
            require(certificate[flag] is True, f"failed certificate {flag}")
        upward = lambda x: math.nextafter(x, math.inf)
        square = upward(gradient * gradient)
        product = upward(261 * square)
        absolute = upward(upward(product / 2.) / lam)
        relative_bound = upward(absolute / max(1., abs(objective)))
        require(relative_bound <= 1e-10, "recomputed relative gap fails")
        require(certificate["upward_rounded_relative_gap_bound"] == relative_bound, "certificate gap arithmetic differs")
        require(certificate["upward_rounded_absolute_gap_bound"] == absolute, "certificate absolute gap differs")
    else:
        raise RuntimeError("unknown solver witness")
    return kind


def audit(root):
    require(os.environ.get("SLURM_JOB_ID") and "-ln" not in socket.gethostname(), "postflight requires root-owned compute allocation")
    scratch_episode = Path(os.environ["SCRATCH"]).resolve() / "br_autoresearch/episode10_single_cell_coprojection"
    expected = scratch_episode / f"v3_m1_real_fit_single_thread_v3_job_{JOB}"
    require(root.absolute() == expected and root.is_dir() and not root.is_symlink(), "wrong current producer root")
    contract = yaml.safe_load((STAGE0 / "v3_m1_model_contract.yaml").read_text())
    panel = contract["fixed_panel"]["ordered_coordinate_ids"]
    require(len(panel) == 16, "panel length mismatch")
    summary = load(root / "v3_m1_fit_summary.json")
    manifest = load(root / "v3_m1_real_single_thread_v3_checkpoint_manifest.json")
    full = load(root / "v3_m1_full_refit.json")
    telemetry = load(root / "v3_m1_solver_telemetry.json")
    require(summary["status"] == full["status"] == telemetry["status"] == "pass", "nonpassing producer")
    require(manifest["status"] == manifest["phase"] == "complete", "nonterminal manifest")
    require(str(manifest["SLURM_job_id"]) == JOB, "foreign manifest")
    require(manifest["completed_warm_paths"] == 78 and manifest["completed_zero_refits"] == 13, "incomplete paths")
    require(summary["development_cells"] == manifest["development_cells"] == 194 and
            summary["development_groups"] == manifest["development_groups"] == 12, "population mismatch")
    require(summary["checkpoint_resume_used"] is False, "unexpected checkpoint resume")
    require(summary["outer_boundary_fold_count"] == 0 and summary["full_refit_used_as_performance_estimate"] is False, "boundary/performance misuse")
    for payload in (summary, manifest, full, telemetry):
        require(payload["M0_prediction_or_fit_artifact_opened"] is False, "new process M0 access")
        if "final_or_unassigned_or_SWC_opened" in payload:
            require(payload["final_or_unassigned_or_SWC_opened"] is False, "protected access")
    require(summary["SWC_opened"] is False and summary["final_or_unassigned_opened"] is False, "protected summary access")
    identity = manifest["execution_identity"]
    require(summary["execution_identity"] == identity, "identity drift")
    require(identity["qualification_job_id"] == "46328672" and identity["numerical_refresh_job_id"] == "46330399", "stale qualification")
    require(identity["historical_M0_artifact_access"] == HISTORY and identity["prior_membership_opening_job_id"] == "45609750", "history changed")
    require(identity["openblas_threads"] == 1 and identity["openblas_core"] == "Zen" and "7502" in identity["cpu_model"], "backend mismatch")
    require(all(str(v) == "1" for v in identity["thread_environment"].values()), "thread environment mismatch")
    require([str(x["held_out_group"]) for x in summary["outer_folds"]] == list(GROUPS), "fold identity mismatch")
    for fold in summary["outer_folds"]:
        group = str(fold["held_out_group"])
        require(fold["held_out_cells"] == GROUPS[group] and fold["training_groups"] == fold["inner_folds"] == 11, "fold support mismatch")
        require(fold["selected_lambda_is_grid_boundary"] is False and fold["selected_lambda"] not in (LAMBDAS[0], LAMBDAS[-1]), "boundary fold")
        one_se(fold["inner_selection"], fold["selected_lambda"])
    one_se(summary["full_development_selection"], summary["full_development_selected_lambda"])
    one_se(full["selection"], full["selected_lambda"])
    require(full["selected_lambda"] == summary["full_development_selected_lambda"], "full lambda mismatch")
    require(np.asarray(full["B_17_by_15"]).shape == (17, 15) and np.asarray(full["gamma_length_6"]).shape == (6,), "coefficient dimensions")
    require(all(np.all(np.isfinite(full[k])) for k in ("B_17_by_15", "gamma_length_6", "full_development_geometry_RMS_scales")), "nonfinite refit")
    require(np.all(np.asarray(full["full_development_geometry_RMS_scales"]) > 0), "invalid geometry scaling")
    checkpoint_plan = {}
    for pair in itertools.combinations(GROUPS, 2):
        checkpoint_plan[f"warm__leave_two__{pair[0]}__{pair[1]}.npz"] = ("warm", set(GROUPS) - set(pair), LAMBDAS)
    for group, fold in zip(GROUPS, summary["outer_folds"]):
        checkpoint_plan[f"warm__leave_one__{group}.npz"] = ("warm", set(GROUPS) - {group}, LAMBDAS)
        checkpoint_plan[f"zero__outer_refit__{group}.npz"] = ("zero", set(GROUPS) - {group}, [fold["selected_lambda"]])
    checkpoint_plan["zero__full_refit.npz"] = ("zero", set(GROUPS), [full["selected_lambda"]])
    checkpoint_root = root / "checkpoints"
    require(set(p.name for p in checkpoint_root.iterdir()) == set(checkpoint_plan), "checkpoint coverage mismatch")
    fits, kinds = 0, Counter()
    for filename, (kind, groups, lambdas) in checkpoint_plan.items():
        path = checkpoint_root / filename
        require(path.is_file() and not path.is_symlink(), "nonordinary checkpoint")
        with np.load(path, allow_pickle=False) as data:
            require(str(data["producer_slurm_job_id"].item()) == JOB, "foreign checkpoint producer")
            require(str(data["qualification_job_id"].item()) == "46328672" and str(data["actual_k_job_id"].item()) == "45602325", "stale checkpoint qualification")
            require(str(data["prior_membership_opening_job_id"].item()) == "45609750", "checkpoint history mismatch")
            require(data["training_groups"].tolist() == [g for g in GROUPS if g in groups], "training leakage/group mismatch")
            require(int(data["training_cells"].item()) == sum(GROUPS[g] for g in groups), "training size mismatch")
            require(data["lambdas"].tolist() == lambdas and str(data["kind"].item()) == kind, "checkpoint grid mismatch")
            require(data["theta"].shape == (len(lambdas), 261) and np.all(np.isfinite(data["theta"])), "invalid theta")
            for i, lam in enumerate(lambdas):
                detail = json.loads(str(data["telemetry_json"][i]))
                kinds[witness(float(data["objective"][i]), float(data["gradient"][i]), float(data["relative_change"][i]), detail, lam)] += 1
                require(0 <= int(data["iterations"][i]) <= 2000 and 0 <= int(data["accepted_steps"][i]) <= int(data["iterations"][i]), "optimizer cap mismatch")
                fits += 1
    require(fits == 949, "fit count mismatch")
    with (root / "v3_m1_cross_fitted_predictions.csv").open(newline="") as stream:
        predictions = list(csv.DictReader(stream))
    ids = {r["provider_neuron_id"] for r in predictions}
    require(len(ids) == len(predictions) == 194 and Counter(r["fMOST_brain_id"] for r in predictions) == GROUPS, "prediction identity/support mismatch")
    # Reuse only the existing access-preserving projection parser, not its math.
    spec = importlib.util.spec_from_file_location("ep10_postflight_membership_parser", STAGE0 / "tools/fit_v3_m1_real_v1.py")
    parser = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = parser
    spec.loader.exec_module(parser)
    target_file = scratch_episode / "gao_mop_development_targets_v3_ep10_assets/primary_16_target_calls.csv"
    require(target_file.is_file() and not target_file.is_symlink(), "missing authorized development projection")
    with target_file.open("rb") as stream:
        membership, counts = parser._scan_membership_projection(stream, ids)
    require(counts.total_rows == 200 and counts.matched_rows == 194 and counts.extra_rows == 6, "projection denominator mismatch")
    require(set(membership) == ids, "prediction membership mismatch")
    grouped = {g: [] for g in GROUPS}
    max_k_error = 0.
    for row in predictions:
        ident, group, k = row["provider_neuron_id"], row["fMOST_brain_id"], int(row["K"])
        mg, mk, detected = membership[ident]
        require(mg == group and mk == k and sum(detected) == k, "K/group/target mismatch")
        marginals = np.asarray([float(row[f"exact_inclusion_marginal__{t}"]) for t in panel])
        require(np.all(np.isfinite(marginals)) and np.all(marginals >= -1e-12) and np.all(marginals <= 1+1e-12), "invalid marginals")
        k_error = abs(math.fsum(marginals) - k)
        max_k_error = max(max_k_error, k_error)
        require(k_error <= 1e-10, "marginal sum-to-K mismatch")
        score = float(row["M1_outer_cross_fitted_log2_probability"])
        uniform = -math.log2(math.comb(16, k))
        require(math.isfinite(score) and score <= 1e-12, "invalid log probability")
        close(float(row["uniform_fixed_K_log2_probability"]), uniform, "uniform identity mismatch")
        close(float(row["M1_minus_uniform_gain_bits"]), score-uniform, "gain identity mismatch")
        if k in (0,16):
            require(score == uniform == 0., "nonzero degenerate score")
        grouped[group].append((score, uniform, np.asarray(detected)-marginals))
    scores = summary["scores_and_calibration"]
    group_scores, group_uniforms = [], []
    for group, rows in grouped.items():
        score = math.fsum(r[0] for r in rows)/len(rows)
        uniform = math.fsum(r[1] for r in rows)/len(rows)
        residual = np.asarray([r[2] for r in rows])
        recorded = scores["per_group"][group]
        close(recorded["mean_M1_log2_probability"], score, "group score mismatch")
        close(recorded["uniform_fixed_K_mean_log2_probability"], uniform, "group uniform mismatch")
        close(recorded["paired_M1_minus_uniform_gain_bits_per_cell"], score-uniform, "group gain mismatch")
        close(recorded["marginal_Brier_M1"], float(np.mean(residual**2)), "Brier mismatch")
        for target, value in zip(panel, np.mean(residual, axis=0)):
            close(recorded["observed_minus_expected_by_target"][target], value, "calibration residual mismatch")
        group_scores.append(score)
        group_uniforms.append(uniform)
    close(scores["aggregate_equal_group_mean_M1_log2_probability"], math.fsum(group_scores)/12, "aggregate score mismatch")
    close(scores["aggregate_equal_group_uniform_fixed_K_mean_log2_probability"], math.fsum(group_uniforms)/12, "aggregate uniform mismatch")
    close(scores["aggregate_equal_group_paired_M1_minus_uniform_gain_bits_per_cell"], math.fsum(a-b for a,b in zip(group_scores,group_uniforms))/12, "aggregate gain mismatch")
    return {"status": "pass", "gate": GATE, "producer_job_id": JOB,
            "audit_job_id": os.environ["SLURM_JOB_ID"], "audited_at_utc": datetime.now(timezone.utc).isoformat(),
            "development_cells": 194, "development_groups": 12, "checkpoints": 91, "fits": fits,
            "witness_counts": dict(kinds), "max_inclusion_marginal_sum_to_K_error": max_k_error,
            "one_SE_decisions_recomputed_from_saved_grid": True, "calibration_recomputed_from_selective_development_membership": True,
            "historical_M0_artifact_access": HISTORY, "new_M0_artifact_opened": False,
            "SWC_final_or_unassigned_opened": False,
            "independence": "artifact/numerical checks authored separately from fitter; original selective parser used only for permitted field projection"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--m1-root", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    report = audit(args.m1_root)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
