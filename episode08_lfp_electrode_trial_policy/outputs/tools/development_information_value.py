#!/usr/bin/env python3
"""EP08 approved development-only electrode information-value analysis."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import signal
import socket
import time

import numpy as np

from development_reader import selected_metadata, read_training_arrays, read_evaluation_arrays
from prepare_development_trial_roles import DEVELOPMENT
from electrode_information_value import (BASE_FEATURES, CONDITIONAL_FEATURE, PREFIXES,
    contexts, predictor_rows, fit_removals, removal_values, cross_session_report)


def save_json(path, value):
    pending = path.with_suffix(path.suffix+".partial")
    pending.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    pending.replace(path)


def write_summary(output, report):
    lines = ["# Exploratory conditional electrode-value prediction", "",
             "Complete eight-session development analysis. The same 40 evaluation trials per session",
             "were exposed during the earlier 41-comparison search; this is not new confirmation.", "",
             "Positive MSE reduction means that adding the conditional cross-validated gain improves",
             "prediction of subsequent fixed-decoder removal loss beyond the fixed quality/marginal",
             "association baseline. All quantities are model-specific predictive values.", "",
             "| Fitting-prefix trials | Transfer | Chewie MSE reduction | Mihili MSE reduction | Equal-animal mean |",
             "| --- | --- | ---: | ---: | ---: |"]
    for result in report["comparisons"]:
        animals = {r["animal"]: r for r in result["animals"]}
        lines.append(f"| {result['prefix_trials']} | {result['transfer']} | "
                     f"{animals['Chewie']['MSE_reduction']:.8g} | {animals['Mihili']['MSE_reduction']:.8g} | "
                     f"{result['equal_animal_MSE_reduction']:.8g} |")
    lines += ["", "The 32-trial leave-one-session-out comparison is primary. The 64-trial information state",
              "and opposite-animal transfers are prespecified sensitivities, not opportunities to select",
              "the most favorable result. Session omissions use fixed predictions and are not confidence intervals.",
              "Electrodes, contexts and sessions within two animals do not establish new-animal replication.",
              "Signal-quality covariates are recorded-feature proxies; artifact fraction, line noise,",
              "physical sensor noise and spike contamination are not separately quantified here.",
              "Four reserved sessions remain closed. No acquisition-efficiency or causal-importance claim follows.",
              "", "Full electrode/context values, predictions, per-session errors and unavailable rank correlations",
              "are retained in `information_value.json`. Historical policy-search results remain unchanged."]
    (output/"information_value.md").write_text("\n".join(lines)+"\n")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return {"status": "unavailable", "reason": "matplotlib not available; complete numerical report retained"}
    figure_dir = Path(__file__).resolve().parents[2]/"outputs/figures/offline_information_value"/output.name
    figure_dir.mkdir(parents=True, exist_ok=False)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, transfer in zip(axes, ("leave_one_session_out", "opposite_animal")):
        for prefix, offset, color in ((32, -.14, "#2768a3"), (64, .14, "#bc6933")):
            comparison = next(r for r in report["comparisons"] if r["prefix_trials"] == prefix and r["transfer"] == transfer)
            for j, animal in enumerate(("Chewie", "Mihili")):
                values = [r["MSE_reduction"] for r in comparison["sessions"] if r["animal"] == animal]
                ax.scatter(j+offset+np.linspace(-.035, .035, 4), values, color=color, alpha=.65, s=25)
                ax.scatter(j+offset, np.mean(values), color=color, marker="D", s=45, label=f"{prefix} fitting trials" if j == 0 else None)
        ax.axhline(0, color="#777777", linewidth=.8)
        ax.set_xticks([0, 1], ["Chewie", "Mihili"])
        ax.set_title(transfer.replace("_", " "))
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("Removal-value prediction MSE reduction")
    figure.suptitle("Exploratory development data; conditional gain added to quality and marginal association")
    figure.tight_layout()
    for extension in ("png", "pdf"):
        figure.savefig(figure_dir/f"information_value.{extension}", dpi=160)
    plt.close(figure)
    return {"status": "complete", "files": [str(figure_dir/f"information_value.{extension}") for extension in ("png", "pdf")]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "metadata-report", "trial-roles", "output-dir"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, default=1800)
    parser.add_argument("--max-rss-gib", type=float, default=5.)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise RuntimeError("Coordinator-owned verified compute allocation required")
    if any(os.environ.get(k) != "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")):
        raise RuntimeError("One numerical worker with explicit BLAS thread limits required")
    if not 0 < args.max_seconds <= 3600 or not 0 < args.max_rss_gib <= 5 or args.output_dir.exists():
        raise ValueError("Fresh output; at most3600seconds and5GiB RSS required")
    args.output_dir.mkdir(parents=True)
    start, cpu = time.monotonic(), time.process_time()
    def stop(*unused):
        raise TimeoutError("Selected information-value deadline reached")
    def check():
        if time.monotonic()-start >= args.max_seconds:
            stop()
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2 > args.max_rss_gib:
            raise MemoryError("Selected information-value process RSS guard reached")
    signal.signal(signal.SIGALRM, stop); signal.signal(signal.SIGTERM, stop)
    signal.alarm(args.max_seconds)
    access = {"logical_neural_bytes": 0, "fitting_trials_read": 0, "evaluation_trials_read": 0,
              "evaluation_neural_elements_read": False}
    progress = {"ridge_fits": 0}
    operational = {"analysis_id": "offline_information_value", "status": "running",
                   "started_utc": datetime.now(timezone.utc).isoformat(), "host": socket.gethostname(),
                   "slurm_job_id": os.environ["SLURM_JOB_ID"], "completed_sessions": [],
                   "progress": progress, "access": access, "held_session_accessed": False,
                   "scientific_scores_released": False, "valid_evaluations_charged": 0}
    save_json(args.output_dir/"operational.json", operational)
    try:
        metadata = json.loads(args.metadata_report.read_text())
        roles = json.loads(args.trial_roles.read_text())
        if len(roles["sessions"]) != 8 or {s["filename"] for s in roles["sessions"]} != set(DEVELOPMENT):
            raise ValueError("Exactly the original eight development roles required")
        sessions = []
        for filename in DEVELOPMENT:
            check()
            source, role, records = selected_metadata(metadata, roles, filename)
            x, y, channels, guides = read_training_arrays(args.source_root/filename, source, role, records, check, access)
            ids = role["nested_training_sets"]["96"]["training_trials"]
            directions = np.asarray([records[t]["direction_index"] for t in ids])
            context_menu = contexts(filename, channels)
            rows = []
            for prefix in PREFIXES:
                prefix_ids = role["nested_training_sets"][str(prefix)]["training_trials"]
                selected = np.asarray([ids.index(t) for t in prefix_ids])
                rows.extend(predictor_rows(x[selected], y[selected], directions[selected], channels,
                                           context_menu, check, progress))
            frozen = fit_removals(x, y, directions, context_menu, check, progress)
            # Every predictor and both decoder versions are fixed before this read.
            ex, ey, eval_channels, _ = read_evaluation_arrays(args.source_root/filename, source, role, records, check, access, guides)
            if channels != eval_channels or ex.shape != (40, 288):
                raise ValueError("Original complete evaluation support required")
            labels = np.asarray([records[t]["direction_index"] for t in role["evaluation_trials"]])
            values, context_metrics = removal_values(frozen, ex, ey, labels, channels)
            for row in rows:
                row["removal_value"] = values[(row["context"], row["electrode"])]
            sessions.append({"filename": filename, "animal": role["animal"], "rows": rows,
                             "contexts": [{**c, "physical_channels": [int(channels[i]) for i in c["indices"]]} for c in context_menu],
                             "context_metrics": context_metrics, "target_units": y.shape[1],
                             "fitting_trials": ids, "evaluation_trials": role["evaluation_trials"],
                             "predictors_and_decoders_frozen_before_evaluation_read": True})
            operational["completed_sessions"].append(filename)
            save_json(args.output_dir/"operational.json", operational)
            print(json.dumps({"session_completed": filename, "ridge_fits": progress["ridge_fits"],
                              "elapsed_seconds": time.monotonic()-start, "scientific_scores_released": False}), flush=True)
            del x, y, ex, ey, frozen
        if access["fitting_trials_read"] != 768 or access["evaluation_trials_read"] != 320:
            raise ArithmeticError("Complete original96+40 trial support in all8 sessions required")
        # Each prefix:4*(96 marginal +19 full +384 drop), plus403 outer per session.
        if progress["ridge_fits"] != 8*(2*4*(96+19+384)+403):
            raise ArithmeticError("Complete fixed fitting menu required")
        comparisons = cross_session_report(sessions, check)
        report = {"analysis_id": "offline_information_value", "status": "complete",
                  "specification": "outputs/information_value_design.md", "sessions": sessions,
                  "comparisons": comparisons, "base_predictors": list(BASE_FEATURES),
                  "added_predictor": CONDITIONAL_FEATURE, "ridge_alpha": 1.,
                  "added_score_donor_residualized_against_baseline": True,
                  "added_score_residual_restandardized": False,
                  "primary": {"prefix_trials": 32, "transfer": "leave_one_session_out", "endpoint": "equal_animal_MSE_reduction"},
                  "exposure": access, "all_source_neural_elements_read_once": True,
                  "prior_policy_comparisons": 41, "prior_policy_terminal_disposition_preserved": True,
                  "evaluation_outcomes_previously_exposed": True, "held_sessions_opened": 0,
                  "valid_evaluations_charged": 1, "shared_valid_evaluation_total": 42,
                  "claim": "Exploratory offline prediction of model-specific conditional removal value",
                  "independent_biological_units": "two animals; eight sessions clustered within animals",
                  "uncertainty": "per-session results and fixed-prediction omissions; no electrode-level inference",
                  "selection_or_hyperparameter_search": False, "progress": progress}
        check()
        report["figure"] = write_summary(args.output_dir, report)
        report["elapsed_seconds"] = time.monotonic()-start
        save_json(args.output_dir/"information_value.json", report)
        operational.update(status="complete", scientific_scores_released=True, valid_evaluations_charged=1)
    except Exception as exc:
        operational.update(status="incomplete", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        signal.alarm(0)
        operational.update(finished_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic()-start,
                           process_CPU_seconds=time.process_time()-cpu,
                           maximum_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        save_json(args.output_dir/"operational.json", operational)
    print(json.dumps({"status": operational["status"], "elapsed_seconds": operational["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
