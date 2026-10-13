"""Render two logged optimizer results; this is partial synthetic QC only."""
from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


episode = Path(os.environ["OAK"]) / "users/zijiao/br_autoresearch/episode11_projection_types_vs_gradients"
snapshot = episode / "outputs/stage0/population_h_resume_20260930"
log_path = snapshot / "population-h-recovery-45736553.out"
log = log_path.read_text()
matches = re.findall(
    r"R=8192 H start (\d)/3 \(([^)]+)\) completed: objective=([\d.eE+-]+), KKT_converged=(True|False)",
    log,
)
rows = [
    {"start_index": int(index), "start": name, "training_objective": float(value),
     "physical_KKT_converged": passed == "True"}
    for index, name, value, passed in matches
]
if len(rows) != 2 or {r["start"] for r in rows} != {"embedded_fitted_C", "truth_based_nonzero_signature"}:
    raise RuntimeError("The expected partial log changed; inspect before plotting.")
by_name = {r["start"]: r for r in rows}
c_objective = by_name["embedded_fitted_C"]["training_objective"]
gain = c_objective - by_name["truth_based_nonzero_signature"]["training_objective"]
observed = datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(timespec="seconds")
summary = {
    "analysis_id": "population_h_optimizer_recovery",
    "job_id": "45736553",
    "evidence_class": "QC_partial_synthetic_optimizer_execution",
    "rendered_at": observed,
    "source_log": str(log_path.relative_to(episode)),
    "completed_logged_starts": rows,
    "empirical_start_result": None,
    "truth_initialized_training_improvement_relative_to_embedded_C": gain,
    "fresh_verification_scores": None,
    "frozen_H_gate_adjudication": None,
    "Panel_A_complete": False,
    "real_projection_outcomes_opened": False,
    "interpretation": "Training improvement depends on initialization; no held-out recovery or biological conclusion follows from this partial log.",
}
(snapshot / "partial_optimizer_progress.json").write_text(json.dumps(summary, indent=2) + "\n")

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(6.3, 3.25))
ax.scatter([0, gain], [1, 0], s=75, c=["#737b84", "#2b6a9b"], zorder=3)
ax.axvline(0, color="#aeb4ba", linewidth=0.9)
ax.set_yticks([1, 0], ["Embedded fitted C", "Truth-initialized H"])
ax.set_xlim(-0.01, 0.16)
ax.set_ylim(-0.55, 1.55)
ax.set_xticks([0, 0.05, 0.10, 0.15])
ax.set_xlabel("Training improvement relative to embedded C (nats/neuron)")
ax.text(gain + 0.004, 0, f"{gain:.3f}", va="center", fontsize=10)
ax.text(0.005, 1, "0.000", va="center", fontsize=10)
ax.set_title("QC · Synthetic H optimization: two completed starts", loc="left", fontsize=11)
ax.grid(axis="x", alpha=0.18)
fig.text(0.02, 0.025, "Empirical start remains pending. Fresh verification and frozen gates are unavailable.\nTraining scores are partial optimizer evidence; real projection outcomes remain closed.", fontsize=8.6)
fig.subplots_adjust(left=0.28, right=0.97, top=0.86, bottom=0.32)
figures = episode / "outputs/figures"
figures.mkdir(exist_ok=True)
out = figures / "population_h_partial_optimizer_qc_20260930.png"
fig.savefig(out, dpi=180)
plt.close(fig)
print(out)
