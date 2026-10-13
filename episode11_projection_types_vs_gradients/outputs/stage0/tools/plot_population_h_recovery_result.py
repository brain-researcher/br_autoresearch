"""Plot the actual terminal H diagnostic; uncertainty is Monte Carlo only."""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

episode = Path(os.environ["OAK"]) / "users/zijiao/br_autoresearch/episode11_projection_types_vs_gradients"
source = episode / "outputs/stage0/population_h_recovery_summary.json"
summary = json.loads(source.read_text())
starts = summary["optimizer"]["starts"]
verification = summary["fresh_paired_verification"]
gap = verification["contrasts"]["H_best_minus_H_second"]
truth_upper = verification["truth_minus_H_one_sided_95_upper"]
thresholds = summary["thresholds"]
cap = summary["optimizer_contract"]["main_maximum_iterations_per_start_frozen"]

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.7, 3.9), gridspec_kw={"width_ratios": [1.1, 1]})
names = ["Embedded C", "Truth-initialized H", "Empirical H"]
colors = ["#737b84", "#2b6a9b", "#b14d4d"]
positions = [2, 1, 0]
ax1.barh(positions, [s["iterations"] for s in starts], color=colors, height=0.42)
ax1.set_yticks(positions, names)
ax1.axvline(cap, color="#b14d4d", linestyle="--", linewidth=1)
for y, row in zip(positions, starts):
    ax1.text(row["iterations"] + 20, y, str(row["iterations"]), va="center", fontsize=9)
ax1.set_xlim(0, cap * 1.18)
ax1.set_ylim(-0.6, 2.65)
ax1.set_xlabel("Main-start iterations")
ax1.set_title("a  Empirical start reached the cap", loc="left", fontsize=10)
ax1.grid(axis="x", alpha=0.15)

stability_limit = thresholds["best_two_H_fresh_score_difference_nats_per_neuron"]
ax2.errorbar(gap["mean"], 0, xerr=1.96 * gap["mcse"], fmt="o", color="#b14d4d", capsize=5, markersize=7)
ax2.axvline(stability_limit, color="#737b84", linestyle="--", linewidth=1)
ax2.text(0.065, 0.27, f'Fresh gap: {gap["mean"]:.5f}\nMCSE: {gap["mcse"]:.6f}', ha="center", fontsize=9)
ax2.text(stability_limit + 0.005, -0.28, f"Tolerance: {stability_limit:.3f}", fontsize=9)
ax2.set_xlim(0, 0.15)
ax2.set_ylim(-0.62, 0.63)
ax2.set_yticks([])
ax2.set_xlabel("Best-minus-second H score (nats/neuron)")
ax2.set_title("b  Converged starts remain distinct", loc="left", fontsize=10)
ax2.grid(axis="x", alpha=0.15)

fig.suptitle("QC · Synthetic fitted-H recovery failed two frozen gates", x=0.02, ha="left", fontsize=12)
fig.text(0.02, 0.06, f"Best H is close to planted truth: one-sided 95% upper deficit {truth_upper:.5f} < 0.010.\n512 fresh paired draws; error bar = ±1.96 MCSE. This is optimizer QC; real outcomes remain closed.", fontsize=9)
fig.subplots_adjust(left=0.18, right=0.98, top=0.80, bottom=0.29, wspace=0.42)
out = episode / "outputs/figures/population_h_recovery_terminal_qc_20260930.png"
fig.savefig(out, dpi=180, bbox_inches="tight")
plt.close(fig)
print(out)
