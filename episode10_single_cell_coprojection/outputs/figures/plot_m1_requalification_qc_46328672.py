"""Actual warm-versus-zero qualification metrics; no fits or synthetic generation."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[2]
source = root / "outputs/stage0/v3_m1_synthetic_warm_zero_diagnostic_v2_job_46328672/v3_m1_synthetic_warm_zero_diagnostic.json"
report = json.loads(source.read_text())
rows = report["warm_zero_comparisons"]
assert report["status"] == "pass" and report["completed_fit_count"] == 36
assert len(rows) == 12 and all(row["status"] == "pass" for row in rows)
ridge = [row["lambda"] for row in rows]
error = [row["zero_start_log_probability_max_absolute_difference"] for row in rows]
limit = 1e-4  # Frozen numerical_execution_addendum_v3.required_qualification.
plt.rcParams.update({"font.size": 10, "svg.fonttype": "none"})
fig, ax = plt.subplots(figsize=(7, 3.8))
fig.subplots_adjust(left=.15, right=.97, top=.74, bottom=.27)
ax.plot(ridge, error, "o-", color="#237b83", markersize=4, linewidth=1.4)
ax.set_xscale("log")
ax.set_yscale("symlog", linthresh=1e-15)
ax.set_xlim(3000, 3e-9)
ax.set_ylim(-.1e-15, 5e-4)
ax.set_yticks([0, 1e-12, 1e-9, 1e-6, 1e-4], ["0", r"$10^{-12}$", r"$10^{-9}$", r"$10^{-6}$", r"$10^{-4}$"])
ax.axhline(limit, color="#555555", linestyle="--", linewidth=1)
ax.text(2e3, limit*1.6, "Frozen maximum", color="#555555", fontsize=9)
ax.set_xlabel("Ridge λ (frozen descending grid)")
ax.set_ylabel("Maximum |Δ log probability|\nWarm versus zero start (nats)")
ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("QC · All 12 warm-versus-zero comparisons pass", x=.02, ha="left", fontsize=13, weight="bold")
fig.text(.02, .84, "Job 46328672 · 36/36 fits pass · 68 prescribed synthetic rows · one thread", fontsize=9)
fig.text(.02, .105, f"Largest difference: {max(error):.2e} nats; limit: 1e−4. Repeated warm estimates match exactly.", fontsize=9)
fig.text(.02, .035, "Required numerical qualification only; population recovery, throughput and empirical results remain gated.", fontsize=9, color="#555555")
stem=Path(__file__).with_name("v3_m1_requalification_qc_job_46328672")
fig.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight", pad_inches=.14)
fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=.14)
plt.close(fig)
print(stem.with_suffix(".png"))
