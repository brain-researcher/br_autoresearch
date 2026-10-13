"""Plot actual scalar failure telemetry; no fits or generated observations."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[2]
source = root / "outputs/stage0/v3_m2_falsifier_permutation_diagnostic_job_46270474/permutation_only_result.json"
report = json.loads(source.read_text())
c = report["optimizer_scalar_context"]
cert = c["certificate"]
values = [c["current_gradient_infinity_norm"], c["prior_relative_objective_change"]]
limits = [cert["gradient_infinity_tolerance"], cert["relative_gap_tolerance"]]
ratios = [value / limit for value, limit in zip(values, limits)]
assert report["status"] == "failed" and not cert["certificate_pass"]

plt.rcParams.update({"font.size": 10, "svg.fonttype": "none"})
fig, (ax, note) = plt.subplots(1, 2, figsize=(8.0, 3.25), gridspec_kw={"width_ratios": [1.55, 1]})
fig.subplots_adjust(left=.23, right=.98, top=.76, bottom=.24, wspace=.28)
ax.set_xscale("log")
ax.set_xlim(1e-10, 4)
ax.set_ylim(-.55, 1.55)
ax.axvline(1, color="#555555", linestyle="--", linewidth=1)
ax.scatter(ratios, [1, 0], s=58, color="#237b83", zorder=3)
for ratio, y, value in zip(ratios, [1, 0], values):
    ax.annotate(f"{value:.2e}", (ratio, y), xytext=(0, 13), textcoords="offset points", ha="center", fontsize=9)
ax.set_yticks([1, 0], ["Gradient infinity norm", "Relative objective change"])
ax.set_xticks([1e-9, 1e-6, 1e-3, 1], [r"$10^{-9}$", r"$10^{-6}$", r"$10^{-3}$", "1"])
ax.set_xlabel("Observed value / frozen tolerance")
ax.text(1, 1.50, "limit", ha="center", va="bottom", fontsize=9)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.tick_params(axis="y", length=0)
note.axis("off")
note.text(0, .96, "Confirmation FAILED", color="#9d3d2a", weight="bold", va="top")
note.text(0, .70, "Direct derivative = 0\nDescent condition rejected\nCertificate ineligible", linespacing=1.6, va="top")
note.text(0, .13, "No accepted confirmation step\nNo passing permutation probe", color="#555555", fontsize=9, va="top")
fig.suptitle("QC · Convergence tolerances passed; required confirmation failed", x=.02, ha="left", fontsize=12, weight="bold")
fig.text(.02, .83, "Job 46270474 · warm path 64 · ridge λ = 1000 · frozen one-thread backend", fontsize=9, color="#555555")
fig.text(.02, .035, "Actual failed-run telemetry. This diagnoses numerical execution; it is not biological evidence.", fontsize=9, color="#555555")
stem = Path(__file__).with_name("v3_m1_permutation_confirmation_qc_job_46270474")
fig.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight", pad_inches=.14)
fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=.14)
plt.close(fig)
print(stem.with_suffix(".png"))
