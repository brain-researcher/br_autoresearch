#!/usr/bin/env python3
"""QC plot from actual job 46330399; no new observations or fits."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
source = root / "stage0/v3_m1_affected_numerics_refresh_v1_job_46330399/v3_m1_affected_numerics_refresh.json"
result = json.loads(source.read_text())
estimate = result["resource_projection"]["projected_production_wall_hours"]
request = result["resource_projection"]["frozen_production_request_hours"]
fig, ax = plt.subplots(figsize=(7.2, 2.65), layout="constrained")
ax.barh([0], [estimate], height=.34, color="#b75b38")
ax.axvline(.8 * request, color="#365f7c", lw=1.8, label="Frozen limit with headroom: 6.40 h")
ax.axvline(request, color="#737373", lw=1.2, ls="--", label="Frozen job request: 8.00 h")
ax.text(estimate + .10, 0, f"{estimate:.2f} h", va="center", fontsize=11, weight="bold")
ax.set(yticks=[0], yticklabels=["157-path\nprojection"], xlim=(0, 9.4),
       ylim=(-.48,.64), xlabel="Projected M1 production wall time (hours)")
ax.set_title("QC · Numerical fits pass; production time gate fails", loc="left", fontsize=12, pad=14)
ax.legend(loc="upper left", bbox_to_anchor=(0, -.33), frameon=False, fontsize=9)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.grid(axis="x", alpha=.14)
ax.set_axisbelow(True)
fig.savefig(root / "figures/v3_m1_refresh_resource_gate_qc_job_46330399.png", dpi=180, bbox_inches="tight")
fig.savefig(root / "figures/v3_m1_refresh_resource_gate_qc_job_46330399.svg", bbox_inches="tight")

