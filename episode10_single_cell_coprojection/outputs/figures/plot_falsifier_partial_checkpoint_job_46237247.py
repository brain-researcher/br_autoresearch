#!/usr/bin/env python3
"""Plot preserved partial timing checkpoints only; no numerical fitting."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


FIGURES = Path(__file__).resolve().parent
CHECKPOINTS = (
    FIGURES.parent / "stage0" / "v3_m2_falsifier_timing_v1_job_46237247"
    / "falsifier_timing_checkpoints"
)
PROBES = (
    ("core_1", "Core 1"), ("core_2", "Core 2"), ("core_3", "Core 3"),
    ("core_4", "Core 4"), ("core_5", "Core 5"),
    ("low_rank", "Low rank"), ("hybrid", "Hybrid"),
)
REQUIRED_FOLDS = 12  # Frozen grouped timing probe requirement.
records = [json.loads((CHECKPOINTS / f"{probe}.progress.json").read_text())
           for probe, _ in PROBES]
permutation = json.loads((CHECKPOINTS / "permutation.progress.json").read_text())
counts = [record["completed_fold_count"] for record in records]
for (probe, _), record, count in zip(PROBES, records, counts):
    if record["probe"] != probe or type(count) is not int or not 0 <= count <= REQUIRED_FOLDS:
        raise ValueError("Unexpected preserved checkpoint identity/count")
if permutation["status"] != "failed" or permutation["failed_during"] != "permutation_M1_started":
    raise ValueError("Permutation checkpoint no longer matches this failure figure")
complete_probes = sum(record["status"] == "pass" for record in [*records, permutation])

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "svg.fonttype": "none",
})
fig, ax = plt.subplots(figsize=(8.2, 5.1))
fig.subplots_adjust(left=0.16, right=0.88, top=0.80, bottom=0.30)
fig.text(0.06, 0.94, "Partial timing checkpoint coverage", size=17, weight="bold")
fig.text(0.06, 0.885, "QC · Job 46237247 · incomplete", size=11, color="#525c66")

positions = list(range(len(PROBES)))
ax.barh(positions, [REQUIRED_FOLDS] * len(PROBES), height=0.56, color="#e8edf1")
ax.barh(positions, counts, height=0.56, color="#287b9e")
ax.set_yticks(positions, [label for _, label in PROBES])
ax.invert_yaxis()
ax.set_xlim(0, REQUIRED_FOLDS)
ax.set_xticks([0, 3, 6, 9, 12])
ax.set_xlabel("Recorded completed folds per Rademacher probe", labelpad=9)
ax.tick_params(axis="y", length=0, pad=8)
ax.tick_params(axis="x", colors="#525c66")
ax.spines["bottom"].set_color("#c0c8cf")
for position, count in zip(positions, counts):
    ax.text(12.2, position, f"{count}/12", va="center", size=11)

fig.text(0.06, 0.16, f"Permutation failed during M1 ({permutation['exception_type']}).", size=10)
fig.text(0.06, 0.105,
         f"{complete_probes}/8 complete timing probes; the performance projection remains unavailable.", size=10)
fig.text(0.06, 0.05, "Checkpoint coverage is numerical QC; empirical M2 and biology were not evaluated.",
         size=10, color="#525c66")

stem = FIGURES / "v3_m2_falsifier_partial_checkpoint_job_46237247"
fig.savefig(stem.with_suffix(".png"), dpi=170, facecolor="white")
fig.savefig(stem.with_suffix(".svg"), facecolor="white")
plt.close(fig)
print(json.dumps({"png": str(stem.with_suffix(".png")),
                  "svg": str(stem.with_suffix(".svg")),
                  "recorded_completed_folds": dict(zip((p for p, _ in PROBES), counts)),
                  "complete_timing_probes": complete_probes}, indent=2))
