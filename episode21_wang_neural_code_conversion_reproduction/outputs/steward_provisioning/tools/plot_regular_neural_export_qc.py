"""Plot only the completed steward's redacted structural report."""
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

episode = Path(__file__).resolve().parents[3]
source = episode / "outputs/steward_provisioning/regular_neural_structural_report.json"
report = json.loads(source.read_text())
assert report["publication"] == "complete"
assert report["contains_neural_values"] is False
assert report["model_fitting_scoring_run"] is False
subjects = report["subjects"]
assert len(subjects) == 5
roles = ("subject_unique", "tau_train", "tau_test")
colors = ("#427aab", "#5ca995", "#ddaa56")
labels = ("Unique training", "Shared tau train", "Shared tau test")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "svg.fonttype": "none", "axes.spines.top": False,
                     "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(8.4, 3.8), layout="constrained")
left = [0] * len(subjects)
for role, color, label in zip(roles, colors, labels):
    rows = [s["roles"][role]["rows"] for s in subjects]
    ax.barh(range(len(subjects)), rows, left=left, height=.62, color=color, label=label)
    left = [a + b for a, b in zip(left, rows)]
maximum = max(left)
for i, s in enumerate(subjects):
    voxels = {s["roles"][r]["voxels"] for r in roles}
    assert len(voxels) == 1
    voxels = voxels.pop()
    nonfinite = sum(s["roles"][r]["nonfinite_elements"] for r in roles)
    elements = left[i] * voxels
    assert 0 <= nonfinite <= elements
    text = f"{left[i]:,} rows | {voxels:,} voxels\n{nonfinite:,} nonfinite ({100 * nonfinite / elements:.3g}%)"
    ax.text(maximum * 1.025, i, text, va="center", fontsize=9)
ax.set_yticks(range(len(subjects)), [s["participant"] for s in subjects])
ax.invert_yaxis()
ax.set_xlim(0, maximum * 1.58)
ax.set_xticks([0, 10000, 20000, 30000])
ax.set_xlabel("Exported regular trial repetitions")
ax.set_title("QC · Sealed regular neural export", loc="left", fontweight="bold")
ax.legend(loc="upper center", bbox_to_anchor=(.5, -.19), ncol=3, frameon=False,
          fontsize=9)
figures = episode / "outputs/figures"
render_date = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d")
stem = figures / f"regular_neural_export_qc_{render_date}"
fig.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
stem.with_suffix(".json").write_text(json.dumps({
    "evidence_status": "QC", "rendered_on": render_date, "source_artifact": str(source),
    "job_id": report["job_id"], "neural_values_read": False,
    "interpretation": "Exported repetitions, role support and finite coverage establish input availability only; no conversion performance or upstream independence is established."
}, indent=2) + "\n")
print(str(stem.with_suffix(".png")))
