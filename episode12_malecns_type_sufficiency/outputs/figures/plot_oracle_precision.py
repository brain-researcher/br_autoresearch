"""Plot a published oracle aggregate only; never open connectivity or shards."""
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


root = Path(__file__).resolve().parents[2]
source = root / "outputs/scientific_qualification/stochastic_oracle_v3/qualification_attempt001/qualification_look06.json"
report = json.loads(source.read_text())
cells = sorted(report["cells"], key=lambda c: c["oracle_enclosure"]["maximum_half_width"], reverse=True)
widths = [c["oracle_enclosure"]["maximum_half_width"] for c in cells]
tolerance = cells[0]["oracle_enclosure"]["integration_tolerance"]
passed = sum(c["oracle_enclosure"]["precision_met"] for c in cells)
out = Path(__file__).resolve().parent

with (out / "oracle_look06_precision.csv").open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["source", "cell_id", "maximum_half_width", "precision_met", "tolerance"])
    for cell in cells:
        enclosure = cell["oracle_enclosure"]
        writer.writerow([str(source.relative_to(root)), cell["cell"]["cell_id"], enclosure["maximum_half_width"], enclosure["precision_met"], tolerance])

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(7.2, 3.7), constrained_layout=True)
colors = ["#cf5d35" if not c["oracle_enclosure"]["precision_met"] else "#267b91" for c in cells]
ax.scatter(range(1, len(cells) + 1), widths, c=colors, s=33, zorder=3)
ax.axhline(tolerance, color="#5a6268", linestyle="--", linewidth=1)
ax.text(34.8, tolerance + 0.00013, "Frozen tolerance: 0.002", ha="right", color="#444c52", fontsize=9)
ax.annotate("Largest uncertainty: sparse curved continua\n5 or 7 neurons per side", xy=(1.5, sum(widths[:2]) / 2), xytext=(9, 0.00545), arrowprops={"arrowstyle": "-", "color": "#687279"}, fontsize=9, color="#444c52")
ax.set(xlim=(0, len(cells) + 1), ylim=(0, max(widths) * 1.15), xlabel="Calibration scenarios, sorted by uncertainty", ylabel="Maximum oracle interval half-width\n(nats per count)")
ax.set_xticks([1, 10, 20, 30, 35])
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="#e5e8eb", linewidth=0.6)
ax.set_axisbelow(True)
fig.suptitle("QC · Oracle precision at completed look 6", x=0.02, ha="left", fontsize=13, fontweight="bold")
ax.set_title(f"Generated data · 4,608 replicates per scenario · {passed}/{len(cells)} meet precision", loc="left", fontsize=10, pad=10)
fig.savefig(out / "oracle_look06_precision.png", dpi=180)
fig.savefig(out / "oracle_look06_precision.pdf")
print(f"Rendered published generated-data QC: {passed}/{len(cells)} cells meet precision.")
