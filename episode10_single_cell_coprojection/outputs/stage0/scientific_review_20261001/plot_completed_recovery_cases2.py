"""Plot the two actual completed recovery cases; no experiment or data access."""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ep = Path("/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode10_single_cell_coprojection")
case_root = ep / "outputs/stage0/v3_m2_optional_family_qualification_job_46115632/completed_cases"
rows = []
for rank in [1, 2]:
    path = case_root / f"low_rank_population_recovery_rank_{rank}.json"
    result = json.loads(path.read_text())
    case = result["cases"][str(rank)]
    rows.append({"rank": rank, "source": str(path),
                 "KL_bits": case["mean_conditional_KL_bits"],
                 "threshold_bits": result["mean_conditional_KL_maximum_bits"],
                 "repeat_bit_exact": case["deterministic_repeat_bit_exact"],
                 "status": case["status"]})
stem = ep / "outputs/figures/v3_m2_required_recovery_partial_job_46115632_cases2"
with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    for row, y in zip(rows, [1, 0]):
        ax.scatter(row["KL_bits"], y, s=55, color="#226B9D")
        ax.annotate(f'{row["KL_bits"]:.2e}', (row["KL_bits"], y),
                    xytext=(0, 10), textcoords="offset points", ha="center")
    ax.axvline(rows[0]["threshold_bits"], color="#9F442F", ls="--", lw=1.3)
    ax.text(rows[0]["threshold_bits"] / 1.1, 1.43, "Frozen KL limit", color="#9F442F",
            ha="right", va="center", fontsize=9)
    ax.set_xscale("log")
    ax.set_xlim(1e-8, 1.5e-5)
    ax.set_ylim(-0.35, 1.65)
    ax.set_yticks([1, 0], ["Low-rank, rank 1", "Low-rank, rank 2"])
    ax.set_xlabel("Mean conditional KL (bits; log scale)")
    ax.grid(axis="x", alpha=0.2)
    fig.suptitle("QC — required population recovery, 2 / 12 cases", y=0.99, fontsize=12)
    fig.text(0.02, 0.015,
             "Both completed cases pass and repeat bit-exactly. Remaining cases are pending.\nContract-required synthetic qualification; no empirical M2 result.", fontsize=8)
    fig.tight_layout(rect=(0, 0.15, 1, 0.95))
    fig.savefig(str(stem)+".png", dpi=180)
    fig.savefig(str(stem)+".svg")
    plt.close(fig)
Path(str(stem)+".json").write_text(json.dumps({
    "classification": "QC", "job_id": 46115632,
    "required_cases": 12, "completed_cases_shown": 2,
    "actual_completed_case_reports": rows,
    "empirical_M2_result": False, "overall_qualification": "pending"
}, indent=2)+"\n")
print(str(stem)+".png")

