#!/bin/bash
#SBATCH --job-name=ep23_stage2_plot
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode23_spectral_operator_convergence/outputs/runs/%x-%j.out
# EP23 stage-2 figure, rendered from collected reports and A1 cells only.
set -euo pipefail
ml devel python/3.12.1 math py-numpy/1.26.3 viz py-matplotlib/3.8.3_py312
E=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode23_spectral_operator_convergence
R=$E/outputs/runs
export MPLCONFIGDIR=$SCRATCH/br_autoresearch/episode23_spectral_operator_convergence/mplconfig
mkdir -p "$MPLCONFIGDIR"
python3 -P -s "$E/outputs/code/plot_stage2.py" \
  --a1-cells "$R/a1_metric_family/20261014-attempt01/cells" "$R/a1_metric_family/20261014-rank32-attempt01/cells" \
  --a1-reports "$R/a1_metric_family/20261014-attempt01/a1_report.json" "$R/a1_metric_family/20261014-rank32-attempt01/a1_report.json" \
  --a2-report "$R/a2_coarse_fine/20261014-attempt01/a2_report.json" \
  --a3-report "$R/a3_pooling_simulation/20261014-attempt01/a3_report.json" \
  --out-dir "$E/outputs/figures/stage2/20261014-attempt01"
