#!/bin/bash
#SBATCH --job-name=ep23_a0_plot
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode23_spectral_operator_convergence/outputs/runs/a0_operator_alignment/%x-%j.out
# EP23 A0 figure, rendered from the collected report only.
set -euo pipefail
ml devel python/3.12.1 math py-numpy/1.26.3 viz py-matplotlib/3.8.3_py312
E=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode23_spectral_operator_convergence
export MPLCONFIGDIR=$SCRATCH/br_autoresearch/episode23_spectral_operator_convergence/mplconfig
mkdir -p "$MPLCONFIGDIR"
python3 -P -s "$E/outputs/code/plot_a0.py" --report "$E/outputs/runs/a0_operator_alignment/${RUN:?set RUN}/a0_report.json" \
  --out-dir "$E/outputs/figures/a0_operator_alignment/$RUN"
