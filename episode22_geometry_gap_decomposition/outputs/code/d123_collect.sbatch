#!/bin/bash
#SBATCH --job-name=ep22-d123-collect
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --chdir=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode22_geometry_gap_decomposition
#SBATCH --output=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode22_geometry_gap_decomposition/outputs/runs/d123_development/20261012-attempt02/slurm-collect-%j.out
# Aggregate the 25 D1–D3 cells, render the result figure, re-render the D0
# figure (shorter tick labels only), and validate the ASTRA program.
set -uo pipefail
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} start=$(date -u +%FT%TZ)"
module purge
module load devel python/3.12.1 math py-numpy/1.26.3 viz py-matplotlib/3.8.3_py312
export OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
attempt=outputs/runs/d123_development/20261012-attempt02
python3.12 -u -B outputs/code/d123_collect.py --cells "$attempt/cells" --out-dir "$attempt/summary" \
  --figure-dir outputs/figures/d123_development/20261012-attempt02
collect_exit=$?
python3.12 -B outputs/code/plot_d0_spectra.py \
  --summary outputs/runs/d0_outcome_blind_spectra/20261012-attempt01/results/d0_spectra_summary.json \
  --out-dir outputs/figures/d0_outcome_blind_spectra/20261012-attempt01
echo "d0_plot_exit=$?"
astra="${GROUP_HOME:?}/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv/bin/astra-tools"
timeout --signal=TERM --kill-after=10s 60s "$astra" validate outputs/astra/v0.0.14/astra.yaml --json > "$attempt/astra_validation.json" 2>&1
echo "astra_exit=$? collect_exit=$collect_exit end=$(date -u +%FT%TZ)"
exit $collect_exit
