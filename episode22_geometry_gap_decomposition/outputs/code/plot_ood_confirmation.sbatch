#!/bin/bash
#SBATCH --job-name=ep22-ood-figure
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --chdir=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode22_geometry_gap_decomposition
#SBATCH --output=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode22_geometry_gap_decomposition/outputs/figures/ood_confirmation/20261013-opening01/slurm-%j.out
# Render the EP22 confirmation figure from the read-only confirmation report only.
module purge
module load devel python/3.12.1 math py-numpy/1.26.3 viz py-matplotlib/3.8.3_py312
python3.12 -B outputs/code/plot_ood_confirmation.py \
  --report outputs/runs/ood_confirmation/20261013-opening01/confirmation_report.json \
  --out-dir outputs/figures/ood_confirmation/20261013-opening01
echo "plot_exit=$?"
