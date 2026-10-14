#!/bin/bash
#SBATCH --job-name=ep23_a1g
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --array=0-4
#SBATCH --output=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode23_spectral_operator_convergence/outputs/runs/a1_metric_family/%x-%A_%a.out
# EP23 A1, one participant per task: its 5 held folds in sequence (cell index = 5*participant + fold).
# Rank-64 cells measured 8.6-10.3 min and <=1.5 GB each (job 47703628).
set -euo pipefail
ml devel python/3.12.1 math py-numpy/1.26.3 py-scipy/1.12.0_py312
E=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode23_spectral_operator_convergence
cd "$SCRATCH/br_autoresearch/episode23_spectral_operator_convergence"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
OUT=$E/outputs/runs/a1_metric_family/${RUN:?set RUN}/cells; mkdir -p "$OUT"
for fold in 0 1 2 3 4; do
  python3 -P -s "$E/outputs/code/a1_metric_family.py" --task-index $((5 * SLURM_ARRAY_TASK_ID + fold)) \
    --out-dir "$OUT" --rank "${RANK:?set RANK}"
done
