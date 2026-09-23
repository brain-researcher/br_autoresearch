#!/bin/bash
#SBATCH --job-name=narps-aggregate
#SBATCH --partition=russpold
#SBATCH --account=russpold
#SBATCH --qos=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/oak/stanford/groups/russpold/users/zijiao/NARPS/outputs/logs/aggregate_%j.out
#SBATCH --error=/oak/stanford/groups/russpold/users/zijiao/NARPS/outputs/logs/aggregate_%j.err

set -euo pipefail

workspace=/oak/stanford/groups/russpold/users/zijiao/NARPS
output_root="$workspace/outputs"

cd "$workspace"
mkdir -p "$output_root/logs" "$output_root/tmp" \
  "$output_root/cache/matplotlib" "$output_root/figures" \
  "$output_root/group_maps"

export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$output_root/tmp"
export XDG_CACHE_HOME="$output_root/cache"
export MPLCONFIGDIR="$output_root/cache/matplotlib"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

module load python/3.12.1

# Submit with an afterok dependency on the subject array when appropriate, e.g.:
# sbatch --dependency=afterok:<fit_array_job_id> outputs/code/aggregate_results.sbatch
# The diagnostic and aggregation stages independently require and validate all
# 108 immutable checkpoints. The diagnostic reads checkpoints/sidecars only;
# it never reopens source BOLD or changes a primary map.
/home/users/zijiao/pcvenv/bin/python outputs/code/audit_edge_scaling.py

/home/users/zijiao/pcvenv/bin/python outputs/code/aggregate_results.py \
  --gram-voxel-chunk 8192
