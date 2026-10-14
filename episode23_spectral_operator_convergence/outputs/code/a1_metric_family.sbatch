#!/bin/bash
#SBATCH --job-name=ep23_a1
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --array=0-24
#SBATCH --output=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode23_spectral_operator_convergence/outputs/runs/a1_metric_family/%x-%A_%a.out
# EP23 A1: metric families per participant x held fold (task = participant-major, fold-minor).
set -euo pipefail
ml devel python/3.12.1 math py-numpy/1.26.3 py-scipy/1.12.0_py312
E=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode23_spectral_operator_convergence
EP04=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode04_shared_scene_geometry
M=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch_data/steward_acquisition/things_metadata_osf_jum2f_20260922/payload
ROWS=$EP04/outputs/runs/source_readiness/20261003-development_embeddings01/development_rows.jsonl
cd "$SCRATCH/br_autoresearch/episode23_spectral_operator_convergence"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
OUT=$E/outputs/runs/a1_metric_family/${RUN:?set RUN}/cells; mkdir -p "$OUT"
python3 -P -s "$E/outputs/code/a1_metric_family.py" --task-index "$SLURM_ARRAY_TASK_ID" --out-dir "$OUT" --rank "${RANK:-64}"
