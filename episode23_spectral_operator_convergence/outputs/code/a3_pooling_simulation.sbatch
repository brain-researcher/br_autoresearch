#!/bin/bash
#SBATCH --job-name=ep23_a3
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --array=0-3
#SBATCH --output=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode23_spectral_operator_convergence/outputs/runs/a3_pooling_simulation/%x-%A_%a.out
# EP23 A3: synthetic pooling simulations (no biological claims).
set -euo pipefail
ml devel python/3.12.1 math py-numpy/1.26.3 py-scipy/1.12.0_py312
E=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode23_spectral_operator_convergence
EP04=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode04_shared_scene_geometry
M=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch_data/steward_acquisition/things_metadata_osf_jum2f_20260922/payload
ROWS=$EP04/outputs/runs/source_readiness/20261003-development_embeddings01/development_rows.jsonl
cd "$SCRATCH/br_autoresearch/episode23_spectral_operator_convergence"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
OUT=$E/outputs/runs/a3_pooling_simulation/${RUN:?set RUN}/tasks; mkdir -p "$OUT"
python3 -P -s "$E/outputs/code/a3_pooling_simulation.py" --task-index "$SLURM_ARRAY_TASK_ID" --n-tasks 4 --out "$OUT/task_$SLURM_ARRAY_TASK_ID.json"
