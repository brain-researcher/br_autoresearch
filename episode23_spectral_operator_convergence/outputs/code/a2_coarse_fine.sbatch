#!/bin/bash
#SBATCH --job-name=ep23_a2
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --array=0-5
#SBATCH --output=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode23_spectral_operator_convergence/outputs/runs/a2_coarse_fine/%x-%A_%a.out
# EP23 A2: coarse vs fine (tasks 0-4 participants; 5 pooled model-model).
set -euo pipefail
ml devel python/3.12.1 math py-numpy/1.26.3 py-scipy/1.12.0_py312
E=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode23_spectral_operator_convergence
EP04=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode04_shared_scene_geometry
M=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch_data/steward_acquisition/things_metadata_osf_jum2f_20260922/payload
ROWS=$EP04/outputs/runs/source_readiness/20261003-development_embeddings01/development_rows.jsonl
cd "$SCRATCH/br_autoresearch/episode23_spectral_operator_convergence"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
P=(sub-01 sub-03 sub-05 sub-06 sub-07 pooled); T=${P[$SLURM_ARRAY_TASK_ID]}
OUT=$E/outputs/runs/a2_coarse_fine/${RUN:?set RUN}/tasks; mkdir -p "$OUT"
BR=$SCRATCH/br_autoresearch/episode23_spectral_operator_convergence/a0/20261013-attempt01/brain_patterns
COMMON=(--embeddings-dir "$EP04/outputs/runs/source_readiness/20261003-development_embeddings01" --dev-rows "$ROWS" --things-meta "$M/concepts-metadata_things.tsv" --things-c27 "$M/category27_manual.tsv")
if [ "$T" = pooled ]; then
  python3 -P -s "$E/outputs/code/a2_coarse_fine.py" --mode pooled "${COMMON[@]}" --out "$OUT/pooled.json"
else
  python3 -P -s "$E/outputs/code/a2_coarse_fine.py" --mode participant --participant "$T" --brain-npz "$BR/brain_patterns_$T.npz" "${COMMON[@]}" --out "$OUT/$T.json"
fi
