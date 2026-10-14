#!/bin/bash
#SBATCH --job-name=ep23_a0_brain
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --array=0-4
#SBATCH --output=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode23_spectral_operator_convergence/outputs/runs/a0_operator_alignment/%x-%A_%a.out
# EP23 A0 stage 2: split-half concept-level brain patterns from EP04's development single-trial cache.
set -euo pipefail
ml devel python/3.12.1 math py-numpy/1.26.3 py-scipy/1.12.0_py312
P=(sub-01 sub-03 sub-05 sub-06 sub-07); PART=${P[$SLURM_ARRAY_TASK_ID]}
E=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode23_spectral_operator_convergence
EP04=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode04_shared_scene_geometry
M=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch_data/steward_acquisition/things_metadata_osf_jum2f_20260922/payload
ROWS=$EP04/outputs/runs/source_readiness/20261003-development_embeddings01/development_rows.jsonl
OUT=$SCRATCH/br_autoresearch/episode23_spectral_operator_convergence/a0/${RUN:?set RUN}/brain_patterns
REP=$E/outputs/runs/a0_operator_alignment/$RUN/brain_patterns
mkdir -p "$OUT" "$REP"; cd "$SCRATCH/br_autoresearch/episode23_spectral_operator_convergence"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
/usr/bin/time -v python3 -P -s "$E/outputs/code/a0_brain_patterns.py" --participant "$PART" --ep04-root "$EP04" \
  --things-meta "$M/concepts-metadata_things.tsv" --dev-rows "$ROWS" --out-dir "$OUT" 2>&1 | tee "$REP/stdout_$PART.txt"
cp "$OUT/brain_patterns_$PART.json" "$REP/"
