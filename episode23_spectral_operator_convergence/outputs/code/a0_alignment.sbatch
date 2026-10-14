#!/bin/bash
#SBATCH --job-name=ep23_a0_align
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-5
#SBATCH --output=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode23_spectral_operator_convergence/outputs/runs/a0_operator_alignment/%x-%A_%a.out
# EP23 A0 stage 3: operator-subspace alignment for each participant (0-4) and the pooled model task (5).
set -euo pipefail
ml devel python/3.12.1 math py-numpy/1.26.3 py-scipy/1.12.0_py312
T=(sub-01 sub-03 sub-05 sub-06 sub-07 pooled); TASK=${T[$SLURM_ARRAY_TASK_ID]}
E=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode23_spectral_operator_convergence
EP04=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode04_shared_scene_geometry
M=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch_data/steward_acquisition/things_metadata_osf_jum2f_20260922/payload
R=$E/outputs/runs/a0_operator_alignment/${RUN:?set RUN}
mkdir -p "$R/tasks"; cd "$SCRATCH/br_autoresearch/episode23_spectral_operator_convergence"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
/usr/bin/time -v python3 -P -s "$E/outputs/code/a0_alignment.py" --task "$TASK" --operators "$R/${OPDIR:-operators}/operators.npz" \
  --brain-dir "$SCRATCH/br_autoresearch/episode23_spectral_operator_convergence/a0/$RUN/brain_patterns" \
  --embeddings-dir "$EP04/outputs/runs/source_readiness/20261003-development_embeddings01" \
  --lowlevel-dir "$EP04/outputs/runs/source_readiness/20261005-low_level_full01" \
  --things-meta "$M/concepts-metadata_things.tsv" --things-c27 "$M/category27_manual.tsv" \
  --dev-rows "$EP04/outputs/runs/source_readiness/20261003-development_embeddings01/development_rows.jsonl" \
  --out "$R/tasks/$TASK.json"
