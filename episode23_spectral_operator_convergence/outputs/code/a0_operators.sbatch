#!/bin/bash
#SBATCH --job-name=ep23_a0_ops
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode23_spectral_operator_convergence/outputs/runs/a0_operator_alignment/%x-%j.out
# EP23 A0 stage 1: WordNet crosswalk plus Visual Genome and COCO-caption operators.
# Reads annotation files and THINGS metadata only (no model or neural data).
set -euo pipefail
ml devel python/3.12.1 math py-numpy/1.26.3 py-scipy/1.12.0_py312
E=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode23_spectral_operator_convergence
D=$SCRATCH/br_autoresearch/episode23_spectral_operator_convergence/feasibility/downloads
M=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch_data/steward_acquisition/things_metadata_osf_jum2f_20260922/payload
ROWS=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode04_shared_scene_geometry/outputs/runs/source_readiness/20261003-development_embeddings01/development_rows.jsonl
OUT=$E/outputs/runs/a0_operator_alignment/${RUN:?set RUN}/${OPDIR:-operators}
mkdir -p "$OUT"; cd "$SCRATCH/br_autoresearch/episode23_spectral_operator_convergence"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
/usr/bin/time -v python3 -P -s "$E/outputs/code/a0_operators.py" --things-meta "$M/concepts-metadata_things.tsv" \
  --dev-rows "$ROWS" --wordnet-zip "$D/wordnet/wordnet.zip" --vg-objects "$D/visual_genome/objects.json.zip" \
  --coco-annotations "$D/coco/annotations_trainval2017.zip" --out-dir "$OUT" ${EXTRA:-} 2>&1 | tee "$OUT/stdout.txt"
