#!/bin/bash
#SBATCH --job-name=ep23_opfeas
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/scratch/users/zijiao/br_autoresearch/episode23_spectral_operator_convergence/feasibility/logs/feasibility_%j.out
# EP23 operator feasibility: THINGS coverage and spectral diagnostics of
# annotation-derived co-occurrence operators. No model or neural data.
# Usage: RUN=<run-name> [EXTRA="--limit-images 3000"] sbatch operator_feasibility.sbatch
set -euo pipefail
ml devel python/3.12.1 math py-numpy/1.26.3 py-scipy/1.12.0_py312
E=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode23_spectral_operator_convergence
D=$SCRATCH/br_autoresearch/episode23_spectral_operator_convergence/feasibility/downloads
M=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch_data/steward_acquisition/things_metadata_osf_jum2f_20260922/payload
ROWS=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode04_shared_scene_geometry/outputs/runs/source_readiness/20261003-development_embeddings01/development_rows.jsonl
OUT=$E/outputs/runs/operator_feasibility/${RUN:?set RUN}
mkdir -p "$OUT"
cd "$SCRATCH/br_autoresearch/episode23_spectral_operator_convergence/feasibility"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
/usr/bin/time -v python3 -P -s "$E/outputs/code/operator_feasibility.py" \
  --things-meta "$M/concepts-metadata_things.tsv" --things-c27 "$M/category27_manual.tsv" \
  --dev-rows "$ROWS" \
  --vg-objects "$D/visual_genome/objects.json.zip" \
  --lvis-train "$D/lvis/lvis_v1_train.json.zip" --lvis-val "$D/lvis/lvis_v1_val.json.zip" \
  --coco-annotations "$D/coco/annotations_trainval2017.zip" \
  --openimages-classes "$D/open_images/oidv7-class-descriptions.csv" \
  --out "$OUT/feasibility_report.json" ${EXTRA:-} 2>&1 | tee "$OUT/stdout.txt"
