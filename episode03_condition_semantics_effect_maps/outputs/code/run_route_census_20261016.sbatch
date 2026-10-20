#!/bin/bash
#SBATCH --job-name=ep03-route-census
#SBATCH --partition=russpold,normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/users/zijiao/br_autoresearch/episode03_condition_semantics_effect_maps/route_census_20261016/slurm-%j.out

# EP03 per-route execution-record census of the NeuroEffect candidate routes.
# Read-only git log/show, JSON/HTML text, events TSVs and sidecars; no maps opened.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
EP=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode03_condition_semantics_effect_maps
FL=/oak/stanford/groups/russpold/data/OpenNeuro_analyses/openneuro_fitlins
PY=/home/groups/russpold/zijiao/miniforge3/bin/python3
OUT=$EP/outputs/runs/source_handoff_qualification/20261016-route-census-$SLURM_JOB_ID
echo "host=$(hostname) job=$SLURM_JOB_ID start=$(date -u +%FT%TZ)"
mkdir -p "$OUT"
$PY -I -B $EP/outputs/code/census_route_provenance.py \
  --candidates $EP/outputs/runs/source_handoff_qualification/20261016-source-facts-48169324/neuroeffect_candidate_provenance.json \
  --analysis-root $FL/analyses \
  --input-root $FL/input \
  --output-dir "$OUT"
echo "census_exit=$? end=$(date -u +%FT%TZ)"
