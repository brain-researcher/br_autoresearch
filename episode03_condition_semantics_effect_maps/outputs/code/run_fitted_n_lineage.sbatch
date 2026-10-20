#!/bin/bash
#SBATCH --job-name=ep03-fitted-n
#SBATCH --partition=russpold,normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/users/zijiao/br_autoresearch/episode03_condition_semantics_effect_maps/fitted_n/slurm-%j.out

# EP03 fitted-N lineage check (first target load after the 2026-10-16 orientation freeze).
# Reads Dataset-level t/z/effect/variance images and the feeding node's effect images only.
# Usage: sbatch run_fitted_n_lineage.sbatch <label> [route ...]   (no routes = all routes)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
EP=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode03_condition_semantics_effect_maps
FL=/oak/stanford/groups/russpold/data/OpenNeuro_analyses/openneuro_fitlins
PY=/home/groups/russpold/zijiao/miniforge3/envs/encoding_model/bin/python3
LABEL=${1:?label required}; shift
OUT=$EP/outputs/runs/source_handoff_qualification/20261016-fitted-n-$LABEL-$SLURM_JOB_ID
echo "host=$(hostname) job=$SLURM_JOB_ID label=$LABEL routes=$* start=$(date -u +%FT%TZ)"
ROUTES=(); [ $# -gt 0 ] && ROUTES=(--routes "$@")
$PY -I -B $EP/outputs/code/check_group_df_recompute.py \
  --census-maps $EP/outputs/runs/source_handoff_qualification/20261016-route-census-48185964/dataset_maps.json \
  --analysis-root $FL/analyses \
  "${ROUTES[@]}" \
  --output "$OUT/fitted_n_lineage.json"
echo "check_exit=$? end=$(date -u +%FT%TZ)"
