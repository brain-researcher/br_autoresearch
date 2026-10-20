#!/bin/bash
#SBATCH --job-name=ep03-route-manifest
#SBATCH --partition=russpold,normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/users/zijiao/br_autoresearch/episode03_condition_semantics_effect_maps/route_census_20261016/slurm-%j.out

# EP03 per-route manifest and proposed orientation table (EP03 records only),
# plus the ds003545 dummy-volume onset check (events TSVs and BOLD JSON only).
export OMP_NUM_THREADS=1
EP=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode03_condition_semantics_effect_maps
FL=/oak/stanford/groups/russpold/data/OpenNeuro_analyses/openneuro_fitlins
PY=/home/groups/russpold/zijiao/miniforge3/bin/python3
RUNS=$EP/outputs/runs/source_handoff_qualification
OUT=$RUNS/20261016-orientation-proposal
echo "host=$(hostname) job=$SLURM_JOB_ID start=$(date -u +%FT%TZ)"
$PY -I -B $EP/outputs/code/build_route_manifest.py \
  --census-dir $RUNS/20261016-route-census-48185964 \
  --roles $OUT/condition_roles.json \
  --edits $OUT/event_edit_review.json \
  --output-dir $OUT
echo "manifest_exit=$?"
$PY -I -B $EP/outputs/code/check_dummy_onsets.py \
  --raw-root $FL/analyses/ds003545/sourcedata/raw \
  --basic-details $FL/analyses/ds003545/code/openneuro_glmfitlins/statsmodel_specs/ds003545/ds003545_basic-details.json \
  --tasks compL1 compLn prodL1 prodLn \
  --output $OUT/ds003545_dummy_onsets.json
echo "onset_check_exit=$? end=$(date -u +%FT%TZ)"
