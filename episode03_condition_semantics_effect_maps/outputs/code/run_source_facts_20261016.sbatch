#!/bin/bash
#SBATCH --job-name=ep03-source-facts
#SBATCH --partition=russpold,normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/users/zijiao/br_autoresearch/episode03_condition_semantics_effect_maps/srcfacts_20261016/slurm-%j.out

# EP03 source-fact checks: BART events construction and NeuroEffect candidate
# provenance. Events TSVs, metadata JSON and filenames only; no maps opened.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
EP=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode03_condition_semantics_effect_maps
FL=/oak/stanford/groups/russpold/data/OpenNeuro_analyses/openneuro_fitlins
NB=/scratch/users/zijiao/neuroeffect-full51-spatial-build/20260917T201500Z.2230efb/release-stage-v1
PY=/home/groups/russpold/zijiao/miniforge3/bin/python3
OUT=$EP/outputs/runs/source_handoff_qualification/20261016-source-facts-$SLURM_JOB_ID
echo "host=$(hostname) job=$SLURM_JOB_ID start=$(date -u +%FT%TZ)"
mkdir -p "$OUT"

$PY -I -B $EP/outputs/code/check_bart_event_construction.py \
  --used-root $FL/input/ds000001 \
  --source-root $FL/analyses/ds000001/sourcedata/raw \
  --output "$OUT/bart_event_construction.json"
echo "events_check_exit=$?"

$PY -I -B $EP/outputs/code/reconstruct_neuroeffect_candidates.py \
  --partitions $NB/subject_partitions.json \
  --release-metadata $NB/release_metadata.json \
  --design-receipt $NB/inputs/design_sidecars.runtime.json \
  --metadata-census $EP/outputs/runs/source_handoff_qualification/20261003-metadata01/source_metadata_report.json \
  --analysis-root $FL/analyses \
  --output "$OUT/neuroeffect_candidate_provenance.json"
echo "census_exit=$? end=$(date -u +%FT%TZ)"
