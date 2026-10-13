#!/bin/bash
# Run one coordinator-selected phase inside the approved attended allocation.
# Usage: bash outputs/code/run_low_level_control.sh representative|full SECONDS OUTPUT_DIR
set -euo pipefail
[[ $# == 3 ]] || { echo 'Expected MODE INTERNAL_SECONDS FRESH_OUTPUT_DIR' >&2; exit 2; }
EP04_LOW_MODE=$1
EP04_LOW_SECONDS=$2
EP04_LOW_OUT=$3
[[ "$EP04_LOW_MODE" == representative || "$EP04_LOW_MODE" == full ]]
[[ "$EP04_LOW_SECONDS" =~ ^[1-9][0-9]*$ ]]
if [[ "$EP04_LOW_MODE" == representative ]]; then [[ "$EP04_LOW_SECONDS" -le 300 ]]; fi
EP04_LOW_HOST=$(hostname)
printf 'HOST=%s JOB=%s CPUS=%s MODE=%s INTERNAL_SECONDS=%s\n' \
  "$EP04_LOW_HOST" "${SLURM_JOB_ID:?}" "${SLURM_CPUS_PER_TASK:?}" "$EP04_LOW_MODE" "$EP04_LOW_SECONDS"
[[ "$EP04_LOW_HOST" != *-ln* && "$SLURM_CPUS_PER_TASK" == 1 ]]
cd "${OAK:?}/users/zijiao/br_autoresearch/episode04_shared_scene_geometry"
module purge
module load devel python/3.12.1 py-h5py/3.10.0_py312 math py-numpy/1.26.3 viz py-pillow/10.2.0_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
EP04_LOW_SOURCE="$OAK/users/zijiao/br_autoresearch_data/steward_acquisition/laion_fmri_raw_stimuli_dua_20260922/payload"
exec timeout --signal=TERM --kill-after=10s "$((EP04_LOW_SECONDS + 10))s" \
  python3.12 -u -B outputs/code/prepare_low_level_proposal.py \
  --metadata-dir outputs/runs/source_readiness/20261003-metadata_folds04 \
  --raw-metadata "$EP04_LOW_SOURCE/task-images_metadata.csv" \
  --raw-h5 "$EP04_LOW_SOURCE/task-images_stimuli.h5" \
  --mode "$EP04_LOW_MODE" --max-seconds "$EP04_LOW_SECONDS" \
  --output-dir "$EP04_LOW_OUT"
