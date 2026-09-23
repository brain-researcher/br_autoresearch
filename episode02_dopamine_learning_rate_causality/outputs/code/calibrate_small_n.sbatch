#!/bin/bash
#SBATCH --job-name=ep02-small-n
#SBATCH --account=russpold
#SBATCH --partition=normal
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

EPISODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUNTIME_DIR="${SCRATCH:?SCRATCH must be set}/br_autoresearch/${EPISODE_DIR##*/}/generic-calibration-${SLURM_JOB_ID:?}"
OBSERVED_ON="${EP02_OBSERVED_ON:-$(date -u +%F)}"

mkdir -p "${RUNTIME_DIR}/output"
exec >"${RUNTIME_DIR}/job.log" 2>&1

"${PYTHON_BIN}" "${EPISODE_DIR}/outputs/code/calibrate_small_n.py" \
  --output-dir "${RUNTIME_DIR}/output" \
  --selection-draws 50000 \
  --validation-draws 50000 \
  --batch-size 1000 \
  --selection-seed 20260922 \
  --validation-seed 120260922 \
  --observed-on "${OBSERVED_ON}"

EP02_GENERIC_CALIBRATION_DIR="${RUNTIME_DIR}/output" "${PYTHON_BIN}" -m unittest discover \
  -s "${EPISODE_DIR}/outputs/tests" \
  -p 'test_small_n_calibration.py' \
  -v
