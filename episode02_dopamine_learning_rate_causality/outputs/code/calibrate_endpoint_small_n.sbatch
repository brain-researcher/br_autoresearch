#!/bin/bash
#SBATCH --job-name=ep02_endpoint_cal
#SBATCH --account=russpold
#SBATCH --partition=normal
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

EPISODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUNTIME_DIR="${SCRATCH:?SCRATCH must be set}/br_autoresearch/${EPISODE_DIR##*/}/endpoint-calibration-${SLURM_JOB_ID:?}"
RECEIPT_DIRECTORY="${EP02_RECEIPT_DIRECTORY:-${RUNTIME_DIR}/receipts}"
SIGNOFF_PATH="${EP02_SIGNOFF_PATH:-${RUNTIME_DIR}/scientist-signoff.json}"

mkdir -p "${RUNTIME_DIR}/output" "${RECEIPT_DIRECTORY}"
exec >"${RUNTIME_DIR}/job.log" 2>&1

"${PYTHON_BIN}" "${EPISODE_DIR}/outputs/code/calibrate_endpoint_small_n.py" \
  --output-directory "${RUNTIME_DIR}/output" \
  --receipt-directory "${RECEIPT_DIRECTORY}" \
  --signoff "${SIGNOFF_PATH}" \
  --draws 20000 \
  --batch-size 1000

EP02_ENDPOINT_CALIBRATION_DIR="${RUNTIME_DIR}/output" "${PYTHON_BIN}" -m unittest discover \
  -s "${EPISODE_DIR}/outputs/tests" \
  -p 'test_endpoint_small_n_calibration.py' \
  -v
