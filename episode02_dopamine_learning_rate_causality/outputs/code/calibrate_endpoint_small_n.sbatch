#!/bin/bash
#SBATCH --job-name=ep02_endpoint_cal
#SBATCH --account=russpold
#SBATCH --partition=normal
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=endpoint-calibration-%j.out
#SBATCH --error=endpoint-calibration-%j.err

set -euo pipefail

EPISODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUNTIME_DIR="${SCRATCH:?SCRATCH must be set}/autoresearch/episode02_dopamine_learning_rate_causality/endpoint-calibration-${SLURM_JOB_ID:?}"

mkdir -p "${RUNTIME_DIR}/output" "${RUNTIME_DIR}/receipts"

"${PYTHON_BIN}" "${EPISODE_DIR}/outputs/code/calibrate_endpoint_small_n.py" \
  --output-directory "${RUNTIME_DIR}/output" \
  --receipt-directory "${RUNTIME_DIR}/receipts" \
  --signoff "${RUNTIME_DIR}/scientist-signoff.json" \
  --draws 20000 \
  --batch-size 1000

"${PYTHON_BIN}" -m unittest discover \
  -s "${EPISODE_DIR}/outputs/tests" \
  -p 'test_endpoint_small_n_calibration.py' \
  -v
