#!/bin/bash
#SBATCH --job-name=ep02-dudman-p0
#SBATCH --account=russpold
#SBATCH --partition=normal
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

EPISODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SOURCE_MAT="${DUDMAN_SOURCE_MAT:?DUDMAN_SOURCE_MAT must name the checksum-pinned MAT file}"
RUNTIME_DIR="${SCRATCH:?SCRATCH must be set}/br_autoresearch/${EPISODE_DIR##*/}/phase0-${SLURM_JOB_ID:?}"
OBSERVED_ON="${EP02_OBSERVED_ON:-$(date -u +%F)}"

mkdir -p "${RUNTIME_DIR}/output"
exec >"${RUNTIME_DIR}/job.log" 2>&1

"${PYTHON_BIN}" "${EPISODE_DIR}/outputs/code/phase0_inventory.py" \
  --mat "${SOURCE_MAT}" \
  --cohort-map "${EPISODE_DIR}/COHORT_MAP.json" \
  --output-dir "${RUNTIME_DIR}/output" \
  --expected-bytes 271014539 \
  --expected-md5 23b0b229d92ab9bf26ab9989946eafb4 \
  --expected-sha256 1039b9555ef50b80e01cc923b372a7960daade774fdb6a236c34cafcedf75866 \
  --observed-on "${OBSERVED_ON}"

EP02_PHASE0_PACKET_DIR="${RUNTIME_DIR}/output" "${PYTHON_BIN}" -m unittest discover \
  -s "${EPISODE_DIR}/outputs/tests" \
  -p 'test_phase0_inventory.py' \
  -v
