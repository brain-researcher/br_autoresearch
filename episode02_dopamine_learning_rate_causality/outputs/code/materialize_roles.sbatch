#!/bin/bash
#SBATCH --job-name=ep02-role-mat
#SBATCH --account=russpold
#SBATCH --partition=normal
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

EPISODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SOURCE_MAT="${DUDMAN_SOURCE_MAT:?DUDMAN_SOURCE_MAT must name the checksum-pinned MAT file}"
RUNTIME_DIR="${SCRATCH:?SCRATCH must be set}/br_autoresearch/${EPISODE_DIR##*/}/materialize-${SLURM_JOB_ID:?}"
OBSERVED_ON="${EP02_OBSERVED_ON:-$(date -u +%F)}"

mkdir -p "${RUNTIME_DIR}"
exec >"${RUNTIME_DIR}/job.log" 2>&1

"${PYTHON_BIN}" "${EPISODE_DIR}/outputs/code/materialize_roles.py" \
  --mat "${SOURCE_MAT}" \
  --cohort-map "${EPISODE_DIR}/COHORT_MAP.json" \
  --development-dir "${RUNTIME_DIR}/candidate/development_full" \
  --audit-structural-dir "${RUNTIME_DIR}/candidate/audit_structural" \
  --vault-root "${RUNTIME_DIR}/evaluator" \
  --public-output-dir "${RUNTIME_DIR}/public" \
  --expected-bytes 271014539 \
  --expected-md5 23b0b229d92ab9bf26ab9989946eafb4 \
  --expected-sha256 1039b9555ef50b80e01cc923b372a7960daade774fdb6a236c34cafcedf75866 \
  --expected-cohort-map-sha256 0a29751c0b26fe453fc7cb556173544586bffb53ccbc09efd79fc0b52cd0cf90 \
  --observed-on "${OBSERVED_ON}"

EP02_ROLE_PUBLIC_DIR="${RUNTIME_DIR}/public" \
EP02_ROLE_DEVELOPMENT_DIR="${RUNTIME_DIR}/candidate/development_full" \
EP02_ROLE_STRUCTURAL_DIR="${RUNTIME_DIR}/candidate/audit_structural" \
EP02_ROLE_VAULT_DIR="${RUNTIME_DIR}/evaluator" \
"${PYTHON_BIN}" -m unittest discover \
  -s "${EPISODE_DIR}/outputs/tests" \
  -p 'test_role_materialization.py' \
  -v
