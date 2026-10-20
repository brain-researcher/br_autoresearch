#!/bin/bash
# One stage of the EP02 primary structural gate, run inside an attended Slurm
# allocation, e.g.
#   srun -p dev -c 1 --mem=4G -t 00:30:00 bash outputs/code/primary_structural_gate_job.sh gate
# Stages: astra (ASTRA 0.0.14 validation) | test (synthetic fixtures) | gate (steward).
set -euo pipefail

STAGE="${1:?stage required}"
: "${SLURM_JOB_ID:?run inside a Slurm allocation}"
EPISODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_DIR="${EPISODE_DIR}/outputs/runs/primary_structural_gate_job_${SLURM_JOB_ID}"
WORK_DIR="${SCRATCH:?}/br_autoresearch/${EPISODE_DIR##*/}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "${RUN_DIR}"
type module >/dev/null 2>&1 || source "${LMOD_PKG:-/share/software/user/open/lmod/lmod}/init/bash"
echo "$(hostname) SLURM_JOB_ID=${SLURM_JOB_ID} stage=${STAGE} start=$(date -u +%FT%TZ)" >>"${RUN_DIR}/runtime.txt"

cd "${EPISODE_DIR}"
rc=0
case "${STAGE}" in
  astra)
    # linkml-runtime 1.12 (October 8, 2026) requires pyoxigraph>=0.5.11, which has no glibc-2.17
    # wheel and does not build here; the 1.11 line satisfies astra-tools 0.2.18 (>=1.8) without it.
    # gcc 12 covers any C++17 source build (rapidfuzz) on this OS.
    module load devel uv/0.8.4 python/3.12.1 gcc/12.4.0 cmake/3.31.4 >/dev/null 2>&1
    CACHE="${WORK_DIR}/astra-validator"
    JOBS="${SLURM_CPUS_PER_TASK:-1}"
    mkdir -p "${CACHE}/tmp" "${CACHE}/cache" "${CACHE}/tools"
    TMPDIR="${CACHE}/tmp" UV_CACHE_DIR="${CACHE}/cache" UV_TOOL_DIR="${CACHE}/tools" UV_PYTHON_DOWNLOADS=never \
      CMAKE_BUILD_PARALLEL_LEVEL="${JOBS}" CC=gcc CXX=g++ \
      timeout --signal=TERM --kill-after=5s "${ASTRA_TIMEOUT_SECONDS:-600}s" "$(command -v uvx)" --no-managed-python --no-python-downloads \
      --python "$(command -v python3.12)" --from astra-tools==0.2.18 --with astra-spec==0.0.14 --with 'linkml-runtime<1.12' \
      astra validate outputs/astra/v0.0.14/astra.yaml --skip-evidence >"${RUN_DIR}/astra_validation.txt" 2>&1 || rc=$? ;;
  test)
    module load math python/3.12.1 py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 >/dev/null 2>&1
    timeout --signal=TERM --kill-after=5s 300s python3.12 -B -m unittest discover -s outputs/tests \
      -p 'test_primary_structural_gate.py' -v >"${RUN_DIR}/tests.txt" 2>&1 || rc=$? ;;
  gate)
    # Steward role: the only stage that opens the trusted mixed-role MAT.
    module load math python/3.12.1 py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 >/dev/null 2>&1
    timeout --signal=TERM --kill-after=5s 900s python3.12 -B outputs/code/primary_structural_gate.py \
      --mat /oak/stanford/groups/russpold/data/br_autoresearch_data/dudman_learning_rate_2023/figshare-21816054-v1/source/seshMerge.mat \
      --cohort-map COHORT_MAP.json \
      --development-pack "${WORK_DIR}/source_semantics_20261016/development_full" \
      --output "${RUN_DIR}/gate.json" \
      --expected-bytes 271014539 --expected-md5 23b0b229d92ab9bf26ab9989946eafb4 \
      --expected-sha256 1039b9555ef50b80e01cc923b372a7960daade774fdb6a236c34cafcedf75866 \
      --expected-cohort-map-sha256 0a29751c0b26fe453fc7cb556173544586bffb53ccbc09efd79fc0b52cd0cf90 \
      >"${RUN_DIR}/gate.stdout" 2>"${RUN_DIR}/gate.stderr" || rc=$? ;;
  *) echo "unknown stage ${STAGE}" >&2; exit 2 ;;
esac
echo "${STAGE}_exit=${rc} $(date -u +%FT%TZ)" >>"${RUN_DIR}/status.txt"
exit "${rc}"
