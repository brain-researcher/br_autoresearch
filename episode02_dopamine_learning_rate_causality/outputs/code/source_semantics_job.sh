#!/bin/bash
# One stage of the October 16 EP02 source-semantics checks, run as a step of
# an existing attended Slurm allocation, e.g.
#   srun --jobid=<id> --overlap -n1 bash outputs/code/source_semantics_job.sh test
# Stages: test | steward | structure | format | semantics <theta>.
set -euo pipefail

STAGE="${1:?stage required}"
: "${SLURM_JOB_ID:?run inside a Slurm allocation}"
EPISODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_DIR="${EPISODE_DIR}/outputs/runs/source_semantics_checks_job_${SLURM_JOB_ID}"
WORK_DIR="${SCRATCH:?}/br_autoresearch/${EPISODE_DIR##*/}/source_semantics_20261016"
export EP02_RELEASED_DEVELOPMENT_ROOT="${WORK_DIR}/development_full"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "${RUN_DIR}"

type module >/dev/null 2>&1 || source "${LMOD_PKG:-/share/software/user/open/lmod/lmod}/init/bash"
module load math python/3.12.1 py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 >/dev/null 2>&1
{ hostname; echo "SLURM_JOB_ID=${SLURM_JOB_ID} stage=${STAGE} $(date -u +%FT%TZ)";
  python3.12 -c 'import sys, numpy, scipy; print(sys.version.split()[0], numpy.__version__, scipy.__version__)'; } >>"${RUN_DIR}/runtime.txt"

cd "${EPISODE_DIR}"
case "${STAGE}" in
  test)
    OUT="${RUN_DIR}/tests.txt"; [[ -e "${OUT}" ]] && OUT="${RUN_DIR}/tests_retry.txt"
    timeout --signal=TERM --kill-after=5s 300s python3.12 -B -m unittest discover -s outputs/tests \
      -p 'test_check_source_semantics.py' -v >"${OUT}" 2>&1 ;;
  steward)
    # Steward role: the only step that opens the trusted mixed-role MAT.
    timeout --signal=TERM --kill-after=5s 1200s python3.12 -B outputs/code/materialize_development_pack.py \
      --mat /oak/stanford/groups/russpold/data/br_autoresearch_data/dudman_learning_rate_2023/figshare-21816054-v1/source/seshMerge.mat \
      --cohort-map COHORT_MAP.json --development-dir "${EP02_RELEASED_DEVELOPMENT_ROOT}" \
      --expected-bytes 271014539 --expected-md5 23b0b229d92ab9bf26ab9989946eafb4 \
      --expected-sha256 1039b9555ef50b80e01cc923b372a7960daade774fdb6a236c34cafcedf75866 \
      --expected-cohort-map-sha256 0a29751c0b26fe453fc7cb556173544586bffb53ccbc09efd79fc0b52cd0cf90 \
      --observed-on 2026-10-16 >"${RUN_DIR}/steward.json" 2>"${RUN_DIR}/steward.stderr" ;;
  structure|format|semantics)
    # Candidate role: reads only the released development pack via the reader.
    EXTRA=(); [[ "${STAGE}" == semantics ]] && EXTRA=(--peak-threshold "${2:?theta from the format stage}")
    timeout --signal=TERM --kill-after=5s 1200s python3.12 -B outputs/code/check_source_semantics.py "${STAGE}" \
      --development-root "${EP02_RELEASED_DEVELOPMENT_ROOT}" --output "${RUN_DIR}/${STAGE}.json" ${EXTRA[@]+"${EXTRA[@]}"} \
      2>"${RUN_DIR}/${STAGE}.stderr" ;;
  *) echo "unknown stage ${STAGE}" >&2; exit 2 ;;
esac
echo "${STAGE}_exit=0 $(date -u +%FT%TZ)" >>"${RUN_DIR}/status.txt"
