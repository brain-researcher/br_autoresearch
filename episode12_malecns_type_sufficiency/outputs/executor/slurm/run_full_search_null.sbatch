#!/bin/bash
#SBATCH --job-name=ep12-full-null
#SBATCH --partition=normal
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:SIGUSR1@46800
#SBATCH --output=/scratch/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency/full-null-%A_%a.out

# Exact-once launchers use a sanitized environment; initialize Sherlock Lmod
# explicitly before enabling nounset.
# shellcheck source=/etc/profile
. /etc/profile >/dev/null
set -euo pipefail
if [[ "$#" -ne 5 ]]; then
  echo "usage: sbatch --array=0-98 $0 POLICY PROCEDURE_LOCK RUNTIME_BUNDLE WORK_ROOT DURABLE_ROOT" >&2
  exit 2
fi
: "${SLURM_ARRAY_TASK_ID:?replicate wrapper requires an array task}"

EP12_POLICY=$1
EP12_PRE_NULL_PROCEDURE_LOCK_PATH=$2
EP12_FULL_NULL_BUNDLE=$3
EP12_FULL_NULL_WORK_ROOT=$4
EP12_FULL_NULL_DURABLE_ROOT=$5
EP12_ROOT=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency
EP12_SCRATCH=/scratch/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency

cd "${EP12_ROOT}"
module purge
module load python/3.12.1 \
  py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 \
  py-scikit-learn/1.5.1_py312 py-pandas/2.2.1_py312 \
  py-pyarrow/18.1.0_py312
export PYTHONPATH="${EP12_ROOT}/outputs/executor${PYTHONPATH:+:${PYTHONPATH}}"
export EP12_SEARCH_POLICY="${EP12_POLICY}"
export EP12_FULL_NULL_RUNTIME_PROVIDER=ep12_executor.full_search_runtime:load_runtime_from_environment
export EP12_PRE_NULL_PROCEDURE_LOCK_PATH EP12_FULL_NULL_BUNDLE
export EP12_FULL_NULL_WORK_ROOT EP12_FULL_NULL_DURABLE_ROOT
export EP12_PRE_NULL_PROCEDURE_LOCK_SHA256
EP12_PRE_NULL_PROCEDURE_LOCK_SHA256=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["procedure_lock_sha256"])' "${EP12_PRE_NULL_PROCEDURE_LOCK_PATH}")
export EP12_FULL_NULL_BUNDLE_SHA256
EP12_FULL_NULL_BUNDLE_SHA256=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["bundle_sha256"])' "${EP12_FULL_NULL_BUNDLE}/runtime_bundle.json")
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

# BEGIN EP12_INTERRUPTIBLE_CHILD_WAIT
ep12_wait_for_child_after_usr1() {
  local child_pid=$1
  local child_rc
  # Bash trap handlers use dynamic scope, so the SIGUSR1 trap below marks this
  # invocation only when it actually interrupts this wait.
  local EP12_USR1_WAIT_INTERRUPTED=0
  while true; do
    wait "${child_pid}"
    child_rc=$?
    # A trapped SIGUSR1 interrupts Bash's wait with 128 + SIGUSR1 (10). The
    # child remains waitable even if it exits before any subsequent probe, so
    # wait again and recover its real status instead of consulting kill -0.
    if [[ "${child_rc}" -eq 138 && "${EP12_USR1_WAIT_INTERRUPTED}" -eq 1 ]]; then
      EP12_USR1_WAIT_INTERRUPTED=0
      continue
    fi
    return "${child_rc}"
  done
}
# END EP12_INTERRUPTIBLE_CHILD_WAIT

EP12_RESTART_ROOT=${EP12_SCRATCH}/restart_requests
mkdir -p "${EP12_RESTART_ROOT}" "${EP12_FULL_NULL_WORK_ROOT}"
EP12_RESTART_COUNT=${SLURM_RESTART_COUNT:-0}
EP12_RESTART_SENTINEL=${EP12_RESTART_ROOT}/full-null-${SLURM_ARRAY_JOB_ID:-job}_${SLURM_ARRAY_TASK_ID}-${EP12_RESTART_COUNT}.request
EP12_RESTART_SENTINEL_TMP=${EP12_RESTART_SENTINEL}.tmp
ep12_handle_usr1() {
  printf "%s\n" "ep12-safe-requeue-request-v1" > "${EP12_RESTART_SENTINEL_TMP}"
  mv -f -- "${EP12_RESTART_SENTINEL_TMP}" "${EP12_RESTART_SENTINEL}"
  EP12_USR1_WAIT_INTERRUPTED=1
}
trap ep12_handle_usr1 SIGUSR1
set +e
python3 -m ep12_executor.full_search_null \
  --runtime-provider "${EP12_FULL_NULL_RUNTIME_PROVIDER}" \
  --policy "${EP12_POLICY}" \
  --output-dir "${EP12_FULL_NULL_WORK_ROOT}" \
  --replicate-index "${SLURM_ARRAY_TASK_ID}" \
  --restart-sentinel "${EP12_RESTART_SENTINEL}" &
EP12_CHILD_PID=$!
# Bash services SIGUSR1 immediately while waiting for a background child. If
# wait is interrupted by the trap, keep waiting so Python can observe the
# authenticated sentinel at its next trial boundary and exit with RC 75.
ep12_wait_for_child_after_usr1 "${EP12_CHILD_PID}"
EP12_RC=$?
set -e
trap - SIGUSR1
rm -f -- "${EP12_RESTART_SENTINEL_TMP}"
if [[ "${EP12_RC}" -eq 75 ]]; then
  scontrol requeue "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
  exit 0
fi
exit "${EP12_RC}"
