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

# This entrypoint uses the fail-closed production runtime provider. Its bundle
# must be materialized only after the observed schedule, selected per-(type,side)
# T/U generators, expansion plan, seed manifest, and procedure lock exist:
#
#   # The authenticated dispatcher exports the bundle/work/lock identities.
#   array_id=$(sbatch --parsable --array=0-98 run_full_search_null.sbatch replicate)
#   sbatch --dependency=afterok:${array_id} run_full_search_null.sbatch aggregate
#
# Each trial is an immutable hash-chained checkpoint. A batch-shell-only signal
# requests a stop at the next trial boundary, after which this exact array task
# is requeued under the same Slurm identity and resumes after replay verification.
# The 48-hour normal-partition allocation sends that signal 13 hours before its
# limit, leaving one complete 12-hour trial allowance plus a checkpoint reserve.
# Aggregate mode fails unless all 99 authenticated checkpoints exist. Creating
# this file does not submit work and cannot access final connectivity.
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "usage: sbatch [--array=0-98] $0 {replicate|aggregate}" >&2
  exit 2
fi

for INHERITED_EP12_NAME in $(compgen -A variable EP12_ || true); do
  case "${INHERITED_EP12_NAME}" in
    EP12_FULL_NULL_RUNTIME_PROVIDER|EP12_FULL_NULL_BUNDLE|EP12_FULL_NULL_BUNDLE_SHA256|EP12_FULL_NULL_WORK_ROOT|EP12_FULL_NULL_DURABLE_ROOT|EP12_PRE_NULL_PROCEDURE_LOCK_PATH|EP12_PRE_NULL_PROCEDURE_LOCK_SHA256|EP12_PRE_NULL_PROCEDURE_LOCK_FILE_SHA256|EP12_SOURCE_FREEZE_MANIFEST|EP12_SOURCE_FREEZE_MANIFEST_SHA256|EP12_FULL_NULL_LAUNCH_PERMIT|EP12_FULL_NULL_LAUNCH_PERMIT_SHA256|EP12_FULL_NULL_MAIN_WRAPPER_SHA256|EP12_SECURE_PUBLISH_LOCK_FD|EP12_SECURE_PUBLISH_LOCK_PATH) ;;
    *)
      echo "refusing ambient EP12 override: ${INHERITED_EP12_NAME}" >&2
      exit 2
      ;;
  esac
done
: "${EP12_FULL_NULL_BUNDLE:?set to the authenticated runtime bundle}"
: "${EP12_FULL_NULL_BUNDLE_SHA256:?set to the authenticated bundle identity}"
: "${EP12_FULL_NULL_WORK_ROOT:?set to the dedicated EP12 scratch work root}"
: "${EP12_FULL_NULL_DURABLE_ROOT:?set to the locked EP12 durable output root}"
: "${EP12_PRE_NULL_PROCEDURE_LOCK_PATH:?set to the upstream lock file}"
: "${EP12_PRE_NULL_PROCEDURE_LOCK_SHA256:?set to the upstream lock identity}"
: "${EP12_PRE_NULL_PROCEDURE_LOCK_FILE_SHA256:?set to the upstream lock bytes identity}"
: "${EP12_SOURCE_FREEZE_MANIFEST:?source-freeze manifest is required}"
: "${EP12_SOURCE_FREEZE_MANIFEST_SHA256:?source-freeze file SHA-256 is required}"
: "${EP12_FULL_NULL_LAUNCH_PERMIT:?authorization-bound launch permit is required}"
: "${EP12_FULL_NULL_LAUNCH_PERMIT_SHA256:?launch-permit file SHA-256 is required}"
: "${EP12_FULL_NULL_RUNTIME_PROVIDER:?runtime provider is required}"
if [[ "${EP12_FULL_NULL_RUNTIME_PROVIDER}" != "ep12_executor.full_search_runtime:load_runtime_from_environment" ]]; then
  echo "full-null runtime provider differs from the frozen implementation" >&2
  exit 2
fi
if [[ ! "${EP12_SOURCE_FREEZE_MANIFEST_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "source-freeze file SHA-256 is malformed" >&2
  exit 2
fi

EP12_MODE=$1
EP12_ROOT=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency
EP12_SCRATCH=/scratch/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency
EP12_WORK_ROOT="${EP12_FULL_NULL_WORK_ROOT}"
EP12_DURABLE_ROOT="${EP12_FULL_NULL_DURABLE_ROOT}"

cd "${EP12_ROOT}"

export MODULEPATH=/share/software/modules/system:/home/groups/russpold/modules:/home/groups/russpold/neurodesk/local/containers/modules:/share/software/modules/math:/share/software/modules/devel:/share/software/modules/categories
export LMOD_SYSTEM_DEFAULT_MODULES=devel,math
if ! type module >/dev/null 2>&1; then
  LMOD_INIT=/share/software/user/open/lmod/lmod/init/bash
  if [[ ! -r "${LMOD_INIT}" || ! -f "${LMOD_INIT}" || -L "${LMOD_INIT}" ]]; then
    echo "validated Lmod initialization is unavailable: ${LMOD_INIT}" >&2
    exit 2
  fi
  source "${LMOD_INIT}"
fi
type module >/dev/null 2>&1 || { echo "Lmod initialization failed" >&2; exit 2; }
module purge
unset PYTHONPATH
module load python/3.12.1 \
  py-numpy/1.26.3_py312 \
  py-scipy/1.12.0_py312 \
  py-scikit-learn/1.5.1_py312 \
  py-pandas/2.2.1_py312 \
  py-pyarrow/18.1.0_py312
EP12_MODULE_PYTHONPATH="${PYTHONPATH:-}"
EP12_PYTHON_BIN="$(command -v python3 || true)"
if [[ "${EP12_PYTHON_BIN}" != "/share/software/user/open/python/3.12.1/bin/python3" ]] || \
   [[ "$(python3 -c 'import sys; print(sys.version.split()[0])')" != "3.12.1" ]]; then
  echo "validated Python 3.12.1 is unavailable" >&2
  exit 2
fi

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TMPDIR="${EP12_SCRATCH}"
export EP12_FULL_NULL_RUNTIME_PROVIDER
export EP12_FULL_NULL_WORK_ROOT="${EP12_WORK_ROOT}"
EP12_SOURCE_STAGE_PARENT="${EP12_SCRATCH}/source_stages"
EP12_PYCACHE_PARENT="${EP12_SCRATCH}/source_pycache"
mkdir -p "${EP12_SOURCE_STAGE_PARENT}" "${EP12_PYCACHE_PARENT}"
for EP12_PRIVATE_PARENT in "${EP12_SOURCE_STAGE_PARENT}" "${EP12_PYCACHE_PARENT}"; do
  if [[ ! -d "${EP12_PRIVATE_PARENT}" || -L "${EP12_PRIVATE_PARENT}" ]]; then
    echo "unsafe private Scratch directory: ${EP12_PRIVATE_PARENT}" >&2
    exit 2
  fi
done
EP12_PYCACHE_ROOT="$(mktemp -d "${EP12_PYCACHE_PARENT}/job-${SLURM_JOB_ID:-nojob}-XXXXXX")"
export PYTHONPYCACHEPREFIX="${EP12_PYCACHE_ROOT}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${EP12_ROOT}/outputs/executor${EP12_MODULE_PYTHONPATH:+:${EP12_MODULE_PYTHONPATH}}"
EP12_FROZEN_SOURCE_ROOT="$(python3 -B -m ep12_executor.source_freeze stage \
  --manifest "${EP12_SOURCE_FREEZE_MANIFEST}" \
  --expected-file-sha256 "${EP12_SOURCE_FREEZE_MANIFEST_SHA256}" \
  --stage-parent "${EP12_SOURCE_STAGE_PARENT}")"
case "${EP12_FROZEN_SOURCE_ROOT}" in
  "${EP12_SOURCE_STAGE_PARENT}"/source-*) ;;
  *) echo "source stage escaped its private parent" >&2; exit 2 ;;
esac
if [[ ! -d "${EP12_FROZEN_SOURCE_ROOT}/outputs/executor" || -L "${EP12_FROZEN_SOURCE_ROOT}" ]]; then
  echo "source stage is missing or unsafe" >&2
  exit 2
fi
export PYTHONPATH="${EP12_FROZEN_SOURCE_ROOT}/outputs/executor${EP12_MODULE_PYTHONPATH:+:${EP12_MODULE_PYTHONPATH}}"

python3 -m ep12_executor.full_null_launch_permit verify \
  --permit "${EP12_FULL_NULL_LAUNCH_PERMIT}" \
  --expected-file-sha256 "${EP12_FULL_NULL_LAUNCH_PERMIT_SHA256}" \
  --mode "${EP12_MODE}"

case "${EP12_MODE}" in
  replicate)
    : "${SLURM_ARRAY_TASK_ID:?replicate mode requires sbatch --array=0-98}"
    : "${SLURM_ARRAY_JOB_ID:?replicate mode requires an array job identity}"
    if [[ ! "${SLURM_ARRAY_TASK_ID}" =~ ^[0-9]+$ ]] || \
       (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID > 98 )); then
      echo "replicate array index must be in 0..98" >&2
      exit 2
    fi
    if [[ ! "${SLURM_ARRAY_JOB_ID}" =~ ^[0-9]+$ ]]; then
      echo "replicate array job identity must be numeric" >&2
      exit 2
    fi
    EP12_RESTART_COUNT="${SLURM_RESTART_COUNT:-0}"
    if [[ ! "${EP12_RESTART_COUNT}" =~ ^[0-9]+$ ]]; then
      echo "Slurm restart count must be a nonnegative integer" >&2
      exit 2
    fi
    EP12_RESTART_TOKEN=ep12-safe-requeue-request-v1
    EP12_RESTART_ROOT="${EP12_SCRATCH}/restart_requests"
    mkdir -p "${EP12_RESTART_ROOT}"
    if [[ -L "${EP12_RESTART_ROOT}" || ! -d "${EP12_RESTART_ROOT}" ]]; then
      echo "restart request root is unsafe" >&2
      exit 2
    fi
    EP12_RESTART_SENTINEL="${EP12_RESTART_ROOT}/full-null-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}-restart-${EP12_RESTART_COUNT}.request"
    if [[ -e "${EP12_RESTART_SENTINEL}" || -L "${EP12_RESTART_SENTINEL}" ]]; then
      echo "restart sentinel already exists for this Slurm incarnation" >&2
      exit 2
    fi
    ep12_request_restart() {
      if [[ -e "${EP12_RESTART_SENTINEL}" || -L "${EP12_RESTART_SENTINEL}" ]]; then
        if [[ -f "${EP12_RESTART_SENTINEL}" && ! -L "${EP12_RESTART_SENTINEL}" ]] && \
           [[ "$(<"${EP12_RESTART_SENTINEL}")" == "${EP12_RESTART_TOKEN}" ]]; then
          return 0
        fi
        echo "restart sentinel changed unexpectedly" >&2
        return 1
      fi
      (
        umask 077
        set -o noclobber
        printf '%s\n' "${EP12_RESTART_TOKEN}" >"${EP12_RESTART_SENTINEL}"
      )
    }
    trap ep12_request_restart SIGUSR1
    set +e
    python3 -m ep12_executor.full_search_null \
      --runtime-provider "${EP12_FULL_NULL_RUNTIME_PROVIDER}" \
      --policy "${EP12_ROOT}/SEARCH_POLICY.yaml" \
      --output-dir "${EP12_WORK_ROOT}" \
      --replicate-index "${SLURM_ARRAY_TASK_ID}" \
      --restart-sentinel "${EP12_RESTART_SENTINEL}" &
    EP12_PYTHON_PID=$!
    while true; do
      wait "${EP12_PYTHON_PID}"
      EP12_REPLICATE_RC=$?
      if (( EP12_REPLICATE_RC > 128 )) && \
         [[ -f "${EP12_RESTART_SENTINEL}" ]]; then
        continue
      fi
      break
    done
    set -e
    trap - SIGUSR1
    if (( EP12_REPLICATE_RC == 75 )); then
      if [[ ! -f "${EP12_RESTART_SENTINEL}" || \
            -L "${EP12_RESTART_SENTINEL}" ]] || \
         [[ "$(<"${EP12_RESTART_SENTINEL}")" != "${EP12_RESTART_TOKEN}" ]]; then
        echo "refusing requeue without the authenticated restart sentinel" >&2
        exit 2
      fi
      scontrol requeue "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
      exit 0
    fi
    exit "${EP12_REPLICATE_RC}"
    ;;
  aggregate)
    if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
      echo "aggregate mode must be a non-array afterok dependency job" >&2
      exit 2
    fi
    python3 -m ep12_executor.full_search_null \
      --runtime-provider "${EP12_FULL_NULL_RUNTIME_PROVIDER}" \
      --policy "${EP12_ROOT}/SEARCH_POLICY.yaml" \
      --output-dir "${EP12_WORK_ROOT}" \
      --aggregate-only
    EP12_WORK_MANIFEST="${EP12_WORK_ROOT}/manifest.json"
    mkdir -p "$(dirname "${EP12_DURABLE_ROOT}")"
    exec python3 -m ep12_executor.secure_publish_lock exec \
      --lock-path "${EP12_DURABLE_ROOT}.publish.lock" -- \
      /bin/bash "$0" publish-aggregate
    ;;
  publish-aggregate)
    if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
      echo "aggregate publication must not run as an array task" >&2
      exit 2
    fi
    : "${EP12_SECURE_PUBLISH_LOCK_FD:?secure publication lock descriptor is missing}"
    : "${EP12_SECURE_PUBLISH_LOCK_PATH:?secure publication lock path is missing}"
    if [[ "${EP12_SECURE_PUBLISH_LOCK_PATH}" != "${EP12_DURABLE_ROOT}.publish.lock" ]]; then
      echo "secure publication lock path differs from the durable target" >&2
      exit 2
    fi
    python3 -m ep12_executor.secure_publish_lock verify \
      --lock-path "${EP12_SECURE_PUBLISH_LOCK_PATH}" \
      --fd "${EP12_SECURE_PUBLISH_LOCK_FD}"
    EP12_WORK_MANIFEST="${EP12_WORK_ROOT}/manifest.json"
    if [[ -d "${EP12_DURABLE_ROOT}" ]]; then
      python3 -m ep12_executor.full_search_null \
        --runtime-provider "${EP12_FULL_NULL_RUNTIME_PROVIDER}" \
        --policy "${EP12_ROOT}/SEARCH_POLICY.yaml" \
        --output-dir "${EP12_DURABLE_ROOT}" \
        --aggregate-only
      if ! cmp -s "${EP12_WORK_MANIFEST}" "${EP12_DURABLE_ROOT}/manifest.json"; then
        echo "existing durable aggregate differs from authenticated scratch" >&2
        exit 2
      fi
      exit 0
    fi
    if [[ -e "${EP12_DURABLE_ROOT}" ]]; then
      echo "durable publish target exists but is not a directory" >&2
      exit 2
    fi
    EP12_STAGE_ROOT="${EP12_DURABLE_ROOT}.staging-${SLURM_JOB_ID}"
    python3 -m ep12_executor.full_search_null \
      --runtime-provider "${EP12_FULL_NULL_RUNTIME_PROVIDER}" \
      --policy "${EP12_ROOT}/SEARCH_POLICY.yaml" \
      --output-dir "${EP12_WORK_ROOT}" \
      --publish-stage "${EP12_STAGE_ROOT}"
    if ! cmp -s "${EP12_WORK_MANIFEST}" "${EP12_STAGE_ROOT}/manifest.json"; then
      echo "staged aggregate differs from authenticated scratch" >&2
      exit 2
    fi
    mv -T "${EP12_STAGE_ROOT}" "${EP12_DURABLE_ROOT}"
    python3 -m ep12_executor.full_search_null \
      --runtime-provider "${EP12_FULL_NULL_RUNTIME_PROVIDER}" \
      --policy "${EP12_ROOT}/SEARCH_POLICY.yaml" \
      --output-dir "${EP12_DURABLE_ROOT}" \
      --aggregate-only
    if ! cmp -s "${EP12_WORK_MANIFEST}" "${EP12_DURABLE_ROOT}/manifest.json"; then
      echo "published aggregate differs from authenticated scratch" >&2
      exit 2
    fi
    ;;
  *)
    echo "mode must be replicate or aggregate" >&2
    exit 2
    ;;
esac
