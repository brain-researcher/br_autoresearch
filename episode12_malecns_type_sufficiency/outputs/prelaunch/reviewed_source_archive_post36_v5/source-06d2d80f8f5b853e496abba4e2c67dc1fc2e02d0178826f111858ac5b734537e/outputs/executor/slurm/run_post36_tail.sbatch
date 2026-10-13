#!/bin/bash
#SBATCH --job-name=ep12-post36-tail
#SBATCH --partition=normal
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency/outputs/development_run002/slurm-post36-tail-%j.out

# Submit the first invocation only after the production-readiness job succeeds:
#
#   sbatch --dependency=afterany:<readiness-job-id> \
#     outputs/executor/slurm/run_post36_tail.sbatch
#
# Each successful invocation realizes at most one valid slot, checkpoints it,
# and dependency-submits exactly one successor when the controller says
# continue.  Each child reauthenticates the already committed context, so the
# dynamic dependency is afterany and a lost parent-side acknowledgement cannot
# strand an accepted child.  A failed/invalid attempt or unsupported control
# never advances an ordinal and never auto-retries.  This file is an entrypoint
# only; creating it does not submit any job.
set -euo pipefail

for INHERITED_EP12_NAME in $(compgen -A variable EP12_ || true); do
  case "${INHERITED_EP12_NAME}" in
    EP12_SOURCE_FREEZE_MANIFEST|EP12_SOURCE_FREEZE_MANIFEST_SHA256|EP12_SOURCE_STAGE_ROOT) ;;
    *)
      echo "refusing ambient EP12 override: ${INHERITED_EP12_NAME}" >&2
      exit 2
      ;;
  esac
done
: "${EP12_SOURCE_FREEZE_MANIFEST:?source-freeze manifest is required}"
: "${EP12_SOURCE_FREEZE_MANIFEST_SHA256:?source-freeze file SHA-256 is required}"
: "${EP12_SOURCE_STAGE_ROOT:?immutable source-stage root is required}"
if [[ ! "${EP12_SOURCE_FREEZE_MANIFEST_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "source-freeze file SHA-256 is malformed" >&2
  exit 2
fi
EP12_RESTART_COUNT="${SLURM_RESTART_COUNT:-0}"
if [[ ! "${EP12_RESTART_COUNT}" =~ ^[0-9]+$ ]]; then
  echo "Slurm restart count must be a nonnegative integer" >&2
  exit 2
fi

EP12_ROOT=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency
EP12_SCRATCH=/scratch/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency
EP12_RUN_ROOT="${EP12_ROOT}/outputs/development_run002"
EP12_TAIL_ROOT="${EP12_RUN_ROOT}/post36_tail"
EP12_TAIL_SCRATCH="${EP12_SCRATCH}/development_run002/post36_tail"
EP12_TRANSITION_ROOT="${EP12_ROOT}/outputs/prelaunch/scientific_contract_transition"
EP12_HANDOFF_ROOT="${EP12_RUN_ROOT}/slurm_handoffs"
EP12_AUTO_CHAIN=1
EP12_AUTO_LAUNCH_PROCEDURE_LOCK=1
EP12_AUTO_LAUNCH_FULL_NULL=0

mkdir -p "${EP12_TAIL_SCRATCH}"
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
EP12_EXPECTED_SOURCE_STAGE_ROOT="${EP12_SOURCE_STAGE_PARENT}/source-${EP12_SOURCE_FREEZE_MANIFEST_SHA256}"
if [[ "${EP12_SOURCE_STAGE_ROOT}" != "${EP12_EXPECTED_SOURCE_STAGE_ROOT}" || \
      ! -d "${EP12_SOURCE_STAGE_ROOT}/outputs/executor" || \
      -L "${EP12_SOURCE_STAGE_ROOT}" ]]; then
  echo "immutable source-stage root does not match the freeze content address" >&2
  exit 2
fi
export PYTHONPATH="${EP12_SOURCE_STAGE_ROOT}/outputs/executor${EP12_MODULE_PYTHONPATH:+:${EP12_MODULE_PYTHONPATH}}"
EP12_FROZEN_SOURCE_ROOT="$(python3 -B -m ep12_executor.source_freeze stage \
  --manifest "${EP12_SOURCE_FREEZE_MANIFEST}" \
  --expected-file-sha256 "${EP12_SOURCE_FREEZE_MANIFEST_SHA256}" \
  --stage-parent "${EP12_SOURCE_STAGE_PARENT}")"
case "${EP12_FROZEN_SOURCE_ROOT}" in
  "${EP12_SOURCE_STAGE_PARENT}"/source-*) ;;
  *) echo "source stage escaped its private parent" >&2; exit 2 ;;
esac
if [[ "${EP12_FROZEN_SOURCE_ROOT}" != "${EP12_SOURCE_STAGE_ROOT}" ]]; then
  echo "verified source stage differs from the submitted immutable stage" >&2
  exit 2
fi
if [[ ! -d "${EP12_FROZEN_SOURCE_ROOT}/outputs/executor" || -L "${EP12_FROZEN_SOURCE_ROOT}" ]]; then
  echo "source stage is missing or unsafe" >&2
  exit 2
fi
export PYTHONPATH="${EP12_FROZEN_SOURCE_ROOT}/outputs/executor${EP12_MODULE_PYTHONPATH:+:${EP12_MODULE_PYTHONPATH}}"
EP12_PROCEDURE_LOCK_DISPATCHER="${EP12_FROZEN_SOURCE_ROOT}/outputs/executor/slurm/run_procedure_lock.sbatch"

set +e
python3 -m ep12_executor.post36_tail run-one \
  --plan "${EP12_FROZEN_SOURCE_ROOT}/outputs/executor/OBSERVED_POST36_TAIL_PLAN.json" \
  --policy "${EP12_FROZEN_SOURCE_ROOT}/SEARCH_POLICY.yaml" \
  --prefix-contract "${EP12_FROZEN_SOURCE_ROOT}/outputs/executor/SCIENTIFIC_CONTRACT.yaml" \
  --active-contract "${EP12_FROZEN_SOURCE_ROOT}/outputs/executor/SCIENTIFIC_CONTRACT_POST36.yaml" \
  --contract-transition-receipt "${EP12_TRANSITION_ROOT}/receipt.json" \
  --contract-transition-spec "${EP12_FROZEN_SOURCE_ROOT}/outputs/executor/CONTRACT_TRANSITION_SPEC.json" \
  --contract-transition-manifest "${EP12_TRANSITION_ROOT}/manifest.json" \
  --readiness-dir "${EP12_RUN_ROOT}/production_readiness" \
  --run-root "${EP12_RUN_ROOT}" \
  --observed-finalization-dir "${EP12_RUN_ROOT}/observed_finalization" \
  --tail-root "${EP12_TAIL_ROOT}" \
  --snapshot-dir "${EP12_SCRATCH}/development_snapshot_v1" \
  --scratch-root "${EP12_TAIL_SCRATCH}"
EP12_CONTROLLER_RC=$?
set -e

case "${EP12_CONTROLLER_RC}" in
  0)
    # A stop and realized schedule were durably committed.  Hand the separate
    # non-trial gates to their fail-closed dispatcher; this script itself never
    # opens final connectivity.  The dispatcher authenticates eligibility and
    # emits a noneligible dispatch instead of launching a null when required.
    if [[ "${EP12_AUTO_LAUNCH_PROCEDURE_LOCK}" == "1" ]]; then
      if [[ ! -f "${EP12_PROCEDURE_LOCK_DISPATCHER}" ]]; then
        echo "procedure-lock dispatcher is unavailable: ${EP12_PROCEDURE_LOCK_DISPATCHER}" >&2
        exit 5
      fi
      : "${SLURM_JOB_ID:?automatic procedure-lock launch requires a Slurm job ID}"
      python3 -m ep12_executor.slurm_handoff submit \
        --kind post36_tail_to_procedure_lock \
        --parent-job-id "${SLURM_JOB_ID}" \
        --context-root "${EP12_TAIL_ROOT}" \
        --transaction-root "${EP12_HANDOFF_ROOT}" \
        --child-script "${EP12_PROCEDURE_LOCK_DISPATCHER}" \
        --export-env "EP12_SOURCE_FREEZE_MANIFEST=${EP12_SOURCE_FREEZE_MANIFEST},EP12_SOURCE_FREEZE_MANIFEST_SHA256=${EP12_SOURCE_FREEZE_MANIFEST_SHA256},EP12_SOURCE_STAGE_ROOT=${EP12_FROZEN_SOURCE_ROOT},EP12_PROCEDURE_LOCK_LAUNCH_MODE=hold,EP12_AUTO_LAUNCH_FULL_NULL=0"
    fi
    exit 0
    ;;
  10)
    if [[ "${EP12_AUTO_CHAIN}" == "1" ]]; then
      : "${SLURM_JOB_ID:?automatic tail continuation requires a Slurm job ID}"
      python3 -m ep12_executor.slurm_handoff submit \
        --kind post36_tail_to_next_tail \
        --parent-job-id "${SLURM_JOB_ID}" \
        --context-root "${EP12_TAIL_ROOT}" \
        --transaction-root "${EP12_HANDOFF_ROOT}" \
        --child-script "${EP12_FROZEN_SOURCE_ROOT}/outputs/executor/slurm/run_post36_tail.sbatch" \
        --export-env "EP12_SOURCE_FREEZE_MANIFEST=${EP12_SOURCE_FREEZE_MANIFEST},EP12_SOURCE_FREEZE_MANIFEST_SHA256=${EP12_SOURCE_FREEZE_MANIFEST_SHA256},EP12_SOURCE_STAGE_ROOT=${EP12_FROZEN_SOURCE_ROOT}"
      exit 0
    fi
    echo "one slot committed; continuation is required but auto-chain is disabled" >&2
    exit 10
    ;;
  12)
    # A durable wall-deadline stop/pending receipt exists.  Neither another
    # tail slot nor the procedure-lock dispatcher is authorized from it.
    exit 0
    ;;
  3)
    echo "slot attempt failed without advancing; diagnose before resubmission" >&2
    exit 3
    ;;
  4)
    echo "frozen tail is blocked by the recorded capability audit" >&2
    exit 4
    ;;
  *)
    exit "${EP12_CONTROLLER_RC}"
    ;;
esac
