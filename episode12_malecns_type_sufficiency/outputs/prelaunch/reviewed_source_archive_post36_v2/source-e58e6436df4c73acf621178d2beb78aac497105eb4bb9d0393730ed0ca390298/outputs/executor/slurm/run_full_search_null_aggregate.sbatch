#!/bin/bash
#SBATCH --job-name=ep12-full-null-aggregate
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --no-requeue
#SBATCH --output=/scratch/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency/full-null-aggregate-%j.out

# The aggregate verifies 99 immutable checkpoint trees, computes the compact
# summary, and publishes authenticated bytes.  It performs no model fitting and
# has no parallel worker pool, so it uses one CPU.  The 32-GiB/12-hour envelope
# leaves explicit headroom for runtime state, checkpoint metadata, hashing, and
# the final Scratch-to-Oak publication without inheriting replicate-scale
# 32-CPU/192-GiB/48-hour/requeue settings.
set -euo pipefail

if [[ "$#" -ne 1 || "$1" != "aggregate" ]]; then
  echo "usage: sbatch $0 aggregate" >&2
  exit 2
fi
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "full-null aggregate must not run as an array task" >&2
  exit 2
fi

EP12_EXPECTED_MAIN_WRAPPER_SHA256="${EP12_FULL_NULL_MAIN_WRAPPER_SHA256:-}"
if [[ ! "${EP12_EXPECTED_MAIN_WRAPPER_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "full-null aggregate requires the exact dispatcher-frozen main-wrapper SHA-256" >&2
  exit 2
fi

: "${EP12_SOURCE_FREEZE_MANIFEST:?source-freeze manifest is required}"
: "${EP12_SOURCE_FREEZE_MANIFEST_SHA256:?source-freeze file SHA-256 is required}"
: "${EP12_FULL_NULL_LAUNCH_PERMIT:?authorization-bound launch permit is required}"
: "${EP12_FULL_NULL_LAUNCH_PERMIT_SHA256:?launch-permit file SHA-256 is required}"
if [[ ! "${EP12_SOURCE_FREEZE_MANIFEST_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "source-freeze file SHA-256 is malformed" >&2
  exit 2
fi
EP12_ROOT=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency
EP12_SCRATCH=/scratch/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency
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
module load python/3.12.1
EP12_MODULE_PYTHONPATH="${PYTHONPATH:-}"
EP12_PYTHON_BIN="$(command -v python3 || true)"
if [[ "${EP12_PYTHON_BIN}" != "/share/software/user/open/python/3.12.1/bin/python3" ]] || \
   [[ "$(python3 -c 'import sys; print(sys.version.split()[0])')" != "3.12.1" ]]; then
  echo "validated Python 3.12.1 is unavailable" >&2
  exit 2
fi
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
EP12_AGGREGATE_ENTRYPOINT="${EP12_FROZEN_SOURCE_ROOT}/outputs/executor/slurm/run_full_search_null.sbatch"
if [[ ! -f "${EP12_AGGREGATE_ENTRYPOINT}" || -L "${EP12_AGGREGATE_ENTRYPOINT}" ]]; then
  echo "full-null aggregate entrypoint is missing or unsafe" >&2
  exit 2
fi
EP12_OBSERVED_MAIN_WRAPPER_SHA256="$(sha256sum -- "${EP12_AGGREGATE_ENTRYPOINT}" | awk '{print $1}')"
if [[ "${EP12_OBSERVED_MAIN_WRAPPER_SHA256}" != "${EP12_EXPECTED_MAIN_WRAPPER_SHA256}" ]]; then
  echo "full-null aggregate entrypoint differs from the dispatcher-frozen main-wrapper SHA-256" >&2
  exit 2
fi
python3 -m ep12_executor.full_null_launch_permit verify \
  --permit "${EP12_FULL_NULL_LAUNCH_PERMIT}" \
  --expected-file-sha256 "${EP12_FULL_NULL_LAUNCH_PERMIT_SHA256}" \
  --mode aggregate

exec /bin/bash "${EP12_AGGREGATE_ENTRYPOINT}" aggregate
