#!/bin/bash
#SBATCH --job-name=ep12-transition-publish
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency/slurm-transition-publish-%j.out

set -euo pipefail

EP12_ROOT=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency
EP12_EXECUTOR_ROOT=${EP12_ROOT}/outputs/executor
EP12_SCRATCH=/scratch/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency
: "${EP12_TRANSITION_SYNTHETIC_INVARIANCE:?export the authenticated comparison.v1 artifact path}"

case "${EP12_TRANSITION_SYNTHETIC_INVARIANCE}" in
  "${EP12_ROOT}"/outputs/*) ;;
  *) echo "synthetic invariance evidence must stay under EP12 outputs" >&2; exit 2 ;;
esac
if [[ ! -f "${EP12_TRANSITION_SYNTHETIC_INVARIANCE}" || \
      -L "${EP12_TRANSITION_SYNTHETIC_INVARIANCE}" ]]; then
  echo "synthetic invariance evidence is missing, nonregular, or a symlink" >&2
  exit 2
fi

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
if [[ "$(command -v python3 || true)" != "/share/software/user/open/python/3.12.1/bin/python3" ]] || \
   [[ "$(python3 -c 'import sys; print(sys.version.split()[0])')" != "3.12.1" ]]; then
  echo "validated Python 3.12.1 is unavailable" >&2
  exit 2
fi

EP12_TRANSITION_TMP=${EP12_SCRATCH}/transition_publish/tmp
EP12_TRANSITION_PYCACHE_PARENT=${EP12_SCRATCH}/transition_publish/pycache
mkdir -p "${EP12_TRANSITION_TMP}" "${EP12_TRANSITION_PYCACHE_PARENT}"
for EP12_PRIVATE_DIRECTORY in \
  "${EP12_TRANSITION_TMP}" \
  "${EP12_TRANSITION_PYCACHE_PARENT}"; do
  if [[ ! -d "${EP12_PRIVATE_DIRECTORY}" || -L "${EP12_PRIVATE_DIRECTORY}" ]]; then
    echo "unsafe transition-publication Scratch directory" >&2
    exit 2
  fi
done
export TMPDIR="${EP12_TRANSITION_TMP}"
export PYTHONPYCACHEPREFIX
PYTHONPYCACHEPREFIX="$(mktemp -d "${EP12_TRANSITION_PYCACHE_PARENT}/job-${SLURM_JOB_ID:-nojob}-XXXXXX")"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export PYTHONNOUSERSITE=1
export LANG=C
export LC_ALL=C
export TZ=UTC
export PYTHONPATH="${EP12_EXECUTOR_ROOT}"

python3 -B -m ep12_executor.scientific_transition_publish \
  --episode-root "${EP12_ROOT}" \
  --executor-root "${EP12_EXECUTOR_ROOT}" \
  --synthetic-invariance "${EP12_TRANSITION_SYNTHETIC_INVARIANCE}"
