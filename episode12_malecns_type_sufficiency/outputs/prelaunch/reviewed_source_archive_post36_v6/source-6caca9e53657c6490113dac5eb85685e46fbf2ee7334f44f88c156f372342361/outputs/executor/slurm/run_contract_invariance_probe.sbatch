#!/bin/bash
#SBATCH --job-name=ep12-contract-probe
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --output=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency/outputs/prelaunch/slurm-contract-probe-%j.out

set -euo pipefail

EP12_ROOT=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency
EP12_SCRATCH=/scratch/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency
EP12_PROBE_PROGRAM=${EP12_ROOT}/outputs/executor/ep12_executor/contract_invariance_probe.py
EP12_PROBE_SPEC=${EP12_ROOT}/outputs/executor/CONTRACT_TRANSITION_PROBE_SPEC.json

: "${EP12_PROBE_PHASE:?export EP12_PROBE_PHASE as legacy or post36}"
: "${EP12_PROBE_SOURCE_ROOT:?export the explicit extracted or frozen executor root}"
: "${EP12_PROBE_OUTPUT:?export the durable probe JSON output path}"

case "${EP12_PROBE_PHASE}" in
  legacy|post36) ;;
  *) echo "EP12_PROBE_PHASE must be legacy or post36" >&2; exit 2 ;;
esac

case "${EP12_PROBE_SOURCE_ROOT}" in
  "${EP12_ROOT}"/outputs/*|"${EP12_SCRATCH}"/*) ;;
  *) echo "EP12_PROBE_SOURCE_ROOT must remain under EP12 outputs or scratch" >&2; exit 2 ;;
esac
case "${EP12_PROBE_OUTPUT}" in
  "${EP12_ROOT}"/outputs/*) ;;
  *) echo "EP12_PROBE_OUTPUT must be a durable EP12 outputs path" >&2; exit 2 ;;
esac

mkdir -p "${EP12_SCRATCH}/contract_transition_probe/tmp"
mkdir -p "$(dirname "${EP12_PROBE_OUTPUT}")"

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
  py-scikit-learn/1.5.1_py312
command -v python3 >/dev/null 2>&1 || {
  echo "python3 is unavailable after the frozen Sherlock module load" >&2
  exit 2
}

export BLIS_NUM_THREADS=1
export LANG=C
export LC_ALL=C
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export PYTHONNOUSERSITE=1
export TZ=UTC
export VECLIB_MAXIMUM_THREADS=1
export TMPDIR="${EP12_SCRATCH}/contract_transition_probe/tmp"

python3 "${EP12_PROBE_PROGRAM}" run \
  --phase "${EP12_PROBE_PHASE}" \
  --source-executor-root "${EP12_PROBE_SOURCE_ROOT}" \
  --spec "${EP12_PROBE_SPEC}" \
  --output "${EP12_PROBE_OUTPUT}"

python3 "${EP12_PROBE_PROGRAM}" verify \
  --artifact "${EP12_PROBE_OUTPUT}" \
  --spec "${EP12_PROBE_SPEC}"
