#!/bin/bash
# Reporting only, within the existing coordinated compute allocation.
set -euo pipefail
[[ $# == 0 && ${SLURM_CPUS_PER_TASK:?} == 1 ]]
ep12_figure_host=$(hostname)
[[ "$ep12_figure_host" != *-ln* && -n ${SLURM_JOB_ID:?} ]]
printf 'HOST=%s JOB=%s STEP=%s\n' "$ep12_figure_host" "$SLURM_JOB_ID" "${SLURM_STEP_ID:?}"
cd "${OAK:?}/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency"
source /share/software/user/open/lmod/lmod/init/bash
module purge
module load devel python/3.12.1 math py-numpy/1.26.3 viz py-matplotlib/3.8.3_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${L_SCRATCH_JOB:?}/ep12-description-figure-${SLURM_STEP_ID}"
mkdir -p "$MPLCONFIGDIR"
exec python3.12 -u -B outputs/tools/plot_observed_outgoing_description.py \
  --output-dir outputs/figures/observed_outgoing_description/20261005-description01 \
  --max-seconds 120
