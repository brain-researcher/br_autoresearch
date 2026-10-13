#!/bin/bash
# Complete scientific aggregation and one fixed decision, then report that decision.
# Root supplies afterok on the complete array; no retry or scientific successor.
#SBATCH --job-name=ep04-collect-screen
#SBATCH --partition=russpold,normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:40:00
#SBATCH --no-requeue
#SBATCH --output=outputs/code/remaining_controls_selected/20261005-full01/collection-%j.out
#SBATCH --error=outputs/code/remaining_controls_selected/20261005-full01/collection-%j.err
set -euo pipefail
[[ $# == 0 ]] || exit 2
ep04_final_host=$(hostname)
printf 'HOST=%s JOB=%s CPUS=%s MEM_MIB=%s\n' "$ep04_final_host" \
  "${SLURM_JOB_ID:?}" "${SLURM_CPUS_PER_TASK:?}" "${SLURM_MEM_PER_NODE:?}"
[[ "$ep04_final_host" != *-ln* && "$SLURM_CPUS_PER_TASK" == 1 \
   && "$SLURM_MEM_PER_NODE" == 4096 && "${SLURM_JOB_CPUS_PER_NODE:?}" == 1 ]]
cd "${OAK:?}/users/zijiao/br_autoresearch/episode04_shared_scene_geometry"
source /share/software/user/open/lmod/lmod/init/bash
module purge
module load devel python/3.12.1 math py-numpy/1.26.3 py-scipy/1.12.0_py312 viz py-pillow/10.2.0_py312 py-matplotlib/3.8.3_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${L_SCRATCH_JOB:?}/ep04-complete-panel-figure"
mkdir -p "$MPLCONFIGDIR"
bash -n outputs/code/collect_and_screen_full01.sh
bash -n outputs/code/collect_and_screen_full01.sbatch

ep04_final_astra="${GROUP_HOME:?}/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv/bin/astra-tools"
set +e
timeout --signal=TERM --kill-after=10s 30s "$ep04_final_astra" validate \
  outputs/astra/v0.0.14/astra.yaml --json \
  > "outputs/code/remaining_controls_selected/20261005-full01/collection-astra-${SLURM_JOB_ID}.json" 2>&1
ep04_final_astra_status=$?
set -e
printf 'astra_validation_exit=%s\n' "$ep04_final_astra_status"

# The entrypoint stops dependent stages on scientific non-evaluability or error.
bash outputs/code/collect_and_screen_full01.sh

# Reporting never changes the published decision. Its exit is recorded separately.
set +e
timeout --signal=TERM --kill-after=10s 150s python3.12 -u -B \
  outputs/code/plot_registered_panel_decision.py --max-seconds 120
ep04_final_plot_status=$?
set -e
printf 'decision_figure_exit=%s\n' "$ep04_final_plot_status"
exit 0
