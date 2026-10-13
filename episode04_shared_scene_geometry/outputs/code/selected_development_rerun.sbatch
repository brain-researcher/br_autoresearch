#!/bin/bash
# Conditional preparation only. Root submits after an actual selected decision.
# Array mode: --array=1-24%24 --time=01:00:00, argument unit.
# Dependent finish mode: afterok on that array, argument finish; slot0 + collection.
# One substantive cell accompanies final collection; no collect-only allocation.
#SBATCH --job-name=ep04-selected-rerun
#SBATCH --partition=russpold,normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:30:00
#SBATCH --no-requeue
#SBATCH --output=outputs/code/selected-rerun-%A_%a.out
#SBATCH --error=outputs/code/selected-rerun-%A_%a.err
set -euo pipefail
[[ $# == 1 ]] || exit 2
ep04_rerun_mode=$1
case "$ep04_rerun_mode" in
  unit)
    [[ "${SLURM_ARRAY_TASK_ID-}" =~ ^([1-9]|1[0-9]|2[0-4])$ ]] || exit 2
    ep04_rerun_slot=$SLURM_ARRAY_TASK_ID
    ;;
  finish)
    [[ -z "${SLURM_ARRAY_JOB_ID-}" ]] || exit 2
    ep04_rerun_slot=0
    ;;
  *) exit 2 ;;
esac
ep04_rerun_host=$(hostname)
printf 'HOST=%s JOB=%s CPUS=%s MEM_MIB=%s MODE=%s SLOT=%s\n' \
  "$ep04_rerun_host" "${SLURM_JOB_ID:?}" "${SLURM_CPUS_PER_TASK:?}" \
  "${SLURM_MEM_PER_NODE:?}" "$ep04_rerun_mode" "$ep04_rerun_slot"
[[ "$ep04_rerun_host" != *-ln* && "$SLURM_CPUS_PER_TASK" == 1 \
   && "$SLURM_MEM_PER_NODE" == 4096 && "${SLURM_JOB_CPUS_PER_NODE:?}" == 1 ]]
cd "${OAK:?}/users/zijiao/br_autoresearch/episode04_shared_scene_geometry"
source /share/software/user/open/lmod/lmod/init/bash
module purge
module load devel python/3.12.1 math py-numpy/1.26.3 py-scipy/1.12.0_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
bash -n outputs/code/selected_development_rerun.sbatch
timeout --signal=TERM --kill-after=10s 60s python3.12 -B -m unittest discover \
  -s outputs/tests -p test_selected_development_rerun.py -v
timeout --signal=TERM --kill-after=10s 3330s python3.12 -u -B \
  outputs/code/run_selected_development_rerun.py unit \
  --slot "$ep04_rerun_slot" --max-seconds 3300
if [[ "$ep04_rerun_mode" == finish ]]; then
  exec timeout --signal=TERM --kill-after=10s 610s python3.12 -u -B \
    outputs/code/run_selected_development_rerun.py collect --max-seconds 600
fi
