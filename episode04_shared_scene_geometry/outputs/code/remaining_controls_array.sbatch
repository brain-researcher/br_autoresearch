#!/bin/bash
# Preparation only. No guessed partition, memory, walltime, array size or throttle.
# prepare in remaining_controls_tasks.py prints explicit sbatch resource arguments.
# One participant/fold bundle runs once; no submission, retry, collection or successor.
#SBATCH --job-name=ep04-controls
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --no-requeue
set -euo pipefail
[[ $# == 1 ]] || exit 2
ep04_remaining_manifest=$1
ep04_remaining_host=$(hostname)
printf 'HOST=%s JOB=%s ARRAY=%s SLOT=%s CPUS=%s\n' "$ep04_remaining_host" \
  "${SLURM_JOB_ID:?}" "${SLURM_ARRAY_JOB_ID:?}" "${SLURM_ARRAY_TASK_ID:?}" "${SLURM_CPUS_PER_TASK:?}"
[[ "$ep04_remaining_host" != *-ln* && "$SLURM_CPUS_PER_TASK" == 1 ]]
cd "${OAK:?}/users/zijiao/br_autoresearch/episode04_shared_scene_geometry"
source /share/software/user/open/lmod/lmod/init/bash
module purge
module load devel python/3.12.1 math py-numpy/1.26.3 py-scipy/1.12.0_py312 viz py-pillow/10.2.0_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
exec python3.12 -u -B outputs/code/remaining_controls_tasks.py run \
  --manifest "$ep04_remaining_manifest"
