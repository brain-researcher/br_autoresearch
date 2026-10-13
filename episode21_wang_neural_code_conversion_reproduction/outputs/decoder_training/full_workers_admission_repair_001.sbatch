#!/bin/bash
# One explicitly reviewed replacement for the unstarted global assignment 7.
# Submitted by the budget helper; this wrapper cannot submit or retry jobs.
#SBATCH --job-name=ep21-full-fit-repair
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --time=08:00:00
#SBATCH --no-requeue
set -euo pipefail
[[ $# -eq 0 ]] || exit 2
ep21_repair_host=$(hostname)
printf 'REPAIR_HOST=%s JOB=%s ARRAY=%s INDEX=%s CPUS=%s MEM_MIB=%s\n' \
  "$ep21_repair_host" "${SLURM_JOB_ID:?}" "${SLURM_ARRAY_JOB_ID:?}" \
  "${SLURM_ARRAY_TASK_ID:?}" "${SLURM_CPUS_PER_TASK:?}" "${SLURM_MEM_PER_NODE:?}"
[[ "$ep21_repair_host" != *-ln* && "$SLURM_ARRAY_TASK_ID" == 7 \
   && "$SLURM_CPUS_PER_TASK" == 1 && "$SLURM_MEM_PER_NODE" == 7168 ]]
cd "${OAK:?}/users/zijiao/br_autoresearch/episode21_wang_neural_code_conversion_reproduction"
source /share/software/user/open/lmod/lmod/init/bash
module purge
module load devel python/3.12.1 math py-numpy/1.26.3_py312 py-h5py/3.10.0_py312 py-scipy/1.12.0_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
export TMPDIR="${L_SCRATCH_JOB:?}"
ep21_repair_check_dir="${PWD}/outputs/decoder_training/admission_repair_checks_job_${SLURM_JOB_ID}"
mkdir "$ep21_repair_check_dir"
(
  cd outputs/decoder_training/tests
  timeout --signal=TERM --kill-after=5s 180s python3.12 -B -m unittest -v \
    test_decoder_array_allocation test_submit_full_decoder_jobs \
    test_reviewed_pre_admission_replacement
) > "$ep21_repair_check_dir/checks.txt" 2>&1
printf 'Required_array_identity_and_reviewed_replacement_checks=passed\n'
# Reuse the unchanged scientific worker and actual-EndTime-minus-300s budget.
exec bash outputs/decoder_training/full_workers_001.sbatch
