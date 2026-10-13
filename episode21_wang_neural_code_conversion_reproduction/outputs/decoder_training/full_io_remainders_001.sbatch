#!/bin/bash
# One manually registered exact complement of an inspected I/O-failed fragment.
# Existing failed fragments are retained and never opened by this fitting job.
#SBATCH --job-name=ep21-full-workers-001
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --time=08:00:00
#SBATCH --no-requeue
#SBATCH --output=outputs/decoder_training/full_io_remainders_001-%A_%a.out
#SBATCH --error=outputs/decoder_training/full_io_remainders_001-%A_%a.err
set -euo pipefail
[[ $# -eq 0 ]] || exit 2
ep21_io_host=$(hostname)
printf 'IO_REMAINDER_HOST=%s JOB=%s ARRAY=%s INDEX=%s CPUS=%s MEM_MIB=%s\n' \
  "$ep21_io_host" "${SLURM_JOB_ID:?}" "${SLURM_ARRAY_JOB_ID:?}" \
  "${SLURM_ARRAY_TASK_ID:?}" "${SLURM_CPUS_PER_TASK:?}" "${SLURM_MEM_PER_NODE:?}"
[[ "$ep21_io_host" != *-ln* && "$SLURM_CPUS_PER_TASK" == 1 && "$SLURM_MEM_PER_NODE" == 7168 ]]
case "$SLURM_ARRAY_TASK_ID" in
  31|32|46|60|61) ;;
  *) exit 2 ;;
esac
printf -v ep21_io_work 'work_%04d' "$((SLURM_ARRAY_TASK_ID + 1))"
cd "${OAK:?}/users/zijiao/br_autoresearch/episode21_wang_neural_code_conversion_reproduction"
source /share/software/user/open/lmod/lmod/init/bash
module purge
unset PYTHONPATH PYTHONHOME
module load devel python/3.12.1 math py-numpy/1.26.3_py312 py-h5py/3.10.0_py312 py-scipy/1.12.0_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
[[ -n "${L_SCRATCH_JOB:?}" && -d "$L_SCRATCH_JOB" ]]
export TMPDIR="$L_SCRATCH_JOB"
ep21_io_checks="${PWD}/outputs/decoder_training/io_remainder_checks_job_${SLURM_JOB_ID}"
mkdir "$ep21_io_checks"
printf 'job=%s\nparent_charged_once_to=EP21\nlogical_member=%s\n' \
  "$SLURM_JOB_ID" "$ep21_io_work" > "$ep21_io_checks/status.txt"
set +e
(
  cd outputs/decoder_training/tests || exit 2
  timeout --signal=TERM --kill-after=5s 180s python3.12 -B -m unittest -v \
    test_reviewed_io_recovery test_io_recovery_accounting test_io_remainder_worker test_submit_io_remainders \
    test_worker_scratch_policy test_work5_quarantine_remainder test_decoder_remainder_bundle
) > "$ep21_io_checks/checks.txt" 2>&1
ep21_io_checks_exit=$?
set -e
printf 'required_checks_exit=%s\n' "$ep21_io_checks_exit" >> "$ep21_io_checks/status.txt"
[[ "$ep21_io_checks_exit" == 0 ]]
printf 'Required_exact_IO_remainder_checks=passed\n'

# Actual EndTime minus300s is enforced by the existing worker budget, including
# startup time. These limits never extend the original eight-hour allocation.
exec timeout --signal=TERM --kill-after=10s 28530s python3.12 -u -B \
  outputs/decoder_training/tools/train_decoder_remainder.py \
  --authorization outputs/decoder_training_full_authorization.json \
  --manifest outputs/decoder_training/full_continuation_001/work_manifest.json \
  --io-recovery-review outputs/decoder_training/full_continuation_001/reviewed_io_salvage_001.json \
  --work-item "$ep21_io_work" --max-seconds 28500
