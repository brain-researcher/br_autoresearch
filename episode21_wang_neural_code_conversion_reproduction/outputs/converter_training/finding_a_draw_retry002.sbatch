#!/bin/bash
# Only the eight newly diagnosed failed draws; no decode or automatic resubmission.
#SBATCH --job-name=ep21-draw-retry
#SBATCH --partition=russpold
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --array=8-15%3
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/finding_a_feature_decoding/amended_full001/draw_retries/attempt002/slurm-%A_%a.out
#SBATCH --error=outputs/runs/finding_a_feature_decoding/amended_full001/draw_retries/attempt002/slurm-%A_%a.err
set -euo pipefail
[[ $# == 0 && "${SLURM_ARRAY_TASK_ID:-}" =~ ^(8|9|1[0-5])$ ]] || exit 2
printf 'HOST=%s JOB=%s TASK=%s\n' "$(hostname)" "${SLURM_JOB_ID:-}" "$SLURM_ARRAY_TASK_ID"
cd "${OAK:?}/users/zijiao/br_autoresearch/episode21_wang_neural_code_conversion_reproduction"
printf -v ep21_retry_index '%03d' "$SLURM_ARRAY_TASK_ID"
ep21_retry_run="$PWD/outputs/runs/finding_a_feature_decoding/amended_full001/draw_retries/attempt002/batch_$ep21_retry_index"
[[ -d "$ep21_retry_run" && -d "${L_SCRATCH_JOB:?}" ]] || exit 2
umask 077
mkdir "$ep21_retry_run/batch_claim"
( set -o noclobber
  printf 'parent_episode=EP21\nparent_job=%s\nphase=draw\nbatch_started=true\nmaximum_allocated_CPU_hours=144\n' \
    "$SLURM_JOB_ID" > "$ep21_retry_run/batch_status.txt" )
ep21_retry_phase_rc=not_run
ep21_retry_finish() {
  ep21_retry_exit=$?
  trap - EXIT
  printf 'phase_exit=%s\nbatch_exit=%s\nstatus=closed\n' \
    "$ep21_retry_phase_rc" "$ep21_retry_exit" >> "$ep21_retry_run/batch_status.txt"
  exit "$ep21_retry_exit"
}
trap ep21_retry_finish EXIT
trap 'exit 143' TERM
ulimit -c 0
ulimit -S -f 16384
TZ=UTC timeout --signal=TERM --kill-after=5s 15s scontrol show job -o \
  "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" > "$ep21_retry_run/allocation.txt"
source /share/software/user/open/lmod/lmod/init/bash
module purge
unset PYTHONPATH PYTHONHOME
module load devel python/3.12.1 math py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 py-h5py/3.10.0_py312 py-pytorch/2.4.1_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=''
export TMPDIR="$L_SCRATCH_JOB/ep21_draw_retry_attempt002_$ep21_retry_index"
mkdir "$TMPDIR"
export XDG_CACHE_HOME="$TMPDIR/cache" TORCH_HOME="$TMPDIR/torch"
ep21_retry_entry="$PWD/outputs/converter_training/tools/run_amended_draw_retry002.py"
ep21_retry_args=(--phase draw --batch-index "$SLURM_ARRAY_TASK_ID")
timeout 60s python3.12 -B "$ep21_retry_entry" "${ep21_retry_args[@]}" --check-admission \
  > "$ep21_retry_run/admission.txt" 2>&1
ep21_retry_remaining=$(timeout 60s python3.12 -B "$ep21_retry_entry" \
  "${ep21_retry_args[@]}" --remaining-seconds 2> "$ep21_retry_run/remaining_stderr.txt")
[[ "$ep21_retry_remaining" =~ ^[0-9]+$ && "$ep21_retry_remaining" -gt 20 ]] || exit 2
ep21_retry_remaining=$((ep21_retry_remaining - 15))
if timeout --signal=TERM --kill-after=10s "${ep21_retry_remaining}s" python3.12 -u -B \
    "$ep21_retry_entry" "${ep21_retry_args[@]}" > "$ep21_retry_run/execution.txt" 2>&1; then
  ep21_retry_phase_rc=0
else
  ep21_retry_phase_rc=$?
fi
exit "$ep21_retry_phase_rc"
