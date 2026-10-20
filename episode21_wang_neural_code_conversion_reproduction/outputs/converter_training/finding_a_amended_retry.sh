#!/bin/bash
# Future manually selected full retry only; no resource request or submission.
# The coordinator creates a separate activated selection and fresh phase dirs,
# binds actual jobs, and supplies finite Slurm resources after terminal diagnosis.
# Scientific work stops before the selected cleanup reserve. TERM occurs ten
# seconds before the hard cutoff; its KILL grace stays inside that cutoff.
# Never substitute this wrapper into the live job.
set -euo pipefail
[[ $# -eq 1 ]] || exit 2
ep21_retry_phase=$1
ep21_retry_args=(--phase "$ep21_retry_phase")
case "$ep21_retry_phase" in
  decode|finalize)
    ep21_retry_relative=$ep21_retry_phase
    ep21_retry_scratch="ep21_amended_full002_$ep21_retry_phase"
    [[ -z "${SLURM_ARRAY_JOB_ID:-}" && -z "${SLURM_ARRAY_TASK_ID:-}" ]] || exit 2
    ;;
  draw)
    [[ "${SLURM_ARRAY_JOB_ID:-}" =~ ^[1-9][0-9]*$ \
      && "${SLURM_ARRAY_TASK_ID:-}" =~ ^(0|[1-9][0-9]?)$ ]] || exit 2
    (( SLURM_ARRAY_TASK_ID <= 62 )) || exit 2
    printf -v ep21_retry_index '%03d' "$SLURM_ARRAY_TASK_ID"
    ep21_retry_relative="draws/batch_$ep21_retry_index"
    ep21_retry_scratch="ep21_amended_full002_draw_$ep21_retry_index"
    ep21_retry_args+=(--batch-index "$SLURM_ARRAY_TASK_ID")
    ;;
  *) exit 2 ;;
esac
ep21_retry_host=$(hostname)
printf 'HOST=%s JOB=%s PHASE=%s\n' "$ep21_retry_host" "${SLURM_JOB_ID:-}" "$ep21_retry_phase"
[[ "$ep21_retry_host" != *-ln* && "${SLURM_JOB_ID:-}" =~ ^[1-9][0-9]*$ ]] || exit 2
cd "${OAK:?}/users/zijiao/br_autoresearch/episode21_wang_neural_code_conversion_reproduction"
[[ -f outputs/amended_readout_retry_authorization.json ]] || exit 2
ep21_retry_root="$PWD/outputs/runs/finding_a_feature_decoding/amended_full002"
ep21_retry_run="$ep21_retry_root/$ep21_retry_relative"
[[ -d "$ep21_retry_run" && -d "${L_SCRATCH_JOB:?}" ]] || exit 2
umask 077
mkdir "$ep21_retry_run/batch_claim"
( set -o noclobber
  printf 'parent_episode=EP21\nparent_job=%s\nphase=%s\ncurrent_tests_executed=false\n' \
    "$SLURM_JOB_ID" "$ep21_retry_phase" > "$ep21_retry_run/batch_status.txt" )
ep21_retry_phase_rc=not_run
ep21_retry_finish() {
  ep21_retry_exit=$?
  trap - EXIT
  if [[ "$ep21_retry_phase_rc" == running ]]; then ep21_retry_phase_rc=$ep21_retry_exit; fi
  printf 'phase_exit=%s\nbatch_exit=%s\nstatus=closed\n' "$ep21_retry_phase_rc" "$ep21_retry_exit" \
    >> "$ep21_retry_run/batch_status.txt"
  exit "$ep21_retry_exit"
}
trap ep21_retry_finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
ulimit -c 0
ulimit -S -f 16384
# Capture this own allocation once; no scheduler polling loop.
TZ=UTC timeout --signal=TERM --kill-after=5s 15s scontrol show job -o "$SLURM_JOB_ID" \
  > "$ep21_retry_run/allocation.txt"
source /share/software/user/open/lmod/lmod/init/bash
module purge
unset PYTHONPATH PYTHONHOME
module load devel python/3.12.1 math py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 py-h5py/3.10.0_py312 py-pytorch/2.4.1_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=''
export TMPDIR="$L_SCRATCH_JOB/$ep21_retry_scratch"
mkdir "$TMPDIR"
export XDG_CACHE_HOME="$TMPDIR/cache" TORCH_HOME="$TMPDIR/torch"
ep21_retry_entry="$PWD/outputs/converter_training/tools/run_amended_readout_retry.py"
timeout --signal=TERM --kill-after=5s 60s python3.12 -B "$ep21_retry_entry" \
  "${ep21_retry_args[@]}" --check-admission > "$ep21_retry_run/admission.txt" 2>&1
ep21_retry_hard_epoch=$(timeout --signal=TERM --kill-after=5s 60s python3.12 -B "$ep21_retry_entry" \
  "${ep21_retry_args[@]}" --hard-stop-epoch 2> "$ep21_retry_run/remaining_stderr.txt")
[[ "$ep21_retry_hard_epoch" =~ ^[0-9]+$ ]] || exit 2
# Reserve kill grace plus one second for whole-second clock rounding.
ep21_retry_remaining=$((ep21_retry_hard_epoch - $(date -u +%s) - 11))
(( ep21_retry_remaining > 0 )) || exit 124
printf 'hard_stop_epoch=%s\nterm_timeout_seconds=%s\nkill_grace_seconds=10\n' \
  "$ep21_retry_hard_epoch" "$ep21_retry_remaining" >> "$ep21_retry_run/batch_status.txt"
ep21_retry_phase_rc=running
if timeout --signal=TERM --kill-after=10s "${ep21_retry_remaining}s" python3.12 -u -B \
    "$ep21_retry_entry" "${ep21_retry_args[@]}" > "$ep21_retry_run/execution.txt" 2>&1; then
  ep21_retry_phase_rc=0
else
  ep21_retry_phase_rc=$?
fi
[[ "$ep21_retry_phase_rc" -eq 0 ]] || exit "$ep21_retry_phase_rc"
if [[ "$ep21_retry_phase" == finalize ]]; then
  # Scientific publication survives a plotting failure. Use only completed JSON
  # and retain the same finite parent clock for the figure and three-line summary.
  module purge
  unset PYTHONPATH PYTHONHOME
  module load devel python/3.12.1 math py-numpy/1.26.3_py312 viz py-matplotlib/3.8.3_py312
  export MPLCONFIGDIR="$TMPDIR/matplotlib"
  mkdir "$MPLCONFIGDIR"
  ep21_retry_render_remaining=$((ep21_retry_hard_epoch - $(date -u +%s) - 6))
  (( ep21_retry_render_remaining > 0 )) || exit 124
  if (( ep21_retry_render_remaining > 240 )); then ep21_retry_render_remaining=240; fi
  timeout --signal=TERM --kill-after=5s "${ep21_retry_render_remaining}s" python3.12 -B \
    outputs/converter_training/tools/render_finding_a_retry_result.py \
    > "$ep21_retry_run/render_execution.txt" 2>&1
fi
