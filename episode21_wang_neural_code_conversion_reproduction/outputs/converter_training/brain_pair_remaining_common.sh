#!/bin/bash
# One fixed, independently granted original brain pair per six-hour parent.
# No retry, next-pair dispatch, held-out evaluation or scientific override.
set -euo pipefail
[[ $# -eq 1 ]] || exit 2
ep21_remaining_stage=$1
case "$ep21_remaining_stage" in
  remaining-009) ep21_remaining_stem=sub05_to_sub03_001 ;;
  remaining-010) ep21_remaining_stem=sub05_to_sub06_001 ;;
  remaining-011) ep21_remaining_stem=sub05_to_sub07_001 ;;
  remaining-013) ep21_remaining_stem=sub06_to_sub03_001 ;;
  remaining-014) ep21_remaining_stem=sub06_to_sub05_001 ;;
  remaining-015) ep21_remaining_stem=sub06_to_sub07_001 ;;
  remaining-017) ep21_remaining_stem=sub07_to_sub03_001 ;;
  remaining-018) ep21_remaining_stem=sub07_to_sub05_001 ;;
  remaining-019) ep21_remaining_stem=sub07_to_sub06_001 ;;
  *) exit 2 ;;
esac
ep21_remaining_host=$(hostname)
printf 'HOST=%s JOB=%s CPUS=%s MEM_MIB=%s STAGE=%s\n' "$ep21_remaining_host" \
  "${SLURM_JOB_ID:?}" "${SLURM_CPUS_PER_TASK:?}" "${SLURM_MEM_PER_NODE:?}" "$ep21_remaining_stage"
[[ "$ep21_remaining_host" != *-ln* && "$SLURM_JOB_ID" =~ ^[0-9]+$ \
  && "$SLURM_CPUS_PER_TASK" == 3 && "$SLURM_MEM_PER_NODE" == 16384 \
  && "${SLURM_NTASKS:-1}" == 1 && "${SLURM_NNODES:-1}" == 1 \
  && -z "${SLURM_JOB_GPUS:-}" && -z "${SLURM_STEP_GPUS:-}" \
  && "${SLURM_GPUS:-0}" == 0 && "${SLURM_GPUS_ON_NODE:-0}" == 0 ]] || exit 2
cd "${OAK:?}/users/zijiao/br_autoresearch/episode21_wang_neural_code_conversion_reproduction"
ep21_remaining_base=$PWD
ep21_remaining_run="$PWD/outputs/runs/brain_pair_training/$ep21_remaining_stem"
[[ -d "$ep21_remaining_run" ]] || exit 2
umask 077
mkdir "$ep21_remaining_run/batch_claim" || exit 2
( set -o noclobber
  printf 'parent_episode=EP21\nparent_job=%s\nstage=%s\nmaximum_CPU_hours=18\nmaximum_GPU_hours=0\nstatus=started\n' \
    "$SLURM_JOB_ID" "$ep21_remaining_stage" > "$ep21_remaining_run/status.txt" )
ep21_remaining_finish() {
  ep21_remaining_rc=$?
  trap - EXIT
  printf 'batch_exit=%s\nstatus=closed\n' "$ep21_remaining_rc" >> "$ep21_remaining_run/status.txt"
  exit "$ep21_remaining_rc"
}
trap ep21_remaining_finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
ulimit -c 0
ulimit -S -f 2097152
TZ=UTC timeout --signal=TERM --kill-after=5s 15s scontrol show job -o "$SLURM_JOB_ID" \
  > "$ep21_remaining_run/allocation.txt"
ep21_remaining_allocation=$(< "$ep21_remaining_run/allocation.txt")
[[ "$ep21_remaining_allocation" =~ StartTime=([^[:space:]]+) ]] || exit 2
ep21_remaining_start=$(date -u -d "${BASH_REMATCH[1]}" +%s)
[[ "$ep21_remaining_allocation" =~ EndTime=([^[:space:]]+) ]] || exit 2
ep21_remaining_end=$(date -u -d "${BASH_REMATCH[1]}" +%s)
ep21_remaining_now=$(date -u +%s)
(( ep21_remaining_start <= ep21_remaining_now && ep21_remaining_now < ep21_remaining_end )) || exit 2
ep21_remaining_deadline=$((ep21_remaining_start + 21000))
if (( ep21_remaining_end - 60 < ep21_remaining_deadline )); then
  ep21_remaining_deadline=$((ep21_remaining_end - 60))
fi
printf 'absolute_deadline_epoch=%s\n' "$ep21_remaining_deadline" >> "$ep21_remaining_run/status.txt"
ep21_remaining_limit() {
  local requested=$1 available
  available=$((ep21_remaining_deadline - $(date -u +%s) - 15))
  (( available > 0 )) || return 124
  if (( available < requested )); then requested=$available; fi
  printf '%s\n' "$requested"
}
source /share/software/user/open/lmod/lmod/init/bash
module purge
unset PYTHONPATH PYTHONHOME
module load devel python/3.12.1 math py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 py-h5py/3.10.0_py312 py-pytorch/2.4.1_py312
# Keep the qualified module-provided PYTHONPATH for all scientific children.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
[[ -d "${L_SCRATCH_JOB:?}" ]] || exit 2
ep21_remaining_free_kib=$(df -Pk "$L_SCRATCH_JOB" | awk 'NR==2 {print $4}')
[[ "$ep21_remaining_free_kib" =~ ^[0-9]+$ && "$ep21_remaining_free_kib" -ge 262144 ]] || exit 2
export TMPDIR="$L_SCRATCH_JOB/ep21_brain_pair_$ep21_remaining_stem"
mkdir "$TMPDIR"
export XDG_CACHE_HOME="$TMPDIR/cache" TORCH_HOME="$TMPDIR/torch"
ep21_remaining_entry="$ep21_remaining_base/outputs/converter_training/tools/run_brain_pair_remaining.py"
ep21_remaining_rc=0
ep21_remaining_timeout=$(ep21_remaining_limit 60)
if timeout --signal=TERM --kill-after=5s "${ep21_remaining_timeout}s" python3.12 -B \
    "$ep21_remaining_entry" --stage "$ep21_remaining_stage" --check-admission \
    > "$ep21_remaining_run/admission.txt" 2>&1; then :; else ep21_remaining_rc=$?; fi
printf 'admission_exit=%s\n' "$ep21_remaining_rc" >> "$ep21_remaining_run/status.txt"
[[ "$ep21_remaining_rc" -eq 0 ]] || exit "$ep21_remaining_rc"
printf 'prerequisites=reused_13_passing_methods_job_47334643_and_completed_original_pair5\n' \
  >> "$ep21_remaining_run/status.txt"

ep21_remaining_checks_rc=0
if [[ "$ep21_remaining_stage" == remaining-009 ]]; then
  ep21_remaining_timeout=$(ep21_remaining_limit 90)
  if (
    ulimit -S -f 32768
    cd "$ep21_remaining_base/outputs/converter_training/tests"
    timeout --signal=TERM --kill-after=5s "${ep21_remaining_timeout}s" python3.12 -B -m unittest -v \
      test_brain_pair_remaining
  ) > "$ep21_remaining_run/brain_admission_checks.txt" 2>&1; then :; else ep21_remaining_checks_rc=$?; fi
  printf 'remaining_checks_exit=%s\n' "$ep21_remaining_checks_rc" >> "$ep21_remaining_run/status.txt"
  [[ "$ep21_remaining_checks_rc" -eq 0 ]] || exit "$ep21_remaining_checks_rc"

  # Documentary validation is recorded once, separately from scientific fit
  # admission. An ordinary ASTRA failure does not veto the selected fit.
  ep21_remaining_astra="${GROUP_HOME:?}/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv"
  ep21_remaining_astra_deps="${GROUP_HOME}/users/zijiao/ep18_model_runtime_cpu241_py312_46842156/lib/python3.12/site-packages"
  ep21_remaining_astra_rc=0
  ep21_remaining_timeout=$(ep21_remaining_limit 30)
  if (
    ulimit -S -f 32768
    env PYTHONPATH="$ep21_remaining_astra/lib/python3.12/site-packages:$ep21_remaining_astra_deps" \
      timeout --signal=TERM --kill-after=5s "${ep21_remaining_timeout}s" "$ep21_remaining_astra/bin/astra-tools" \
      validate outputs/astra/v0.0.14/astra.yaml --json
  ) > "$ep21_remaining_run/astra_validation.json" 2>&1; then :; else ep21_remaining_astra_rc=$?; fi
  printf 'astra_exit=%s\n' "$ep21_remaining_astra_rc" >> "$ep21_remaining_run/status.txt"
  [[ "$ep21_remaining_astra_rc" -ne 137 ]] || exit 137
else
  # The coordinator submits these separately selected pairs only after
  # observing pair009's shared three-method PASS, without waiting for its fit.
  ep21_remaining_shared="$ep21_remaining_base/outputs/runs/brain_pair_training/sub05_to_sub03_001/status.txt"
  [[ -f "$ep21_remaining_shared" ]] || exit 2
  ep21_remaining_passes=0
  while IFS='=' read -r key value; do
    if [[ "$key" == remaining_checks_exit ]]; then
      [[ "$value" == 0 ]] || exit 2
      ep21_remaining_passes=$((ep21_remaining_passes + 1))
    fi
  done < "$ep21_remaining_shared"
  [[ "$ep21_remaining_passes" -eq 1 ]] || exit 2
  printf 'remaining_checks_reused_from=../sub05_to_sub03_001/brain_admission_checks.txt\nremaining_checks_exit=0\n' \
    >> "$ep21_remaining_run/status.txt"
fi

# Re-admission cannot reset the selected parent clock after checks/ASTRA.
ep21_remaining_timeout=$(ep21_remaining_limit 60)
ep21_remaining_admission_rc=0
if ep21_remaining_seconds=$(timeout --signal=TERM --kill-after=5s "${ep21_remaining_timeout}s" \
    python3.12 -B "$ep21_remaining_entry" --stage "$ep21_remaining_stage" --remaining-seconds \
    2> "$ep21_remaining_run/remaining_stderr.txt"); then :; else ep21_remaining_admission_rc=$?; fi
printf 'remaining_admission_exit=%s\n' "$ep21_remaining_admission_rc" >> "$ep21_remaining_run/status.txt"
[[ "$ep21_remaining_admission_rc" -eq 0 && "$ep21_remaining_seconds" =~ ^[0-9]+$ \
  && "$ep21_remaining_seconds" -gt 0 ]] || exit 2
ep21_remaining_timeout=$(ep21_remaining_limit "$ep21_remaining_seconds")
printf 'brain_pair_timeout_seconds=%s\n' "$ep21_remaining_timeout" >> "$ep21_remaining_run/status.txt"
ep21_remaining_rc=0
if timeout --signal=TERM --kill-after=10s "${ep21_remaining_timeout}s" python3.12 -u -B \
    "$ep21_remaining_entry" --stage "$ep21_remaining_stage" \
    > "$ep21_remaining_run/fit.txt" 2>&1; then :; else ep21_remaining_rc=$?; fi
printf 'brain_pair_exit=%s\n' "$ep21_remaining_rc" >> "$ep21_remaining_run/status.txt"
exit "$ep21_remaining_rc"
