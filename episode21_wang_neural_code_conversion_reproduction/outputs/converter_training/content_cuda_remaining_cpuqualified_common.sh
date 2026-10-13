#!/bin/bash
# Prospective CPU-qualified route for the fifteen unsubmitted original pairs.
# Submitted remaining000 and its common/checks/accounting remain unchanged.
set -euo pipefail
[[ $# -eq 1 ]] || exit 2
ep21_content_remaining_stage=$1
case "$ep21_content_remaining_stage" in
  remaining-001) ep21_content_remaining_stem=sub01_to_sub05_cuda001; ep21_content_remaining_suffix=001 ;;
  remaining-002) ep21_content_remaining_stem=sub01_to_sub06_cuda001; ep21_content_remaining_suffix=002 ;;
  remaining-003) ep21_content_remaining_stem=sub01_to_sub07_cuda001; ep21_content_remaining_suffix=003 ;;
  remaining-008) ep21_content_remaining_stem=sub05_to_sub01_cuda001; ep21_content_remaining_suffix=008 ;;
  remaining-009) ep21_content_remaining_stem=sub05_to_sub03_cuda001; ep21_content_remaining_suffix=009 ;;
  remaining-010) ep21_content_remaining_stem=sub05_to_sub06_cuda001; ep21_content_remaining_suffix=010 ;;
  remaining-011) ep21_content_remaining_stem=sub05_to_sub07_cuda001; ep21_content_remaining_suffix=011 ;;
  remaining-012) ep21_content_remaining_stem=sub06_to_sub01_cuda001; ep21_content_remaining_suffix=012 ;;
  remaining-013) ep21_content_remaining_stem=sub06_to_sub03_cuda001; ep21_content_remaining_suffix=013 ;;
  remaining-014) ep21_content_remaining_stem=sub06_to_sub05_cuda001; ep21_content_remaining_suffix=014 ;;
  remaining-015) ep21_content_remaining_stem=sub06_to_sub07_cuda001; ep21_content_remaining_suffix=015 ;;
  remaining-016) ep21_content_remaining_stem=sub07_to_sub01_cuda001; ep21_content_remaining_suffix=016 ;;
  remaining-017) ep21_content_remaining_stem=sub07_to_sub03_cuda001; ep21_content_remaining_suffix=017 ;;
  remaining-018) ep21_content_remaining_stem=sub07_to_sub05_cuda001; ep21_content_remaining_suffix=018 ;;
  remaining-019) ep21_content_remaining_stem=sub07_to_sub06_cuda001; ep21_content_remaining_suffix=019 ;;
  *) exit 2 ;;
esac
ep21_content_remaining_host=$(hostname)
printf 'HOST=%s JOB=%s CPUS=%s MEM_MIB=%s GPUS=%s STAGE=%s\n' "$ep21_content_remaining_host" \
  "${SLURM_JOB_ID:?}" "${SLURM_CPUS_PER_TASK:?}" "${SLURM_MEM_PER_NODE:?}" \
  "${SLURM_GPUS_ON_NODE:?}" "$ep21_content_remaining_stage"
[[ "$ep21_content_remaining_host" != *-ln* && "$SLURM_JOB_ID" =~ ^[0-9]+$ \
  && "$SLURM_CPUS_PER_TASK" == 2 && "$SLURM_MEM_PER_NODE" == 16384 \
  && "$SLURM_GPUS_ON_NODE" == 1 && "${SLURM_NTASKS:-1}" == 1 \
  && "${SLURM_NNODES:-1}" == 1 ]] || exit 2
cd "${OAK:?}/users/zijiao/br_autoresearch/episode21_wang_neural_code_conversion_reproduction"
ep21_content_remaining_base=$PWD
ep21_content_remaining_run="$PWD/outputs/runs/content_pair_training/$ep21_content_remaining_stem"
[[ -d "$ep21_content_remaining_run" ]] || exit 2
umask 077
mkdir "$ep21_content_remaining_run/batch_claim" || exit 2
( set -o noclobber
  printf 'parent_episode=EP21\nparent_job=%s\nstage=%s\nmaximum_CPU_hours=96\nmaximum_GPU_hours=48\nstatus=started\n' \
    "$SLURM_JOB_ID" "$ep21_content_remaining_stage" > "$ep21_content_remaining_run/status.txt" )
ep21_content_remaining_finish() {
  ep21_content_remaining_rc=$?
  trap - EXIT
  printf 'batch_exit=%s\nstatus=closed\n' "$ep21_content_remaining_rc" >> "$ep21_content_remaining_run/status.txt"
  exit "$ep21_content_remaining_rc"
}
trap ep21_content_remaining_finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
ulimit -c 0
ulimit -S -f 1048576
TZ=UTC timeout --signal=TERM --kill-after=5s 15s scontrol show job -o "$SLURM_JOB_ID" \
  > "$ep21_content_remaining_run/allocation.txt"
ep21_content_remaining_allocation=$(< "$ep21_content_remaining_run/allocation.txt")
[[ "$ep21_content_remaining_allocation" =~ StartTime=([^[:space:]]+) ]] || exit 2
ep21_content_remaining_start=$(date -u -d "${BASH_REMATCH[1]}" +%s)
[[ "$ep21_content_remaining_allocation" =~ EndTime=([^[:space:]]+) ]] || exit 2
ep21_content_remaining_end=$(date -u -d "${BASH_REMATCH[1]}" +%s)
ep21_content_remaining_now=$(date -u +%s)
(( ep21_content_remaining_start <= ep21_content_remaining_now && ep21_content_remaining_now < ep21_content_remaining_end )) || exit 2
ep21_content_remaining_deadline=$((ep21_content_remaining_start + 169200))
if (( ep21_content_remaining_end - 60 < ep21_content_remaining_deadline )); then
  ep21_content_remaining_deadline=$((ep21_content_remaining_end - 60))
fi
printf 'absolute_deadline_epoch=%s\n' "$ep21_content_remaining_deadline" >> "$ep21_content_remaining_run/status.txt"
ep21_content_remaining_limit() {
  local requested=$1 available
  available=$((ep21_content_remaining_deadline - $(date -u +%s) - 15))
  (( available > 0 )) || return 124
  if (( available < requested )); then requested=$available; fi
  printf '%s\n' "$requested"
}
source /share/software/user/open/lmod/lmod/init/bash
module purge
unset PYTHONPATH PYTHONHOME
module load devel python/3.12.1 math py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 py-h5py/3.10.0_py312 py-pytorch/2.4.1_py312
# Preserve the module-provided PYTHONPATH and Slurm's assigned GPU visibility.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
[[ -d "${L_SCRATCH_JOB:?}" ]] || exit 2
ep21_content_remaining_free_kib=$(df -Pk "$L_SCRATCH_JOB" | awk 'NR==2 {print $4}')
[[ "$ep21_content_remaining_free_kib" =~ ^[0-9]+$ && "$ep21_content_remaining_free_kib" -ge 1048576 ]] || exit 2
export TMPDIR="$L_SCRATCH_JOB/ep21_content_cuda_remaining${ep21_content_remaining_suffix}"
mkdir "$TMPDIR"
export XDG_CACHE_HOME="$TMPDIR/cache" CUDA_CACHE_PATH="$TMPDIR/cuda-cache" TORCH_HOME="$TMPDIR/torch"
ep21_content_remaining_entry="$ep21_content_remaining_base/outputs/converter_training/tools/run_content_cuda_remaining.py"
ep21_content_remaining_rc=0
ep21_content_remaining_timeout=$(ep21_content_remaining_limit 60)
if timeout --signal=TERM --kill-after=5s "${ep21_content_remaining_timeout}s" python3.12 -B \
    "$ep21_content_remaining_entry" --stage "$ep21_content_remaining_stage" --check-admission \
    > "$ep21_content_remaining_run/admission.txt" 2>&1; then :; else ep21_content_remaining_rc=$?; fi
printf 'admission_exit=%s\n' "$ep21_content_remaining_rc" >> "$ep21_content_remaining_run/status.txt"
[[ "$ep21_content_remaining_rc" -eq 0 ]] || exit "$ep21_content_remaining_rc"

ep21_content_remaining_shared="$ep21_content_remaining_base/outputs/runs/content_launcher_qualification/cpu001/status.txt"
[[ -f "$ep21_content_remaining_shared" ]] || exit 2
ep21_content_remaining_passes=0
ep21_content_remaining_qualification_passes=0
while IFS='=' read -r key value; do
  case "$key" in
    remaining_checks_exit)
      [[ "$value" == 0 ]] || exit 2
      ep21_content_remaining_passes=$((ep21_content_remaining_passes + 1)) ;;
    qualification_exit)
      [[ "$value" == 0 ]] || exit 2
      ep21_content_remaining_qualification_passes=$((ep21_content_remaining_qualification_passes + 1)) ;;
  esac
done < "$ep21_content_remaining_shared"
[[ "$ep21_content_remaining_passes" -eq 1 && "$ep21_content_remaining_qualification_passes" -eq 1 ]] || exit 2
printf 'production_checks_reused_from=outputs/runs/content_launcher_qualification/cpu001/remaining_checks.txt\nproduction_checks_exit=0\n' \
  >> "$ep21_content_remaining_run/status.txt"

# Re-admission uses the same actual StartTime, including all earlier children.
ep21_content_remaining_timeout=$(ep21_content_remaining_limit 60)
ep21_content_remaining_admission_rc=0
if ep21_content_remaining_seconds=$(timeout --signal=TERM --kill-after=5s "${ep21_content_remaining_timeout}s" \
    python3.12 -B "$ep21_content_remaining_entry" --stage "$ep21_content_remaining_stage" --remaining-seconds \
    2> "$ep21_content_remaining_run/remaining_stderr.txt"); then :; else ep21_content_remaining_admission_rc=$?; fi
printf 'remaining_admission_exit=%s\n' "$ep21_content_remaining_admission_rc" >> "$ep21_content_remaining_run/status.txt"
[[ "$ep21_content_remaining_admission_rc" -eq 0 && "$ep21_content_remaining_seconds" =~ ^[0-9]+$ \
  && "$ep21_content_remaining_seconds" -gt 0 ]] || exit 2
ep21_content_remaining_timeout=$(ep21_content_remaining_limit "$ep21_content_remaining_seconds")
printf 'production_timeout_seconds=%s\n' "$ep21_content_remaining_timeout" >> "$ep21_content_remaining_run/status.txt"
ep21_content_remaining_rc=0
if timeout --signal=TERM --kill-after=10s "${ep21_content_remaining_timeout}s" python3.12 -u -B \
    "$ep21_content_remaining_entry" --stage "$ep21_content_remaining_stage" \
    > "$ep21_content_remaining_run/training.txt" 2>&1; then :; else ep21_content_remaining_rc=$?; fi
printf 'training_exit=%s\n' "$ep21_content_remaining_rc" >> "$ep21_content_remaining_run/status.txt"
exit "$ep21_content_remaining_rc"
