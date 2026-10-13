#!/bin/bash
# Invoked only by one of the three separately selected original-pair batches.
set -euo pipefail
[[ $# -eq 1 ]] || exit 2
ep21_roster_stage=$1
case "$ep21_roster_stage" in
  roster-005) ep21_roster_target=sub05; ep21_roster_suffix=005 ;;
  roster-006) ep21_roster_target=sub06; ep21_roster_suffix=006 ;;
  roster-007) ep21_roster_target=sub07; ep21_roster_suffix=007 ;;
  *) exit 2 ;;
esac
ep21_roster_host=$(hostname)
printf 'HOST=%s JOB=%s CPUS=%s MEM_MIB=%s GPUS=%s STAGE=%s\n' "$ep21_roster_host" \
  "${SLURM_JOB_ID:?}" "${SLURM_CPUS_PER_TASK:?}" "${SLURM_MEM_PER_NODE:?}" \
  "${SLURM_GPUS_ON_NODE:?}" "$ep21_roster_stage"
[[ "$ep21_roster_host" != *-ln* && "$SLURM_CPUS_PER_TASK" == 2 \
  && "$SLURM_MEM_PER_NODE" == 16384 && "$SLURM_GPUS_ON_NODE" == 1 ]] || exit 2
cd "${OAK:?}/users/zijiao/br_autoresearch/episode21_wang_neural_code_conversion_reproduction"
ep21_roster_base=$PWD
ep21_roster_run="$PWD/outputs/runs/content_pair_training/sub03_to_${ep21_roster_target}_cuda001"
[[ -d "$ep21_roster_run" ]] || exit 2
umask 077
mkdir "$ep21_roster_run/batch_claim" || exit 2
( set -o noclobber
  printf 'parent_episode=EP21\nparent_job=%s\nstage=%s\nmaximum_CPU_hours=96\nmaximum_GPU_hours=48\nstatus=started\n' \
    "$SLURM_JOB_ID" "$ep21_roster_stage" > "$ep21_roster_run/status.txt" )
ep21_roster_finish() {
  ep21_roster_rc=$?
  trap - EXIT
  printf 'batch_exit=%s\nstatus=closed\n' "$ep21_roster_rc" >> "$ep21_roster_run/status.txt"
  exit "$ep21_roster_rc"
}
trap ep21_roster_finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
ulimit -c 0
ulimit -S -f 1048576
TZ=UTC timeout --signal=TERM --kill-after=5s 15s scontrol show job -o "$SLURM_JOB_ID" \
  > "$ep21_roster_run/allocation.txt"
ep21_roster_allocation=$(< "$ep21_roster_run/allocation.txt")
[[ "$ep21_roster_allocation" =~ StartTime=([^[:space:]]+) ]] || exit 2
ep21_roster_start=$(date -u -d "${BASH_REMATCH[1]}" +%s)
[[ "$ep21_roster_allocation" =~ EndTime=([^[:space:]]+) ]] || exit 2
ep21_roster_end=$(date -u -d "${BASH_REMATCH[1]}" +%s)
ep21_roster_deadline=$((ep21_roster_start + 169200))
if (( ep21_roster_end - 60 < ep21_roster_deadline )); then
  ep21_roster_deadline=$((ep21_roster_end - 60))
fi
printf 'absolute_deadline_epoch=%s\n' "$ep21_roster_deadline" >> "$ep21_roster_run/status.txt"
ep21_roster_limit() {
  local requested=$1 available
  available=$((ep21_roster_deadline - $(date -u +%s) - 15))
  (( available > 0 )) || return 124
  if (( available < requested )); then requested=$available; fi
  printf '%s\n' "$requested"
}
source /share/software/user/open/lmod/lmod/init/bash
module purge
unset PYTHONPATH PYTHONHOME
module load devel python/3.12.1 math py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 py-h5py/3.10.0_py312 py-pytorch/2.4.1_py312
# Keep the module-provided PYTHONPATH for every scientific child.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
[[ -d "${L_SCRATCH_JOB:?}" ]] || exit 2
export TMPDIR="$L_SCRATCH_JOB/ep21_content_cuda_roster${ep21_roster_suffix}"
mkdir "$TMPDIR"
export XDG_CACHE_HOME="$TMPDIR/cache" CUDA_CACHE_PATH="$TMPDIR/cuda-cache" TORCH_HOME="$TMPDIR/torch"
ep21_roster_entry="$ep21_roster_base/outputs/converter_training/tools/run_content_cuda_roster.py"
ep21_roster_rc=0
ep21_roster_timeout=$(ep21_roster_limit 60)
if timeout --signal=TERM --kill-after=5s "${ep21_roster_timeout}s" python3.12 -B \
    "$ep21_roster_entry" --stage "$ep21_roster_stage" --check-admission \
    > "$ep21_roster_run/admission.txt" 2>&1; then :; else ep21_roster_rc=$?; fi
printf 'admission_exit=%s\n' "$ep21_roster_rc" >> "$ep21_roster_run/status.txt"
[[ "$ep21_roster_rc" -eq 0 ]] || exit "$ep21_roster_rc"

ep21_roster_tests() {
  local directory=$1 seconds=$2 limit
  shift 2
  limit=$(ep21_roster_limit "$seconds") || return $?
  (
    ulimit -S -f 32768
    export CUDA_VISIBLE_DEVICES=''
    cd "$directory"
    timeout --signal=TERM --kill-after=5s "${limit}s" python3.12 -B - "$@" <<'PYTEST'
import sys
import torch
import unittest
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
suite = unittest.defaultTestLoader.loadTestsFromNames(sys.argv[1:])
result = unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(0 if result.wasSuccessful() else 1)
PYTEST
  )
}
ep21_roster_astra="${GROUP_HOME:?}/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv"
ep21_roster_astra_deps="${GROUP_HOME}/users/zijiao/ep18_model_runtime_cpu241_py312_46842156/lib/python3.12/site-packages"
ep21_roster_validate() {
  local limit
  limit=$(ep21_roster_limit 30) || return $?
  env PYTHONPATH="$ep21_roster_astra/lib/python3.12/site-packages:$ep21_roster_astra_deps" \
    timeout --signal=TERM --kill-after=5s "${limit}s" "$ep21_roster_astra/bin/astra-tools" \
    validate outputs/astra/v0.0.14/astra.yaml --json
}

ep21_roster_checks_rc=0
if [[ "$ep21_roster_stage" == roster-005 ]]; then
  if ep21_roster_tests "$ep21_roster_base/outputs/converter_training/tests" 180 \
      test_content_cuda_roster_admission test_content_cuda_roster_training \
      > "$ep21_roster_run/production_checks.txt" 2>&1; then :; else ep21_roster_checks_rc=$?; fi
  printf 'production_checks_exit=%s\n' "$ep21_roster_checks_rc" >> "$ep21_roster_run/status.txt"
  [[ "$ep21_roster_checks_rc" -ne 137 ]] || exit 137
else
  # These separately selected submissions occur only after the coordinator
  # observes pair5's shared five-method PASS. Preserve that exact evidence.
  ep21_roster_shared="$ep21_roster_base/outputs/runs/content_pair_training/sub03_to_sub05_cuda001/status.txt"
  [[ -f "$ep21_roster_shared" ]] || exit 2
  ep21_roster_passes=0
  while IFS='=' read -r key value; do
    if [[ "$key" == production_checks_exit ]]; then
      [[ "$value" == 0 ]] || exit 2
      ep21_roster_passes=$((ep21_roster_passes + 1))
    fi
  done < "$ep21_roster_shared"
  [[ "$ep21_roster_passes" -eq 1 ]] || exit 2
  printf 'production_checks_reused_from=../sub03_to_sub05_cuda001/production_checks.txt\n' >> "$ep21_roster_run/status.txt"
fi
ep21_roster_astra_rc=0
if ep21_roster_validate > "$ep21_roster_run/astra_validation.json" 2>&1; then :; else ep21_roster_astra_rc=$?; fi
printf 'astra_exit=%s\n' "$ep21_roster_astra_rc" >> "$ep21_roster_run/status.txt"
[[ "$ep21_roster_astra_rc" -ne 137 ]] || exit 137

# Independently selected software-only companions are added only in pair5;
# their ordinary failures are recorded separately and cannot veto this fit.
if [[ "$ep21_roster_stage" == roster-005 ]]; then
  source "$ep21_roster_base/outputs/converter_training/content_cuda_roster_companions.sh"
fi

[[ "$ep21_roster_checks_rc" -eq 0 ]] || exit "$ep21_roster_checks_rc"
ep21_roster_timeout=$(ep21_roster_limit 60)
ep21_roster_remaining_rc=0
if ep21_roster_remaining=$(timeout --signal=TERM --kill-after=5s "${ep21_roster_timeout}s" \
    python3.12 -B "$ep21_roster_entry" --stage "$ep21_roster_stage" --remaining-seconds \
    2> "$ep21_roster_run/remaining_stderr.txt"); then :; else ep21_roster_remaining_rc=$?; fi
printf 'remaining_admission_exit=%s\n' "$ep21_roster_remaining_rc" >> "$ep21_roster_run/status.txt"
[[ "$ep21_roster_remaining_rc" -eq 0 && "$ep21_roster_remaining" =~ ^[0-9]+$ \
  && "$ep21_roster_remaining" -gt 0 ]] || exit 2
ep21_roster_timeout=$(ep21_roster_limit "$ep21_roster_remaining")
printf 'production_timeout_seconds=%s\n' "$ep21_roster_timeout" >> "$ep21_roster_run/status.txt"
ep21_roster_rc=0
if timeout --signal=TERM --kill-after=10s "${ep21_roster_timeout}s" python3.12 -u -B \
    "$ep21_roster_entry" --stage "$ep21_roster_stage" \
    > "$ep21_roster_run/training.txt" 2>&1; then :; else ep21_roster_rc=$?; fi
printf 'training_exit=%s\n' "$ep21_roster_rc" >> "$ep21_roster_run/status.txt"
exit "$ep21_roster_rc"
