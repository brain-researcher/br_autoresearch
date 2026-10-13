#!/bin/bash
# Sourced by the three fixed wrappers only. No submission, retry or resume.
# The runner admits the activated profile, exact procedural job binding and
# captured allocation before scientific imports or input access. The coordinator
# creates phase directories and supplies dependencies at submission time.
set -euo pipefail
[[ $# -eq 1 ]] || exit 2
ep21_full_phase=$1
ep21_full_args=(--phase "$ep21_full_phase")
case "$ep21_full_phase" in
  decode)
    ep21_full_cpus=3; ep21_full_mem=16384; ep21_full_seconds=255600
    ep21_full_hours=216; ep21_full_relative=decode
    ep21_full_scratch=ep21_finding_a_full001_decode
    [[ -z "${SLURM_ARRAY_JOB_ID:-}" && -z "${SLURM_ARRAY_TASK_ID:-}" ]] || exit 2
    ;;
  draw)
    ep21_full_cpus=3; ep21_full_mem=16384; ep21_full_seconds=84600
    ep21_full_hours=72
    [[ "${SLURM_ARRAY_JOB_ID:-}" =~ ^[1-9][0-9]*$ \
      && "${SLURM_ARRAY_TASK_ID:-}" =~ ^(0|[1-9][0-9]?)$ \
      && "${SLURM_ARRAY_TASK_COUNT:-}" == 63 \
      && "${SLURM_ARRAY_TASK_MIN:-}" == 0 \
      && "${SLURM_ARRAY_TASK_MAX:-}" == 62 ]] || exit 2
    (( SLURM_ARRAY_TASK_ID <= 62 )) || exit 2
    printf -v ep21_full_index '%03d' "$SLURM_ARRAY_TASK_ID"
    ep21_full_relative="draws/batch_$ep21_full_index"
    ep21_full_scratch="ep21_finding_a_full001_draw_$ep21_full_index"
    ep21_full_args+=(--batch-index "$SLURM_ARRAY_TASK_ID")
    ;;
  finalize)
    ep21_full_cpus=1; ep21_full_mem=7168; ep21_full_seconds=6900
    ep21_full_hours=2; ep21_full_relative=finalize
    ep21_full_scratch=ep21_finding_a_full001_finalize
    [[ -z "${SLURM_ARRAY_JOB_ID:-}" && -z "${SLURM_ARRAY_TASK_ID:-}" ]] || exit 2
    ;;
  *) exit 2 ;;
esac
ep21_full_host=$(hostname)
printf 'HOST=%s JOB=%s PHASE=%s CPUS=%s MEM_MIB=%s\n' "$ep21_full_host" \
  "${SLURM_JOB_ID:-}" "$ep21_full_phase" "${SLURM_CPUS_PER_TASK:-}" "${SLURM_MEM_PER_NODE:-}"
[[ "$ep21_full_host" != *-ln* && "${SLURM_JOB_ID:-}" =~ ^[1-9][0-9]*$ \
  && "${SLURM_JOB_PARTITION:-}" == russpold \
  && "${SLURM_CPUS_PER_TASK:-}" == "$ep21_full_cpus" \
  && "${SLURM_MEM_PER_NODE:-}" == "$ep21_full_mem" \
  && "${SLURM_NTASKS:-1}" == 1 && "${SLURM_JOB_NUM_NODES:-1}" == 1 \
  && "${SLURM_GPUS:-0}" == 0 && "${SLURM_GPUS_ON_NODE:-0}" == 0 \
  && -z "${SLURM_JOB_GPUS:-}" && -z "${SLURM_STEP_GPUS:-}" ]] || exit 2
cd "${OAK:?}/users/zijiao/br_autoresearch/episode21_wang_neural_code_conversion_reproduction"
ep21_full_base=$PWD
ep21_full_root="$ep21_full_base/outputs/runs/finding_a_feature_decoding/full001"
ep21_full_run="$ep21_full_root/$ep21_full_relative"
[[ -d "$ep21_full_run" ]] || exit 2
umask 077
mkdir "$ep21_full_run/batch_claim" || exit 2
( set -o noclobber
  printf 'parent_episode=EP21\nparent_job=%s\nphase=%s\nmaximum_allocated_CPU_hours=%s\nbatch_started=true\n' \
    "$SLURM_JOB_ID" "$ep21_full_phase" "$ep21_full_hours" > "$ep21_full_run/batch_status.txt" )
ep21_full_phase_rc=not_run
ep21_full_finish() {
  ep21_full_exit=$?
  trap - EXIT
  if [[ "$ep21_full_phase_rc" == running ]]; then ep21_full_phase_rc=$ep21_full_exit; fi
  printf 'phase_exit=%s\nbatch_exit=%s\n' "$ep21_full_phase_rc" "$ep21_full_exit" \
    >> "$ep21_full_run/batch_status.txt"
  if (( ep21_full_exit > 128 && ep21_full_exit < 193 )); then
    printf 'terminating_signal=%s\n' "$((ep21_full_exit - 128))" >> "$ep21_full_run/batch_status.txt"
  fi
  printf 'status=closed\n' >> "$ep21_full_run/batch_status.txt"
  exit "$ep21_full_exit"
}
trap ep21_full_finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
ulimit -c 0
# Logs and fixtures inherit a 16-MiB soft limit. Keep the hard limit intact:
# isolated fixtures and the admitted native-HDF runner select their own finite
# soft caps and signal semantics. No SIGXFSZ handler is installed here.
ulimit -S -f 16384
# In an array SLURM_JOB_ID is the actual element, not SLURM_ARRAY_JOB_ID.
TZ=UTC timeout --signal=TERM --kill-after=5s 15s scontrol show job -o "$SLURM_JOB_ID" \
  > "$ep21_full_run/allocation.txt"
ep21_full_allocation=$(< "$ep21_full_run/allocation.txt")
[[ "$ep21_full_allocation" =~ (^|[[:space:]])JobId=([0-9]+)($|[[:space:]]) \
  && "${BASH_REMATCH[2]}" == "$SLURM_JOB_ID" ]] || exit 2
[[ "$ep21_full_allocation" =~ StartTime=([^[:space:]]+) ]] || exit 2
ep21_full_start=$(date -u -d "${BASH_REMATCH[1]}" +%s)
[[ "$ep21_full_allocation" =~ EndTime=([^[:space:]]+) ]] || exit 2
ep21_full_end=$(date -u -d "${BASH_REMATCH[1]}" +%s)
ep21_full_now=$(date -u +%s)
(( ep21_full_start <= ep21_full_now && ep21_full_now < ep21_full_end )) || exit 2
ep21_full_deadline=$((ep21_full_start + ep21_full_seconds))
if (( ep21_full_end - 60 < ep21_full_deadline )); then
  ep21_full_deadline=$((ep21_full_end - 60))
fi
printf 'absolute_deadline_epoch=%s\n' "$ep21_full_deadline" >> "$ep21_full_run/batch_status.txt"
ep21_full_limit() {
  local requested=$1 available
  available=$((ep21_full_deadline - $(date -u +%s) - 15))
  (( available > 0 )) || return 124
  if (( available < requested )); then requested=$available; fi
  printf '%s\n' "$requested"
}

source /share/software/user/open/lmod/lmod/init/bash
module purge
unset PYTHONPATH PYTHONHOME
module load devel python/3.12.1 math py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 py-h5py/3.10.0_py312 py-pytorch/2.4.1_py312
# Preserve module-provided PYTHONPATH for every scientific Python child.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=''
[[ -d "${L_SCRATCH_JOB:?}" ]] || exit 2
export TMPDIR="$L_SCRATCH_JOB/$ep21_full_scratch"
mkdir "$TMPDIR"
export XDG_CACHE_HOME="$TMPDIR/cache" TORCH_HOME="$TMPDIR/torch"
ep21_full_entry="$ep21_full_base/outputs/converter_training/tools/run_finding_a_full_readout.py"
ep21_full_rc=0
ep21_full_timeout=$(ep21_full_limit 60)
if timeout --signal=TERM --kill-after=5s "${ep21_full_timeout}s" python3.12 -B \
    "$ep21_full_entry" "${ep21_full_args[@]}" --check-admission \
    > "$ep21_full_run/admission.txt" 2>&1; then :; else ep21_full_rc=$?; fi
printf 'admission_exit=%s\n' "$ep21_full_rc" >> "$ep21_full_run/batch_status.txt"
[[ "$ep21_full_rc" -eq 0 ]] || exit "$ep21_full_rc"

ep21_full_check() {
  local module_name=$1 receipt=$2 requested=$3 limit rc=0
  limit=$(ep21_full_limit "$requested") || return $?
  if (
    cd "$ep21_full_base/outputs/converter_training/tests"
    timeout --signal=TERM --kill-after=5s "${limit}s" python3.12 -B - "$module_name" <<'PYTEST'
import sys
import torch
import unittest
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
suite = unittest.defaultTestLoader.loadTestsFromName(sys.argv[1])
result = unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(0 if result.wasSuccessful() else 1)
PYTEST
  ) > "$ep21_full_run/${receipt%_exit}.txt" 2>&1; then :; else rc=$?; fi
  printf '%s=%s\n' "$receipt" "$rc" >> "$ep21_full_run/batch_status.txt"
  return "$rc"
}
if [[ "$ep21_full_phase" == decode ]]; then
  # Exactly 23 new component methods plus 3 admission methods; one serial pass.
  # Every receipt and ASTRA must pass before the runner can read payloads.
  ep21_full_check test_finding_a_full_empirical_inputs full_input_checks_exit 180
  ep21_full_check test_finding_a_full_models full_model_checks_exit 180
  ep21_full_check test_finding_a_full_readout full_composition_checks_exit 180
  ep21_full_check test_finding_a_draw_batches draw_batch_checks_exit 180
  ep21_full_check test_finding_a_grouped_collection grouped_collection_checks_exit 180
  ep21_full_check test_finding_a_draw_artifacts draw_artifact_checks_exit 180
  ep21_full_check test_finding_a_staged_collection staged_collection_checks_exit 180
  ep21_full_check test_finding_a_distributed_readout distributed_composition_checks_exit 120
  ep21_full_check test_finding_a_full_readout_admission full_readout_admission_checks_exit 180
  ep21_full_astra="${GROUP_HOME:?}/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv"
  ep21_full_astra_deps="${GROUP_HOME}/users/zijiao/ep18_model_runtime_cpu241_py312_46842156/lib/python3.12/site-packages"
  ep21_full_timeout=$(ep21_full_limit 30)
  ep21_full_rc=0
  if env PYTHONPATH="$ep21_full_astra/lib/python3.12/site-packages:$ep21_full_astra_deps" \
      timeout --signal=TERM --kill-after=5s "${ep21_full_timeout}s" "$ep21_full_astra/bin/astra-tools" \
      validate outputs/astra/v0.0.14/astra.yaml --json \
      > "$ep21_full_run/astra_validation.json" 2>&1; then :; else ep21_full_rc=$?; fi
  printf 'astra_exit=%s\n' "$ep21_full_rc" >> "$ep21_full_run/batch_status.txt"
  [[ "$ep21_full_rc" -eq 0 ]] || exit "$ep21_full_rc"
fi
# Draw/finalize admission reuses the exact decode receipts and predecessor
# evidence. No test rerun, scheduler polling or automatic follow-on occurs.
ep21_full_timeout=$(ep21_full_limit 60)
ep21_full_rc=0
if ep21_full_remaining=$(timeout --signal=TERM --kill-after=5s "${ep21_full_timeout}s" \
    python3.12 -B "$ep21_full_entry" "${ep21_full_args[@]}" --remaining-seconds \
    2> "$ep21_full_run/remaining_stderr.txt"); then :; else ep21_full_rc=$?; fi
printf 'remaining_admission_exit=%s\n' "$ep21_full_rc" >> "$ep21_full_run/batch_status.txt"
[[ "$ep21_full_rc" -eq 0 && "$ep21_full_remaining" =~ ^[0-9]+$ \
  && "$ep21_full_remaining" -gt 0 ]] || exit 2
ep21_full_timeout=$(ep21_full_limit "$ep21_full_remaining")
printf 'phase_timeout_seconds=%s\n' "$ep21_full_timeout" >> "$ep21_full_run/batch_status.txt"
ep21_full_phase_rc=running
if timeout --signal=TERM --kill-after=10s "${ep21_full_timeout}s" python3.12 -u -B \
    "$ep21_full_entry" "${ep21_full_args[@]}" > "$ep21_full_run/execution.txt" 2>&1; then ep21_full_phase_rc=0
else ep21_full_phase_rc=$?; fi
exit "$ep21_full_phase_rc"
