#!/bin/bash
# Separate amended readout wiring; sourced by its three new wrappers only.
# No grant, submission, dependency selection, retry or resume is created here.
# The coordinator supplies fresh phase directories and forty actual model jobs.
set -euo pipefail
[[ $# -eq 1 ]] || exit 2
ep21_amended_phase=$1
ep21_amended_args=(--phase "$ep21_amended_phase")
case "$ep21_amended_phase" in
  decode)
    ep21_amended_cpus=3; ep21_amended_mem=16384; ep21_amended_seconds=255600
    ep21_amended_hours=216; ep21_amended_relative=decode
    ep21_amended_scratch=ep21_amended_full001_decode
    [[ -z "${SLURM_ARRAY_JOB_ID:-}" && -z "${SLURM_ARRAY_TASK_ID:-}" ]] || exit 2
    ;;
  draw)
    ep21_amended_cpus=3; ep21_amended_mem=16384; ep21_amended_seconds=84600
    ep21_amended_hours=72
    [[ "${SLURM_ARRAY_JOB_ID:-}" =~ ^[1-9][0-9]*$ \
      && "${SLURM_ARRAY_TASK_ID:-}" =~ ^(0|[1-9][0-9]?)$ \
      && "${SLURM_ARRAY_TASK_COUNT:-}" == 63 \
      && "${SLURM_ARRAY_TASK_MIN:-}" == 0 \
      && "${SLURM_ARRAY_TASK_MAX:-}" == 62 \
      && "${SLURM_ARRAY_TASK_STEP:-}" == 1 ]] || exit 2
    (( SLURM_ARRAY_TASK_ID <= 62 )) || exit 2
    printf -v ep21_amended_index '%03d' "$SLURM_ARRAY_TASK_ID"
    ep21_amended_relative="draws/batch_$ep21_amended_index"
    ep21_amended_scratch="ep21_amended_full001_draw_$ep21_amended_index"
    ep21_amended_args+=(--batch-index "$SLURM_ARRAY_TASK_ID")
    ;;
  finalize)
    ep21_amended_cpus=1; ep21_amended_mem=7168; ep21_amended_seconds=6900
    ep21_amended_hours=2; ep21_amended_relative=finalize
    ep21_amended_scratch=ep21_amended_full001_finalize
    [[ -z "${SLURM_ARRAY_JOB_ID:-}" && -z "${SLURM_ARRAY_TASK_ID:-}" ]] || exit 2
    ;;
  *) exit 2 ;;
esac
ep21_amended_host=$(hostname)
printf 'HOST=%s JOB=%s PHASE=%s CPUS=%s MEM_MIB=%s\n' "$ep21_amended_host" \
  "${SLURM_JOB_ID:-}" "$ep21_amended_phase" "${SLURM_CPUS_PER_TASK:-}" "${SLURM_MEM_PER_NODE:-}"
[[ "$ep21_amended_host" != *-ln* && "${SLURM_JOB_ID:-}" =~ ^[1-9][0-9]*$ \
  && "${SLURM_JOB_PARTITION:-}" == russpold \
  && "${SLURM_CPUS_PER_TASK:-}" == "$ep21_amended_cpus" \
  && "${SLURM_MEM_PER_NODE:-}" == "$ep21_amended_mem" \
  && "${SLURM_NTASKS:-1}" == 1 && "${SLURM_JOB_NUM_NODES:-1}" == 1 \
  && "${SLURM_GPUS:-0}" == 0 && "${SLURM_GPUS_ON_NODE:-0}" == 0 \
  && -z "${SLURM_JOB_GPUS:-}" && -z "${SLURM_STEP_GPUS:-}" ]] || exit 2
cd "${OAK:?}/users/zijiao/br_autoresearch/episode21_wang_neural_code_conversion_reproduction"
ep21_amended_base=$PWD
ep21_amended_root="$ep21_amended_base/outputs/runs/finding_a_feature_decoding/amended_full001"
ep21_amended_run="$ep21_amended_root/$ep21_amended_relative"
[[ -d "$ep21_amended_run" ]] || exit 2
umask 077
mkdir "$ep21_amended_run/batch_claim" || exit 2
( set -o noclobber
  printf 'parent_episode=EP21\nparent_job=%s\nphase=%s\nmodel_policy=exact_constant_brain_comparator_all20_fresh_fits\nmaximum_allocated_CPU_hours=%s\nbatch_started=true\ncurrent_tests_executed=false\n' \
    "$SLURM_JOB_ID" "$ep21_amended_phase" "$ep21_amended_hours" > "$ep21_amended_run/batch_status.txt" )
ep21_amended_phase_rc=not_run
ep21_amended_finish() {
  ep21_amended_exit=$?
  trap - EXIT
  if [[ "$ep21_amended_phase_rc" == running ]]; then ep21_amended_phase_rc=$ep21_amended_exit; fi
  printf 'phase_exit=%s\nbatch_exit=%s\n' "$ep21_amended_phase_rc" "$ep21_amended_exit" \
    >> "$ep21_amended_run/batch_status.txt"
  if (( ep21_amended_exit > 128 && ep21_amended_exit < 193 )); then
    printf 'terminating_signal=%s\n' "$((ep21_amended_exit - 128))" >> "$ep21_amended_run/batch_status.txt"
  fi
  printf 'status=closed\n' >> "$ep21_amended_run/batch_status.txt"
  exit "$ep21_amended_exit"
}
trap ep21_amended_finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
ulimit -c 0
# Keep the hard limit intact. The isolated admitted runner selects the finite
# collection soft cap and fatal SIGXFSZ behavior before scientific imports.
ulimit -S -f 16384
# Capture this actual element once; subsequent admission reads this record.
TZ=UTC timeout --signal=TERM --kill-after=5s 15s scontrol show job -o "$SLURM_JOB_ID" \
  > "$ep21_amended_run/allocation.txt"
ep21_amended_allocation=$(< "$ep21_amended_run/allocation.txt")
[[ "$ep21_amended_allocation" =~ (^|[[:space:]])JobId=([0-9]+)($|[[:space:]]) \
  && "${BASH_REMATCH[2]}" == "$SLURM_JOB_ID" ]] || exit 2
[[ "$ep21_amended_allocation" =~ StartTime=([^[:space:]]+) ]] || exit 2
ep21_amended_start=$(date -u -d "${BASH_REMATCH[1]}" +%s)
[[ "$ep21_amended_allocation" =~ EndTime=([^[:space:]]+) ]] || exit 2
ep21_amended_end=$(date -u -d "${BASH_REMATCH[1]}" +%s)
ep21_amended_now=$(date -u +%s)
(( ep21_amended_start <= ep21_amended_now && ep21_amended_now < ep21_amended_end )) || exit 2
ep21_amended_deadline=$((ep21_amended_start + ep21_amended_seconds))
if (( ep21_amended_end - 60 < ep21_amended_deadline )); then
  ep21_amended_deadline=$((ep21_amended_end - 60))
fi
printf 'absolute_deadline_epoch=%s\n' "$ep21_amended_deadline" >> "$ep21_amended_run/batch_status.txt"
ep21_amended_limit() {
  local requested=$1 available
  available=$((ep21_amended_deadline - $(date -u +%s) - 15))
  (( available > 0 )) || return 124
  if (( available < requested )); then requested=$available; fi
  printf '%s\n' "$requested"
}

source /share/software/user/open/lmod/lmod/init/bash
module purge
unset PYTHONPATH PYTHONHOME
module load devel python/3.12.1 math py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 py-h5py/3.10.0_py312 py-pytorch/2.4.1_py312
# Preserve the scientific module path, with one numerical thread per process.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=''
[[ -d "${L_SCRATCH_JOB:?}" ]] || exit 2
export TMPDIR="$L_SCRATCH_JOB/$ep21_amended_scratch"
mkdir "$TMPDIR"
export XDG_CACHE_HOME="$TMPDIR/cache" TORCH_HOME="$TMPDIR/torch"
ep21_amended_entry="$ep21_amended_base/outputs/converter_training/tools/run_amended_full_readout.py"
ep21_amended_rc=0
ep21_amended_timeout=$(ep21_amended_limit 60)
if timeout --signal=TERM --kill-after=5s "${ep21_amended_timeout}s" python3.12 -B \
    "$ep21_amended_entry" "${ep21_amended_args[@]}" --check-admission \
    > "$ep21_amended_run/admission.txt" 2>&1; then :; else ep21_amended_rc=$?; fi
printf 'admission_exit=%s\n' "$ep21_amended_rc" >> "$ep21_amended_run/batch_status.txt"
[[ "$ep21_amended_rc" -eq 0 ]] || exit "$ep21_amended_rc"

if [[ "$ep21_amended_phase" == decode ]]; then
  # Reuse the actual eight-method PASS once. This is bounded metadata parsing,
  # not a test rerun. Copy only the two verified original check-result lines;
  # record source/job/date separately so no current test execution is implied.
  ep21_amended_evidence="$ep21_amended_base/outputs/runs/software_integration/amended_readout_companion001"
  ep21_amended_timeout=$(ep21_amended_limit 15)
  timeout --signal=TERM --kill-after=5s "${ep21_amended_timeout}s" python3.12 -B - \
    "$ep21_amended_evidence" <<'PY' > "$ep21_amended_run/reused_check_evidence.txt"
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
def bounded(path, cap):
    with path.open('rb') as stream:
        value = stream.read(cap + 1)
    if len(value) > cap:
        raise ValueError('prior_check_metadata_cap')
    return value.decode('utf-8')

lines = bounded(root / 'status.txt', 65536).splitlines()
required = {'amended_readout_checks_exit': '0', 'amended_readout_check_methods': '8',
            'phase': 'completed', 'job_id': '47584681', 'empirical_inputs': 'false'}
for key, expected in required.items():
    matches = [line for line in lines if line.startswith(key + '=')]
    if matches != [key + '=' + expected]:
        raise ValueError('prior_eight_method_status_not_complete')
checks = bounded(root / 'checks.txt', 131072)
methods = [line for line in checks.splitlines() if line.startswith('test_')]
if (len(methods) != 8 or len(set(methods)) != 8
        or any(not line.endswith(' ... ok') for line in methods)
        or len(re.findall(r'^Ran 8 tests in [0-9.]+s$', checks, re.MULTILINE)) != 1
        or checks.rstrip().splitlines()[-1] != 'OK'
        or 'job_id=47584681' not in checks.splitlines()[0]):
    raise ValueError('prior_eight_method_execution_evidence_incomplete')
for key in ('amended_readout_checks_exit', 'amended_readout_check_methods'):
    print(next(line for line in lines if line.startswith(key + '=')))
print('amended_readout_checks_reused=true')
print('amended_readout_checks_prior_job=47584681')
print('amended_readout_checks_recorded_date=2026-10-12_Pacific')
print('amended_readout_checks_status_source=' + str(root / 'status.txt'))
print('amended_readout_checks_execution_source=' + str(root / 'checks.txt'))
print('amended_readout_checks_prior_charge=EP18_recovery_once')
PY
  cat "$ep21_amended_run/reused_check_evidence.txt" >> "$ep21_amended_run/batch_status.txt"
  # The changed ASTRA document is a tooling record, not a scientific gate.
  ep21_amended_astra="${GROUP_HOME:?}/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv"
  ep21_amended_astra_deps="${GROUP_HOME}/users/zijiao/ep18_model_runtime_cpu241_py312_46842156/lib/python3.12/site-packages"
  ep21_amended_timeout=$(ep21_amended_limit 30)
  ep21_amended_rc=0
  if env PYTHONPATH="$ep21_amended_astra/lib/python3.12/site-packages:$ep21_amended_astra_deps" \
      timeout --signal=TERM --kill-after=5s "${ep21_amended_timeout}s" "$ep21_amended_astra/bin/astra-tools" \
      validate outputs/astra/v0.0.14/astra.yaml --json \
      > "$ep21_amended_run/astra_validation.json" 2>&1; then :; else ep21_amended_rc=$?; fi
  printf 'astra_exit=%s\nastra_is_scientific_gate=false\n' "$ep21_amended_rc" >> "$ep21_amended_run/batch_status.txt"
fi
# Later phases reuse decode's original PASS provenance and require all closed
# successful predecessors. No tests, polling loop or automatic successor.
ep21_amended_timeout=$(ep21_amended_limit 60)
ep21_amended_rc=0
if ep21_amended_remaining=$(timeout --signal=TERM --kill-after=5s "${ep21_amended_timeout}s" \
    python3.12 -B "$ep21_amended_entry" "${ep21_amended_args[@]}" --remaining-seconds \
    2> "$ep21_amended_run/remaining_stderr.txt"); then :; else ep21_amended_rc=$?; fi
printf 'remaining_admission_exit=%s\n' "$ep21_amended_rc" >> "$ep21_amended_run/batch_status.txt"
[[ "$ep21_amended_rc" -eq 0 && "$ep21_amended_remaining" =~ ^[0-9]+$ \
  && "$ep21_amended_remaining" -gt 0 ]] || exit 2
ep21_amended_timeout=$(ep21_amended_limit "$ep21_amended_remaining")
printf 'phase_timeout_seconds=%s\n' "$ep21_amended_timeout" >> "$ep21_amended_run/batch_status.txt"
ep21_amended_phase_rc=running
if timeout --signal=TERM --kill-after=10s "${ep21_amended_timeout}s" python3.12 -u -B \
    "$ep21_amended_entry" "${ep21_amended_args[@]}" > "$ep21_amended_run/execution.txt" 2>&1; then ep21_amended_phase_rc=0
else ep21_amended_phase_rc=$?; fi
exit "$ep21_amended_phase_rc"
