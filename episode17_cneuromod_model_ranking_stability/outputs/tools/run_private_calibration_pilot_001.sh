#!/bin/bash
# Run once inside the coordinator's attended allocation. No scheduling,
# waiting, retry, reader release, full-mode escalation or sibling-EP child.
set -euo pipefail
[[ $# -eq 0 ]] || exit 2
ep17_calibration_host=$(hostname)
printf 'HOST=%s JOB=%s CPUS=%s\n' "$ep17_calibration_host" \
  "${SLURM_JOB_ID:?}" "${SLURM_CPUS_PER_TASK:?}"
[[ "$ep17_calibration_host" == sh[0-9][0-9]-[0-9][0-9]n* &&
   "$ep17_calibration_host" != *-ln* && "$SLURM_CPUS_PER_TASK" == 1 &&
   "$SLURM_JOB_ID" =~ ^[0-9]+$ ]]
[[ -z "${SLURM_JOB_GPUS:-}" && -z "${SLURM_STEP_GPUS:-}" &&
   "${SLURM_GPUS_ON_NODE:-0}" == 0 && "${SLURM_GPUS:-0}" == 0 ]]

# Inherit the already selected module runtime; do not install, reload or
# replace the coordinator's interpreter/dependencies in this child wrapper.
for ep17_calibration_module in python/3.12.1 py-numpy/1.26.3_py312 \
  py-scipy/1.12.0_py312 py-h5py/3.10.0_py312; do
  [[ ":${LOADEDMODULES:-}:" == *":${ep17_calibration_module}:"* ]] || {
    printf 'Required inherited module missing: %s\n' "$ep17_calibration_module" >&2
    exit 2
  }
done
cd "${OAK:?}/users/zijiao/br_autoresearch/episode17_cneuromod_model_ranking_stability"
export LC_ALL=C OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
umask 077
[[ -d "${L_SCRATCH_JOB:?}" ]]
export TMPDIR="$L_SCRATCH_JOB/ep17_calibration_pilot_${SLURM_JOB_ID}"
mkdir "$TMPDIR"
# Normalize only this newly created empty task-owned directory, including a
# possible inherited setgid bit; existing directories were refused by mkdir.
chmod u=rwx,go=,g-s "$TMPDIR"
ep17_calibration_run="$PWD/outputs/runs/calibration_support/compute_job_${SLURM_JOB_ID}"
mkdir -p "$PWD/outputs/runs/calibration_support"
mkdir "$ep17_calibration_run"
printf 'job=%s\nhost=%s\nchild_virtual_memory_KiB=2097152\nreader_handoffs_activated=false\n' \
  "$SLURM_JOB_ID" "$ep17_calibration_host" > "$ep17_calibration_run/status.txt"

ep17_calibration_finish() {
  ep17_calibration_wrapper_exit=$?
  printf 'wrapper_exit=%s\n' "$ep17_calibration_wrapper_exit" >> "$ep17_calibration_run/status.txt"
}
trap ep17_calibration_finish EXIT

# Six new source-free checks only; established numerical/copy checks are reused.
set +e
(
  ulimit -v 2097152 || exit 2
  exec timeout --signal=TERM --kill-after=10s 90s python3.12 -B \
    -m unittest discover -s outputs/tests -p test_calibrate_private_roles.py -v
) > "$ep17_calibration_run/adapter_tests.txt" 2>&1
ep17_calibration_tests_exit=$?
set -e
printf 'adapter_tests_exit=%s\n' "$ep17_calibration_tests_exit" >> "$ep17_calibration_run/status.txt"
[[ "$ep17_calibration_tests_exit" == 0 ]] || exit "$ep17_calibration_tests_exit"

# Exact existing pinned validator/runtime supplied by the coordinator. A
# tooling/schema result is retained for review; it does not release data roles
# or create an additional scientific gate for the separately selected pilot.
ep17_calibration_astra="${GROUP_HOME:?}/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv"
ep17_calibration_astra_dependency="$GROUP_HOME/users/zijiao/ep18_model_runtime_cpu241_py312_46842156/lib/python3.12/site-packages"
set +e
(
  ulimit -v 2097152 || exit 2
  export PYTHONPATH="$ep17_calibration_astra/lib/python3.12/site-packages:$ep17_calibration_astra_dependency"
  exec timeout --signal=TERM --kill-after=5s 30s "$ep17_calibration_astra/bin/astra-tools" \
    validate outputs/astra/v0.0.14/astra.yaml --json
) > "$ep17_calibration_run/astra_validation.json" 2> "$ep17_calibration_run/astra_validation.err"
ep17_calibration_astra_exit=$?
set -e
printf 'astra_validation_exit=%s\n' "$ep17_calibration_astra_exit" >> "$ep17_calibration_run/status.txt"

# Adapter itself admits only completed full attempt002 and the fixed first128
# sub-01/left original columns, all eligible images/repeats, B/C/D. It enforces
# 300s internal, 2GiB virtual memory and 64MiB derived-output bounds.
set +e
(
  ulimit -v 2097152 || exit 2
  exec timeout --signal=TERM --kill-after=10s 360s python3.12 -B \
    outputs/tools/calibrate_private_roles.py --mode pilot
) > "$ep17_calibration_run/calibration_pilot.txt" 2>&1
ep17_calibration_pilot_exit=$?
set -e
printf 'calibration_pilot_exit=%s\n' "$ep17_calibration_pilot_exit" >> "$ep17_calibration_run/status.txt"
exit "$ep17_calibration_pilot_exit"
