#!/bin/bash
# One attended, separately selected software-only step in the existing EP21
# parent. No submissions, empirical calls, CUDA arithmetic, retries or fits.
set -euo pipefail
[[ $# -eq 0 ]] || exit 2
ep21_cpu_host=$(hostname)
[[ "${ep21_cpu_host%%.*}" == sh03-06n11 && "${SLURM_JOB_ID:-}" == 47376911 \
  && "${SLURM_STEP_ID:-}" =~ ^[0-9]+$ && "${SLURM_CPUS_PER_TASK:-}" == 1 \
  && "${SLURM_MEM_PER_NODE:-}" == 16384 && "${SLURM_NTASKS:-1}" == 1 ]] || exit 2
# The coordinator requested1CPU/4GiB/10min for this step. Slurm inherits the
# parent's16GiB environment value; this is not a claim of hard memory isolation.
ep21_cpu_started=$(date -u +%s)
ep21_cpu_deadline=$((ep21_cpu_started + 600))
if (( 1791744915 < ep21_cpu_deadline )); then ep21_cpu_deadline=1791744915; fi
# This attended step was already opened; reserve shutdown before its14:25:48UTC end.
if (( 1791728730 < ep21_cpu_deadline )); then ep21_cpu_deadline=1791728730; fi
(( ep21_cpu_started < ep21_cpu_deadline - 15 )) || exit 124
cd "${OAK:?}/users/zijiao/br_autoresearch/episode21_wang_neural_code_conversion_reproduction"
ep21_cpu_base=$PWD
ep21_cpu_run="$ep21_cpu_base/outputs/runs/content_launcher_qualification/cpu001"
umask 077
mkdir -p "$ep21_cpu_base/outputs/runs/content_launcher_qualification"
mkdir "$ep21_cpu_run" || exit 2
set -o noclobber
printf 'parent_episode=EP21\nparent_job=%s\nstep=%s\nhost=%s\nrequested_step_cpus=1\nrequested_step_memory_mib=4096\ninherited_parent_memory_mib=%s\nstarted_epoch=%s\nabsolute_deadline_epoch=%s\ndurable_allowance_bytes=67108864\nlocal_allowance_bytes=1073741824\nempirical_inputs=false\nqualification_started=true\n' \
  "$SLURM_JOB_ID" "$SLURM_STEP_ID" "$ep21_cpu_host" "$SLURM_MEM_PER_NODE" \
  "$ep21_cpu_started" "$ep21_cpu_deadline" > "$ep21_cpu_run/status.txt"
ep21_cpu_finish() {
  ep21_cpu_exit=$?
  trap - EXIT
  printf 'qualification_exit=%s\nstatus=closed\n' "$ep21_cpu_exit" >> "$ep21_cpu_run/status.txt"
  exit "$ep21_cpu_exit"
}
trap ep21_cpu_finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
ulimit -c 0
ulimit -S -f 32768
ep21_cpu_limit() {
  local requested=$1 available
  available=$((ep21_cpu_deadline - $(date -u +%s) - 10))
  (( available > 0 )) || return 124
  if (( available < requested )); then requested=$available; fi
  printf '%s\n' "$requested"
}
source /share/software/user/open/lmod/lmod/init/bash
module purge
unset PYTHONPATH PYTHONHOME
module load devel python/3.12.1 math py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 py-h5py/3.10.0_py312 py-pytorch/2.4.1_py312
# Preserve loaded numerical module paths. These existing fixtures explicitly
# stub CUDA; composition fixtures create only tiny CPU objects and mock fits.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=''
[[ -d "${L_SCRATCH_JOB:?}" ]] || exit 2
export TMPDIR="$L_SCRATCH_JOB/ep21_content_launcher_cpu001"
mkdir "$TMPDIR"
export XDG_CACHE_HOME="$TMPDIR/cache" TORCH_HOME="$TMPDIR/torch"
ep21_cpu_checks() {
  local group=$1 limit rc=0
  shift
  limit=$(ep21_cpu_limit 180) || return $?
  if (
    cd "$ep21_cpu_base/outputs/converter_training/tests"
    /usr/bin/time -f 'elapsed_seconds=%e\nmaximum_resident_set_kib=%M\ncommand_exit=%x' \
      -o "$ep21_cpu_run/${group}_time.txt" \
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
  ) > "$ep21_cpu_run/${group}_checks.txt" 2>&1; then :; else rc=$?; fi
  printf '%s_checks_exit=%s\n' "$group" "$rc" >> "$ep21_cpu_run/status.txt"
  return "$rc"
}
ep21_cpu_roster_rc=0
if ep21_cpu_checks roster test_content_cuda_roster_admission test_content_cuda_roster_training; then :
else ep21_cpu_roster_rc=$?; fi
[[ "$ep21_cpu_roster_rc" -eq 0 ]] || exit "$ep21_cpu_roster_rc"
ep21_cpu_remaining_rc=0
if ep21_cpu_checks remaining test_content_cuda_remaining_admission test_content_cuda_remaining_training; then :
else ep21_cpu_remaining_rc=$?; fi
[[ "$ep21_cpu_remaining_rc" -eq 0 ]] || exit "$ep21_cpu_remaining_rc"

# Root adds the prospective references before invoking this final validation.
ep21_cpu_astra="${GROUP_HOME:?}/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv"
ep21_cpu_astra_deps="${GROUP_HOME}/users/zijiao/ep18_model_runtime_cpu241_py312_46842156/lib/python3.12/site-packages"
ep21_cpu_astra_rc=0
ep21_cpu_timeout=$(ep21_cpu_limit 30)
if /usr/bin/time -f 'elapsed_seconds=%e\nmaximum_resident_set_kib=%M\ncommand_exit=%x' \
    -o "$ep21_cpu_run/astra_time.txt" \
    env PYTHONPATH="$ep21_cpu_astra/lib/python3.12/site-packages:$ep21_cpu_astra_deps" \
    timeout --signal=TERM --kill-after=5s "${ep21_cpu_timeout}s" "$ep21_cpu_astra/bin/astra-tools" \
    validate outputs/astra/v0.0.14/astra.yaml --json \
    > "$ep21_cpu_run/astra_validation.json" 2>&1; then :; else ep21_cpu_astra_rc=$?; fi
printf 'astra_exit=%s\n' "$ep21_cpu_astra_rc" >> "$ep21_cpu_run/status.txt"
# Preserve the existing independent ASTRA convention. Its ordinary failure is
# recorded and does not change these ten software checks; OOM terminates work.
[[ "$ep21_cpu_astra_rc" -ne 137 ]] || exit 137
exit 0
