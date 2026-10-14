#!/bin/bash
# Serial fixture-only companion in the selected EP18 recovery allocation.
# The coordinator wraps this whole script in timeout TERM180s/kill-after5s,
# shortened when necessary by the parent's original absolute deadline.
# Its exit is recorded independently and never gates EP18 image extraction.
# No scheduler operation, empirical input, readout grant or training is used.
set -euo pipefail
umask 077

[[ $# == 1 ]]
ep21_parent_deadline=$1
[[ "$ep21_parent_deadline" =~ ^[0-9]+$ ]]
[[ -n "${SLURM_JOB_ID:-}" && "$(hostname)" != *-ln* ]]
[[ "${SLURM_CPUS_PER_TASK:-}" == 2 && "${SLURM_MEM_PER_NODE:-}" == 15360 ]]
[[ "${SLURM_JOB_NUM_NODES:-}" == 1 && "${SLURM_JOB_CPUS_PER_NODE:-}" == 2 ]]
[[ -z "${SLURM_JOB_GPUS:-}" && -z "${SLURM_STEP_GPUS:-}" ]]
: "${OAK:?}" "${L_SCRATCH_JOB:?}" "${SLURM_JOB_START_TIME:?}" "${SLURM_JOB_END_TIME:?}"
[[ "$SLURM_JOB_START_TIME" =~ ^[0-9]+$ && "$SLURM_JOB_END_TIME" =~ ^[0-9]+$ ]]
ep21_started=$(date +%s)
(( SLURM_JOB_START_TIME <= ep21_started && ep21_started < SLURM_JOB_END_TIME ))
(( ep21_parent_deadline <= SLURM_JOB_START_TIME + 14400 - 60 ))
(( ep21_parent_deadline <= SLURM_JOB_END_TIME - 60 && ep21_parent_deadline <= 1791926990 ))
(( ep21_parent_deadline > ep21_started + 5 ))
ep21_deadline=$((ep21_started+175))
if (( ep21_parent_deadline-5 < ep21_deadline )); then ep21_deadline=$((ep21_parent_deadline-5)); fi
ep21_episode="$OAK/users/zijiao/br_autoresearch/episode21_wang_neural_code_conversion_reproduction"
ep21_tools="$ep21_episode/outputs/converter_training/tools"
ep21_tests="$ep21_episode/outputs/converter_training/tests"
ep21_operations="$ep21_episode/outputs/runs/software_integration/amended_readout_companion001"
mkdir -p -m 700 "$ep21_episode/outputs/runs/software_integration"
mkdir -m 700 "$ep21_operations"
ep21_stage=runtime
ep21_methods=0
finish() {
  ep21_exit=$?
  trap - EXIT
  if [[ -f "$ep21_operations/method_count.txt" ]]; then
    read -r ep21_methods < "$ep21_operations/method_count.txt" || ep21_methods=0
    [[ "$ep21_methods" =~ ^[0-9]+$ ]] || ep21_methods=0
  fi
  printf 'amended_readout_checks_exit=%s\namended_readout_check_methods=%s\nphase=%s\njob_id=%s\nactual_parent_cpus=2\nactual_parent_memory_mib=15360\nnumerical_threads=1\nseparate_RSS_guard=false\ncharged_once_to=EP18_recovery\nEP18_gate=false\nempirical_inputs=false\nparent_deadline_unix=%s\ncompanion_deadline_unix=%s\n' \
    "$ep21_exit" "$ep21_methods" "$ep21_stage" "$SLURM_JOB_ID" "$ep21_parent_deadline" "$ep21_deadline" \
    > "$ep21_operations/status.txt"
  exit "$ep21_exit"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
exec > "$ep21_operations/checks.txt" 2>&1
printf 'hostname=%s job_id=%s parent_cpus=2 parent_memory_mib=15360 fixtures_only=true\n' \
  "$(hostname)" "$SLURM_JOB_ID"

source /share/software/user/open/lmod/lmod/init/bash
module purge
# Remove inherited interpreter routing BEFORE loading the qualified modules.
# The module-provided PYTHONPATH is required and remains in effect below.
unset PYTHONPATH PYTHONHOME
module load devel python/3.12.1 math py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 py-h5py/3.10.0_py312 py-pytorch/2.4.1_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
ep21_local="$L_SCRATCH_JOB/ep21-amended-readout-companion001"
mkdir -m 700 "$ep21_local"
mkdir -m 700 "$ep21_local/tmp" "$ep21_local/cache"
export TMPDIR="$ep21_local/tmp" XDG_CACHE_HOME="$ep21_local/cache"
cd "$ep21_local"
ep21_now=$(date +%s)
ep21_remaining=$((ep21_deadline-ep21_now))
(( ep21_remaining > 0 ))
ep21_stage=fixture_checks
set +e
timeout --signal=TERM --kill-after=5s "${ep21_remaining}s" python3 -B - \
  "$ep21_tools" "$ep21_tests" "$ep21_operations/method_count.txt" <<'PY'
import sys
import unittest
from pathlib import Path

sys.path.insert(0, sys.argv[1])
sys.path.insert(0, sys.argv[2])
names = ("test_amended_brain_readout", "test_amended_distributed_readout",
         "test_amended_full_readout_admission")
suite = unittest.defaultTestLoader.loadTestsFromNames(names)
count = suite.countTestCases()
with Path(sys.argv[3]).open("x", encoding="utf-8") as output:
    output.write(str(count) + "\n")
if count != 8 or unittest.defaultTestLoader.errors:
    raise RuntimeError("Exactly the eight selected fixture methods must load")
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(0 if result.wasSuccessful() and result.testsRun == 8 and not result.skipped else 1)
PY
ep21_checks_exit=$?
set -e
if (( ep21_checks_exit == 0 )); then ep21_stage=completed; else ep21_stage=fixture_checks_failed; fi
exit "$ep21_checks_exit"
