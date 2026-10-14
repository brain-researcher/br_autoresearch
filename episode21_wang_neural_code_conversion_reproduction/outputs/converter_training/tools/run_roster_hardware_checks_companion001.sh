#!/bin/bash
# One attended fixture-only step in the already selected EP18 recovery parent.
set -euo pipefail
umask 077
[[ "${SLURM_JOB_ID:-}" == 47584681 && "$(hostname)" != *-ln* ]]
[[ -n "${SLURM_STEP_ID:-}" && "${SLURM_CPUS_PER_TASK:-}" == 1 ]]
[[ -z "${SLURM_JOB_GPUS:-}" && -z "${SLURM_STEP_GPUS:-}" ]]
(( $(date +%s) + 185 < 1791867777 ))
: "${OAK:?}" "${L_SCRATCH_JOB:?}"
ep21_root="$OAK/users/zijiao/br_autoresearch/episode21_wang_neural_code_conversion_reproduction"
ep21_code="$ep21_root/outputs/converter_training"
ep21_ops="$ep21_root/outputs/runs/software_integration/roster_hardware_companion001"
mkdir -m 700 "$ep21_ops"
ep21_phase=runtime
finish() {
  ep21_exit=$?
  trap - EXIT
  printf 'exit=%s\nphase=%s\njob_id=%s\nstep_id=%s\nstep_cpus=1\nstep_memory_mib=2048\nmethods_selected=5\nempirical_inputs=false\ncharged_once_to=EP18_recovery\n' "$ep21_exit" "$ep21_phase" "$SLURM_JOB_ID" "$SLURM_STEP_ID" > "$ep21_ops/status.txt"
  exit "$ep21_exit"
}
trap finish EXIT
trap 'exit 143' TERM
exec > "$ep21_ops/checks.txt" 2>&1
printf 'hostname=%s job_id=%s step_id=%s\n' "$(hostname)" "$SLURM_JOB_ID" "$SLURM_STEP_ID"
source /share/software/user/open/lmod/lmod/init/bash
module purge
unset PYTHONPATH PYTHONHOME
module load devel python/3.12.1 math py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 py-h5py/3.10.0_py312 py-pytorch/2.4.1_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES='' PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
ep21_local="$L_SCRATCH_JOB/ep21-roster-hardware-companion001"
mkdir -m 700 "$ep21_local"
export TMPDIR="$ep21_local" XDG_CACHE_HOME="$ep21_local/cache"
cd "$ep21_local"
ep21_phase=fixture_checks
python3 -B - "$ep21_code/tests" <<'PY'
import sys
import unittest
sys.path.insert(0, sys.argv[1])
suite = unittest.defaultTestLoader.loadTestsFromNames(("test_content_cuda_roster_admission", "test_content_cuda_roster_hardware"))
if suite.countTestCases() != 5 or unittest.defaultTestLoader.errors:
    raise RuntimeError("Exactly five selected methods must load")
result = unittest.TextTestRunner(verbosity=2).run(suite)
if not result.wasSuccessful() or result.testsRun != 5 or result.skipped:
    raise SystemExit(1)
PY
ep21_phase=wrapper_syntax
bash -n "$ep21_code/finding_a_amended_full_common.sh" "$ep21_code/finding_a_amended_full_decode.sbatch" "$ep21_code/finding_a_amended_full_draw.sbatch" "$ep21_code/finding_a_amended_full_finalize.sbatch"
ep21_phase=completed
