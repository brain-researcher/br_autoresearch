#!/bin/bash
# One fresh finite L40S measurement; no production job or old attempt changes.
#SBATCH --job-name=ep21-l40s-cost
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gpus=1
#SBATCH --constraint="GPU_SKU:L40S&GPU_MEM:48GB"
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/content_resource_measurement/cuda_l40s001/slurm-%j.out
#SBATCH --error=outputs/runs/content_resource_measurement/cuda_l40s001/slurm-%j.err
set -euo pipefail
[[ $# -eq 0 ]] || exit 2
ep21_l40s_host=$(hostname)
printf 'HOST=%s JOB=%s CPUS=%s MEM_MIB=%s GPUS=%s\n' "$ep21_l40s_host" \
  "${SLURM_JOB_ID:?}" "${SLURM_CPUS_PER_TASK:?}" "${SLURM_MEM_PER_NODE:?}" "${SLURM_GPUS_ON_NODE:?}"
[[ "$ep21_l40s_host" != *-ln* && "$SLURM_CPUS_PER_TASK" == 2 \
  && "$SLURM_MEM_PER_NODE" == 16384 && "$SLURM_GPUS_ON_NODE" == 1 ]] || exit 2
cd "${OAK:?}/users/zijiao/br_autoresearch/episode21_wang_neural_code_conversion_reproduction"
ep21_l40s_run="outputs/runs/content_resource_measurement/cuda_l40s001"
[[ -d "$ep21_l40s_run" ]] || exit 2
umask 077
mkdir "$ep21_l40s_run/batch_claim" || exit 2
( set -o noclobber
  printf 'parent_episode=EP21\nparent_job=%s\nstage=l40s-001\nstatus=started\n' \
    "$SLURM_JOB_ID" > "$ep21_l40s_run/status.txt"
)
ep21_l40s_close() {
  ep21_l40s_exit=$?
  trap - EXIT
  printf 'batch_exit=%s\nstatus=closed\n' "$ep21_l40s_exit" >> "$ep21_l40s_run/status.txt"
  exit "$ep21_l40s_exit"
}
trap ep21_l40s_close EXIT
ulimit -c 0
ulimit -S -f 32768
TZ=UTC scontrol show job -o "$SLURM_JOB_ID" > "$ep21_l40s_run/allocation.txt"
source /share/software/user/open/lmod/lmod/init/bash
module purge
unset PYTHONPATH PYTHONHOME
module load devel python/3.12.1 math py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 py-h5py/3.10.0_py312 py-pytorch/2.4.1_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
[[ -d "${L_SCRATCH_JOB:?}" ]] || exit 2
export TMPDIR="$L_SCRATCH_JOB/ep21_content_cuda_l40s001"
mkdir "$TMPDIR"
export XDG_CACHE_HOME="$TMPDIR/cache" CUDA_CACHE_PATH="$TMPDIR/cuda-cache" TORCH_HOME="$TMPDIR/torch"
ep21_l40s_entry="outputs/converter_training/tools/run_content_cuda_l40s_measurement.py"
ep21_l40s_deadline=$(python3.12 -B - "$ep21_l40s_run/allocation.txt" <<'PY'
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
path = Path(sys.argv[1])
assert path.is_file() and path.stat().st_size <= 65536
fields = dict(token.split('=', 1) for token in path.read_text().split() if '=' in token)
def utc_epoch(value):
    stamp = datetime.fromisoformat(value)
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    assert stamp.utcoffset().total_seconds() == 0
    return int(stamp.timestamp())
start, end = utc_epoch(fields['StartTime']), utc_epoch(fields['EndTime'])
assert start <= time.time() < end
deadline = min(start + 6300, end - 60)
assert deadline - time.time() > 15
print(deadline)
PY
)
[[ "$ep21_l40s_deadline" =~ ^[0-9]+$ ]] || exit 2
printf 'absolute_deadline_epoch=%s\n' "$ep21_l40s_deadline" >> "$ep21_l40s_run/status.txt"
ep21_l40s_limit() {
  local ep21_l40s_now ep21_l40s_available ep21_l40s_cap
  ep21_l40s_now=$(date +%s)
  ep21_l40s_available=$((ep21_l40s_deadline - ep21_l40s_now - 15))
  [[ "$ep21_l40s_available" -gt 0 ]] || return 2
  ep21_l40s_cap=$1
  if [[ "$ep21_l40s_available" -lt "$ep21_l40s_cap" ]]; then ep21_l40s_cap=$ep21_l40s_available; fi
  printf '%s\n' "$ep21_l40s_cap"
}
ep21_l40s_admission_rc=0
if timeout --signal=TERM --kill-after=10s "$(ep21_l40s_limit 60)s" \
    python3.12 -B "$ep21_l40s_entry" --check-admission \
    > "$ep21_l40s_run/admission.txt" 2>&1; then :; else ep21_l40s_admission_rc=$?; fi
printf 'admission_exit=%s\n' "$ep21_l40s_admission_rc" >> "$ep21_l40s_run/status.txt"
[[ "$ep21_l40s_admission_rc" -eq 0 ]] || exit "$ep21_l40s_admission_rc"
ep21_l40s_checks_rc=0
if (
  cd outputs/converter_training/tests
  timeout --signal=TERM --kill-after=10s "$(ep21_l40s_limit 120)s" python3.12 -B - <<'PY'
import sys
import torch
import unittest
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
print('torch=' + str(torch.__version__), flush=True)
suite = unittest.defaultTestLoader.loadTestsFromNames([
    'test_content_cuda_l40s_admission',
    'test_content_cuda.ContentCudaTests.test_pinned_converter_cuda_sparse_step_matches_CPU_dense_autograd',
])
result = unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(0 if result.wasSuccessful() and result.testsRun == 2 and not result.skipped else 1)
PY
) > "$ep21_l40s_run/cuda_checks.txt" 2>&1; then :; else ep21_l40s_checks_rc=$?; fi
printf 'cuda_checks_exit=%s\ncuda_check_methods=2\n' "$ep21_l40s_checks_rc" >> "$ep21_l40s_run/status.txt"
[[ "$ep21_l40s_checks_rc" -eq 0 ]] || exit "$ep21_l40s_checks_rc"
ep21_l40s_astra="${GROUP_HOME:?}/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv"
ep21_l40s_astra_deps="${GROUP_HOME}/users/zijiao/ep18_model_runtime_cpu241_py312_46842156/lib/python3.12/site-packages"
ep21_l40s_astra_rc=0
if PYTHONPATH="$ep21_l40s_astra/lib/python3.12/site-packages:$ep21_l40s_astra_deps" \
    timeout --signal=TERM --kill-after=10s "$(ep21_l40s_limit 30)s" "$ep21_l40s_astra/bin/astra-tools" \
    validate outputs/astra/v0.0.14/astra.yaml --json \
    > "$ep21_l40s_run/astra_validation.json" 2>&1; then :; else ep21_l40s_astra_rc=$?; fi
printf 'astra_exit=%s\nastra_is_scientific_gate=false\n' "$ep21_l40s_astra_rc" >> "$ep21_l40s_run/status.txt"
ep21_l40s_remaining=$(ep21_l40s_limit 6300)
printf 'measurement_timeout_seconds=%s\n' "$ep21_l40s_remaining" >> "$ep21_l40s_run/status.txt"
ep21_l40s_measurement_rc=0
if timeout --signal=TERM --kill-after=10s "${ep21_l40s_remaining}s" \
    python3.12 -u -B "$ep21_l40s_entry" > "$ep21_l40s_run/measurement.txt" 2>&1; then :
else ep21_l40s_measurement_rc=$?; fi
printf 'measurement_exit=%s\n' "$ep21_l40s_measurement_rc" >> "$ep21_l40s_run/status.txt"
exit "$ep21_l40s_measurement_rc"
