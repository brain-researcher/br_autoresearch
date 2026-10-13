#!/bin/bash
# One finite empirical cache measurement; all children share its absolute clock.
#SBATCH --job-name=ep21-cuda-cache-r
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gpus=1
#SBATCH --constraint="GPU_MEM:24GB|GPU_MEM:32GB|GPU_MEM:48GB|GPU_MEM:80GB"
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/content_resource_measurement/cuda_cache002/slurm-%j.out
#SBATCH --error=outputs/runs/content_resource_measurement/cuda_cache002/slurm-%j.err
set -euo pipefail
[[ $# -eq 0 ]] || exit 2
ep21_cache_host=$(hostname)
printf 'HOST=%s JOB=%s CPUS=%s MEM_MIB=%s GPUS=%s\n' "$ep21_cache_host" \
  "${SLURM_JOB_ID:?}" "${SLURM_CPUS_PER_TASK:?}" "${SLURM_MEM_PER_NODE:?}" "${SLURM_GPUS_ON_NODE:?}"
[[ "$ep21_cache_host" != *-ln* && "$SLURM_CPUS_PER_TASK" == 2 \
  && "$SLURM_MEM_PER_NODE" == 16384 && "$SLURM_GPUS_ON_NODE" == 1 ]] || exit 2
cd "${OAK:?}/users/zijiao/br_autoresearch/episode21_wang_neural_code_conversion_reproduction"
ep21_cache_run="outputs/runs/content_resource_measurement/cuda_cache002"
[[ -d "$ep21_cache_run" ]] || exit 2
mkdir "$ep21_cache_run/batch_claim" || exit 2
printf 'parent_episode=EP21\nparent_job=%s\nseparate_measurement_grant=true\nstatus=started\n' \
  "$SLURM_JOB_ID" > "$ep21_cache_run/status.txt"
ulimit -c 0
ulimit -S -f 32768
TZ=UTC scontrol show job -o "$SLURM_JOB_ID" > "$ep21_cache_run/allocation.txt"
source /share/software/user/open/lmod/lmod/init/bash
module purge
unset PYTHONPATH PYTHONHOME
module load devel python/3.12.1 math py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 py-h5py/3.10.0_py312 py-pytorch/2.4.1_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
[[ -d "${L_SCRATCH_JOB:?}" ]] || exit 2
export TMPDIR="$L_SCRATCH_JOB/ep21_content_cuda_cache002"
umask 077
mkdir "$TMPDIR"
export XDG_CACHE_HOME="$TMPDIR/cache" CUDA_CACHE_PATH="$TMPDIR/cuda-cache" TORCH_HOME="$TMPDIR/torch"
ep21_cache_entry="outputs/converter_training/tools/run_content_cuda_measurement.py"
ep21_cache_admission_rc=0
if timeout --signal=TERM --kill-after=5s 60s python3.12 -B "$ep21_cache_entry" \
    --stage cache-002 --check-admission > "$ep21_cache_run/admission.txt" 2>&1; then :
else ep21_cache_admission_rc=$?; fi
printf 'admission_exit=%s\n' "$ep21_cache_admission_rc" >> "$ep21_cache_run/status.txt"
if [[ "$ep21_cache_admission_rc" -ne 0 ]]; then
  printf 'status=closed\n' >> "$ep21_cache_run/status.txt"
  exit "$ep21_cache_admission_rc"
fi

ep21_cache_cuda_rc=0
if (
  cd outputs/converter_training/tests
  timeout --signal=TERM --kill-after=5s 120s python3.12 -B - <<'PY'
import sys
import torch
import unittest
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
print('torch=' + str(torch.__version__), flush=True)
suite = unittest.defaultTestLoader.loadTestsFromNames(['test_content_cuda_admission', 'test_content_cuda'])
result = unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(0 if result.wasSuccessful() else 1)
PY
) > "$ep21_cache_run/cuda_checks.txt" 2>&1; then :; else ep21_cache_cuda_rc=$?; fi
printf 'cuda_checks_exit=%s\n' "$ep21_cache_cuda_rc" >> "$ep21_cache_run/status.txt"

ep21_cache_integration_rc=0
if (
  export CUDA_VISIBLE_DEVICES=''
  cd outputs/converter_training/tests
  timeout --signal=TERM --kill-after=5s 90s python3.12 -B - <<'PYTEST'
import sys
import torch
import unittest
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
suite = unittest.defaultTestLoader.loadTestsFromNames([
    'test_content_channel_cache.ContentChannelCacheTests.test_exact_two_pass_arrays_and_loss_adjoint_with_one_upstream_pass'])
result = unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(0 if result.wasSuccessful() else 1)
PYTEST
) > "$ep21_cache_run/cache_order_checks.txt" 2>&1; then :; else ep21_cache_integration_rc=$?; fi
printf 'cache_order_checks_exit=%s\n' "$ep21_cache_integration_rc" >> "$ep21_cache_run/status.txt"

ep21_cache_astra="${GROUP_HOME:?}/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv"
ep21_cache_astra_deps="${GROUP_HOME}/users/zijiao/ep18_model_runtime_cpu241_py312_46842156/lib/python3.12/site-packages"
ep21_cache_astra_rc=0
if PYTHONPATH="$ep21_cache_astra/lib/python3.12/site-packages:$ep21_cache_astra_deps" \
    timeout --signal=TERM --kill-after=5s 30s "$ep21_cache_astra/bin/astra-tools" \
    validate outputs/astra/v0.0.14/astra.yaml --json \
    > "$ep21_cache_run/astra_validation.json" 2>&1; then :; else ep21_cache_astra_rc=$?; fi
printf 'astra_exit=%s\n' "$ep21_cache_astra_rc" >> "$ep21_cache_run/status.txt"

ep21_cache_fit_rc=2
if [[ "$ep21_cache_cuda_rc" -eq 0 && "$ep21_cache_integration_rc" -eq 0 ]]; then
  if ep21_cache_remaining=$(timeout --signal=TERM --kill-after=5s 60s python3.12 -B \
      "$ep21_cache_entry" --stage cache-002 --remaining-seconds \
      2> "$ep21_cache_run/remaining_stderr.txt") \
      && [[ "$ep21_cache_remaining" =~ ^[0-9]+$ && "$ep21_cache_remaining" -gt 0 ]]; then
    printf 'measurement_timeout_seconds=%s\n' "$ep21_cache_remaining" >> "$ep21_cache_run/status.txt"
    if timeout --signal=TERM --kill-after=10s "${ep21_cache_remaining}s" python3.12 -u -B \
        "$ep21_cache_entry" --stage cache-002 > "$ep21_cache_run/measurement.txt" 2>&1; then
      ep21_cache_fit_rc=0
    else ep21_cache_fit_rc=$?; fi
  fi
fi
printf 'measurement_exit=%s\nstatus=closed\n' "$ep21_cache_fit_rc" >> "$ep21_cache_run/status.txt"
exit "$ep21_cache_fit_rc"
