#!/bin/bash
# One complete original CUDA pair; all children share its absolute allocation clock.
#SBATCH --job-name=ep21-content-full
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gpus=1
#SBATCH --constraint="GPU_SKU:V100_SXM2&GPU_MEM:32GB"
#SBATCH --time=2-00:00:00
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/content_pair_training/cuda001/slurm-%j.out
#SBATCH --error=outputs/runs/content_pair_training/cuda001/slurm-%j.err
set -euo pipefail
[[ $# -eq 0 ]] || exit 2
ep21_full_host=$(hostname)
printf 'HOST=%s JOB=%s CPUS=%s MEM_MIB=%s GPUS=%s\n' "$ep21_full_host" \
  "${SLURM_JOB_ID:?}" "${SLURM_CPUS_PER_TASK:?}" "${SLURM_MEM_PER_NODE:?}" "${SLURM_GPUS_ON_NODE:?}"
[[ "$ep21_full_host" != *-ln* && "$SLURM_CPUS_PER_TASK" == 2 \
  && "$SLURM_MEM_PER_NODE" == 16384 && "$SLURM_GPUS_ON_NODE" == 1 ]] || exit 2
cd "${OAK:?}/users/zijiao/br_autoresearch/episode21_wang_neural_code_conversion_reproduction"
ep21_full_run="outputs/runs/content_pair_training/cuda001"
[[ -d "$ep21_full_run" ]] || exit 2
mkdir "$ep21_full_run/batch_claim" || exit 2
printf 'parent_episode=EP21\nparent_job=%s\nseparate_full_production_grant=true\nstatus=started\n' \
  "$SLURM_JOB_ID" > "$ep21_full_run/status.txt"
ulimit -c 0
ulimit -S -f 1048576
TZ=UTC scontrol show job -o "$SLURM_JOB_ID" > "$ep21_full_run/allocation.txt"
source /share/software/user/open/lmod/lmod/init/bash
module purge
unset PYTHONPATH PYTHONHOME
module load devel python/3.12.1 math py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 py-h5py/3.10.0_py312 py-pytorch/2.4.1_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
[[ -d "${L_SCRATCH_JOB:?}" ]] || exit 2
export TMPDIR="$L_SCRATCH_JOB/ep21_content_cuda_pair001"
umask 077
mkdir "$TMPDIR"
export XDG_CACHE_HOME="$TMPDIR/cache" CUDA_CACHE_PATH="$TMPDIR/cuda-cache" TORCH_HOME="$TMPDIR/torch"
ep21_full_entry="outputs/converter_training/tools/run_content_cuda_pair.py"
ep21_full_admission_rc=0
if timeout --signal=TERM --kill-after=5s 60s python3.12 -B "$ep21_full_entry" \
    --check-admission > "$ep21_full_run/admission.txt" 2>&1; then :
else ep21_full_admission_rc=$?; fi
printf 'admission_exit=%s\n' "$ep21_full_admission_rc" >> "$ep21_full_run/status.txt"
if [[ "$ep21_full_admission_rc" -ne 0 ]]; then
  printf 'status=closed\n' >> "$ep21_full_run/status.txt"
  exit "$ep21_full_admission_rc"
fi

ep21_production_checks_rc=0
if (
  ulimit -S -f 32768
  export CUDA_VISIBLE_DEVICES=''
  cd outputs/converter_training/tests
  timeout --signal=TERM --kill-after=5s 120s python3.12 -B - <<'PYTEST'
import sys
import torch
import unittest
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
suite = unittest.defaultTestLoader.loadTestsFromNames([
    'test_content_cuda_pair_admission', 'test_content_cuda_pair_training'])
result = unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(0 if result.wasSuccessful() else 1)
PYTEST
) > "$ep21_full_run/production_checks.txt" 2>&1; then :; else ep21_production_checks_rc=$?; fi
printf 'production_checks_exit=%s\n' "$ep21_production_checks_rc" >> "$ep21_full_run/status.txt"

ep21_full_astra="${GROUP_HOME:?}/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv"
ep21_full_astra_deps="${GROUP_HOME}/users/zijiao/ep18_model_runtime_cpu241_py312_46842156/lib/python3.12/site-packages"
ep21_full_astra_rc=0
if PYTHONPATH="$ep21_full_astra/lib/python3.12/site-packages:$ep21_full_astra_deps" \
    timeout --signal=TERM --kill-after=5s 30s "$ep21_full_astra/bin/astra-tools" \
    validate outputs/astra/v0.0.14/astra.yaml --json \
    > "$ep21_full_run/astra_validation.json" 2>&1; then :; else ep21_full_astra_rc=$?; fi
printf 'astra_exit=%s\n' "$ep21_full_astra_rc" >> "$ep21_full_run/status.txt"

ep21_full_fit_rc=2
if [[ "$ep21_production_checks_rc" -eq 0 ]]; then
  if ep21_full_remaining=$(timeout --signal=TERM --kill-after=5s 60s python3.12 -B \
      "$ep21_full_entry" --remaining-seconds \
      2> "$ep21_full_run/remaining_stderr.txt") \
      && [[ "$ep21_full_remaining" =~ ^[0-9]+$ && "$ep21_full_remaining" -gt 0 ]]; then
    printf 'production_timeout_seconds=%s\n' "$ep21_full_remaining" >> "$ep21_full_run/status.txt"
    if timeout --signal=TERM --kill-after=10s "${ep21_full_remaining}s" python3.12 -u -B \
        "$ep21_full_entry" > "$ep21_full_run/training.txt" 2>&1; then
      ep21_full_fit_rc=0
    else ep21_full_fit_rc=$?; fi
  fi
fi
printf 'training_exit=%s\nstatus=closed\n' "$ep21_full_fit_rc" >> "$ep21_full_run/status.txt"
exit "$ep21_full_fit_rc"
