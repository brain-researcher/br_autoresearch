#!/bin/bash
# One deliberately selected replacement; test-only file-cap correction.
#SBATCH --job-name=ep21-content-io
#SBATCH --partition=russpold
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/content_resource_measurement/column_stripe002/slurm-%j.out
#SBATCH --error=outputs/runs/content_resource_measurement/column_stripe002/slurm-%j.err
set -euo pipefail
[[ $# -eq 0 ]] || exit 2
ep21_io_host=$(hostname)
printf 'HOST=%s JOB=%s CPUS=%s MEM_MIB=%s\n' "$ep21_io_host" \
  "${SLURM_JOB_ID:?}" "${SLURM_CPUS_PER_TASK:?}" "${SLURM_MEM_PER_NODE:?}"
[[ "$ep21_io_host" != *-ln* && "$SLURM_CPUS_PER_TASK" == 3 \
   && "$SLURM_MEM_PER_NODE" == 16384 ]] || exit 2
cd "${OAK:?}/users/zijiao/br_autoresearch/episode21_wang_neural_code_conversion_reproduction"
source /share/software/user/open/lmod/lmod/init/bash
module purge
unset PYTHONPATH PYTHONHOME
module load devel python/3.12.1 math py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 py-h5py/3.10.0_py312 py-pytorch/2.4.1_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
[[ -d "${L_SCRATCH_JOB:?}" ]] || exit 2
export TMPDIR="$L_SCRATCH_JOB/ep21_content_measurement_column_stripe002"
mkdir "$TMPDIR"
umask 077
ep21_io_run="outputs/runs/content_resource_measurement/column_stripe002"
[[ -d "$ep21_io_run" ]] || exit 2
mkdir "$ep21_io_run/batch_claim" || exit 2
TZ=UTC scontrol show job -o "$SLURM_JOB_ID" > "$ep21_io_run/allocation.txt"
python3.12 -B outputs/converter_training/tools/run_content_measurement.py \
  --stage column-stripe-002 --check-admission
printf 'parent_episode=EP21\nparent_job=%s\nseparate_measurement_grant=true\nstatus=started\n' \
  "$SLURM_JOB_ID" > "$ep21_io_run/status.txt"
ulimit -S -f 1024
ep21_io_checks_rc=0
if (
  # The unchanged fixture contains1,979,040 feature bytes plus HDF5 metadata.
  # Raise only this child's soft cap; retain aggregate grant storage limits.
  ulimit -S -f 8192
  cd outputs/converter_training/tests
  timeout --signal=TERM --kill-after=5s 180s python3.12 -B - <<'PY'
import sys
import torch
import unittest
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
print('torch=' + str(torch.__version__), flush=True)
suite = unittest.defaultTestLoader.loadTestsFromNames([
    'test_feature_stripes', 'test_content_measurement_admission'])
result = unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(0 if result.wasSuccessful() else 1)
PY
) > "$ep21_io_run/checks.txt" 2>&1; then :; else ep21_io_checks_rc=$?; fi
printf 'checks_exit=%s\n' "$ep21_io_checks_rc" >> "$ep21_io_run/status.txt"

ep21_io_astra="${GROUP_HOME:?}/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv"
ep21_io_astra_deps="${GROUP_HOME}/users/zijiao/ep18_model_runtime_cpu241_py312_46842156/lib/python3.12/site-packages"
ep21_io_astra_rc=0
if PYTHONPATH="$ep21_io_astra/lib/python3.12/site-packages:$ep21_io_astra_deps" \
    timeout --signal=TERM --kill-after=5s 30s "$ep21_io_astra/bin/astra-tools" \
    validate outputs/astra/v0.0.14/astra.yaml --json \
    > "$ep21_io_run/astra_validation.json" 2>&1; then :; else ep21_io_astra_rc=$?; fi
printf 'astra_exit=%s\n' "$ep21_io_astra_rc" >> "$ep21_io_run/status.txt"

# Re-admit remaining time and actual storage before any empirical operation.
ep21_io_rc=2
if [[ "$ep21_io_checks_rc" -eq 0 ]]; then
  if timeout --signal=TERM --kill-after=10s 6320s python3.12 -u -B \
      outputs/converter_training/tools/run_content_measurement.py \
      --stage column-stripe-002 > "$ep21_io_run/measurement.txt" 2>&1; then
    ep21_io_rc=0
  else ep21_io_rc=$?; fi
fi
printf 'measurement_exit=%s\nstatus=closed\n' "$ep21_io_rc" >> "$ep21_io_run/status.txt"
exit "$ep21_io_rc"
