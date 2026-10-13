#!/bin/bash
# One selected CUDA measurement with separately scoped serialized companions.
#SBATCH --job-name=ep21-content-cuda
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gpus=1
#SBATCH --constraint="GPU_MEM:24GB|GPU_MEM:32GB|GPU_MEM:40GB|GPU_MEM:48GB|GPU_MEM:80GB|GPU_MEM:141GB"
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/content_resource_measurement/cuda001/slurm-%j.out
#SBATCH --error=outputs/runs/content_resource_measurement/cuda001/slurm-%j.err
set -euo pipefail
[[ $# -eq 0 ]] || exit 2
ep21_cuda_host=$(hostname)
printf 'HOST=%s JOB=%s CPUS=%s MEM_MIB=%s GPUS=%s\n' "$ep21_cuda_host" \
  "${SLURM_JOB_ID:?}" "${SLURM_CPUS_PER_TASK:?}" "${SLURM_MEM_PER_NODE:?}" "${SLURM_GPUS_ON_NODE:?}"
[[ "$ep21_cuda_host" != *-ln* && "$SLURM_CPUS_PER_TASK" == 2 \
  && "$SLURM_MEM_PER_NODE" == 16384 && "$SLURM_GPUS_ON_NODE" == 1 ]] || exit 2
cd "${OAK:?}/users/zijiao/br_autoresearch/episode21_wang_neural_code_conversion_reproduction"
source /share/software/user/open/lmod/lmod/init/bash
module purge
unset PYTHONPATH PYTHONHOME
module load devel python/3.12.1 math py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 py-h5py/3.10.0_py312 py-pytorch/2.4.1_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
[[ -d "${L_SCRATCH_JOB:?}" ]] || exit 2
export TMPDIR="$L_SCRATCH_JOB/ep21_content_cuda_measurement001"
umask 077
mkdir "$TMPDIR"
export XDG_CACHE_HOME="$TMPDIR/cache" CUDA_CACHE_PATH="$TMPDIR/cuda-cache" TORCH_HOME="$TMPDIR/torch"
ep21_cuda_run="outputs/runs/content_resource_measurement/cuda001"
ep21_brain_diag_run="outputs/runs/brain_pair_prefix_diagnostic/attempt001"
ep19_refit_run="${OAK}/users/zijiao/br_autoresearch/episode19_causal_motor_forecasting_benchmark/outputs/runs/reference_panel/cuda_companion001"
[[ -d "$ep21_cuda_run" && -d "$ep21_brain_diag_run" && -d "$ep19_refit_run" ]] || exit 2
mkdir "$ep21_cuda_run/batch_claim" || exit 2
TZ=UTC scontrol show job -o "$SLURM_JOB_ID" > "$ep21_cuda_run/allocation.txt"
python3.12 -B outputs/converter_training/tools/run_content_cuda_measurement.py --check-admission
printf 'parent_episode=EP21\nparent_job=%s\nseparate_measurement_grant=true\nstatus=started\n' \
  "$SLURM_JOB_ID" > "$ep21_cuda_run/status.txt"
ulimit -S -f 1024
ep21_cuda_checks_rc=0
if (
  ulimit -S -f 32768
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
) > "$ep21_cuda_run/cuda_checks.txt" 2>&1; then :; else ep21_cuda_checks_rc=$?; fi
printf 'cuda_checks_exit=%s\n' "$ep21_cuda_checks_rc" >> "$ep21_cuda_run/status.txt"

ep21_companion_rc=0
if (
  ulimit -S -f 32768
  export CUDA_VISIBLE_DEVICES=''
  cd outputs/converter_training/tests
  timeout --signal=TERM --kill-after=5s 180s python3.12 -B - <<'PY'
import sys
import torch
import unittest
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
suite = unittest.defaultTestLoader.loadTestsFromNames([
    'test_finding_a_collection', 'test_content_pair_admission',
    'test_brain_pair_admission', 'test_brain_prefix_diagnostic_admission'])
result = unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(0 if result.wasSuccessful() else 1)
PY
) > "$ep21_cuda_run/companion_checks.txt" 2>&1; then :; else ep21_companion_rc=$?; fi
printf 'companion_checks_exit=%s\n' "$ep21_companion_rc" >> "$ep21_cuda_run/status.txt"

ep19_refit_rc=0
if (
  ulimit -S -f 32768
  export CUDA_VISIBLE_DEVICES=''
  cd "${OAK}/users/zijiao/br_autoresearch/episode19_causal_motor_forecasting_benchmark/outputs/tests"
  timeout --signal=TERM --kill-after=5s 115s python3.12 -B -m unittest -v -f \
    test_reference_final_refit \
    test_classical_forecast_backend.ClassicalBackendTests.test_forged_fitting_scope_is_rejected_before_batches_or_features \
    test_gru_forecast_backend.GRUBackendTests.test_recipe_scope_and_unresolved_semantics_rejected_before_batches \
    test_tcn_forecast_backend.TCNBackendTests.test_registry_explicit_grid_and_complete_past_windows_gate_the_tcn \
    test_eegnet_forecast_backend.EEGNetBackendTests.test_recipe_source_grid_and_complete_window_admission \
    test_ensemble_forecast_backend.EnsembleForecastTests.test_fitted_seed_reports_bind_recipe_fold_grid_and_incomplete_members
) > "$ep19_refit_run/checks.txt" 2>&1; then :; else ep19_refit_rc=$?; fi
printf 'parent_episode=EP21\nparent_job=%s\nchecks_exit=%s\nempirical_access=false\n' \
  "$SLURM_JOB_ID" "$ep19_refit_rc" > "$ep19_refit_run/status.txt"

ep21_cuda_astra="${GROUP_HOME:?}/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv"
ep21_cuda_astra_deps="${GROUP_HOME}/users/zijiao/ep18_model_runtime_cpu241_py312_46842156/lib/python3.12/site-packages"
ep21_cuda_astra_rc=0
if PYTHONPATH="$ep21_cuda_astra/lib/python3.12/site-packages:$ep21_cuda_astra_deps" \
    timeout --signal=TERM --kill-after=5s 30s "$ep21_cuda_astra/bin/astra-tools" \
    validate outputs/astra/v0.0.14/astra.yaml --json \
    > "$ep21_cuda_run/astra_validation.json" 2>&1; then :; else ep21_cuda_astra_rc=$?; fi
printf 'astra_exit=%s\n' "$ep21_cuda_astra_rc" >> "$ep21_cuda_run/status.txt"
ep19_astra_rc=0
if (
  cd "${OAK}/users/zijiao/br_autoresearch/episode19_causal_motor_forecasting_benchmark"
  PYTHONPATH="$ep21_cuda_astra/lib/python3.12/site-packages:$ep21_cuda_astra_deps" \
    timeout --signal=TERM --kill-after=5s 30s "$ep21_cuda_astra/bin/astra-tools" \
    validate outputs/astra/v0.0.14/astra.yaml --json
) > "$ep19_refit_run/astra_validation.json" 2>&1; then :; else ep19_astra_rc=$?; fi
printf 'astra_exit=%s\nstatus=closed\n' "$ep19_astra_rc" >> "$ep19_refit_run/status.txt"

# Separate diagnostic authority; no full fit is retried. A diagnostic failure
# does not authorize a repair or change the independent content measurement.
ep21_brain_diag_rc=2
if [[ "$ep21_companion_rc" -eq 0 ]]; then
  if timeout --signal=TERM --kill-after=10s 310s python3.12 -u -B \
      outputs/converter_training/tools/run_brain_prefix_diagnostic.py \
      > "$ep21_brain_diag_run/diagnostic.txt" 2>&1; then
    ep21_brain_diag_rc=0
  else ep21_brain_diag_rc=$?; fi
fi
printf 'parent_job=%s\ndiagnostic_exit=%s\nstatus=closed\n' \
  "$SLURM_JOB_ID" "$ep21_brain_diag_rc" > "$ep21_brain_diag_run/status.txt"

ep21_cuda_rc=2
if [[ "$ep21_cuda_checks_rc" -eq 0 ]]; then
  if timeout --signal=TERM --kill-after=10s 6320s python3.12 -u -B \
      outputs/converter_training/tools/run_content_cuda_measurement.py \
      > "$ep21_cuda_run/measurement.txt" 2>&1; then
    ep21_cuda_rc=0
  else ep21_cuda_rc=$?; fi
fi
printf 'measurement_exit=%s\nstatus=closed\n' "$ep21_cuda_rc" >> "$ep21_cuda_run/status.txt"
exit "$ep21_cuda_rc"
