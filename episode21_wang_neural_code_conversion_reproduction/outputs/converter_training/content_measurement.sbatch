#!/bin/bash
# Separate measurement grant adopted October 10; see outputs/experiment_log.md.
# One disposable empirical sizing attempt, then bounded source-free companions.
# No automatic retry, successor, polling or production converter publication.
# Submit from the EP21 root after creating the fixed run directory for logs.
#SBATCH --job-name=ep21-content-size
#SBATCH --partition=russpold
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/content_resource_measurement/attempt002/slurm-%j.out
#SBATCH --error=outputs/runs/content_resource_measurement/attempt002/slurm-%j.err
set -euo pipefail
[[ $# -eq 0 ]] || exit 2
ep21_measure_host=$(hostname)
printf 'HOST=%s JOB=%s CPUS=%s MEM_MIB=%s\n' "$ep21_measure_host" \
  "${SLURM_JOB_ID:?}" "${SLURM_CPUS_PER_TASK:?}" "${SLURM_MEM_PER_NODE:?}"
[[ "$ep21_measure_host" != *-ln* && "$SLURM_CPUS_PER_TASK" == 3 \
   && "$SLURM_MEM_PER_NODE" == 16384 ]] || exit 2
ep21_project="${OAK:?}/users/zijiao/br_autoresearch"
ep21_measure_root="$ep21_project/episode21_wang_neural_code_conversion_reproduction"
cd "$ep21_measure_root"
source /share/software/user/open/lmod/lmod/init/bash
module purge
unset PYTHONPATH PYTHONHOME
module load devel python/3.12.1 math py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 py-h5py/3.10.0_py312 py-pytorch/2.4.1_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
[[ -d "${L_SCRATCH_JOB:?}" ]] || exit 2
export TMPDIR="$L_SCRATCH_JOB/ep21_content_measurement_attempt002"
mkdir "$TMPDIR"
umask 077
# Refuse the unactivated proposal before claiming an attempt or querying Slurm.
python3.12 -B - <<'PY'
import json
from pathlib import Path
p = json.loads(Path("outputs/converter_measurement_proposal.json").read_text())
if p.get("authorized") is not True or not p.get("approval_source"):
    raise SystemExit("UNACTIVATED: a separate converter measurement grant is required")
PY
ep21_measure_run="outputs/runs/content_resource_measurement/attempt002"
[[ -d "$ep21_measure_run" ]] || exit 2
# Atomic, one-shot batch claim also covers prerequisite and companion checks.
mkdir "$ep21_measure_run/batch_claim" || exit 2
TZ=UTC scontrol show job -o "$SLURM_JOB_ID" > "$ep21_measure_run/allocation.txt"
python3.12 -B outputs/converter_training/tools/run_content_measurement.py --check-admission
printf 'parent_episode=EP21\nparent_job=%s\nseparate_measurement_grant=true\nstatus=started\n' \
  "$SLURM_JOB_ID" > "$ep21_measure_run/status.txt"
# Bound each log; the operational report is separately size-checked by its CLI.
ulimit -S -f 1024
ep21_measure_checks_rc=0
if (cd outputs/converter_training/tests && \
    timeout --signal=TERM --kill-after=5s 120s python3.12 -B - <<'PY'
import sys
import torch
import unittest
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
print("torch=" + str(torch.__version__), flush=True)
suite = unittest.defaultTestLoader.loadTestsFromNames([
    "test_empirical_content_inputs", "test_content_pair_measurement",
    "test_content_measurement_admission"])
result = unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(0 if result.wasSuccessful() else 1)
PY
) > "$ep21_measure_run/checks.txt" 2>&1; then :; else ep21_measure_checks_rc=$?; fi
printf 'checks_exit=%s\n' "$ep21_measure_checks_rc" >> "$ep21_measure_run/status.txt"
ep21_measure_rc=2
if [[ "$ep21_measure_checks_rc" -eq 0 ]]; then
  if timeout --signal=TERM --kill-after=10s 6320s python3.12 -u -B \
      outputs/converter_training/tools/run_content_measurement.py \
      > "$ep21_measure_run/measurement.txt" 2>&1; then
    ep21_measure_rc=0
  else
    ep21_measure_rc=$?
  fi
fi
printf 'measurement_exit=%s\n' "$ep21_measure_rc" >> "$ep21_measure_run/status.txt"

# New source-only checks replace companions already completed in other jobs.
# The complete tail remains within the original 630-second companion envelope.
# Each episode owns its output; no sibling scientific payload feeds EP21.
ep21_astra="${GROUP_HOME:?}/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv"
ep21_astra_deps="${GROUP_HOME}/users/zijiao/ep18_model_runtime_cpu241_py312_46842156/lib/python3.12/site-packages"
# Reuse the recorded allocation end without polling the scheduler again.
ep21_end_epoch=$(python3.12 -B - <<'PYEND'
from datetime import datetime, timezone
from pathlib import Path
fields = dict(t.split('=', 1) for t in Path('outputs/runs/content_resource_measurement/attempt002/allocation.txt').read_text().split() if '=' in t)
print(int(datetime.fromisoformat(fields['EndTime']).replace(tzinfo=timezone.utc).timestamp()))
PYEND
)
ep21_astra_rc=0
ep21_now_epoch=$(date -u +%s)
if (( ep21_end_epoch - ep21_now_epoch < 45 )); then
  ep21_astra_rc=2
  printf 'astra_validation=deferred_insufficient_remaining_time\n' >> "$ep21_measure_run/status.txt"
elif PYTHONPATH="$ep21_astra/lib/python3.12/site-packages:$ep21_astra_deps" \
    timeout --signal=TERM --kill-after=5s 30s "$ep21_astra/bin/astra-tools" \
    validate outputs/astra/v0.0.14/astra.yaml --json \
    > "$ep21_measure_run/astra_validation.json" 2>&1; then :; else ep21_astra_rc=$?; fi
printf 'astra_exit=%s\n' "$ep21_astra_rc" >> "$ep21_measure_run/status.txt"

ep21_companion_rc=0
for ep21_companion in episode06_lfp_regional_fingerprints episode17_cneuromod_model_ranking_stability episode19_causal_motor_forecasting_benchmark; do
  if (
    cd "$ep21_project/$ep21_companion" || exit 2
    ep21_child="outputs/runs/scoring_composition_checks_job_${SLURM_JOB_ID}"
    mkdir "$ep21_child" || exit 2
    printf 'parent_episode=EP21\nparent_job=%s\nparent_charged_once=true\n' "$SLURM_JOB_ID" > "$ep21_child/status.txt"
    case "$ep21_companion" in
      episode06*) ep21_test_dir=outputs/tests; ep21_test_names=(test_component_scoring test_component_nomination); ep21_test_limit=60 ;;
      episode17*) ep21_test_dir=outputs/tools; ep21_test_names=(test_endpoint_evaluator test_audit_adjudicator test_nested_prediction); ep21_test_limit=235 ;;
      episode19*)
        ep21_test_dir=outputs/tests
        ep21_test_limit=175
        ep21_test_names=(test_baseline_final_refit
          test_non_neural_forecast_backend.NonNeuralWrapperTests.test_registered_roles_weights_and_scope_rejection_precede_fit
          test_boosted_hazard_backend.BoostedHazardTests.test_model_keeps_canonical_roles_fitting_weights_and_past_only_frozen_offsets
          test_temporal_hazard_backend.TemporalHazardTests.test_group_roles_and_training_only_channel_summary_scaling
          test_ensemble_forecast_backend.EnsembleForecastTests.test_fitted_seed_reports_bind_recipe_fold_grid_and_incomplete_members) ;;
    esac
    ep21_now_epoch=$(date -u +%s)
    if (( ep21_end_epoch - ep21_now_epoch < ep21_test_limit + 50 )); then
      printf 'status=deferred_insufficient_remaining_time\n' >> "$ep21_child/status.txt"
      exit 2
    fi
    ep21_child_rc=0
    if (cd "$ep21_test_dir" && timeout --signal=TERM --kill-after=5s "$ep21_test_limit" \
        python3.12 -B -m unittest -v -f "${ep21_test_names[@]}") \
        > "$ep21_child/tests.txt" 2>&1; then :; else ep21_child_rc=$?; fi
    printf 'tests_exit=%s\n' "$ep21_child_rc" >> "$ep21_child/status.txt"
    if PYTHONPATH="$ep21_astra/lib/python3.12/site-packages:$ep21_astra_deps" \
        timeout --signal=TERM --kill-after=5s 30s "$ep21_astra/bin/astra-tools" \
        validate outputs/astra/v0.0.14/astra.yaml --json \
        > "$ep21_child/astra_validation.json" 2>&1; then
      printf 'astra_exit=0\n' >> "$ep21_child/status.txt"
    else
      printf 'astra_exit=%s\n' "$?" >> "$ep21_child/status.txt"
      ep21_child_rc=2
    fi
    printf 'status=closed\n' >> "$ep21_child/status.txt"
    exit "$ep21_child_rc"
  ); then :; else ep21_companion_rc=2; fi
done
printf 'companion_exit=%s\nstatus=closed\n' "$ep21_companion_rc" >> "$ep21_measure_run/status.txt"
exit "$ep21_measure_rc"
