#!/bin/bash
# Root-attended step inside the existing substantive EP21 allocation.
set -euo pipefail
umask 077
[[ $# == 0 && ${SLURM_JOB_ID:?} == 47579378 ]]
ep18_check_host=$(hostname)
[[ "$ep18_check_host" != *-ln* && ${SLURM_CPUS_PER_TASK:?} == 1 ]]
[[ ${SLURM_STEP_ID:?} != batch && "$SLURM_STEP_ID" != extern ]]
ep18_check_now=$(date -u +%s)
ep18_check_end=$((SLURM_JOB_START_TIME + 21000))
if (( SLURM_JOB_END_TIME - 60 < ep18_check_end )); then ep18_check_end=$((SLURM_JOB_END_TIME - 60)); fi
(( ep18_check_end > ep18_check_now + 240 ))
cd "${OAK:?}/users/zijiao/br_autoresearch/episode18_things_eeg_concept_stability"
ep18_check_run="$PWD/outputs/runs/software_integration/companion003"
mkdir -p -m 700 "$PWD/outputs/runs/software_integration"
mkdir -m 700 "$ep18_check_run"
printf 'parent_job=%s\nstep_id=%s\nhostname=%s\nrequested_step_cpus=1\nrequested_step_memory_mib=512\nparent_charged_once_to=EP21\n' \
  "$SLURM_JOB_ID" "$SLURM_STEP_ID" "$ep18_check_host" >"$ep18_check_run/status.txt"
trap 'ep18_check_exit=$?; printf "exit=%s\nfinished_utc=%s\n" "$ep18_check_exit" "$(date -u +%FT%TZ)" >>"$ep18_check_run/status.txt"' EXIT
ulimit -c 0
ulimit -S -f 2048
source /share/software/user/open/lmod/lmod/init/bash
module purge
unset PYTHONPATH PYTHONHOME
module load devel python/3.12.1 math py-numpy/1.26.3_py312 py-scipy/1.12.0_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export TMPDIR="${L_SCRATCH_JOB:?}/ep18-integration-companion003"
mkdir -m 700 "$TMPDIR"
export XDG_CACHE_HOME="$TMPDIR/cache"
cd outputs/code
timeout --signal=TERM --kill-after=5s 120s python3.12 -B - <<'PY' >"$ep18_check_run/checks.txt" 2>&1
import json
import resource
import unittest
suite = unittest.defaultTestLoader.loadTestsFromNames([
    "test_production_baseline_controller",
])
result = unittest.TextTestRunner(verbosity=2).run(suite)
print(json.dumps({"methods": result.testsRun, "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  "empirical_inputs_read": False, "scientific_scores_computed": False}), flush=True)
raise SystemExit(0 if result.wasSuccessful() and result.testsRun == 2 and not result.skipped else 1)
PY
# Documentary checks remain separate from scientific admission.
ep18_check_astra="${GROUP_HOME:?}/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv"
ep18_check_astra_deps="$GROUP_HOME/users/zijiao/ep18_model_runtime_cpu241_py312_46842156/lib/python3.12/site-packages"
cd "${OAK:?}/users/zijiao/br_autoresearch/episode18_things_eeg_concept_stability"
set +e
env PYTHONPATH="$ep18_check_astra/lib/python3.12/site-packages:$ep18_check_astra_deps" \
  timeout --signal=TERM --kill-after=5s 20s "$ep18_check_astra/bin/astra-tools" \
  validate outputs/astra/v0.0.14/astra.yaml --json >"$ep18_check_run/ep18_astra_validation.json" 2>&1
ep18_check_validation_exit=$?
set -e
printf 'ep18_astra_exit=%s\n' "$ep18_check_validation_exit" >>"$ep18_check_run/status.txt"
cd "${OAK:?}/users/zijiao/br_autoresearch/episode21_wang_neural_code_conversion_reproduction"
set +e
env PYTHONPATH="$ep18_check_astra/lib/python3.12/site-packages:$ep18_check_astra_deps" \
  timeout --signal=TERM --kill-after=5s 20s "$ep18_check_astra/bin/astra-tools" \
  validate outputs/astra/v0.0.14/astra.yaml --json >"$ep18_check_run/ep21_astra_validation.json" 2>&1
ep18_check_validation_exit=$?
set -e
printf 'ep21_astra_exit=%s\nastra_is_scientific_execution_gate=false\n' "$ep18_check_validation_exit" >>"$ep18_check_run/status.txt"
