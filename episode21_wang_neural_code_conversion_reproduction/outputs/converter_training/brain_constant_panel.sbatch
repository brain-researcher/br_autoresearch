#!/bin/bash
# Root submits pair004 first, then all other19 only after observed check PASS.
# Full adapted pair/job; no original artifact reuse, retry, scoring or dispatch.
#SBATCH --job-name=ep21-brain-constant
#SBATCH --partition=russpold
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/brain_constant_training/batch-%A_%a.out
#SBATCH --error=outputs/runs/brain_constant_training/batch-%A_%a.err
set -euo pipefail
umask 077
[[ $# == 0 ]]
ep21_c_host=$(hostname)
ep21_c_index=${SLURM_ARRAY_TASK_ID:?}
[[ "$ep21_c_index" =~ ^([0-9]|1[0-9])$ ]]
printf 'hostname=%s job_id=%s pair_index=%s cpus=%s memory_mib=%s\n' \
  "$ep21_c_host" "${SLURM_JOB_ID:?}" "$ep21_c_index" "${SLURM_CPUS_PER_TASK:?}" "${SLURM_MEM_PER_NODE:?}"
[[ "$ep21_c_host" != *-ln* && "$SLURM_CPUS_PER_TASK" == 3 && "$SLURM_MEM_PER_NODE" == 16384 ]]
[[ "${SLURM_JOB_START_TIME:?}" =~ ^[0-9]+$ && "${SLURM_JOB_END_TIME:?}" =~ ^[0-9]+$ ]]
ep21_c_now=$(date -u +%s)
(( SLURM_JOB_START_TIME <= ep21_c_now && ep21_c_now < SLURM_JOB_END_TIME ))
ep21_c_deadline=$((SLURM_JOB_START_TIME + 21000))
if (( SLURM_JOB_END_TIME - 60 < ep21_c_deadline )); then
  ep21_c_deadline=$((SLURM_JOB_END_TIME - 60))
fi
(( ep21_c_deadline > ep21_c_now + 600 ))
cd "${OAK:?}/users/zijiao/br_autoresearch/episode21_wang_neural_code_conversion_reproduction"
ep21_c_base=$PWD
ep21_c_people=(sub01 sub03 sub05 sub06 sub07)
ep21_c_source=$((ep21_c_index / 4))
ep21_c_target=$((ep21_c_index % 4))
if (( ep21_c_target >= ep21_c_source )); then ep21_c_target=$((ep21_c_target+1)); fi
printf -v ep21_c_stem 'pair%03d_%s_to_%s_001' "$ep21_c_index" \
  "${ep21_c_people[$ep21_c_source]}" "${ep21_c_people[$ep21_c_target]}"
ep21_c_run="$PWD/outputs/runs/brain_constant_training/$ep21_c_stem"
mkdir -m 700 "$ep21_c_run"
printf 'job_id=%s\npair_index=%s\ncpus=3\nmemory_mib=16384\nmaximum_CPU_hours=18\nallocation_start_epoch=%s\nallocation_end_epoch=%s\nwork_deadline_epoch=%s\n' \
  "$SLURM_JOB_ID" "$ep21_c_index" "$SLURM_JOB_START_TIME" "$SLURM_JOB_END_TIME" \
  "$ep21_c_deadline" >"$ep21_c_run/status.txt"
trap 'ep21_c_exit=$?; printf "batch_exit=%s\nfinished_utc=%s\n" "$ep21_c_exit" "$(date -u +%FT%TZ)" >>"$ep21_c_run/status.txt"' EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
ulimit -c 0
ulimit -S -f 2097152
source /share/software/user/open/lmod/lmod/init/bash
module purge
unset PYTHONPATH PYTHONHOME
module load devel python/3.12.1 math py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 py-h5py/3.10.0_py312 py-pytorch/2.4.1_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
ep21_c_free_kib=$(df -Pk "${L_SCRATCH_JOB:?}" | awk 'NR==2 {print $4}')
[[ "$ep21_c_free_kib" =~ ^[0-9]+$ && "$ep21_c_free_kib" -ge 262144 ]]
export TMPDIR="$L_SCRATCH_JOB/ep21_brain_constant_$ep21_c_stem"
mkdir "$TMPDIR"
export XDG_CACHE_HOME="$TMPDIR/cache" TORCH_HOME="$TMPDIR/torch"
ep21_c_entry="$PWD/outputs/converter_training/tools/run_brain_constant_pair.py"
ep21_c_limit() {
  local requested=$1 available
  available=$((ep21_c_deadline - $(date -u +%s) - 15))
  (( available > 0 )) || return 124
  if (( available < requested )); then requested=$available; fi
  printf '%s\n' "$requested"
}
ep21_c_seconds=$(ep21_c_limit 30)
timeout --signal=TERM --kill-after=5s "${ep21_c_seconds}s" python3.12 -B "$ep21_c_entry" \
  --pair-index "$ep21_c_index" --check-admission >"$ep21_c_run/admission.txt" 2>&1
printf 'admission_exit=0\n' >>"$ep21_c_run/status.txt"
if (( ep21_c_index == 4 )); then
  ep21_c_seconds=$(ep21_c_limit 120)
  (
    cd outputs/converter_training
    timeout --signal=TERM --kill-after=5s "${ep21_c_seconds}s" python3.12 -B - <<'PY'
import unittest
import sys
sys.path.insert(0, "tests")
suite = unittest.defaultTestLoader.loadTestsFromNames([
    "test_brain_constant_comparator", "test_brain_constant_runner"])
result = unittest.TextTestRunner(verbosity=2).run(suite)
# Explicit expected method count is frozen after the focused source review.
raise SystemExit(0 if result.wasSuccessful() and result.testsRun == 10 and not result.skipped else 1)
PY
  ) >"$ep21_c_run/constant_checks.txt" 2>&1
  printf 'constant_checks_exit=0\n' >>"$ep21_c_run/status.txt"
  # Separately scoped EP18 engineering checks, serial and independent of EP21.
  ep21_c_seconds=$(ep21_c_limit 120)
  set +e
  (
    cd "${OAK:?}/users/zijiao/br_autoresearch/episode18_things_eeg_concept_stability/outputs/code" || exit 1
    timeout --signal=TERM --kill-after=5s "${ep21_c_seconds}s" python3.12 -B - <<'PY'
import unittest
suite = unittest.defaultTestLoader.loadTestsFromNames([
    "test_streamed_slot_capacity", "test_production_continuous_designs"])
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(0 if result.wasSuccessful() and result.testsRun == 5 and not result.skipped else 1)
PY
  ) >"$ep21_c_run/ep18_checks.txt" 2>&1
  ep21_c_ep18_exit=$?
  set -e
  printf 'ep18_checks_exit=%s\nep18_checks_is_ep21_gate=false\n' "$ep21_c_ep18_exit" >>"$ep21_c_run/status.txt"
  ep21_c_astra="${GROUP_HOME:?}/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv"
  ep21_c_astra_deps="$GROUP_HOME/users/zijiao/ep18_model_runtime_cpu241_py312_46842156/lib/python3.12/site-packages"
  ep21_c_seconds=$(ep21_c_limit 60)
  set +e
  env PYTHONPATH="$ep21_c_astra/lib/python3.12/site-packages:$ep21_c_astra_deps" \
    timeout --signal=TERM --kill-after=5s "${ep21_c_seconds}s" "$ep21_c_astra/bin/astra-tools" \
    validate outputs/astra/v0.0.14/astra.yaml --json >"$ep21_c_run/astra_validation.json" 2>&1
  ep21_c_astra_exit=$?
  set -e
  printf 'astra_exit=%s\nastra_is_scientific_execution_gate=false\n' "$ep21_c_astra_exit" >>"$ep21_c_run/status.txt"
else
  ep21_c_shared="$PWD/outputs/runs/brain_constant_training/pair004_sub03_to_sub01_001/status.txt"
  [[ -f "$ep21_c_shared" ]]
  ep21_c_passes=0
  while IFS='=' read -r ep21_c_key ep21_c_value; do
    if [[ "$ep21_c_key" == constant_checks_exit ]]; then
      [[ "$ep21_c_value" == 0 ]]
      ep21_c_passes=$((ep21_c_passes+1))
    fi
  done <"$ep21_c_shared"
  [[ "$ep21_c_passes" == 1 ]]
  printf 'constant_checks_exit=0\nconstant_checks_reused_from=pair004_sub03_to_sub01_001\n' >>"$ep21_c_run/status.txt"
fi
ep21_c_seconds=$(ep21_c_limit 21000)
set +e
timeout --signal=TERM --kill-after=10s "${ep21_c_seconds}s" python3.12 -u -B "$ep21_c_entry" \
  --pair-index "$ep21_c_index" >"$ep21_c_run/fit.txt" 2>&1
ep21_c_fit_exit=$?
set -e
printf 'fit_exit=%s\n' "$ep21_c_fit_exit" >>"$ep21_c_run/status.txt"
exit "$ep21_c_fit_exit"
