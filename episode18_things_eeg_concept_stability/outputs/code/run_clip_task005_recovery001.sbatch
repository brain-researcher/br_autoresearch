#!/bin/bash
# Coordinator supplies phase-specific partition, resources and dependencies.
# No queue retry and no reuse of failed partial rows.
#SBATCH --job-name=ep18-clip-recovery
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=15G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/feature_preparation/image_recovery_operations_20261012_001/%x-%j.out
#SBATCH --error=outputs/runs/feature_preparation/image_recovery_operations_20261012_001/%x-%j.err
set -euo pipefail
umask 077
ep18_phase=${1:?worker or assembly required}
[[ $# == 1 && -n "${SLURM_JOB_ID:-}" && "$(hostname)" != *-ln* ]]
[[ -z "${SLURM_ARRAY_TASK_ID:-}" && -z "${SLURM_JOB_GPUS:-}" ]]
: "${OAK:?}" "${GROUP_HOME:?}" "${L_SCRATCH_JOB:?}"
case "$ep18_phase" in
  worker) ep18_cpus=2; ep18_mem=15360; ep18_cap=14400 ;;
  assembly) ep18_cpus=1; ep18_mem=7168; ep18_cap=3600 ;;
  *) exit 2 ;;
esac
[[ "${SLURM_CPUS_PER_TASK:-}" == "$ep18_cpus" && "${SLURM_MEM_PER_NODE:-}" == "$ep18_mem" ]]
[[ "${SLURM_JOB_NUM_NODES:-}" == 1 && "${SLURM_JOB_CPUS_PER_NODE:-}" == "$ep18_cpus" ]]
ep18_episode="$OAK/users/zijiao/br_autoresearch/episode18_things_eeg_concept_stability"
ep18_code="$ep18_episode/outputs/code"
ep18_grant="$ep18_episode/outputs/config/image_task005_recovery_20261012_001.json"
ep18_ops="$ep18_episode/outputs/runs/feature_preparation/image_recovery_operations_20261012_001/${ep18_phase}_${SLURM_JOB_ID}"
mkdir -m 700 "$ep18_ops"
ep18_stage=allocation_admission
ep18_started=$SECONDS
ep18_deadline=0
finish() {
  ep18_exit=$?
  trap - EXIT
  printf '{"phase":"%s","stage":"%s","exit_code":%s,"elapsed_seconds":%s,"job_id":"%s","deadline_unix":%s,"automatic_retry":false}\n' \
    "$ep18_phase" "$ep18_stage" "$ep18_exit" "$((SECONDS-ep18_started))" "$SLURM_JOB_ID" "$ep18_deadline" > "$ep18_ops/exit.json"
  exit "$ep18_exit"
}
trap finish EXIT
ep18_start=${SLURM_JOB_START_TIME:?}
ep18_end=${SLURM_JOB_END_TIME:?}
[[ "$ep18_start" =~ ^[0-9]+$ && "$ep18_end" =~ ^[0-9]+$ ]]
ep18_now=$(date +%s)
(( ep18_start > 0 && ep18_start <= ep18_now && ep18_end > ep18_now ))
ep18_deadline=$((ep18_start+ep18_cap-60))
if (( ep18_end-60 < ep18_deadline )); then ep18_deadline=$((ep18_end-60)); fi
if (( 1791926990 < ep18_deadline )); then ep18_deadline=1791926990; fi
remaining_seconds() {
  local ep18_left
  ep18_left=$((ep18_deadline-$(date +%s)))
  (( ep18_left > 0 )) || return 2
  printf '%s\n' "$ep18_left"
}
ep18_remaining=$(remaining_seconds)
printf 'hostname=%s job_id=%s phase=%s cpus=%s memory_mib=%s start=%s end=%s deadline=%s\n' \
  "$(hostname)" "$SLURM_JOB_ID" "$ep18_phase" "$ep18_cpus" "$ep18_mem" "$ep18_start" "$ep18_end" "$ep18_deadline"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES='' PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
ep18_local="$L_SCRATCH_JOB/ep18-recovery-wrapper-${ep18_phase}"
mkdir -m 700 "$ep18_local"
mkdir -m 700 "$ep18_local/tmp" "$ep18_local/cache" "$ep18_local/hf"
export TMPDIR="$ep18_local/tmp" XDG_CACHE_HOME="$ep18_local/cache" HF_HOME="$ep18_local/hf"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
if [[ "$ep18_phase" == worker ]]; then
  # Independent, serial EP21 fixture companion. Failure does not gate EP18.
  ep18_stage=ep21_readout_companion
  ep18_remaining=$(remaining_seconds)
  ep18_check_seconds=180
  if (( ep18_remaining < ep18_check_seconds )); then ep18_check_seconds=$ep18_remaining; fi
  set +e
  timeout --signal=TERM --kill-after=5s "${ep18_check_seconds}s" bash \
    "$OAK/users/zijiao/br_autoresearch/episode21_wang_neural_code_conversion_reproduction/outputs/converter_training/tools/run_amended_readout_checks_companion001.sh" "$ep18_deadline" \
    > "$ep18_ops/ep21_companion_stdout.txt" 2> "$ep18_ops/ep21_companion_stderr.txt"
  ep18_companion_exit=$?
  set -e
  printf 'ep21_companion_exit=%s\n' "$ep18_companion_exit" > "$ep18_ops/ep21_companion_status.txt"
fi
source /share/software/user/open/lmod/lmod/init/bash
module purge
module load devel python/3.12.1
unset PYTHONPATH PYTHONHOME
ep18_python="$GROUP_HOME/users/zijiao/ep18_model_runtime_cpu241_py312_46842156/bin/python"
[[ -x "$ep18_python" ]]
if [[ "$ep18_phase" == worker ]]; then
  ep18_stage=recovery_checks
  cd "$ep18_code"
  ep18_remaining=$(remaining_seconds)
  ep18_check_seconds=120
  if (( ep18_remaining < ep18_check_seconds )); then ep18_check_seconds=$ep18_remaining; fi
  timeout --signal=TERM --kill-after=5s "${ep18_check_seconds}s" "$ep18_python" -B - <<'PY' > "$ep18_ops/recovery_checks.txt" 2>&1
import unittest
suite = unittest.defaultTestLoader.loadTestsFromName('test_recover_clip_task005_001')
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(0 if result.wasSuccessful() and result.testsRun == 2 and not result.skipped else 1)
PY
  # Changed documentary programs are checked once; this does not admit data.
  ep18_stage=astra_validation
  ep18_astra="$GROUP_HOME/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv"
  for ep18_document in episode18_things_eeg_concept_stability episode21_wang_neural_code_conversion_reproduction; do
    ep18_remaining=$(remaining_seconds)
    ep18_check_seconds=20
    if (( ep18_remaining < ep18_check_seconds )); then ep18_check_seconds=$ep18_remaining; fi
    cd "$OAK/users/zijiao/br_autoresearch/$ep18_document"
    set +e
    env PYTHONPATH="$ep18_astra/lib/python3.12/site-packages:$GROUP_HOME/users/zijiao/ep18_model_runtime_cpu241_py312_46842156/lib/python3.12/site-packages" \
      timeout --signal=TERM --kill-after=5s "${ep18_check_seconds}s" "$ep18_astra/bin/astra-tools" \
      validate outputs/astra/v0.0.14/astra.yaml --json > "$ep18_ops/${ep18_document}_astra.json" 2>&1
    ep18_validation_exit=$?
    set -e
    printf '%s_exit=%s\n' "$ep18_document" "$ep18_validation_exit" >> "$ep18_ops/astra_status.txt"
  done
fi
ep18_stage="${ep18_phase}_execution"
ep18_remaining=$(remaining_seconds)
cd "$ep18_local"
ep18_arguments=(--grant "$ep18_grant" --phase "$ep18_phase" --deadline-unix "$ep18_deadline")
if [[ "$ep18_phase" == worker ]]; then ep18_arguments+=(--local-dir "$L_SCRATCH_JOB/ep18-clip-recovery001"); fi
timeout --signal=TERM --kill-after=5s "${ep18_remaining}s" env -i \
  PATH="$PATH" LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" LANG=C LC_ALL=C PYTHONUTF8=1 \
  SLURM_JOB_ID="$SLURM_JOB_ID" SLURM_CPUS_PER_TASK="$ep18_cpus" SLURM_MEM_PER_NODE="$ep18_mem" \
  SLURM_JOB_START_TIME="$ep18_start" SLURM_JOB_END_TIME="$ep18_end" SLURM_JOB_NUM_NODES=1 SLURM_JOB_CPUS_PER_NODE="$ep18_cpus" \
  L_SCRATCH_JOB="$L_SCRATCH_JOB" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 \
  CUDA_VISIBLE_DEVICES='' PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false \
  TMPDIR="$ep18_local/tmp" XDG_CACHE_HOME="$ep18_local/cache" HF_HOME="$ep18_local/hf" \
  "$ep18_python" -B "$ep18_code/recover_clip_task005_001.py" "${ep18_arguments[@]}" \
  > "$ep18_ops/execution.txt" 2> "$ep18_ops/execution_stderr.txt"
ep18_stage=completed
