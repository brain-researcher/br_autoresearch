#!/bin/bash
# Coordinator supplies the discovered partition and precreates only parents.
# This single allocation includes checks, all30 input binding and the complete
# first-person cost attempt. It neither retries nor finalizes a partial cohort.
#SBATCH --job-name=ep18-core-cost001
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=15G
#SBATCH --time=02:00:00
#SBATCH --dependency=afterok:47584687
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/baseline_models/core_reference_operations_20261013_001/%x-%j.out
#SBATCH --error=outputs/runs/baseline_models/core_reference_operations_20261013_001/%x-%j.err
set -euo pipefail
umask 077

[[ $# == 0 && -n "${SLURM_JOB_ID:-}" && "$(hostname)" != *-ln* ]]
[[ -z "${SLURM_ARRAY_TASK_ID:-}" && -z "${SLURM_JOB_GPUS:-}" && -z "${SLURM_STEP_GPUS:-}" ]]
[[ "${SLURM_CPUS_PER_TASK:-}" == 2 && "${SLURM_MEM_PER_NODE:-}" == 15360 ]]
[[ "${SLURM_JOB_NUM_NODES:-}" == 1 && "${SLURM_JOB_CPUS_PER_NODE:-}" == 2 ]]
: "${OAK:?}" "${GROUP_HOME:?}" "${L_SCRATCH_JOB:?}"
ep18_start=${SLURM_JOB_START_TIME:?}
ep18_end=${SLURM_JOB_END_TIME:?}
[[ "$ep18_start" =~ ^[0-9]+$ && "$ep18_end" =~ ^[0-9]+$ ]]
ep18_now=$(date +%s)
(( ep18_start > 0 && ep18_start <= ep18_now && ep18_now < ep18_end ))
ep18_deadline=$((ep18_start+7140))
if (( ep18_end-60 < ep18_deadline )); then ep18_deadline=$((ep18_end-60)); fi
# Five seconds for cooperative publication, then TERM and five seconds to KILL.
# Even the final native-call KILL remains within the unchanged episode clock.
if (( 1791926990 < ep18_deadline )); then ep18_deadline=1791926990; fi
(( ep18_now < ep18_deadline ))

ep18_episode="$OAK/users/zijiao/br_autoresearch/episode18_things_eeg_concept_stability"
ep18_code="$ep18_episode/outputs/code"
ep18_recipe="$ep18_episode/outputs/config/core_recipe_20261013_001.json"
ep18_release="$ep18_episode/outputs/config/core_release_20261013_001.json"
ep18_run="$ep18_episode/outputs/cache/private/core_reference_20261013_001"
ep18_ops="$ep18_episode/outputs/runs/baseline_models/core_reference_operations_20261013_001/initial_${SLURM_JOB_ID}"
mkdir -m 700 "$ep18_ops"
ep18_stage=allocation_admission
ep18_started=$SECONDS
finish() {
  ep18_exit=$?
  trap - EXIT
  printf '{"stage":"%s","exit_code":%s,"elapsed_seconds":%s,"job_id":"%s","deadline_epoch":%s,"allocated_cpus":2,"memory_mib":15360,"numerical_threads":1,"reserved_cpu_hours":4,"RSS_guard_bytes":12884901888,"RSS_guard":"observed_high_water_at_budget_checks","automatic_retry":false,"cost_only":true}\n' \
    "$ep18_stage" "$ep18_exit" "$((SECONDS-ep18_started))" "$SLURM_JOB_ID" "$ep18_deadline" > "$ep18_ops/exit.json"
  exit "$ep18_exit"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
[[ -f "$ep18_recipe" && -f "$ep18_release" && ! -e "$ep18_run" ]]
printf 'hostname=%s job_id=%s start=%s end=%s deadline=%s cpus=2 memory_mib=15360 numerical_threads=1 reserved_cpu_hours=4\n' \
  "$(hostname)" "$SLURM_JOB_ID" "$ep18_start" "$ep18_end" "$ep18_deadline"

ep18_local="$L_SCRATCH_JOB/ep18-core-reference-cost001"
mkdir -m 700 "$ep18_local"
mkdir -m 700 "$ep18_local/tmp" "$ep18_local/cache"
export TMPDIR="$ep18_local/tmp" XDG_CACHE_HOME="$ep18_local/cache"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
source /share/software/user/open/lmod/lmod/init/bash
module purge
unset PYTHONPATH PYTHONHOME
module load devel python/3.12.1
# The previously qualified complete EP18 environment supplies scientific packages.
unset PYTHONPATH PYTHONHOME
ep18_python="$GROUP_HOME/users/zijiao/ep18_model_runtime_cpu241_py312_46842156/bin/python"
[[ -x "$ep18_python" ]]
cd "$ep18_local"

remaining_seconds() {
  local ep18_left
  ep18_left=$((ep18_deadline-$(date +%s)))
  (( ep18_left > 0 )) || return 2
  printf '%s\n' "$ep18_left"
}

ep18_stage=core_reference_checks
ep18_check_seconds=$(remaining_seconds)
if (( ep18_check_seconds > 180 )); then ep18_check_seconds=180; fi
timeout --signal=TERM --kill-after=5s "${ep18_check_seconds}s" "$ep18_python" -B - "$ep18_code" <<'PY' > "$ep18_ops/core_reference_checks.txt" 2>&1
import sys
import unittest

sys.path.insert(0, sys.argv[1])
suite = unittest.defaultTestLoader.loadTestsFromName('test_production_core_reference')
if suite.countTestCases() != 2 or unittest.defaultTestLoader.errors:
    raise RuntimeError('Exactly the two selected core-reference checks must load')
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(0 if result.wasSuccessful() and result.testsRun == 2 and not result.skipped else 1)
PY

# The unchanged ASTRA program, including this core calibration, passed in
# recovery47584681 at18:04Pacific. Reuse that documentary check.

ep18_stage=all30_core_input_initialization
ep18_remaining=$(remaining_seconds)
ep18_now=$(date +%s)
ep18_initial_deadline=$((ep18_now+295))
if (( ep18_deadline < ep18_initial_deadline )); then ep18_initial_deadline=$ep18_deadline; fi
ep18_timeout=$((ep18_initial_deadline+5-ep18_now))
# Initialize owns at most128MiB; its complete marker binds all30 development
# participants and the complete assembly before any participant EEG is loaded.
timeout --signal=TERM --kill-after=5s "${ep18_timeout}s" "$ep18_python" -B \
  "$ep18_code/production_core_reference.py" initialize \
  --run-dir "$ep18_run" --release "$ep18_release" --recipe "$ep18_recipe" \
  --deadline-epoch "$ep18_initial_deadline" --rss-limit-bytes 12884901888 \
  --durable-byte-limit 134217728 --record-byte-limit 134217728 --threads 1 \
  > "$ep18_ops/initialize_stdout.txt" 2> "$ep18_ops/initialize_stderr.txt"

ep18_stage=complete_sub07_core_reference_cost
ep18_remaining=$(remaining_seconds)
ep18_timeout=$((ep18_remaining+5))
# The 2GiB participant allowance includes its reserved64KiB operational record;
# initialization's128MiB is separate. Retain every complete/partial private
# record. No full-cohort G/T or candidate comparison is produced by this job.
timeout --signal=TERM --kill-after=5s "${ep18_timeout}s" "$ep18_python" -B \
  "$ep18_code/production_core_reference.py" participant \
  --run-dir "$ep18_run" --participant-id sub-07 --cost-only \
  --deadline-epoch "$ep18_deadline" --rss-limit-bytes 12884901888 \
  --durable-byte-limit 2147483648 --record-byte-limit 134217728 --threads 1 \
  > "$ep18_ops/participant_stdout.txt" 2> "$ep18_ops/participant_stderr.txt"
ep18_stage=complete_sub07_cost_attempt
