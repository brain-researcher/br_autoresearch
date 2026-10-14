#!/bin/bash
# Prepared first-grid cost invocation; coordinator selects and submits once.
# Reuses qualified checks and completed original input binding. No successor.
#SBATCH --job-name=ep18-scalar-cost001
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=15G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/baseline_models/core_scalar_operations_20261013_001/slurm-%j.out
#SBATCH --error=outputs/runs/baseline_models/core_scalar_operations_20261013_001/slurm-%j.err
set -euo pipefail
umask 077
[[ $# == 0 && -n "${SLURM_JOB_ID:-}" && "$(hostname)" != *-ln* ]]
[[ "${SLURM_CPUS_PER_TASK:-}" == 4 && "${SLURM_MEM_PER_NODE:-}" == 15360 ]]
[[ "${SLURM_JOB_NUM_NODES:-}" == 1 && "${SLURM_JOB_CPUS_PER_NODE:-}" == 4 ]]
[[ -z "${SLURM_ARRAY_TASK_ID:-}" && -z "${SLURM_JOB_GPUS:-}" && -z "${SLURM_STEP_GPUS:-}" ]]
[[ "${SLURM_RESTART_COUNT:-0}" == 0 ]]
: "${OAK:?}" "${GROUP_HOME:?}" "${L_SCRATCH_JOB:?}"
ep18_start=${SLURM_JOB_START_TIME:?}
ep18_end=${SLURM_JOB_END_TIME:?}
[[ "$ep18_start" =~ ^[0-9]+$ && "$ep18_end" =~ ^[0-9]+$ ]]
ep18_now=$(date +%s)
(( ep18_start > 0 && ep18_start <= ep18_now && ep18_now < ep18_end ))
ep18_deadline=$((ep18_start+7140))
if (( ep18_end-60 < ep18_deadline )); then ep18_deadline=$((ep18_end-60)); fi
if (( 1791926990 < ep18_deadline )); then ep18_deadline=1791926990; fi
(( ep18_deadline-ep18_now > 60 ))
ep18_episode="$OAK/users/zijiao/br_autoresearch/episode18_things_eeg_concept_stability"
ep18_code="$ep18_episode/outputs/code"
ep18_initialized="$ep18_episode/outputs/cache/private/core_reference_20261013_001"
ep18_private="$ep18_episode/outputs/cache/private/core_scalar_cost_20261013_001"
ep18_operations="$ep18_episode/outputs/runs/baseline_models/core_scalar_operations_20261013_001"
[[ -d "$ep18_initialized" && -d "$ep18_operations" && ! -e "$ep18_private" ]]
ep18_batch="$ep18_operations/batch_${SLURM_JOB_ID}"
ep18_public="$ep18_operations/initial_${SLURM_JOB_ID}"
mkdir -m 700 "$ep18_batch"
ep18_stage=allocation_admission
ep18_started=$SECONDS
finish() {
  ep18_exit=$?
  trap - EXIT
  printf '{"stage":"%s","exit_code":%s,"elapsed_seconds":%s,"job_id":"%s","deadline_epoch":%s,"allocated_cpus":4,"memory_mib":15360,"numerical_workers":4,"BLAS_threads_per_worker":1,"reserved_cpu_hours":8,"parent_RSS_guard_bytes":12884901888,"aggregate_PSS_guard_bytes":12884901888,"automatic_retry":false,"cost_only":true,"scientific_release_authorized":false}\n' \
    "$ep18_stage" "$ep18_exit" "$((SECONDS-ep18_started))" "$SLURM_JOB_ID" "$ep18_deadline" > "$ep18_batch/exit.json"
  exit "$ep18_exit"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
ulimit -c 0
# Both specific affected components must already have passed in coordinated compute.
for ep18_checked in scalar_checkpoint001 scalar_cleanup001 scalar_driver002; do
  rg -qx 'checks_exit=0' "$ep18_episode/outputs/runs/software_integration/$ep18_checked/status.txt"
done
ep18_local="$L_SCRATCH_JOB/ep18-core-scalar-cost001"
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
unset PYTHONPATH PYTHONHOME
ep18_python="$GROUP_HOME/users/zijiao/ep18_model_runtime_cpu241_py312_46842156/bin/python"
[[ -x "$ep18_python" ]]
cd "$ep18_local"
ep18_stage=original_first_symmetric_grid_cost
ep18_now=$(date +%s)
(( ep18_deadline-ep18_now > 20 ))
ep18_timeout=$((ep18_deadline+5-ep18_now))
timeout --signal=TERM --kill-after=5s "${ep18_timeout}s" "$ep18_python" -B \
  "$ep18_code/production_core_scalar_cost.py" \
  --initialized-root "$ep18_initialized" --private-dir "$ep18_private" --public-dir "$ep18_public" \
  --workers 4 --deadline-epoch "$ep18_deadline" \
  --rss-limit-bytes 12884901888 --aggregate-pss-limit-bytes 12884901888 \
  --durable-byte-limit 2147483648 --record-byte-limit 134217728 \
  > "$ep18_batch/driver_stdout.txt" 2> "$ep18_batch/driver_stderr.txt"
ep18_stage=complete_first_grid_cost_attempt
