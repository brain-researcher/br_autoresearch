#!/bin/bash
# Explicit selected previously unattempted runs; no automatic retry.
#SBATCH --job-name=ep01-pending-fmri
#SBATCH --partition=russpold,normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=15G
#SBATCH --time=00:45:00
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/exploratory_fmri_specification_sensitivity/pending_20261013_001/slurm-%A_%a.out
#SBATCH --error=outputs/runs/exploratory_fmri_specification_sensitivity/pending_20261013_001/slurm-%A_%a.err
set -euo pipefail
umask 077
[[ -n "${SLURM_JOB_ID:-}" && "$(hostname)" != *-ln* ]]
[[ "${SLURM_CPUS_PER_TASK:-}" == 2 && "${SLURM_MEM_PER_NODE:-}" == 15360 ]]
[[ "${SLURM_ARRAY_TASK_ID:-}" =~ ^[0-3]$ && "${SLURM_RESTART_COUNT:-0}" == 0 ]]
: "${OAK:?}" "${GROUP_HOME:?}" "${L_SCRATCH_JOB:?}"
ep01_base="$OAK/users/zijiao/br_autoresearch/episode01_narps_deep_search"
ep01_root="$ep01_base/outputs/runs/exploratory_fmri_specification_sensitivity"
ep01_pending="$ep01_root/pending_20261013_001"
ep01_ops="$ep01_pending/ops_${SLURM_ARRAY_TASK_ID}_${SLURM_JOB_ID}"
mkdir -m 700 "$ep01_ops"
ep01_begin=$SECONDS
finish() {
  ep01_exit=$?
  trap - EXIT
  printf '{"job_id":"%s","array_task":"%s","exit_code":%s,"elapsed_seconds":%s,"allocated_cpus":2,"memory_mib":15360,"automatic_retry":false}\n' \
    "$SLURM_JOB_ID" "$SLURM_ARRAY_TASK_ID" "$ep01_exit" "$((SECONDS-ep01_begin))" > "$ep01_ops/exit.json"
  exit "$ep01_exit"
}
trap finish EXIT
ulimit -c 0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH PYTHONHOME
export TMPDIR="$L_SCRATCH_JOB/ep01-pending-tmp"
mkdir -m 700 "$TMPDIR"
ep01_python="$GROUP_HOME/zijiao/miniforge3/envs/encoding_model/bin/python"
ep01_seconds=$((SLURM_JOB_END_TIME-$(date +%s)-60))
if (( ep01_seconds > 2640 )); then ep01_seconds=2640; fi
(( ep01_seconds > 60 ))
cd "$ep01_base/outputs/code"
timeout --signal=TERM --kill-after=10s "$((ep01_seconds+10))s" "$ep01_python" -B \
  continue_exploratory_fmri.py --prepared-dir "$ep01_root/prepared_20261013_002" \
  --selection-json "$ep01_pending/selection_${SLURM_ARRAY_TASK_ID}.json" \
  --previous-run-root "$ep01_root/pilot_20261013_47670430/models02" \
  --previous-run-root "$ep01_root/cohort_20261013_001/models" \
  --output-dir "$ep01_pending/pack_${SLURM_ARRAY_TASK_ID}_models" \
  --max-seconds "$ep01_seconds" --max-rss-gib 12 --max-local-gib 12 \
  > "$ep01_ops/fit.log" 2>&1
