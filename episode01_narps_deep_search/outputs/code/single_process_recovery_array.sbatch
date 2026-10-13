#!/bin/bash -l
#SBATCH --partition=normal
#SBATCH --constraint=CPU_SKU:7502
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=01:35:00
#SBATCH --array=1-20%20
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/behavioral_mechanism_kill_test/20261005-single-process-array-%A_%a.out
#SBATCH --error=outputs/runs/behavioral_mechanism_kill_test/20261005-single-process-array-%A_%a.err

# Fixed indices only. The coordinator owns pooling and pending-task curtailment.
# This script never submits, retries, renews, or opens observed responses.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
EP_RECOVERY_MODEL="${1:?Supply MV or MZ}"
case "$EP_RECOVERY_MODEL" in
  MV|MZ) ;;
  *) printf 'Unsupported recovery model\n' >&2; exit 2 ;;
esac
case "${SLURM_ARRAY_TASK_ID-}" in
  ''|*[!0-9]*) printf 'Missing numeric array index\n' >&2; exit 2 ;;
esac
if (( SLURM_ARRAY_TASK_ID < 1 || SLURM_ARRAY_TASK_ID > 20 )); then
  printf 'Array index is outside the selected 1-20 wave\n' >&2
  exit 2
fi
module purge
module load devel python/3.12.1 math py-numpy/1.26.3 py-scikit-learn/1.5.1_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
hostname
printf 'SLURM_JOB_ID=%s SLURM_ARRAY_JOB_ID=%s SLURM_ARRAY_TASK_ID=%s MODEL=%s CPUS=%s\n' \
  "$SLURM_JOB_ID" "$SLURM_ARRAY_JOB_ID" "$SLURM_ARRAY_TASK_ID" "$EP_RECOVERY_MODEL" "$SLURM_CPUS_PER_TASK"
python3.12 --version
exec timeout --signal=TERM --kill-after=10s 5400s python3.12 -B \
  outputs/code/run_single_process_recovery.py \
  --implementation outputs/code/behavioral_implementation.json \
  --design-events outputs/runs/behavioral_mechanism_kill_test/20261003-MC-SESOI-replicate000-attempt003/simulated_events.csv.gz \
  --model "$EP_RECOVERY_MODEL" --replicate-index "$SLURM_ARRAY_TASK_ID" \
  --output-dir "outputs/runs/behavioral_mechanism_kill_test/20261005-${EP_RECOVERY_MODEL}-SESOI-array-${SLURM_ARRAY_JOB_ID}/replicate${SLURM_ARRAY_TASK_ID}"
