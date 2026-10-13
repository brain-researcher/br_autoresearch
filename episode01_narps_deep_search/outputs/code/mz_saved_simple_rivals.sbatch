#!/bin/bash -l
#SBATCH --job-name=ep01-mz-rivals
#SBATCH --partition=normal
#SBATCH --constraint=CPU_SKU:7502
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-saved-rival-%A_%a.out
#SBATCH --error=outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-saved-rival-%A_%a.err

# Root supplies the selected model and unique array indices at submission.
# No simulation, baseline refit, resubmission, or follow-on work.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
EP_RIVAL_MODEL="${1:?Supply MC, MV or MVZ}"
case "$EP_RIVAL_MODEL" in
  MC|MV|MVZ) ;;
  *) printf 'Only the selected MC/MV/MVZ wave is supported\n' >&2; exit 2 ;;
esac
case "${SLURM_ARRAY_TASK_ID-}" in
  ''|*[!0-9]*) printf 'Missing numeric array index\n' >&2; exit 2 ;;
esac
if (( SLURM_ARRAY_TASK_ID < 1 || SLURM_ARRAY_TASK_ID > 199 )); then
  printf 'Expected fixed replicate index1 through199\n' >&2
  exit 2
fi
EP_SOURCE_ARRAY=46539590
if (( SLURM_ARRAY_TASK_ID <= 20 )); then
  EP_SOURCE_ARRAY=46538652
fi
module purge
module load devel python/3.12.1 math py-numpy/1.26.3 py-scikit-learn/1.5.1_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
EP_PYTHON=$(command -v python3.12)
hostname
printf 'SLURM_JOB_ID=%s ARRAY=%s INDEX=%s MODEL=%s CPUS=%s\n' \
  "$SLURM_JOB_ID" "$SLURM_ARRAY_JOB_ID" "$SLURM_ARRAY_TASK_ID" "$EP_RIVAL_MODEL" "$SLURM_CPUS_PER_TASK"
"$EP_PYTHON" --version
exec timeout --signal=TERM --kill-after=10s 3300s "$EP_PYTHON" -B \
  outputs/code/run_mz_saved_replicate_rival.py \
  --model "$EP_RIVAL_MODEL" --replicate-index "$SLURM_ARRAY_TASK_ID" \
  --source-dir "outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-SESOI-array-${EP_SOURCE_ARRAY}/replicate${SLURM_ARRAY_TASK_ID}" \
  --implementation outputs/code/behavioral_implementation.json \
  --output-dir "outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-${EP_RIVAL_MODEL}-saved-array-${SLURM_ARRAY_JOB_ID}/replicate${SLURM_ARRAY_TASK_ID}" \
  --max-seconds 3300
