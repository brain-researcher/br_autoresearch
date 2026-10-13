#!/bin/bash -l
#SBATCH --job-name=ep01-mc-recovery
#SBATCH --partition=normal
#SBATCH --constraint=CPU_SKU:7502
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=01:30:00
#SBATCH --array=1-24%4
#SBATCH --output=outputs/runs/behavioral_mechanism_kill_test/20261003-MC-SESOI-array-%A_%a.out
#SBATCH --error=outputs/runs/behavioral_mechanism_kill_test/20261003-MC-SESOI-array-%A_%a.err

# One fixed, unique replicate per task. No retries or successor submissions.
# The coordinator pools valid reports once and stops pending work if the
# registered 200-replicate cell can no longer reach 90 percent success.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
module purge
module load devel python/3.12.1 math py-numpy/1.26.3 py-scikit-learn/1.5.1_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
hostname
printf 'SLURM_JOB_ID=%s SLURM_ARRAY_TASK_ID=%s\n' "$SLURM_JOB_ID" "$SLURM_ARRAY_TASK_ID"
python3 --version
exec python3 outputs/code/run_criterion_recovery.py \
  --events outputs/runs/public_asset_qualification/20261003-attempt01/original_events.csv.gz \
  --implementation outputs/code/behavioral_implementation.json \
  --replicate-index "$SLURM_ARRAY_TASK_ID" \
  --output-dir "outputs/runs/behavioral_mechanism_kill_test/20261003-MC-SESOI-array-${SLURM_ARRAY_JOB_ID}/replicate${SLURM_ARRAY_TASK_ID}"
