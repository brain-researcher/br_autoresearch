#!/bin/bash -l
#SBATCH --job-name=ep01-mz-pilot
#SBATCH --partition=normal
#SBATCH --constraint=CPU_SKU:7502
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=01:35:00
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-SESOI-representative-%j.out
#SBATCH --error=outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-SESOI-representative-%j.err

# One selected representative, after coordinator review and compute checks.
# No submission, retry, successor or empirical data access from this script.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load devel python/3.12.1 math py-numpy/1.26.3 py-scikit-learn/1.5.1_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
hostname
printf 'SLURM_JOB_ID=%s SLURM_CPUS_PER_TASK=%s\n' "$SLURM_JOB_ID" "$SLURM_CPUS_PER_TASK"
python3.12 --version
exec timeout --signal=TERM --kill-after=10s 5400s python3.12 -B \
  outputs/code/run_single_process_recovery.py \
  --implementation outputs/code/behavioral_implementation.json \
  --design-events outputs/runs/behavioral_mechanism_kill_test/20261003-MC-SESOI-replicate000-attempt003/simulated_events.csv.gz \
  --model MZ --replicate-index 0 \
  --output-dir outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-SESOI-replicate000-attempt001
