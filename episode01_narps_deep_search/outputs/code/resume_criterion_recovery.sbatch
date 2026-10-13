#!/bin/bash -l
#SBATCH --job-name=ep01-mc-resume
#SBATCH --partition=normal
#SBATCH --constraint=CPU_SKU:7502
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --output=outputs/runs/behavioral_mechanism_kill_test/20261003-MC-SESOI-replicate000-resume-%j.out
#SBATCH --error=outputs/runs/behavioral_mechanism_kill_test/20261003-MC-SESOI-replicate000-resume-%j.err

# Submit from the episode root only after the coordinator observes attempt002
# stopped. This single continuation never submits successors or other replicates.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
module purge
module load devel python/3.12.1 math py-numpy/1.26.3 py-scikit-learn/1.5.1_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
hostname
printf 'SLURM_JOB_ID=%s\n' "${SLURM_JOB_ID}"
python3 --version
exec python3 outputs/code/resume_criterion_recovery.py \
  --source-attempt outputs/runs/behavioral_mechanism_kill_test/20261003-MC-SESOI-replicate000-attempt002 \
  --implementation outputs/code/behavioral_implementation.json \
  --replicate-index 0 \
  --output-dir outputs/runs/behavioral_mechanism_kill_test/20261003-MC-SESOI-replicate000-attempt003
