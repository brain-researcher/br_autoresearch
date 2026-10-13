#!/bin/bash
#SBATCH --job-name=ep14-fixed-roi
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/runs/fixed_development_comparison/attempt001/development-%j.out
#SBATCH --error=outputs/runs/fixed_development_comparison/attempt001/development-%j.err

set -euo pipefail
hostname
printf 'SLURM_JOB_ID=%s\n' "${SLURM_JOB_ID:?}"
cd "${OAK:?}/users/zijiao/br_autoresearch/episode14_openbhb_roi_site_generalization"
module purge
module load devel math python/3.12.1 py-numpy/1.26.3_py312
module load py-scikit-learn/1.5.1_py312
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
  /share/software/user/open/python/3.12.1/bin/python3 \
  outputs/tools/run_development.py run \
  --out outputs/runs/fixed_development_comparison/attempt001 \
  --threads "${SLURM_CPUS_PER_TASK}"
