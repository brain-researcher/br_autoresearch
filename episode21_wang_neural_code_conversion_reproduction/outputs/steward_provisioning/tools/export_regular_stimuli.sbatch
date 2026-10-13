#!/bin/bash
#SBATCH --job-name=ep21-regular-stimuli
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=outputs/steward_provisioning/export-%j.log
set -euo pipefail
source /share/software/user/open/lmod/lmod/init/bash
module purge
module load devel math python/3.12.1 py-numpy/1.26.3_py312 py-h5py/3.10.0_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 -B outputs/steward_provisioning/tools/export_regular_stimuli.py
