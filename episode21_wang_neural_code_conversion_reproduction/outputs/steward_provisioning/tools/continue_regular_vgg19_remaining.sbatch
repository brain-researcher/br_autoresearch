#!/bin/bash
# Approved remaining four partitions; root submits after successful sub-05 publication.
#SBATCH --job-name=ep21_vgg19_remaining
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -G 1
#SBATCH --time=10:00:00
#SBATCH --no-requeue
#SBATCH --output=outputs/steward_provisioning/regular_vgg19_continuation_remaining-%j.out
#SBATCH --error=outputs/steward_provisioning/regular_vgg19_continuation_remaining-%j.err
set -euo pipefail
[[ $# -eq 0 ]] || { printf 'This fixed export wrapper takes no arguments.\n' >&2; exit 2; }
task_host=$(hostname)
[[ -n ${SLURM_JOB_ID:-} && $task_host != *-ln* && ${SLURM_CPUS_PER_TASK:-} == 4 ]] || exit 2
printf 'hostname=%s SLURM_JOB_ID=%s cpus=%s\n' "$task_host" "$SLURM_JOB_ID" "$SLURM_CPUS_PER_TASK"
task_episode="$OAK/users/zijiao/br_autoresearch/episode21_wang_neural_code_conversion_reproduction"
cd "$task_episode"
source /share/software/user/open/lmod/lmod/init/bash
module purge >/dev/null 2>&1
module load devel math viz python/3.12.1 py-numpy/1.26.3_py312
module load py-h5py/3.10.0_py312 py-pillow/10.2.0_py312 py-pytorch/2.4.1_py312
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4 NUMEXPR_NUM_THREADS=4 BLIS_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1
task_runtime="$GROUP_HOME/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules"
exec timeout --signal=TERM --kill-after=10s 35900s \
  "$task_runtime/venv/bin/python" -u -B outputs/steward_provisioning/tools/continue_regular_vgg19.py \
  --stage remaining-export \
  --authorization outputs/regular_vgg19_continuation_authorization_2026-10-05.json \
  --max-seconds 35880
