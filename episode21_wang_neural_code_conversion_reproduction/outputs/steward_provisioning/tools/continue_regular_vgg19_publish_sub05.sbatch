#!/bin/bash
# Approved finite publication of the already generated sub-05 partition.
#SBATCH --job-name=ep21_vgg19_publish_sub05
#SBATCH --partition=russpold
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=outputs/steward_provisioning/regular_vgg19_continuation_publish_sub05-%j.out
#SBATCH --error=outputs/steward_provisioning/regular_vgg19_continuation_publish_sub05-%j.err
set -euo pipefail
[[ $# -eq 0 ]] || { printf 'This fixed publication wrapper takes no arguments.\n' >&2; exit 2; }
task_host=$(hostname)
[[ -n ${SLURM_JOB_ID:-} && $task_host != *-ln* && ${SLURM_CPUS_PER_TASK:-} == 1 ]] || exit 2
printf 'hostname=%s SLURM_JOB_ID=%s cpus=%s\n' "$task_host" "$SLURM_JOB_ID" "$SLURM_CPUS_PER_TASK"
task_episode="$OAK/users/zijiao/br_autoresearch/episode21_wang_neural_code_conversion_reproduction"
cd "$task_episode"
source /share/software/user/open/lmod/lmod/init/bash
module purge >/dev/null 2>&1
module load devel math viz python/3.12.1 py-numpy/1.26.3_py312
module load py-h5py/3.10.0_py312 py-pillow/10.2.0_py312 py-pytorch/2.4.1_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
task_runtime="$GROUP_HOME/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules"
# Source syntax only; reuse the reviewed adapter and passed conformance.
bash -n outputs/steward_provisioning/tools/continue_regular_vgg19_publish_sub05.sbatch
bash -n outputs/steward_provisioning/tools/continue_regular_vgg19_remaining.sbatch
"$task_runtime/venv/bin/python" -B -c 'import ast; from pathlib import Path; ast.parse(Path("outputs/steward_provisioning/tools/continue_regular_vgg19.py").read_text())'
# Changed ASTRA recipe validation is optional and cannot authorize data access.
set +e
timeout --signal=TERM --kill-after=10s 30s "$task_runtime/venv/bin/astra-tools" validate \
  outputs/astra/v0.0.14/astra.yaml --json \
  > "outputs/steward_provisioning/regular_vgg19_continuation_astra-${SLURM_JOB_ID}.json" 2>&1
task_astra_exit=$?
set -e
printf 'astra_validation_exit=%s\n' "$task_astra_exit"
exec timeout --signal=TERM --kill-after=10s 7100s \
  "$task_runtime/venv/bin/python" -u -B outputs/steward_provisioning/tools/continue_regular_vgg19.py \
  --stage publish-sub05 \
  --authorization outputs/regular_vgg19_continuation_authorization_2026-10-05.json \
  --max-seconds 7080
