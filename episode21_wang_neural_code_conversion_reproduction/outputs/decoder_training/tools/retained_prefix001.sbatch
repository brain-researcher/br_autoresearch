#!/bin/bash
# Approved2026-10-05; requires the exact bounded decoder-prefix authorization.
# Root submits once with afterok on the complete feature export.
#SBATCH --job-name=ep21-decoder-prefix
#SBATCH --partition=russpold,normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=outputs/decoder_training/retained_prefix001-%j.out
#SBATCH --error=outputs/decoder_training/retained_prefix001-%j.err
set -euo pipefail
[[ $# -eq 0 ]] || exit 2
ep21_prefix_host=$(hostname)
printf 'HOST=%s JOB=%s CPUS=%s MEM_MIB=%s\n' "$ep21_prefix_host" \
  "${SLURM_JOB_ID:?}" "${SLURM_CPUS_PER_TASK:?}" "${SLURM_MEM_PER_NODE:?}"
[[ "$ep21_prefix_host" != *-ln* && "$SLURM_CPUS_PER_TASK" == 1 \
   && "$SLURM_MEM_PER_NODE" == 7168 ]]
cd "${OAK:?}/users/zijiao/br_autoresearch/episode21_wang_neural_code_conversion_reproduction"
[[ -f outputs/decoder_training_prefix_authorization.json ]]
source /share/software/user/open/lmod/lmod/init/bash
module purge
module load devel python/3.12.1 math py-numpy/1.26.3 py-h5py/3.10.0_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
bash -n outputs/decoder_training/tools/retained_prefix001.sbatch
ep21_prefix_astra="${GROUP_HOME:?}/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv/bin/astra-tools"
set +e
timeout --signal=TERM --kill-after=10s 30s "$ep21_prefix_astra" validate \
  outputs/astra/v0.0.14/astra.yaml --json \
  > "outputs/decoder_training/retained_prefix001-astra-${SLURM_JOB_ID}.json" 2>&1
ep21_prefix_astra_exit=$?
set -e
printf 'astra_validation_exit=%s\n' "$ep21_prefix_astra_exit"
timeout --signal=TERM --kill-after=10s 60s python3.12 -B -m unittest discover \
  -s outputs/decoder_training/tests -p test_train_retained_prefix.py -v
exec timeout --signal=TERM --kill-after=10s 6930s python3.12 -u -B \
  outputs/decoder_training/tools/train_retained_prefix.py \
  --authorization outputs/decoder_training_prefix_authorization.json \
  --participant sub-01 --units-per-layer 64 --workers 1 --max-seconds 6900
