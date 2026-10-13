#!/bin/bash
# Approved October 5, 2026 Pacific: one 1CPU/7GiB/24h continuation.
# Coordinator submits once in a discovered eligible nonpreemptible partition.
#SBATCH --job-name=ep21-channel-001
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --time=24:00:00
#SBATCH --no-requeue
#SBATCH --output=outputs/decoder_training/continue_channel001-%j.out
#SBATCH --error=outputs/decoder_training/continue_channel001-%j.err
set -euo pipefail
[[ $# -eq 0 ]] || exit 2
ep21_continuation_host=$(hostname)
printf 'HOST=%s JOB=%s CPUS=%s MEM_MIB=%s\n' "$ep21_continuation_host" \
  "${SLURM_JOB_ID:?}" "${SLURM_CPUS_PER_TASK:?}" "${SLURM_MEM_PER_NODE:?}"
[[ "$ep21_continuation_host" != *-ln* && "$SLURM_CPUS_PER_TASK" == 1 \
   && "$SLURM_MEM_PER_NODE" == 7168 ]]
cd "${OAK:?}/users/zijiao/br_autoresearch/episode21_wang_neural_code_conversion_reproduction"
[[ -f outputs/decoder_training_continuation_authorization.json \
   && ! -e outputs/decoder_training/continuation_001 ]]
source /share/software/user/open/lmod/lmod/init/bash
module purge
module load devel python/3.12.1 math py-numpy/1.26.3 py-h5py/3.10.0_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
[[ -n "${L_SCRATCH_JOB:?}" && -d "$L_SCRATCH_JOB" ]]
export TMPDIR="$L_SCRATCH_JOB"
ep21_continuation_astra="${GROUP_HOME:?}/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv/bin/astra-tools"
set +e
timeout --signal=TERM --kill-after=5s 30s "$ep21_continuation_astra" validate \
  outputs/astra/v0.0.14/astra.yaml --json \
  > "outputs/decoder_training/continue_channel001-astra-${SLURM_JOB_ID}.json" 2>&1
ep21_continuation_astra_exit=$?
set -e
printf 'astra_validation_exit=%s\n' "$ep21_continuation_astra_exit"
# Only the new adapter checks; no repetition of completed source conformance.
timeout --signal=TERM --kill-after=10s 60s python3.12 -B -m unittest discover \
  -s outputs/decoder_training/tests -p test_train_channel_continuation.py -v
exec timeout --signal=TERM --kill-after=10s 86130s python3.12 -u -B \
  outputs/decoder_training/tools/train_channel_continuation.py \
  --authorization outputs/decoder_training_continuation_authorization.json --max-seconds 86100
