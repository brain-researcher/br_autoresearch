#!/bin/bash
# UNACTIVATED: coordinator supplies afterok for the complete fit-array parent.
# This validates/assembles decoders; prediction, scoring and converters stay closed.
#SBATCH --job-name=ep21-full-assembly
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --output=outputs/decoder_training/full_assembly001-%j.out
#SBATCH --error=outputs/decoder_training/full_assembly001-%j.err
set -euo pipefail
[[ $# -eq 0 ]] || exit 2
ep21_assembly_host=$(hostname)
printf 'HOST=%s JOB=%s CPUS=%s MEM_MIB=%s\n' "$ep21_assembly_host" \
  "${SLURM_JOB_ID:?}" "${SLURM_CPUS_PER_TASK:?}" "${SLURM_MEM_PER_NODE:?}"
[[ "$ep21_assembly_host" != *-ln* && "$SLURM_CPUS_PER_TASK" == 1 && "$SLURM_MEM_PER_NODE" == 7168 ]]
cd "${OAK:?}/users/zijiao/br_autoresearch/episode21_wang_neural_code_conversion_reproduction"
[[ -f outputs/decoder_training_full_authorization.json \
   && -f outputs/decoder_training/full_continuation_001/work_manifest.json ]]
source /share/software/user/open/lmod/lmod/init/bash
module purge
module load devel python/3.12.1 math py-numpy/1.26.3_py312 py-h5py/3.10.0_py312 py-scipy/1.12.0_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
[[ -n "${L_SCRATCH_JOB:?}" && -d "$L_SCRATCH_JOB" ]]
export TMPDIR="$L_SCRATCH_JOB"
# Assembly derives its internal deadline from the verified allocation and full cap.
exec timeout --signal=TERM --kill-after=10s 14130s python3.12 -u -B \
  outputs/decoder_training/tools/decoder_assembly.py \
  --authorization outputs/decoder_training_full_authorization.json --block-units 256
