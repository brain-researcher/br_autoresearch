#!/bin/bash
#SBATCH --job-name=ep05-reference02
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
# Slurm directives cannot expand environment variables; fix this episode only.
#SBATCH --chdir=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode05_sensorimotor_lfp
#SBATCH --output=outputs/publication_reproduction_attempt002_%j.out
#SBATCH --error=outputs/publication_reproduction_attempt002_%j.err

set -euo pipefail
EP05_EXEC_HOST=$(hostname)
printf 'hostname=%s SLURM_JOB_ID=%s SLURM_CPUS_PER_TASK=%s\n' \
  "$EP05_EXEC_HOST" "${SLURM_JOB_ID-}" "${SLURM_CPUS_PER_TASK-}"
if [[ -z "${SLURM_JOB_ID-}" || "$EP05_EXEC_HOST" == *-ln* ]]; then
  printf 'EP05 reference execution requires its allocated compute node.\n' >&2
  exit 2
fi
if [[ ! "${SLURM_CPUS_PER_TASK-}" =~ ^[1-9][0-9]*$ ]]; then
  printf 'This selected reference attempt requires at least one allocated CPU.\n' >&2
  exit 2
fi

: "${OAK:?OAK must be set by Sherlock}"
: "${SCRATCH:?SCRATCH must be set by Sherlock}"
EP05_ROOT="$OAK/users/zijiao/br_autoresearch/episode05_sensorimotor_lfp"
cd "$EP05_ROOT"
EP05_WORK="$SCRATCH/br_autoresearch/episode05_sensorimotor_lfp/reference_Mihili_20140217_attempt002"
EP05_OUT=outputs/runs/publication_reproduction/Mihili_20140217_attempt002
if [[ -e "$EP05_WORK" || -e "$EP05_OUT" ]]; then
  printf 'Preserve prior attempt directories; coordinator must diagnose any retry.\n' >&2
  exit 2
fi

# Caches and MATLAB preferences are disposable and private to this allocation.
EP05_TEMP_BASE="${L_SCRATCH_JOB:-$SCRATCH/br_autoresearch/episode05_sensorimotor_lfp}"
EP05_RUNTIME="$EP05_TEMP_BASE/ep05-reference-runtime-$SLURM_JOB_ID"
mkdir -p "$EP05_RUNTIME/tmp" "$EP05_RUNTIME/cache" "$EP05_RUNTIME/matlab-prefs"
export TMPDIR="$EP05_RUNTIME/tmp"
export XDG_CACHE_HOME="$EP05_RUNTIME/cache"
export MATLAB_PREFDIR="$EP05_RUNTIME/matlab-prefs"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1

module purge
module load math devel matlab/R2022b
module load py-scikit-learn/1.5.1_py312 py-h5py/3.10.0_py312
module list
date -u '+start_utc=%Y-%m-%dT%H:%M:%SZ'

# The runner invokes the unchanged MATLAB functions with -singleCompThread.
python3 outputs/tools/run_publication_reproduction.py \
  --source "$OAK/data/br_autoresearch_data/sensorimotor_lfp/dryad-xd2547dkt-v5-payload/source" \
  --trial-table outputs/runs/source_support/attempt002/trial_identity.csv \
  --session Mihili_20140217 \
  --work "$EP05_WORK" \
  --out "$EP05_OUT" \
  --matlab /share/software/user/restricted/matlab/R2022b/bin/matlab

date -u '+finished_utc=%Y-%m-%dT%H:%M:%SZ'
