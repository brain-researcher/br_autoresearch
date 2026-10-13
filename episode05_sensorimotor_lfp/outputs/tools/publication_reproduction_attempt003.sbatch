#!/bin/bash
#SBATCH --job-name=ep05-reference03
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
# Five finite groups of three sessions satisfy the site's short-job aggregation.
#SBATCH --array=1-5%4
#SBATCH --chdir=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode05_sensorimotor_lfp
#SBATCH --output=outputs/publication_reproduction_attempt003_%A_%a.out
#SBATCH --error=outputs/publication_reproduction_attempt003_%A_%a.err

set -euo pipefail
EP05_EXEC_HOST=$(hostname)
printf 'hostname=%s SLURM_JOB_ID=%s SLURM_ARRAY_JOB_ID=%s SLURM_ARRAY_TASK_ID=%s SLURM_CPUS_PER_TASK=%s\n' \
  "$EP05_EXEC_HOST" "${SLURM_JOB_ID-}" "${SLURM_ARRAY_JOB_ID-}" \
  "${SLURM_ARRAY_TASK_ID-}" "${SLURM_CPUS_PER_TASK-}"
if [[ -z "${SLURM_JOB_ID-}" || "$EP05_EXEC_HOST" == *-ln* ]]; then
  printf 'EP05 reference execution requires its allocated compute node.\n' >&2
  exit 2
fi
if [[ ! "${SLURM_CPUS_PER_TASK-}" =~ ^[1-9][0-9]*$ ]]; then
  printf 'This selected reference attempt requires at least one allocated CPU.\n' >&2
  exit 2
fi
if [[ ! "${SLURM_ARRAY_TASK_ID-}" =~ ^[1-5]$ ]]; then
  printf 'Only the five predefined finite session groups are authorized.\n' >&2
  exit 2
fi

: "${OAK:?OAK must be set by Sherlock}"
: "${SCRATCH:?SCRATCH must be set by Sherlock}"
EP05_ROOT="$OAK/users/zijiao/br_autoresearch/episode05_sensorimotor_lfp"
cd "$EP05_ROOT"
# Exact remaining contract order. The completed first session is absent.
EP05_SESSIONS=(
  Mihili_20140218 Mihili_20140303 Mihili_20140304
  Mihili_20140306 Mihili_20140307 Chewie_20160929
  Chewie_20161005 Chewie_20161006 Chewie_20161007
  Chewie_20161014 Chewie_20161021 Chewie_20150313
  Chewie_20150319 Chewie_20150629 Chewie_20150630
)
EP05_FIRST=$(( (SLURM_ARRAY_TASK_ID - 1) * 3 ))
EP05_GROUP=("${EP05_SESSIONS[@]:EP05_FIRST:3}")
# Preserve all previous directories, including partial attempts. No auto-retry.
for EP05_SESSION in "${EP05_GROUP[@]}"; do
  EP05_WORK="$SCRATCH/br_autoresearch/episode05_sensorimotor_lfp/remaining_sessions_attempt003/$EP05_SESSION"
  EP05_OUT="outputs/runs/publication_reproduction/remaining_sessions_attempt003/$EP05_SESSION"
  if [[ -e "$EP05_WORK" || -e "$EP05_OUT" ]]; then
    printf 'Preserve previous directory for %s; coordinator must diagnose any retry.\n' "$EP05_SESSION" >&2
    exit 2
  fi
done

EP05_TEMP_BASE="${L_SCRATCH_JOB:-$SCRATCH/br_autoresearch/episode05_sensorimotor_lfp}"
EP05_RUNTIME="$EP05_TEMP_BASE/ep05-reference-runtime-$SLURM_JOB_ID-$SLURM_ARRAY_TASK_ID"
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
date -u '+group_start_utc=%Y-%m-%dT%H:%M:%SZ'

for EP05_SESSION in "${EP05_GROUP[@]}"; do
  EP05_WORK="$SCRATCH/br_autoresearch/episode05_sensorimotor_lfp/remaining_sessions_attempt003/$EP05_SESSION"
  EP05_OUT="outputs/runs/publication_reproduction/remaining_sessions_attempt003/$EP05_SESSION"
  printf 'selected_session=%s\n' "$EP05_SESSION"
  # Fixed seed is derived inside this unchanged runner from the contract index.
  # Each MATLAB process receives -singleCompThread; sessions are sequential.
  python3 outputs/tools/run_publication_reproduction.py \
    --source "$OAK/data/br_autoresearch_data/sensorimotor_lfp/dryad-xd2547dkt-v5-payload/source" \
    --trial-table outputs/runs/source_support/attempt002/trial_identity.csv \
    --session "$EP05_SESSION" \
    --work "$EP05_WORK" \
    --out "$EP05_OUT" \
    --matlab /share/software/user/restricted/matlab/R2022b/bin/matlab
done

date -u '+group_finished_utc=%Y-%m-%dT%H:%M:%SZ'
