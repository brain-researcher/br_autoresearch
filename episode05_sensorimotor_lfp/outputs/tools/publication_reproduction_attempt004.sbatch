#!/bin/bash
#SBATCH --job-name=ep05-reference04
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:40:00
#SBATCH --no-requeue
#SBATCH --chdir=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode05_sensorimotor_lfp
#SBATCH --output=outputs/publication_reproduction_attempt004_%j.out
#SBATCH --error=outputs/publication_reproduction_attempt004_%j.err

# One remaining session from the timed-out three-session group; no loop/retry.
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
if [[ -n "${SLURM_ARRAY_TASK_ID-}" ]]; then
  printf 'Attempt004 is one exact session, not an array.\n' >&2
  exit 2
fi

: "${OAK:?OAK must be set by Sherlock}"
: "${SCRATCH:?SCRATCH must be set by Sherlock}"
EP05_ROOT="$OAK/users/zijiao/br_autoresearch/episode05_sensorimotor_lfp"
cd "$EP05_ROOT"
EP05_SESSION=Chewie_20150313
EP05_WORK="$SCRATCH/br_autoresearch/episode05_sensorimotor_lfp/remaining_sessions_attempt004/$EP05_SESSION"
EP05_OUT="outputs/runs/publication_reproduction/remaining_sessions_attempt004/$EP05_SESSION"
if [[ -e "$EP05_WORK" || -e "$EP05_OUT" ]]; then
  printf 'Preserve previous attempt004 directories; coordinator must diagnose any retry.\n' >&2
  exit 2
fi

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
date -u '+session_start_utc=%Y-%m-%dT%H:%M:%SZ'
printf 'selected_session=%s qualified_trial_count=1038 unchanged_seed=20261015\n' "$EP05_SESSION"
# The unchanged runner derives seed20261015 from contract index12, retains
# all1038 trials and the pinned revisions, and bounds MATLAB at1500seconds.
python3 outputs/tools/run_publication_reproduction.py \
  --source "$OAK/data/br_autoresearch_data/sensorimotor_lfp/dryad-xd2547dkt-v5-payload/source" \
  --trial-table outputs/runs/source_support/attempt002/trial_identity.csv \
  --session "$EP05_SESSION" \
  --work "$EP05_WORK" \
  --out "$EP05_OUT" \
  --matlab /share/software/user/restricted/matlab/R2022b/bin/matlab
date -u '+session_finished_utc=%Y-%m-%dT%H:%M:%SZ'
