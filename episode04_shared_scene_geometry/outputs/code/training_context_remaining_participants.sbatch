#!/bin/bash -l
#SBATCH --job-name=ep04-context-rest
#SBATCH --partition=normal,russpold
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=20
#SBATCH --no-requeue

# Selected after the coordinator checked sub-01's completed result.
# One finite pack; no submission, retry, resume or later-participant logic.
set -euo pipefail
[[ $# == 0 ]]
cd "${OAK:?}/users/zijiao/br_autoresearch/episode04_shared_scene_geometry"
EP04_REST_HOST=$(hostname)
printf 'HOST=%s JOB=%s NTASKS=%s CPUS_PER_TASK=%s\n' \
  "$EP04_REST_HOST" "${SLURM_JOB_ID:?}" "${SLURM_NTASKS:?}" "${SLURM_CPUS_PER_TASK:?}"
[[ "$SLURM_JOB_ID" =~ ^[0-9]+$ && "$EP04_REST_HOST" != *-ln* \
   && "$SLURM_NTASKS" == 3 && "$SLURM_CPUS_PER_TASK" == 1 ]]
module purge
module load devel python/3.12.1 math py-numpy/1.26.3
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
EP04_REST_ROOT="outputs/runs/training_neural_context/20261005-remaining-context-${SLURM_JOB_ID}"
mkdir "$EP04_REST_ROOT"
EP04_REST_PARTICIPANTS=(sub-03 sub-06 sub-07)
EP04_REST_PIDS=()
for EP04_REST_PARTICIPANT in "${EP04_REST_PARTICIPANTS[@]}"; do
  srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=4G \
    timeout --signal=TERM --kill-after=10s 900s python3.12 -u -B \
    outputs/code/training_neural_context.py \
    --preprocessing-report "outputs/runs/neural_preprocessing/20261003-remaining-folds01/${EP04_REST_PARTICIPANT}/preprocessing.json" \
    --metadata-dir outputs/runs/source_readiness/20261003-metadata_folds04 \
    --output-dir "$EP04_REST_ROOT/$EP04_REST_PARTICIPANT" \
    --max-seconds 840 \
    >"$EP04_REST_ROOT/${EP04_REST_PARTICIPANT}.out" \
    2>"$EP04_REST_ROOT/${EP04_REST_PARTICIPANT}.err" &
  EP04_REST_PIDS+=("$!")
done
EP04_REST_ANY_FAILURE=0
printf 'participant\texit_status\n' >"$EP04_REST_ROOT/participant_exit_status.tsv"
for EP04_REST_INDEX in 0 1 2; do
  if wait "${EP04_REST_PIDS[$EP04_REST_INDEX]}"; then
    EP04_REST_STATUS=0
  else
    EP04_REST_STATUS=$?
    EP04_REST_ANY_FAILURE=1
  fi
  printf '%s\t%s\n' "${EP04_REST_PARTICIPANTS[$EP04_REST_INDEX]}" "$EP04_REST_STATUS" \
    >>"$EP04_REST_ROOT/participant_exit_status.tsv"
done
exit "$EP04_REST_ANY_FAILURE"
