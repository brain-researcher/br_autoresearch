#!/bin/bash -l
#SBATCH --job-name=ep01-offer-refits
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:45:00
#SBATCH --no-requeue

# Selected by coordinator after the successful229.574s EI-offers-01 pilot.
# The fixed remaining15 groups require explicit --array=1-3 at submission.
# Supply Slurm --output/--error paths at submission. No hardware constraint.
# Reuse the passed seven checks and ASTRA; do not rerun them in each task.
# Each task executes five fixed omissions sequentially. Internal1200s,
# external1220s plus20s kill grace => at most6200s plus setup within6300s.
# Failures are preserved and never retried; independent later groups continue.
set -euo pipefail

EP_OFFER_HOST=$(hostname)
printf 'HOST=%s SLURM_JOB_ID=%s ARRAY_JOB_ID=%s ARRAY_TASK_ID=%s NTASKS=%s CPUS_PER_TASK=%s\n' \
  "$EP_OFFER_HOST" "${SLURM_JOB_ID-}" "${SLURM_ARRAY_JOB_ID-}" \
  "${SLURM_ARRAY_TASK_ID-}" "${SLURM_NTASKS-}" "${SLURM_CPUS_PER_TASK-}"
if [[ -z "${SLURM_JOB_ID-}" || "$EP_OFFER_HOST" == *-ln* || \
      ! "${SLURM_ARRAY_JOB_ID-}" =~ ^[1-9][0-9]*$ || \
      "${SLURM_NTASKS-}" != 1 || "${SLURM_CPUS_PER_TASK-}" != 4 || "$#" != 0 ]]; then
  printf 'Requires one actual four-CPU compute array task and no arguments\n' >&2
  exit 2
fi
case "${SLURM_ARRAY_TASK_ID-}" in
  1) EP_OFFER_GROUPS=(EI-offers-02 EI-offers-03 EI-offers-04 EI-offers-05 EI-offers-06) ;;
  2) EP_OFFER_GROUPS=(EI-offers-07 EI-offers-08 ER-offers-01 ER-offers-02 ER-offers-03) ;;
  3) EP_OFFER_GROUPS=(ER-offers-04 ER-offers-05 ER-offers-06 ER-offers-07 ER-offers-08) ;;
  *) printf 'Coordinator must explicitly select array task1,2 or3\n' >&2; exit 2 ;;
esac

cd "$SLURM_SUBMIT_DIR"
module purge
module load devel python/3.12.1 math py-numpy/1.26.3 py-scikit-learn/1.5.1_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="outputs/code${PYTHONPATH:+:$PYTHONPATH}"
EP_OFFER_PYTHON=$(command -v python3.12)
EP_OFFER_ROOT="outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-offer-sequence-refits-${SLURM_ARRAY_JOB_ID}"
EP_OFFER_TASK_ROOT="$EP_OFFER_ROOT/task${SLURM_ARRAY_TASK_ID}"
mkdir -p "$EP_OFFER_ROOT"
mkdir "$EP_OFFER_TASK_ROOT"
printf 'array_job_id\tarray_task_id\tgroup\tinternal_seconds\texternal_seconds\tkill_grace_seconds\toutput_dir\n' \
  > "$EP_OFFER_TASK_ROOT/components.tsv"
printf 'group\tstarted_utc\tfinished_utc\texit_status\n' > "$EP_OFFER_TASK_ROOT/exits.tsv"
printf 'SELECTED_FIXED_SCOPE root=%s groups=%s internal_seconds=1200 maximum_sweeps=500\n' \
  "$EP_OFFER_ROOT" "${EP_OFFER_GROUPS[*]}"

EP_OFFER_EXIT=0
for EP_OFFER_GROUP in "${EP_OFFER_GROUPS[@]}"; do
  EP_OFFER_OUTPUT="$EP_OFFER_ROOT/$EP_OFFER_GROUP"
  printf '%s\t%s\t%s\t1200\t1220\t20\t%s\n' \
    "$SLURM_ARRAY_JOB_ID" "$SLURM_ARRAY_TASK_ID" "$EP_OFFER_GROUP" "$EP_OFFER_OUTPUT" \
    >> "$EP_OFFER_TASK_ROOT/components.tsv"
  EP_OFFER_STARTED=$(date -u +%FT%TZ)
  printf 'START group=%s time=%s output=%s\n' "$EP_OFFER_GROUP" "$EP_OFFER_STARTED" "$EP_OFFER_OUTPUT"
  if srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=8G \
      timeout --signal=TERM --kill-after=20s 1220s "$EP_OFFER_PYTHON" -B \
      outputs/code/run_mz0_offer_sequence_refit.py \
      --omitted-group "$EP_OFFER_GROUP" \
      --source-dir outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-SESOI-replicate000-attempt001 \
      --implementation outputs/code/behavioral_implementation.json \
      --output-dir "$EP_OFFER_OUTPUT" --max-seconds 1200 \
      > "$EP_OFFER_TASK_ROOT/$EP_OFFER_GROUP.stdout" \
      2> "$EP_OFFER_TASK_ROOT/$EP_OFFER_GROUP.stderr"; then
    EP_OFFER_STATUS=0
  else
    EP_OFFER_STATUS=$?
    EP_OFFER_EXIT=1
  fi
  EP_OFFER_FINISHED=$(date -u +%FT%TZ)
  printf '%s\t%s\t%s\t%s\n' "$EP_OFFER_GROUP" "$EP_OFFER_STARTED" "$EP_OFFER_FINISHED" "$EP_OFFER_STATUS" \
    >> "$EP_OFFER_TASK_ROOT/exits.tsv"
  printf 'FINISH group=%s time=%s exit_status=%s\n' "$EP_OFFER_GROUP" "$EP_OFFER_FINISHED" "$EP_OFFER_STATUS"
done
exit "$EP_OFFER_EXIT"
