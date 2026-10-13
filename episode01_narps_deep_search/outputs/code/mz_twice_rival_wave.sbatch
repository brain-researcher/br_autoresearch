#!/bin/bash -l
#SBATCH --job-name=ep01-twice-rival
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --no-requeue

# PREPARATION ONLY: root selects each model's packing, cap, walltime and array
# concurrency after the completed pilot. No time or array selection is implicit.
# Required submission options: --time matching PARENT_SECONDS, --array=1-N
# (with root-selected concurrency), and fresh --output/--error destinations.
# Arguments: MODEL PACKING INTERNAL_SECONDS PARENT_SECONDS
# N=ceil(199/PACKING). Each task owns its one contiguous pack, clipped at199.
# No tests, retries, replacement workers, follow-on submissions or polling here.
# Every component has external INTERNAL_SECONDS+20 plus20 seconds kill grace.
# Reserve120 seconds for parent setup, dispatch and status-file overhead.
set -euo pipefail

if [[ "$#" != 4 ]]; then
  printf 'Usage: MODEL PACKING INTERNAL_SECONDS PARENT_SECONDS; all selections explicit\n' >&2
  exit 2
fi
EP_TWICE_MODEL=$1
EP_TWICE_PACK=$2
EP_TWICE_INTERNAL=$3
EP_TWICE_PARENT=$4
case "$EP_TWICE_MODEL" in
  MC|MV|MVZ|MVC) ;;
  *) printf 'Select exactly MC, MV, MVZ or MVC\n' >&2; exit 2 ;;
esac
for EP_TWICE_VALUE in "$EP_TWICE_PACK" "$EP_TWICE_INTERNAL" "$EP_TWICE_PARENT"; do
  if [[ ! "$EP_TWICE_VALUE" =~ ^[1-9][0-9]{0,8}$ ]]; then
    printf 'Packing and bounds must be explicit positive decimal integers\n' >&2
    exit 2
  fi
done
if (( EP_TWICE_PACK > 199 || EP_TWICE_INTERNAL > 5400 )); then
  printf 'Packing must be1..199 and the unchanged component cap1..5400 seconds\n' >&2
  exit 2
fi
EP_TWICE_EXTERNAL=$((EP_TWICE_INTERNAL + 20))
EP_TWICE_ENVELOPE=$((EP_TWICE_EXTERNAL + 20))
EP_TWICE_SETUP=120
EP_TWICE_MAX_PACK_SECONDS=$((EP_TWICE_PACK * EP_TWICE_ENVELOPE + EP_TWICE_SETUP))
if (( EP_TWICE_MAX_PACK_SECONDS > EP_TWICE_PARENT )); then
  printf 'Selected pack requires%s seconds including setup; declared parent is%s\n' \
    "$EP_TWICE_MAX_PACK_SECONDS" "$EP_TWICE_PARENT" >&2
  exit 2
fi

EP_TWICE_HOST=$(hostname)
EP_TWICE_ENTERED=$(date -u +%s)
EP_TWICE_TASKS=$(((199 + EP_TWICE_PACK - 1) / EP_TWICE_PACK))
printf 'HOST=%s JOB=%s ARRAY=%s TASK=%s CPUS=%s MODEL=%s PACKING=%s INTERNAL=%s PARENT=%s\n' \
  "$EP_TWICE_HOST" "${SLURM_JOB_ID-}" "${SLURM_ARRAY_JOB_ID-}" "${SLURM_ARRAY_TASK_ID-}" \
  "${SLURM_CPUS_PER_TASK-}" "$EP_TWICE_MODEL" "$EP_TWICE_PACK" "$EP_TWICE_INTERNAL" "$EP_TWICE_PARENT"
if [[ -z "${SLURM_JOB_ID-}" || "$EP_TWICE_HOST" == *-ln* || \
      ! "${SLURM_ARRAY_JOB_ID-}" =~ ^[1-9][0-9]*$ || \
      ! "${SLURM_ARRAY_TASK_ID-}" =~ ^[1-9][0-9]*$ || \
      "${SLURM_NTASKS-}" != 1 || "${SLURM_CPUS_PER_TASK-}" != 4 || \
      "${SLURM_ARRAY_TASK_MIN-}" != 1 || "${SLURM_ARRAY_TASK_MAX-}" != "$EP_TWICE_TASKS" || \
      "${SLURM_ARRAY_TASK_COUNT-}" != "$EP_TWICE_TASKS" ]]; then
  printf 'Require an actual four-CPU compute task in the complete selected array1..%s\n' "$EP_TWICE_TASKS" >&2
  exit 2
fi
if (( SLURM_ARRAY_TASK_ID > EP_TWICE_TASKS )); then
  printf 'Array index exceeds the fixed199-cohort packing\n' >&2
  exit 2
fi
EP_TWICE_FIRST=$(((SLURM_ARRAY_TASK_ID - 1) * EP_TWICE_PACK + 1))
EP_TWICE_LAST=$((EP_TWICE_FIRST + EP_TWICE_PACK - 1))
if (( EP_TWICE_LAST > 199 )); then EP_TWICE_LAST=199; fi

# Root verifies the explicit requested parent time before dispatch. Slurm's
# optional epochs are observations, not an equality gate. A valid end epoch
# may only shorten the declared envelope, never extend it.
# BEGIN OPTIONAL_PARENT_DEADLINE
EP_TWICE_DECLARED_DEADLINE=$((EP_TWICE_ENTERED + EP_TWICE_PARENT))
EP_TWICE_DEADLINE=$EP_TWICE_DECLARED_DEADLINE
EP_TWICE_TIME_EVIDENCE=coordinator_verified_declared_parent
EP_TWICE_OBSERVED_START=${SLURM_JOB_START_TIME-unavailable}
EP_TWICE_OBSERVED_END=${SLURM_JOB_END_TIME-unavailable}
EP_TWICE_OBSERVED_PARENT=unavailable
if [[ "${SLURM_JOB_START_TIME-}" =~ ^[1-9][0-9]{0,10}$ && \
      "${SLURM_JOB_END_TIME-}" =~ ^[1-9][0-9]{0,10}$ ]] && \
      (( SLURM_JOB_END_TIME > SLURM_JOB_START_TIME )); then
  EP_TWICE_OBSERVED_PARENT=$((SLURM_JOB_END_TIME - SLURM_JOB_START_TIME))
fi
if [[ "${SLURM_JOB_END_TIME-}" =~ ^[1-9][0-9]{0,10}$ ]]; then
  EP_TWICE_TIME_EVIDENCE=declared_parent_with_slurm_end_observed
  if (( SLURM_JOB_END_TIME < EP_TWICE_DEADLINE )); then
    EP_TWICE_DEADLINE=$SLURM_JOB_END_TIME
    EP_TWICE_TIME_EVIDENCE=declared_parent_shortened_by_slurm_end
  fi
fi
# END OPTIONAL_PARENT_DEADLINE

cd "$SLURM_SUBMIT_DIR"
module purge
module load devel python/3.12.1 math py-numpy/1.26.3 py-scikit-learn/1.5.1_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="outputs/code${PYTHONPATH:+:$PYTHONPATH}"
EP_TWICE_PYTHON=$(command -v python3.12)
EP_TWICE_ROOT="outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-twice-menu-${EP_TWICE_MODEL}-wave-${SLURM_ARRAY_JOB_ID}"
EP_TWICE_TASK_ROOT="$EP_TWICE_ROOT/task${SLURM_ARRAY_TASK_ID}"
mkdir -p "$EP_TWICE_ROOT"
mkdir "$EP_TWICE_TASK_ROOT"
printf 'model\tarray_job_id\tarray_task_id\tpacking\tfirst_index\tlast_index\tinternal_seconds\texternal_seconds\tkill_grace_seconds\tsetup_seconds\tparent_seconds\ttime_evidence\tentered_epoch\tdeclared_deadline_epoch\tslurm_start_epoch_observed\tslurm_end_epoch_observed\tslurm_parent_seconds_observed\teffective_deadline_epoch\n' \
  > "$EP_TWICE_TASK_ROOT/selection.tsv"
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t20\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "$EP_TWICE_MODEL" "$SLURM_ARRAY_JOB_ID" "$SLURM_ARRAY_TASK_ID" "$EP_TWICE_PACK" \
  "$EP_TWICE_FIRST" "$EP_TWICE_LAST" "$EP_TWICE_INTERNAL" "$EP_TWICE_EXTERNAL" \
  "$EP_TWICE_SETUP" "$EP_TWICE_PARENT" "$EP_TWICE_TIME_EVIDENCE" "$EP_TWICE_ENTERED" \
  "$EP_TWICE_DECLARED_DEADLINE" "$EP_TWICE_OBSERVED_START" "$EP_TWICE_OBSERVED_END" \
  "$EP_TWICE_OBSERVED_PARENT" "$EP_TWICE_DEADLINE" >> "$EP_TWICE_TASK_ROOT/selection.tsv"
printf 'replicate_index\tstarted_utc\tfinished_utc\texit_status\tstate\toutput_dir\n' > "$EP_TWICE_TASK_ROOT/exits.tsv"

EP_TWICE_EXIT=0
for ((EP_TWICE_INDEX=EP_TWICE_FIRST; EP_TWICE_INDEX<=EP_TWICE_LAST; EP_TWICE_INDEX++)); do
  EP_TWICE_OUTPUT="$EP_TWICE_ROOT/replicate${EP_TWICE_INDEX}"
  EP_TWICE_STARTED=$(date -u +%FT%TZ)
  EP_TWICE_NOW=$(date -u +%s)
  EP_TWICE_STATE=attempted
  if (( EP_TWICE_DEADLINE - EP_TWICE_NOW < EP_TWICE_ENVELOPE )); then
    # Do not silently shorten a selected scientific cap or start work that
    # cannot receive its external termination allowance. Missing report stays
    # unknown; record this unstarted component and continue finite accounting.
    EP_TWICE_STATUS=125
    EP_TWICE_STATE=not_started_insufficient_parent_time
    EP_TWICE_EXIT=1
  elif srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=8G \
      timeout --signal=TERM --kill-after=20s "${EP_TWICE_EXTERNAL}s" "$EP_TWICE_PYTHON" -B \
      outputs/code/run_twice_mz_rival_component.py \
      --model "$EP_TWICE_MODEL" --replicate-index "$EP_TWICE_INDEX" \
      --implementation outputs/code/behavioral_implementation.json \
      --output-dir "$EP_TWICE_OUTPUT" --max-seconds "$EP_TWICE_INTERNAL" \
      > "$EP_TWICE_TASK_ROOT/replicate${EP_TWICE_INDEX}.stdout" \
      2> "$EP_TWICE_TASK_ROOT/replicate${EP_TWICE_INDEX}.stderr"; then
    EP_TWICE_STATUS=0
  else
    EP_TWICE_STATUS=$?
    EP_TWICE_EXIT=1
  fi
  EP_TWICE_FINISHED=$(date -u +%FT%TZ)
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$EP_TWICE_INDEX" "$EP_TWICE_STARTED" \
    "$EP_TWICE_FINISHED" "$EP_TWICE_STATUS" "$EP_TWICE_STATE" "$EP_TWICE_OUTPUT" >> "$EP_TWICE_TASK_ROOT/exits.tsv"
  printf 'MODEL=%s INDEX=%s STATE=%s EXIT=%s FINISHED=%s\n' \
    "$EP_TWICE_MODEL" "$EP_TWICE_INDEX" "$EP_TWICE_STATE" "$EP_TWICE_STATUS" "$EP_TWICE_FINISHED"
done
exit "$EP_TWICE_EXIT"
