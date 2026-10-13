#!/bin/bash -l
#SBATCH --job-name=ep01-null-simple-wave
#SBATCH --partition=normal
#SBATCH --constraint=CPU_SKU:7502
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=03:00:00
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/behavioral_mechanism_kill_test/20261005-M0-null-simple-group-%A_%a.out
#SBATCH --error=outputs/runs/behavioral_mechanism_kill_test/20261005-M0-null-simple-group-%A_%a.err

# Explicit mode: MC_MV, MC or MV; optional qualification group1(default) or8.
# Submit the single-component entry point
# for MC/MV alone so its allocation is95m. Root alone chooses models, arrays,
# source dependencies and concurrency; this file never submits or polls.
# Group1 uses baseline46579924; groups2–50 use46580387. Root establishes
# terminal sources with dependencies and observes qualification-group checks/replay before
# selecting further groups. Do not combine MC_MV with prior MC/MV selections
# for the same indices: use the remaining single component instead.
# Root may select initial --array=8 --dependency=afterany:46580387_8 with
# arguments MC_MV 8, then explicitly select other groups with the same
# qualification-group argument after observing checks and actual replay.
# Four cohorts/group, except group50=197–199; two sequential pairs of1CPU
# workers. Each worker attempts every selected model once, even after failure.
# Worst-case work including timeout graces: combined9880s<10800s;
# single5060s<5700s. Source/replay/fit failures remain technical unknowns.
set -euo pipefail
EP_NULL_SIMPLE_HOST=$(hostname)
printf 'HOST=%s SLURM_JOB_ID=%s NTASKS=%s CPUS_PER_TASK=%s\n' \
  "$EP_NULL_SIMPLE_HOST" "${SLURM_JOB_ID-}" "${SLURM_NTASKS-}" "${SLURM_CPUS_PER_TASK-}"
if [[ -z "${SLURM_JOB_ID-}" || "$EP_NULL_SIMPLE_HOST" == *-ln* || "${SLURM_CPUS_PER_TASK-}" != 1 ]]; then
  printf 'An actual compute allocation with one CPU per task is required\n' >&2
  exit 2
fi
cd "$SLURM_SUBMIT_DIR"

if [[ "${1-}" == --worker ]]; then
  if (( $# != 3 )); then printf 'Worker requires one index and explicit model mode\n' >&2; exit 2; fi
  EP_NULL_SIMPLE_INDEX="$2"
  EP_NULL_SIMPLE_MODE="$3"
  case "$EP_NULL_SIMPLE_INDEX" in
    ''|*[!0-9]*) printf 'Expected numeric cohort index\n' >&2; exit 2 ;;
  esac
  if (( EP_NULL_SIMPLE_INDEX < 1 || EP_NULL_SIMPLE_INDEX > 199 )) || [[ "${SLURM_NTASKS-}" != 1 ]]; then
    printf 'Worker requires index1–199 and exactly one Slurm task\n' >&2
    exit 2
  fi
  case "$EP_NULL_SIMPLE_MODE" in
    MC_MV) EP_NULL_SIMPLE_MODELS=(MC MV) ;;
    MC) EP_NULL_SIMPLE_MODELS=(MC) ;;
    MV) EP_NULL_SIMPLE_MODELS=(MV) ;;
    *) printf 'Expected MC_MV, MC or MV\n' >&2; exit 2 ;;
  esac
  if (( EP_NULL_SIMPLE_INDEX <= 4 )); then
    EP_NULL_SIMPLE_SOURCE_ARRAY=46579924
  else
    EP_NULL_SIMPLE_SOURCE_ARRAY=46580387
  fi
  EP_NULL_SIMPLE_WORKER_EXIT=0
  for EP_NULL_SIMPLE_MODEL in "${EP_NULL_SIMPLE_MODELS[@]}"; do
    if timeout --signal=TERM --kill-after=10s 2400s "$EP_NULL_SIMPLE_PYTHON" -B \
        outputs/code/run_mz_null_indexed_simple_rival.py \
        --model "$EP_NULL_SIMPLE_MODEL" --replicate-index "$EP_NULL_SIMPLE_INDEX" \
        --source-dir "outputs/runs/behavioral_mechanism_kill_test/20261005-M0-null-baseline-array-${EP_NULL_SIMPLE_SOURCE_ARRAY}/replicate${EP_NULL_SIMPLE_INDEX}" \
        --implementation outputs/code/behavioral_implementation.json \
        --output-dir "outputs/runs/behavioral_mechanism_kill_test/20261005-M0-null-${EP_NULL_SIMPLE_MODEL}-indexed-array-${SLURM_ARRAY_JOB_ID}/replicate${EP_NULL_SIMPLE_INDEX}" \
        --max-seconds 2400 \
        > "outputs/runs/behavioral_mechanism_kill_test/20261005-M0-null-${EP_NULL_SIMPLE_MODEL}-indexed-${SLURM_ARRAY_JOB_ID}-${EP_NULL_SIMPLE_INDEX}.out" \
        2> "outputs/runs/behavioral_mechanism_kill_test/20261005-M0-null-${EP_NULL_SIMPLE_MODEL}-indexed-${SLURM_ARRAY_JOB_ID}-${EP_NULL_SIMPLE_INDEX}.err"; then
      printf 'NULL_INDEXED_SIMPLE_COMPLETED model=%s index=%s\n' "$EP_NULL_SIMPLE_MODEL" "$EP_NULL_SIMPLE_INDEX"
    else
      EP_NULL_SIMPLE_WORKER_EXIT=1
      printf 'NULL_INDEXED_SIMPLE_FAILED model=%s index=%s: preserve attempt; next selected model remains independent; no retry\n' \
        "$EP_NULL_SIMPLE_MODEL" "$EP_NULL_SIMPLE_INDEX" >&2
    fi
  done
  exit "$EP_NULL_SIMPLE_WORKER_EXIT"
fi

if (( $# < 1 || $# > 2 )) || [[ "${SLURM_NTASKS-}" != 2 ]]; then
  printf 'Grouped entry requires two Slurm tasks, explicit mode and optional qualification group\n' >&2
  exit 2
fi
EP_NULL_SIMPLE_MODE="$1"
case "$EP_NULL_SIMPLE_MODE" in
  MC_MV|MC|MV) ;;
  *) printf 'Expected MC_MV, MC or MV\n' >&2; exit 2 ;;
esac
EP_NULL_SIMPLE_QUALIFICATION_GROUP="${2-1}"
case "$EP_NULL_SIMPLE_QUALIFICATION_GROUP" in
  1|8) ;;
  *) printf 'Qualification group must be explicitly1 or8\n' >&2; exit 2 ;;
esac
case "${SLURM_ARRAY_TASK_ID-}" in
  ''|*[!0-9]*) printf 'Expected numeric group index\n' >&2; exit 2 ;;
esac
if (( SLURM_ARRAY_TASK_ID < 1 || SLURM_ARRAY_TASK_ID > 50 )); then
  printf 'Expected group index1 through50\n' >&2
  exit 2
fi
module purge
module load devel python/3.12.1 math py-numpy/1.26.3 py-scikit-learn/1.5.1_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
EP_NULL_SIMPLE_PYTHON=$(command -v python3.12)
export EP_NULL_SIMPLE_PYTHON
EP_NULL_SIMPLE_FIRST=$((4 * (SLURM_ARRAY_TASK_ID - 1) + 1))
EP_NULL_SIMPLE_LAST=$((EP_NULL_SIMPLE_FIRST + 3))
if (( EP_NULL_SIMPLE_LAST > 199 )); then EP_NULL_SIMPLE_LAST=199; fi
printf 'ARRAY=%s GROUP=%s QUALIFICATION_GROUP=%s MODE=%s INDICES=%s-%s\n' \
  "$SLURM_ARRAY_JOB_ID" "$SLURM_ARRAY_TASK_ID" "$EP_NULL_SIMPLE_QUALIFICATION_GROUP" \
  "$EP_NULL_SIMPLE_MODE" "$EP_NULL_SIMPLE_FIRST" "$EP_NULL_SIMPLE_LAST"
"$EP_NULL_SIMPLE_PYTHON" --version

if (( SLURM_ARRAY_TASK_ID == EP_NULL_SIMPLE_QUALIFICATION_GROUP )); then
  bash -n outputs/code/mz_null_indexed_simple_groups.sbatch
  bash -n outputs/code/mz_null_indexed_simple_single_groups.sbatch
  EP_NULL_SIMPLE_CHECKS="outputs/runs/behavioral_mechanism_kill_test/20261005-M0-null-indexed-${EP_NULL_SIMPLE_MODE}-checks-${SLURM_ARRAY_JOB_ID}"
  mkdir "$EP_NULL_SIMPLE_CHECKS"
  timeout --signal=TERM --kill-after=10s 60s "$EP_NULL_SIMPLE_PYTHON" -B \
    -m unittest discover -s outputs/code -p test_mz_null_baseline_wave.py -v \
    > "$EP_NULL_SIMPLE_CHECKS/baseline_pool_tests.txt" 2>&1
  timeout --signal=TERM --kill-after=10s 120s "$EP_NULL_SIMPLE_PYTHON" -B \
    -m unittest discover -s outputs/code -p test_mz_null_indexed_simple_rival.py -v \
    > "$EP_NULL_SIMPLE_CHECKS/tests.txt" 2>&1
  printf 'NULL_INDEXED_SIMPLE_TESTS_PASS mode=%s\n' "$EP_NULL_SIMPLE_MODE"
  EP_NULL_SIMPLE_ASTRA="$GROUP_HOME/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv/bin/astra-tools"
  if timeout --signal=TERM --kill-after=10s 30s "$EP_NULL_SIMPLE_ASTRA" validate \
      outputs/astra/v0.0.14/astra.yaml --json \
      > "outputs/astra/v0.0.14/validation_20261005_null_indexed_${EP_NULL_SIMPLE_MODE}_${SLURM_ARRAY_JOB_ID}.json"; then
    printf 'NULL_INDEXED_SIMPLE_ASTRA_PASS mode=%s\n' "$EP_NULL_SIMPLE_MODE"
  else
    printf 'NULL_INDEXED_SIMPLE_ASTRA_OPEN: coordinator must inspect validation output\n' >&2
  fi
fi

EP_NULL_SIMPLE_GROUP_EXIT=0
for ((EP_NULL_SIMPLE_PAIR=EP_NULL_SIMPLE_FIRST; EP_NULL_SIMPLE_PAIR<=EP_NULL_SIMPLE_LAST; EP_NULL_SIMPLE_PAIR+=2)); do
  EP_NULL_SIMPLE_PIDS=()
  EP_NULL_SIMPLE_INDICES=()
  for ((EP_NULL_SIMPLE_INDEX=EP_NULL_SIMPLE_PAIR; EP_NULL_SIMPLE_INDEX<=EP_NULL_SIMPLE_PAIR+1 && EP_NULL_SIMPLE_INDEX<=EP_NULL_SIMPLE_LAST; EP_NULL_SIMPLE_INDEX++)); do
    srun --nodes=1 --ntasks=1 --cpus-per-task=1 --exclusive --exact --mem=2G --cpu-bind=cores \
      --output="outputs/runs/behavioral_mechanism_kill_test/20261005-M0-null-${EP_NULL_SIMPLE_MODE}-worker-${SLURM_ARRAY_JOB_ID}-${EP_NULL_SIMPLE_INDEX}.out" \
      --error="outputs/runs/behavioral_mechanism_kill_test/20261005-M0-null-${EP_NULL_SIMPLE_MODE}-worker-${SLURM_ARRAY_JOB_ID}-${EP_NULL_SIMPLE_INDEX}.err" \
      bash outputs/code/mz_null_indexed_simple_groups.sbatch --worker "$EP_NULL_SIMPLE_INDEX" "$EP_NULL_SIMPLE_MODE" &
    EP_NULL_SIMPLE_PIDS+=("$!")
    EP_NULL_SIMPLE_INDICES+=("$EP_NULL_SIMPLE_INDEX")
  done
  for EP_NULL_SIMPLE_SLOT in "${!EP_NULL_SIMPLE_PIDS[@]}"; do
    if wait "${EP_NULL_SIMPLE_PIDS[$EP_NULL_SIMPLE_SLOT]}"; then
      printf 'NULL_INDEXED_SIMPLE_WORKER_COMPLETED mode=%s index=%s\n' \
        "$EP_NULL_SIMPLE_MODE" "${EP_NULL_SIMPLE_INDICES[$EP_NULL_SIMPLE_SLOT]}"
    else
      EP_NULL_SIMPLE_GROUP_EXIT=1
      printf 'NULL_INDEXED_SIMPLE_WORKER_FAILED mode=%s index=%s: preserve attempts; no retry\n' \
        "$EP_NULL_SIMPLE_MODE" "${EP_NULL_SIMPLE_INDICES[$EP_NULL_SIMPLE_SLOT]}" >&2
    fi
  done
done
exit "$EP_NULL_SIMPLE_GROUP_EXIT"
