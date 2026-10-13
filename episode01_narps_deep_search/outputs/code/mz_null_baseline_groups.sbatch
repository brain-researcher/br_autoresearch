#!/bin/bash -l
#SBATCH --job-name=ep01-null-baselines
#SBATCH --partition=normal
#SBATCH --constraint=CPU_SKU:7502
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:35:00
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/behavioral_mechanism_kill_test/20261005-M0-null-baseline-group-%A_%a.out
#SBATCH --error=outputs/runs/behavioral_mechanism_kill_test/20261005-M0-null-baseline-group-%A_%a.err

# Root submits group1 first, then groups2–50 only after its changed checks pass.
# Four preassigned cohorts/group (three in group50), two sequential pairs.
# This finite worker schedule does not submit, poll, retry or renew jobs.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
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
EP_NULL_PYTHON=$(command -v python3.12)
EP_NULL_FIRST=$((4 * (SLURM_ARRAY_TASK_ID - 1) + 1))
EP_NULL_LAST=$((EP_NULL_FIRST + 3))
if (( EP_NULL_LAST > 199 )); then EP_NULL_LAST=199; fi
hostname
printf 'SLURM_JOB_ID=%s ARRAY=%s GROUP=%s NTASKS=%s CPUS_PER_TASK=%s INDICES=%s-%s\n' \
  "$SLURM_JOB_ID" "$SLURM_ARRAY_JOB_ID" "$SLURM_ARRAY_TASK_ID" "$SLURM_NTASKS" \
  "$SLURM_CPUS_PER_TASK" "$EP_NULL_FIRST" "$EP_NULL_LAST"
"$EP_NULL_PYTHON" --version

if (( SLURM_ARRAY_TASK_ID == 1 )); then
  bash -n outputs/code/mz_null_baseline_groups.sbatch
  EP_NULL_CHECKS="outputs/runs/behavioral_mechanism_kill_test/20261005-M0-null-indexed-checks-${SLURM_ARRAY_JOB_ID}"
  mkdir "$EP_NULL_CHECKS"
  timeout --signal=TERM --kill-after=10s 120s "$EP_NULL_PYTHON" -B \
    outputs/code/test_mz_null_replicate.py -v > "$EP_NULL_CHECKS/tests.txt" 2>&1
  printf 'NULL_INDEXED_TESTS_PASS\n'
  EP_NULL_ASTRA="$GROUP_HOME/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv/bin/astra-tools"
  if timeout --signal=TERM --kill-after=10s 30s "$EP_NULL_ASTRA" validate \
      outputs/astra/v0.0.14/astra.yaml --json \
      > "outputs/astra/v0.0.14/validation_20261005_null_indexed_${SLURM_ARRAY_JOB_ID}.json"; then
    printf 'NULL_INDEXED_ASTRA_PASS\n'
  else
    printf 'NULL_INDEXED_ASTRA_OPEN: coordinator must inspect validation output\n' >&2
  fi
fi

EP_NULL_GROUP_EXIT=0
for ((EP_NULL_PAIR=EP_NULL_FIRST; EP_NULL_PAIR<=EP_NULL_LAST; EP_NULL_PAIR+=2)); do
  EP_NULL_PIDS=()
  EP_NULL_INDICES=()
  for ((EP_NULL_INDEX=EP_NULL_PAIR; EP_NULL_INDEX<=EP_NULL_PAIR+1 && EP_NULL_INDEX<=EP_NULL_LAST; EP_NULL_INDEX++)); do
    srun --nodes=1 --ntasks=1 --cpus-per-task=1 --exclusive --exact --mem=2G --cpu-bind=cores \
      --output="outputs/runs/behavioral_mechanism_kill_test/20261005-M0-null-baseline-${SLURM_ARRAY_JOB_ID}-${EP_NULL_INDEX}.out" \
      --error="outputs/runs/behavioral_mechanism_kill_test/20261005-M0-null-baseline-${SLURM_ARRAY_JOB_ID}-${EP_NULL_INDEX}.err" \
      timeout --signal=TERM --kill-after=10s 2700s "$EP_NULL_PYTHON" -B \
      outputs/code/run_mz_null_replicate.py --replicate-index "$EP_NULL_INDEX" \
      --design-events outputs/runs/behavioral_mechanism_kill_test/20261003-MC-SESOI-replicate000-attempt003/simulated_events.csv.gz \
      --implementation outputs/code/behavioral_implementation.json \
      --output-dir "outputs/runs/behavioral_mechanism_kill_test/20261005-M0-null-baseline-array-${SLURM_ARRAY_JOB_ID}/replicate${EP_NULL_INDEX}" \
      --max-seconds 2700 &
    EP_NULL_PIDS+=("$!")
    EP_NULL_INDICES+=("$EP_NULL_INDEX")
  done
  for EP_NULL_SLOT in "${!EP_NULL_PIDS[@]}"; do
    if wait "${EP_NULL_PIDS[$EP_NULL_SLOT]}"; then
      printf 'NULL_BASELINE_COMPLETED index=%s\n' "${EP_NULL_INDICES[$EP_NULL_SLOT]}"
    else
      EP_NULL_GROUP_EXIT=1
      printf 'NULL_BASELINE_FAILED index=%s: preserve attempt; no retry\n' "${EP_NULL_INDICES[$EP_NULL_SLOT]}" >&2
    fi
  done
done
exit "$EP_NULL_GROUP_EXIT"
