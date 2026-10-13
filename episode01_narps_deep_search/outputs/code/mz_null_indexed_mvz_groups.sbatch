#!/bin/bash -l
#SBATCH --job-name=ep01-null-mvz-wave
#SBATCH --partition=normal
#SBATCH --constraint=CPU_SKU:7502
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/behavioral_mechanism_kill_test/20261005-M0-null-MVZ-group-%A_%a.out
#SBATCH --error=outputs/runs/behavioral_mechanism_kill_test/20261005-M0-null-MVZ-group-%A_%a.err

# Root submits group1 afterany:46579924, then groups2–50
# afterany:46579924:46580387
# only once the changed checks and actual source replay in group1 pass.
# Four saved cohorts/group (three in group50), two sequential pairs.
# No simulation, baseline refit, polling, job submission or retry occurs here.
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
EP_NULL_MVZ_PYTHON=$(command -v python3.12)
EP_NULL_MVZ_FIRST=$((4 * (SLURM_ARRAY_TASK_ID - 1) + 1))
EP_NULL_MVZ_LAST=$((EP_NULL_MVZ_FIRST + 3))
if (( EP_NULL_MVZ_LAST > 199 )); then EP_NULL_MVZ_LAST=199; fi
if (( SLURM_ARRAY_TASK_ID == 1 )); then
  EP_NULL_MVZ_BASE_ARRAY=46579924
else
  EP_NULL_MVZ_BASE_ARRAY=46580387
fi
hostname
printf 'SLURM_JOB_ID=%s ARRAY=%s GROUP=%s NTASKS=%s CPUS_PER_TASK=%s INDICES=%s-%s SOURCE_ARRAY=%s\n' \
  "$SLURM_JOB_ID" "$SLURM_ARRAY_JOB_ID" "$SLURM_ARRAY_TASK_ID" "$SLURM_NTASKS" \
  "$SLURM_CPUS_PER_TASK" "$EP_NULL_MVZ_FIRST" "$EP_NULL_MVZ_LAST" "$EP_NULL_MVZ_BASE_ARRAY"
"$EP_NULL_MVZ_PYTHON" --version

if (( SLURM_ARRAY_TASK_ID == 1 )); then
  bash -n outputs/code/mz_null_indexed_mvz_groups.sbatch
  EP_NULL_MVZ_CHECKS="outputs/runs/behavioral_mechanism_kill_test/20261005-M0-null-indexed-MVZ-checks-${SLURM_ARRAY_JOB_ID}"
  mkdir "$EP_NULL_MVZ_CHECKS"
  timeout --signal=TERM --kill-after=10s 60s "$EP_NULL_MVZ_PYTHON" -B \
    -m unittest discover -s outputs/code -p test_mz_null_baseline_wave.py -v \
    > "$EP_NULL_MVZ_CHECKS/baseline_pool_tests.txt" 2>&1
  timeout --signal=TERM --kill-after=10s 120s "$EP_NULL_MVZ_PYTHON" -B \
    -m unittest discover -s outputs/code -p test_mz_null_indexed_rival.py -v \
    > "$EP_NULL_MVZ_CHECKS/indexed_rival_tests.txt" 2>&1
  printf 'NULL_INDEXED_MVZ_TESTS_PASS\n'
  EP_NULL_MVZ_ASTRA="$GROUP_HOME/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv/bin/astra-tools"
  if timeout --signal=TERM --kill-after=10s 30s "$EP_NULL_MVZ_ASTRA" validate \
      outputs/astra/v0.0.14/astra.yaml --json \
      > "outputs/astra/v0.0.14/validation_20261005_null_indexed_mvz_${SLURM_ARRAY_JOB_ID}.json"; then
    printf 'NULL_INDEXED_MVZ_ASTRA_PASS\n'
  else
    printf 'NULL_INDEXED_MVZ_ASTRA_OPEN: coordinator must inspect validation output\n' >&2
  fi
fi

EP_NULL_MVZ_GROUP_EXIT=0
if (( SLURM_ARRAY_TASK_ID == 2 )); then
  # Root's remaining-array dependency establishes all saved baselines terminal.
  # This finite loop constructs one argument; it does not poll.
  EP_NULL_MVZ_TERMINAL_INDICES=0
  for ((EP_NULL_MVZ_BASE_INDEX=1; EP_NULL_MVZ_BASE_INDEX<=199; EP_NULL_MVZ_BASE_INDEX++)); do
    EP_NULL_MVZ_TERMINAL_INDICES+=",${EP_NULL_MVZ_BASE_INDEX}"
  done
  if timeout --signal=TERM --kill-after=10s 180s "$EP_NULL_MVZ_PYTHON" -B \
      outputs/code/summarize_mz_null_baseline_wave.py \
      --first-array-job-id 46579924 --remaining-array-job-id 46580387 \
      --terminal-indices "$EP_NULL_MVZ_TERMINAL_INDICES" \
      --implementation outputs/code/behavioral_implementation.json \
      --output "outputs/runs/behavioral_mechanism_kill_test/20261005-M0-null-baseline-wave-${SLURM_ARRAY_JOB_ID}.json"; then
    printf 'NULL_BASELINE_WAVE_POOL_COMPLETED\n'
  else
    EP_NULL_MVZ_GROUP_EXIT=1
    printf 'NULL_BASELINE_WAVE_POOL_INCOMPLETE: preserve error; independent MVZ source validation continues\n' >&2
  fi
fi
for ((EP_NULL_MVZ_PAIR=EP_NULL_MVZ_FIRST; EP_NULL_MVZ_PAIR<=EP_NULL_MVZ_LAST; EP_NULL_MVZ_PAIR+=2)); do
  EP_NULL_MVZ_PIDS=()
  EP_NULL_MVZ_INDICES=()
  for ((EP_NULL_MVZ_INDEX=EP_NULL_MVZ_PAIR; EP_NULL_MVZ_INDEX<=EP_NULL_MVZ_PAIR+1 && EP_NULL_MVZ_INDEX<=EP_NULL_MVZ_LAST; EP_NULL_MVZ_INDEX++)); do
    srun --nodes=1 --ntasks=1 --cpus-per-task=1 --exclusive --exact --mem=2G --cpu-bind=cores \
      --output="outputs/runs/behavioral_mechanism_kill_test/20261005-M0-null-MVZ-${SLURM_ARRAY_JOB_ID}-${EP_NULL_MVZ_INDEX}.out" \
      --error="outputs/runs/behavioral_mechanism_kill_test/20261005-M0-null-MVZ-${SLURM_ARRAY_JOB_ID}-${EP_NULL_MVZ_INDEX}.err" \
      timeout --signal=TERM --kill-after=10s 3300s "$EP_NULL_MVZ_PYTHON" -B \
      outputs/code/run_mz_null_indexed_rival.py --model MVZ --replicate-index "$EP_NULL_MVZ_INDEX" \
      --source-dir "outputs/runs/behavioral_mechanism_kill_test/20261005-M0-null-baseline-array-${EP_NULL_MVZ_BASE_ARRAY}/replicate${EP_NULL_MVZ_INDEX}" \
      --implementation outputs/code/behavioral_implementation.json \
      --output-dir "outputs/runs/behavioral_mechanism_kill_test/20261005-M0-null-MVZ-indexed-array-${SLURM_ARRAY_JOB_ID}/replicate${EP_NULL_MVZ_INDEX}" \
      --max-seconds 3300 &
    EP_NULL_MVZ_PIDS+=("$!")
    EP_NULL_MVZ_INDICES+=("$EP_NULL_MVZ_INDEX")
  done
  for EP_NULL_MVZ_SLOT in "${!EP_NULL_MVZ_PIDS[@]}"; do
    if wait "${EP_NULL_MVZ_PIDS[$EP_NULL_MVZ_SLOT]}"; then
      printf 'NULL_INDEXED_MVZ_COMPLETED index=%s\n' "${EP_NULL_MVZ_INDICES[$EP_NULL_MVZ_SLOT]}"
    else
      EP_NULL_MVZ_GROUP_EXIT=1
      printf 'NULL_INDEXED_MVZ_FAILED index=%s: preserve source/attempt as unknown; no retry\n' "${EP_NULL_MVZ_INDICES[$EP_NULL_MVZ_SLOT]}" >&2
    fi
  done
done
exit "$EP_NULL_MVZ_GROUP_EXIT"
