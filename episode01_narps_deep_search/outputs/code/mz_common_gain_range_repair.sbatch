#!/bin/bash -l
#SBATCH --job-name=ep01-common-gain-repair
#SBATCH --partition=normal
#SBATCH --constraint=CPU_SKU:7502
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:30:00
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-common-gain-repair-%j.out
#SBATCH --error=outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-common-gain-repair-%j.err

# Checks250 + parallel-candidate checks130 + ASTRA40 + four simultaneous
# bounded4210s components + conditional pool130 =4760<5400s before setup.
# Scientific workers use only the original1CPU fitter: fresh M01500/MZ500.
set -euo pipefail
EP_GAIN_REPAIR_HOST=$(hostname)
printf 'HOST=%s SLURM_JOB_ID=%s NTASKS=%s CPUS_PER_TASK=%s\n' \
  "$EP_GAIN_REPAIR_HOST" "${SLURM_JOB_ID-}" "${SLURM_NTASKS-}" "${SLURM_CPUS_PER_TASK-}"
if [[ $# != 0 || -z "${SLURM_JOB_ID-}" || "$EP_GAIN_REPAIR_HOST" == *-ln* \
      || "${SLURM_NTASKS-}" != 4 || "${SLURM_CPUS_PER_TASK-}" != 1 ]]; then exit 2; fi
cd "$SLURM_SUBMIT_DIR"
module purge
module load devel python/3.12.1 math py-numpy/1.26.3 py-scikit-learn/1.5.1_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="outputs/code:outputs/tests${PYTHONPATH:+:$PYTHONPATH}"
EP_GAIN_REPAIR_PYTHON=$(command -v python3.12)
"$EP_GAIN_REPAIR_PYTHON" --version
bash -n outputs/code/mz_common_gain_range_repair.sbatch
EP_GAIN_REPAIR_ROOT="outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-common-gain-repair-${SLURM_JOB_ID}"
EP_GAIN_REPAIR_CHECKS="outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-common-gain-repair-checks-${SLURM_JOB_ID}"
mkdir "$EP_GAIN_REPAIR_ROOT" "$EP_GAIN_REPAIR_CHECKS"
mkdir "$EP_GAIN_REPAIR_ROOT/dir0" "$EP_GAIN_REPAIR_ROOT/dir1"
printf 'stage\texit_status\n' > "$EP_GAIN_REPAIR_CHECKS/stage_exit_status.tsv"

if srun --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=2G --exclusive --exact --cpu-bind=cores \
    timeout --signal=TERM --kill-after=10s 240s "$EP_GAIN_REPAIR_PYTHON" -B \
    -m unittest -v test_mz_common_gain_range_repair test_mz_common_gain_range_repaired_pool \
    > "$EP_GAIN_REPAIR_CHECKS/common_gain_repair_tests.txt" 2>&1; then
  EP_GAIN_REPAIR_STATUS=0
else
  EP_GAIN_REPAIR_STATUS=$?
fi
printf 'common_gain_repair_tests\t%s\n' "$EP_GAIN_REPAIR_STATUS" \
  >> "$EP_GAIN_REPAIR_CHECKS/stage_exit_status.tsv"
if [[ "$EP_GAIN_REPAIR_STATUS" != 0 ]]; then exit "$EP_GAIN_REPAIR_STATUS"; fi
printf 'COMMON_GAIN_REPAIR_TESTS_PASS\n'

# This isolated candidate is not used by the four scientific components.
# Retain a failed candidate check without blocking the independent repair.
if srun --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=4G --exclusive --exact --cpu-bind=cores \
    timeout --signal=TERM --kill-after=10s 120s "$EP_GAIN_REPAIR_PYTHON" -B \
    -m unittest discover -s outputs/tests -p test_hierarchical_behavior_parallel.py -v \
    > "$EP_GAIN_REPAIR_CHECKS/parallel_candidate_tests.txt" 2>&1; then
  EP_GAIN_REPAIR_STATUS=0
  printf 'PARALLEL_FITTER_CANDIDATE_TESTS_PASS\n'
else
  EP_GAIN_REPAIR_STATUS=$?
  printf 'PARALLEL_FITTER_CANDIDATE_TESTS_FAILED: original fitter remains selected\n' >&2
fi
printf 'parallel_candidate_tests\t%s\n' "$EP_GAIN_REPAIR_STATUS" \
  >> "$EP_GAIN_REPAIR_CHECKS/stage_exit_status.tsv"

EP_GAIN_REPAIR_ASTRA="$GROUP_HOME/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv/bin/astra-tools"
if srun --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=2G --exclusive --exact --cpu-bind=cores \
    timeout --signal=TERM --kill-after=10s 30s "$EP_GAIN_REPAIR_ASTRA" \
    validate outputs/astra/v0.0.14/astra.yaml --json \
    > "$EP_GAIN_REPAIR_CHECKS/astra_validation.json" 2> "$EP_GAIN_REPAIR_CHECKS/astra_validation.err"; then
  EP_GAIN_REPAIR_STATUS=0
  printf 'COMMON_GAIN_REPAIR_ASTRA_PASS\n'
else
  EP_GAIN_REPAIR_STATUS=$?
  printf 'COMMON_GAIN_REPAIR_ASTRA_OPEN: coordinator must inspect retained output\n' >&2
fi
printf 'astra_validation\t%s\n' "$EP_GAIN_REPAIR_STATUS" \
  >> "$EP_GAIN_REPAIR_CHECKS/stage_exit_status.tsv"

EP_GAIN_REPAIR_PIDS=()
EP_GAIN_REPAIR_DIRECTIONS=()
EP_GAIN_REPAIR_MODELS=()
for EP_GAIN_REPAIR_DIRECTION in 0 1; do
  for EP_GAIN_REPAIR_MODEL in M0 MZ; do
    srun --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=2G --exclusive --exact --cpu-bind=cores \
      timeout --signal=TERM --kill-after=10s 4200s "$EP_GAIN_REPAIR_PYTHON" -B \
      outputs/code/run_mz_common_gain_range_repair.py \
      --direction "$EP_GAIN_REPAIR_DIRECTION" --model "$EP_GAIN_REPAIR_MODEL" \
      --source-dir outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-SESOI-replicate000-attempt001 \
      --implementation outputs/code/behavioral_implementation.json \
      --output-dir "$EP_GAIN_REPAIR_ROOT/dir${EP_GAIN_REPAIR_DIRECTION}/${EP_GAIN_REPAIR_MODEL}" \
      --max-seconds 4200 \
      > "$EP_GAIN_REPAIR_ROOT/dir${EP_GAIN_REPAIR_DIRECTION}/${EP_GAIN_REPAIR_MODEL}.out" \
      2> "$EP_GAIN_REPAIR_ROOT/dir${EP_GAIN_REPAIR_DIRECTION}/${EP_GAIN_REPAIR_MODEL}.err" &
    EP_GAIN_REPAIR_PIDS+=("$!")
    EP_GAIN_REPAIR_DIRECTIONS+=("$EP_GAIN_REPAIR_DIRECTION")
    EP_GAIN_REPAIR_MODELS+=("$EP_GAIN_REPAIR_MODEL")
  done
done

EP_GAIN_REPAIR_FAILED=0
printf 'direction\tmodel\texit_status\n' > "$EP_GAIN_REPAIR_ROOT/component_exit_status.tsv"
for EP_GAIN_REPAIR_INDEX in {0..3}; do
  if wait "${EP_GAIN_REPAIR_PIDS[$EP_GAIN_REPAIR_INDEX]}"; then
    EP_GAIN_REPAIR_STATUS=0
  else
    EP_GAIN_REPAIR_STATUS=$?
    EP_GAIN_REPAIR_FAILED=1
  fi
  printf '%s\t%s\t%s\n' "${EP_GAIN_REPAIR_DIRECTIONS[$EP_GAIN_REPAIR_INDEX]}" \
    "${EP_GAIN_REPAIR_MODELS[$EP_GAIN_REPAIR_INDEX]}" "$EP_GAIN_REPAIR_STATUS" \
    >> "$EP_GAIN_REPAIR_ROOT/component_exit_status.tsv"
  printf 'COMMON_GAIN_DIRECTION=%s MODEL=%s EXIT_STATUS=%s\n' \
    "${EP_GAIN_REPAIR_DIRECTIONS[$EP_GAIN_REPAIR_INDEX]}" \
    "${EP_GAIN_REPAIR_MODELS[$EP_GAIN_REPAIR_INDEX]}" "$EP_GAIN_REPAIR_STATUS"
done
if [[ "$EP_GAIN_REPAIR_FAILED" != 0 ]]; then exit 1; fi

srun --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=2G --exclusive --exact --cpu-bind=cores \
  timeout --signal=TERM --kill-after=10s 120s "$EP_GAIN_REPAIR_PYTHON" -B \
  outputs/code/summarize_mz_common_gain_range_repaired.py \
  --source-root "$EP_GAIN_REPAIR_ROOT" --job-id "$SLURM_JOB_ID" \
  --implementation outputs/code/behavioral_implementation.json \
  --output "$EP_GAIN_REPAIR_ROOT/common_gain_range_repaired_summary.json" \
  > "$EP_GAIN_REPAIR_ROOT/pool.out" 2> "$EP_GAIN_REPAIR_ROOT/pool.err"
