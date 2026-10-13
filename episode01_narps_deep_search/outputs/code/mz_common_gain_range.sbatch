#!/bin/bash -l
#SBATCH --job-name=ep01-common-gain
#SBATCH --partition=normal
#SBATCH --constraint=CPU_SKU:7502
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:40:00
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-common-gain-range-%j.out
#SBATCH --error=outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-common-gain-range-%j.err

# Checks190 + ASTRA40 + two parallel5410s components + pool130 =5770<6000s.
# Pool only after both children actually exit0; no active-writer reads/retries.
set -euo pipefail
EP_COMMON_GAIN_HOST=$(hostname)
printf 'HOST=%s SLURM_JOB_ID=%s NTASKS=%s CPUS_PER_TASK=%s\n' \
  "$EP_COMMON_GAIN_HOST" "${SLURM_JOB_ID-}" "${SLURM_NTASKS-}" "${SLURM_CPUS_PER_TASK-}"
if [[ -z "${SLURM_JOB_ID-}" || "$EP_COMMON_GAIN_HOST" == *-ln* \
      || "${SLURM_NTASKS-}" != 2 || "${SLURM_CPUS_PER_TASK-}" != 1 ]]; then exit 2; fi
cd "$SLURM_SUBMIT_DIR"
module purge
module load devel python/3.12.1 math py-numpy/1.26.3 py-scikit-learn/1.5.1_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="outputs/code${PYTHONPATH:+:$PYTHONPATH}"
EP_COMMON_GAIN_PYTHON=$(command -v python3.12)
"$EP_COMMON_GAIN_PYTHON" --version
bash -n outputs/code/mz_common_gain_range.sbatch
EP_COMMON_GAIN_ROOT="outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-common-gain-range-${SLURM_JOB_ID}"
EP_COMMON_GAIN_CHECKS="outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-common-gain-range-checks-${SLURM_JOB_ID}"
mkdir "$EP_COMMON_GAIN_ROOT" "$EP_COMMON_GAIN_CHECKS"
srun --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=2G --exclusive --exact --cpu-bind=cores \
  timeout --signal=TERM --kill-after=10s 180s "$EP_COMMON_GAIN_PYTHON" -B \
  -m unittest -v test_mz_common_gain_range test_mz_common_gain_range_pool \
  > "$EP_COMMON_GAIN_CHECKS/tests.txt" 2>&1
printf 'COMMON_GAIN_COMPONENT_AND_POOL_TESTS_PASS\n'
EP_COMMON_GAIN_ASTRA="$GROUP_HOME/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv/bin/astra-tools"
if srun --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=2G --exclusive --exact --cpu-bind=cores \
    timeout --signal=TERM --kill-after=10s 30s "$EP_COMMON_GAIN_ASTRA" \
    validate outputs/astra/v0.0.14/astra.yaml --json \
    > "$EP_COMMON_GAIN_CHECKS/astra_validation.json" 2> "$EP_COMMON_GAIN_CHECKS/astra_validation.err"; then
  printf 'COMMON_GAIN_ASTRA_PASS\n'
else
  printf 'COMMON_GAIN_ASTRA_OPEN: coordinator must inspect retained output\n' >&2
fi
EP_COMMON_GAIN_PIDS=()
for EP_COMMON_GAIN_DIRECTION in 0 1; do
  srun --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=2G --exclusive --exact --cpu-bind=cores \
    timeout --signal=TERM --kill-after=10s 5400s "$EP_COMMON_GAIN_PYTHON" -B \
    outputs/code/run_mz_common_gain_range.py \
    --direction "$EP_COMMON_GAIN_DIRECTION" \
    --source-dir outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-SESOI-replicate000-attempt001 \
    --implementation outputs/code/behavioral_implementation.json \
    --output-dir "$EP_COMMON_GAIN_ROOT/dir${EP_COMMON_GAIN_DIRECTION}" \
    --max-seconds 5400 \
    > "$EP_COMMON_GAIN_ROOT/dir${EP_COMMON_GAIN_DIRECTION}.out" \
    2> "$EP_COMMON_GAIN_ROOT/dir${EP_COMMON_GAIN_DIRECTION}.err" &
  EP_COMMON_GAIN_PIDS+=("$!")
done
EP_COMMON_GAIN_FAILED=0
printf 'direction\texit_status\n' > "$EP_COMMON_GAIN_ROOT/component_exit_status.tsv"
for EP_COMMON_GAIN_DIRECTION in 0 1; do
  if wait "${EP_COMMON_GAIN_PIDS[$EP_COMMON_GAIN_DIRECTION]}"; then
    EP_COMMON_GAIN_STATUS=0
  else
    EP_COMMON_GAIN_STATUS=$?
    EP_COMMON_GAIN_FAILED=1
  fi
  printf '%s\t%s\n' "$EP_COMMON_GAIN_DIRECTION" "$EP_COMMON_GAIN_STATUS" \
    >> "$EP_COMMON_GAIN_ROOT/component_exit_status.tsv"
done
if [[ "$EP_COMMON_GAIN_FAILED" != 0 ]]; then
  printf 'COMMON_GAIN_POOL_UNKNOWN: one or both terminal components failed; no partial pool or active report read\n' >&2
  exit 1
fi
srun --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=2G --exclusive --exact --cpu-bind=cores \
  timeout --signal=TERM --kill-after=10s 120s "$EP_COMMON_GAIN_PYTHON" -B \
  outputs/code/summarize_mz_common_gain_range.py \
  --source-root "$EP_COMMON_GAIN_ROOT" --job-id "$SLURM_JOB_ID" \
  --implementation outputs/code/behavioral_implementation.json \
  --output "$EP_COMMON_GAIN_ROOT/common_gain_range_summary.json" \
  > "$EP_COMMON_GAIN_ROOT/pool.out" 2> "$EP_COMMON_GAIN_ROOT/pool.err"
