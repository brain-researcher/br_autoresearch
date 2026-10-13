#!/bin/bash -l
#SBATCH --job-name=ep01-nonlinear-m0-directions
#SBATCH --partition=normal
#SBATCH --constraint=CPU_SKU:7502
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-nonlinear-M0-directions-%j.out
#SBATCH --error=outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-nonlinear-M0-directions-%j.err

# New checks250 + ASTRA40 + concurrent science4810 + pool130 =5230<5400s.
set -euo pipefail
EP_M0_DIR_HOST=$(hostname)
printf 'HOST=%s SLURM_JOB_ID=%s CPUS_PER_TASK=%s\n' \
  "$EP_M0_DIR_HOST" "${SLURM_JOB_ID-}" "${SLURM_CPUS_PER_TASK-}"
if [[ $# != 0 || "$EP_M0_DIR_HOST" == *-ln* || -z "${SLURM_JOB_ID-}" \
      || "${SLURM_CPUS_PER_TASK-}" != 8 ]]; then exit 2; fi
cd "$SLURM_SUBMIT_DIR"
module purge
module load devel python/3.12.1 math py-numpy/1.26.3 py-scikit-learn/1.5.1_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="outputs/code:outputs/tests${PYTHONPATH:+:$PYTHONPATH}"
EP_M0_DIR_PYTHON=$(command -v python3.12)
"$EP_M0_DIR_PYTHON" --version
bash -n outputs/code/mz_nonlinear_m0_directions_repair.sbatch
EP_M0_DIR_ROOT="outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-nonlinear-M0-directions-repair-${SLURM_JOB_ID}"
EP_M0_DIR_CHECKS="outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-nonlinear-M0-directions-checks-${SLURM_JOB_ID}"
mkdir "$EP_M0_DIR_ROOT" "$EP_M0_DIR_CHECKS"
printf 'stage\texit_status\n' > "$EP_M0_DIR_CHECKS/stage_exit_status.tsv"

if srun --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=2G --exclusive --exact --cpu-bind=cores \
    timeout --signal=TERM --kill-after=10s 240s "$EP_M0_DIR_PYTHON" -B -m unittest -v \
    test_mz_nonlinear_m0_direction_repair test_mz_nonlinear_m0_directions_repaired \
    > "$EP_M0_DIR_CHECKS/tests.txt" 2>&1; then
  EP_M0_DIR_TEST_STATUS=0
else
  EP_M0_DIR_TEST_STATUS=$?
fi
printf 'nonlinear_m0_direction_tests\t%s\n' "$EP_M0_DIR_TEST_STATUS" \
  >> "$EP_M0_DIR_CHECKS/stage_exit_status.tsv"

EP_M0_DIR_ASTRA="$GROUP_HOME/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv/bin/astra-tools"
if srun --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=2G --exclusive --exact --cpu-bind=cores \
    timeout --signal=TERM --kill-after=10s 30s "$EP_M0_DIR_ASTRA" \
    validate outputs/astra/v0.0.14/astra.yaml --json \
    > "$EP_M0_DIR_CHECKS/astra_validation.json" 2> "$EP_M0_DIR_CHECKS/astra_validation.err"; then
  EP_M0_DIR_ASTRA_STATUS=0
else
  EP_M0_DIR_ASTRA_STATUS=$?
fi
printf 'astra_validation\t%s\n' "$EP_M0_DIR_ASTRA_STATUS" >> "$EP_M0_DIR_CHECKS/stage_exit_status.tsv"
if [[ "$EP_M0_DIR_TEST_STATUS" != 0 ]]; then exit 1; fi

EP_M0_DIR_PIDS=()
for EP_M0_DIRECTION in 0 1; do
  srun --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=8G --exclusive --exact --cpu-bind=cores \
    timeout --signal=TERM --kill-after=10s 4800s "$EP_M0_DIR_PYTHON" -B \
    outputs/code/run_mz_nonlinear_m0_direction_repair.py \
    --direction "$EP_M0_DIRECTION" \
    --source-dir outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-SESOI-replicate000-attempt001 \
    --implementation outputs/code/behavioral_implementation.json \
    --output-dir "$EP_M0_DIR_ROOT/dir${EP_M0_DIRECTION}" --max-seconds 4800 \
    > "$EP_M0_DIR_ROOT/dir${EP_M0_DIRECTION}.out" \
    2> "$EP_M0_DIR_ROOT/dir${EP_M0_DIRECTION}.err" &
  EP_M0_DIR_PIDS+=("$!")
done
EP_M0_DIR_SCIENCE_STATUS=0
for EP_M0_DIRECTION in 0 1; do
  if wait "${EP_M0_DIR_PIDS[$EP_M0_DIRECTION]}"; then
    EP_M0_DIR_STATUS=0
  else
    EP_M0_DIR_STATUS=$?
    EP_M0_DIR_SCIENCE_STATUS=1
  fi
  printf 'nonlinear_m0_direction%s\t%s\n' "$EP_M0_DIRECTION" "$EP_M0_DIR_STATUS" \
    >> "$EP_M0_DIR_CHECKS/stage_exit_status.tsv"
done

# Both reports are terminal. A valid technical component yields an unknown
# summary; malformed completion is rejected. Neither permits a contrast.
if srun --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=2G --exclusive --exact --cpu-bind=cores \
    timeout --signal=TERM --kill-after=10s 120s "$EP_M0_DIR_PYTHON" -B \
    outputs/code/summarize_mz_nonlinear_m0_directions_repaired.py \
    --repair-dir "$EP_M0_DIR_ROOT" --repair-job-id "$SLURM_JOB_ID" \
    --implementation outputs/code/behavioral_implementation.json \
    --output "$EP_M0_DIR_ROOT/nonlinear_m0_directions_repaired_summary.json" \
    > "$EP_M0_DIR_ROOT/pool.out" 2> "$EP_M0_DIR_ROOT/pool.err"; then
  EP_M0_DIR_POOL_STATUS=0
else
  EP_M0_DIR_POOL_STATUS=$?
fi
printf 'nonlinear_m0_directions_pool\t%s\n' "$EP_M0_DIR_POOL_STATUS" \
  >> "$EP_M0_DIR_CHECKS/stage_exit_status.tsv"
if [[ "$EP_M0_DIR_SCIENCE_STATUS" != 0 || "$EP_M0_DIR_POOL_STATUS" != 0 \
      || "$EP_M0_DIR_ASTRA_STATUS" != 0 ]]; then exit 1; fi
