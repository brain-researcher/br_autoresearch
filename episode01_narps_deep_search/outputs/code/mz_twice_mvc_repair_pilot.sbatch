#!/bin/bash -l
#SBATCH --job-name=ep01-twice0-MVC-repair
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=100
#SBATCH --no-requeue

# One diagnosed repair only. Failed/completed historical sources are immutable.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
EP_REPAIR_HOST=$(hostname)
printf 'HOSTNAME=%s\n' "$EP_REPAIR_HOST"
printf 'REPAIR_JOB=%s CPUS=%s\n' "$SLURM_JOB_ID" "$SLURM_CPUS_PER_TASK"
[[ "$SLURM_JOB_ID" =~ ^[0-9]+$ && "$EP_REPAIR_HOST" != *-ln* && "$SLURM_CPUS_PER_TASK" == 4 ]]
module purge
module load devel python/3.12.1 math py-numpy/1.26.3 py-scikit-learn/1.5.1_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="outputs/code:outputs/tests${PYTHONPATH:+:$PYTHONPATH}"
EP_REPAIR_ROOT="outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-twice-MVC-repair-${SLURM_JOB_ID}-attempt001"
mkdir "$EP_REPAIR_ROOT"
if timeout --signal=TERM --kill-after=10s 120s python3.12 -B -m unittest discover -s outputs/code -p test_twice_mz_mvc_repair_pilot.py -v >"$EP_REPAIR_ROOT/tests.log" 2>&1; then
  printf 'REPAIR_TEST_EXIT=0\n'
else
  EP_REPAIR_TEST_EXIT=$?
  printf 'REPAIR_TEST_EXIT=%s; scientific repair not entered\n' "$EP_REPAIR_TEST_EXIT"
  exit "$EP_REPAIR_TEST_EXIT"
fi
# Root validates the changed ASTRA program in the already selected count step.
srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=8G \
  timeout --signal=TERM --kill-after=20s 5420s python3.12 -B \
  outputs/code/run_twice_mz_mvc_repair_pilot.py \
  --implementation outputs/code/behavioral_implementation.json \
  --output-dir "$EP_REPAIR_ROOT/pilot" --max-seconds 5400
