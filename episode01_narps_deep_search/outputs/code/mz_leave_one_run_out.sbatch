#!/bin/bash -l
#SBATCH --job-name=ep01-mz-loo
#SBATCH --partition=normal
#SBATCH --constraint=CPU_SKU:7502
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:40:00
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-LOO-%j.out
#SBATCH --error=outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-LOO-%j.err

# Four finite workers, then pool only after all four zero exits; no polling/retry.
# Checks190 + ASTRA40 + parallel workers5410 + pool130 =5770<6000s.
set -euo pipefail
EP_MZ_LOO_HOST=$(hostname)
printf 'HOST=%s SLURM_JOB_ID=%s NTASKS=%s CPUS=%s\n' "$EP_MZ_LOO_HOST" "${SLURM_JOB_ID-}" "${SLURM_NTASKS-}" "${SLURM_CPUS_PER_TASK-}"
if [[ -z "${SLURM_JOB_ID-}" || "$EP_MZ_LOO_HOST" == *-ln* || "${SLURM_NTASKS-}" != 4 || "${SLURM_CPUS_PER_TASK-}" != 1 ]]; then exit 2; fi
cd "$SLURM_SUBMIT_DIR"
module purge
module load devel python/3.12.1 math py-numpy/1.26.3 py-scikit-learn/1.5.1_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
EP_MZ_LOO_PYTHON=$(command -v python3.12)
"$EP_MZ_LOO_PYTHON" --version
bash -n outputs/code/mz_leave_one_run_out.sbatch
EP_MZ_LOO_CHECKS="outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-LOO-checks-${SLURM_JOB_ID}"
EP_MZ_LOO_ROOT="outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-LOO-${SLURM_JOB_ID}"
mkdir "$EP_MZ_LOO_CHECKS"
timeout --signal=TERM --kill-after=10s 180s "$EP_MZ_LOO_PYTHON" -B \
  -m unittest discover -s outputs/code -p 'test_mz_leave_one_run_out*.py' -v \
  > "$EP_MZ_LOO_CHECKS/tests.txt" 2>&1
printf 'MZ_LOO_TESTS_PASS\n'
EP_MZ_LOO_ASTRA="$GROUP_HOME/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv/bin/astra-tools"
if timeout --signal=TERM --kill-after=10s 30s "$EP_MZ_LOO_ASTRA" validate outputs/astra/v0.0.14/astra.yaml --json \
    > "$EP_MZ_LOO_CHECKS/astra_validation.json"; then
  printf 'MZ_LOO_ASTRA_PASS\n'
else
  printf 'MZ_LOO_ASTRA_OPEN: coordinator must inspect retained output\n' >&2
fi

EP_MZ_LOO_PIDS=()
EP_MZ_LOO_RUNS=()
for EP_MZ_LOO_RUN in 1 2 3 4; do
  srun --nodes=1 --ntasks=1 --cpus-per-task=1 --exclusive --exact --mem=2G --cpu-bind=cores \
    --output="$EP_MZ_LOO_CHECKS/heldout${EP_MZ_LOO_RUN}.out" --error="$EP_MZ_LOO_CHECKS/heldout${EP_MZ_LOO_RUN}.err" \
    timeout --signal=TERM --kill-after=10s 5400s "$EP_MZ_LOO_PYTHON" -B \
    outputs/code/run_mz_leave_one_run_out.py --held-out-run "$EP_MZ_LOO_RUN" \
    --source-dir outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-SESOI-replicate000-attempt001 \
    --implementation outputs/code/behavioral_implementation.json \
    --output-dir "$EP_MZ_LOO_ROOT/heldout${EP_MZ_LOO_RUN}" --max-seconds 5400 &
  EP_MZ_LOO_PIDS+=("$!")
  EP_MZ_LOO_RUNS+=("$EP_MZ_LOO_RUN")
done
EP_MZ_LOO_EXIT=0
for EP_MZ_LOO_SLOT in "${!EP_MZ_LOO_PIDS[@]}"; do
  if wait "${EP_MZ_LOO_PIDS[$EP_MZ_LOO_SLOT]}"; then
    printf 'MZ_LOO_COMPLETED held_out_run=%s\n' "${EP_MZ_LOO_RUNS[$EP_MZ_LOO_SLOT]}"
  else
    EP_MZ_LOO_EXIT=1
    printf 'MZ_LOO_FAILED held_out_run=%s: preserve attempt; no retry\n' "${EP_MZ_LOO_RUNS[$EP_MZ_LOO_SLOT]}" >&2
  fi
done
if [[ "$EP_MZ_LOO_EXIT" == 0 ]]; then
  if timeout --signal=TERM --kill-after=10s 120s "$EP_MZ_LOO_PYTHON" -B \
      outputs/code/summarize_mz_leave_one_run_out.py --source-root "$EP_MZ_LOO_ROOT" \
      --job-id "$SLURM_JOB_ID" --implementation outputs/code/behavioral_implementation.json \
      --output "$EP_MZ_LOO_ROOT/pooled_loo.json" > "$EP_MZ_LOO_CHECKS/pool.txt" 2>&1; then
    printf 'MZ_LOO_POOL_COMPLETED\n'
  else
    EP_MZ_LOO_EXIT=1
    printf 'MZ_LOO_POOL_FAILED: retain all reports; no retry\n' >&2
  fi
else
  printf 'MZ_LOO_POOL_UNKNOWN: one or more workers incomplete; no pool computed\n' >&2
fi
exit "$EP_MZ_LOO_EXIT"
