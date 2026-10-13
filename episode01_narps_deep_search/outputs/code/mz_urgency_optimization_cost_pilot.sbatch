#!/bin/bash -l
#SBATCH --job-name=ep01-urgency-opt-cost
#SBATCH --partition=normal
#SBATCH --constraint=CPU_SKU:7502
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-urgency-optimization-cost-%j.out
#SBATCH --error=outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-urgency-optimization-cost-%j.err

# Checks250 + ASTRA40 + eight simultaneous bounded3010s probes =3300<3600s.
# All steps use one CPU and2GiB; no pooling, retries or subsequent submission.
set -euo pipefail
EP_URGENCY_OPT_HOST=$(hostname)
printf 'HOST=%s SLURM_JOB_ID=%s NTASKS=%s CPUS_PER_TASK=%s\n' \
  "$EP_URGENCY_OPT_HOST" "${SLURM_JOB_ID-}" "${SLURM_NTASKS-}" "${SLURM_CPUS_PER_TASK-}"
if [[ -z "${SLURM_JOB_ID-}" || "$EP_URGENCY_OPT_HOST" == *-ln* \
      || "${SLURM_NTASKS-}" != 8 || "${SLURM_CPUS_PER_TASK-}" != 1 ]]; then exit 2; fi
cd "$SLURM_SUBMIT_DIR"
module purge
module load devel python/3.12.1 math py-numpy/1.26.3 py-scikit-learn/1.5.1_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="outputs/code${PYTHONPATH:+:$PYTHONPATH}"
EP_URGENCY_OPT_PYTHON=$(command -v python3.12)
"$EP_URGENCY_OPT_PYTHON" --version
bash -n outputs/code/mz_urgency_optimization_cost_pilot.sbatch
EP_URGENCY_OPT_ROOT="outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-urgency-optimization-cost-${SLURM_JOB_ID}"
EP_URGENCY_OPT_CHECKS="outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-urgency-optimization-cost-checks-${SLURM_JOB_ID}"
mkdir "$EP_URGENCY_OPT_ROOT" "$EP_URGENCY_OPT_CHECKS"
srun --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=2G --exclusive --exact --cpu-bind=cores \
  timeout --signal=TERM --kill-after=10s 240s "$EP_URGENCY_OPT_PYTHON" -B \
  -m unittest discover -s outputs/tests -p test_mz_urgency_optimization_cost_pilot.py -v \
  > "$EP_URGENCY_OPT_CHECKS/tests.txt" 2>&1
printf 'URGENCY_OPTIMIZATION_COST_TESTS_PASS\n'
EP_URGENCY_OPT_ASTRA="$GROUP_HOME/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv/bin/astra-tools"
if srun --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=2G --exclusive --exact --cpu-bind=cores \
    timeout --signal=TERM --kill-after=10s 30s "$EP_URGENCY_OPT_ASTRA" \
    validate outputs/astra/v0.0.14/astra.yaml --json \
    > "$EP_URGENCY_OPT_CHECKS/astra_validation.json" 2> "$EP_URGENCY_OPT_CHECKS/astra_validation.err"; then
  printf 'URGENCY_OPTIMIZATION_COST_ASTRA_PASS\n'
else
  printf 'URGENCY_OPTIMIZATION_COST_ASTRA_OPEN: coordinator must inspect retained output\n' >&2
fi
EP_URGENCY_OPT_PIDS=()
for EP_URGENCY_OPT_INDEX in {0..7}; do
  srun --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=2G --exclusive --exact --cpu-bind=cores \
    timeout --signal=TERM --kill-after=10s 3000s "$EP_URGENCY_OPT_PYTHON" -B \
    outputs/code/run_mz_urgency_optimization_cost_pilot.py \
    --probe-index "$EP_URGENCY_OPT_INDEX" \
    --source-dir outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-SESOI-replicate000-attempt001 \
    --implementation outputs/code/behavioral_implementation.json \
    --output-dir "$EP_URGENCY_OPT_ROOT/probe${EP_URGENCY_OPT_INDEX}" \
    --max-seconds 3000 \
    > "$EP_URGENCY_OPT_ROOT/probe${EP_URGENCY_OPT_INDEX}.out" \
    2> "$EP_URGENCY_OPT_ROOT/probe${EP_URGENCY_OPT_INDEX}.err" &
  EP_URGENCY_OPT_PIDS+=("$!")
done
EP_URGENCY_OPT_FAILED=0
printf 'probe_index\texit_status\n' > "$EP_URGENCY_OPT_ROOT/probe_exit_status.tsv"
for EP_URGENCY_OPT_INDEX in {0..7}; do
  if wait "${EP_URGENCY_OPT_PIDS[$EP_URGENCY_OPT_INDEX]}"; then
    EP_URGENCY_OPT_STATUS=0
  else
    EP_URGENCY_OPT_STATUS=$?
    EP_URGENCY_OPT_FAILED=1
  fi
  printf '%s\t%s\n' "$EP_URGENCY_OPT_INDEX" "$EP_URGENCY_OPT_STATUS" \
    >> "$EP_URGENCY_OPT_ROOT/probe_exit_status.tsv"
  printf 'URGENCY_OPTIMIZATION_PROBE=%s EXIT_STATUS=%s\n' "$EP_URGENCY_OPT_INDEX" "$EP_URGENCY_OPT_STATUS"
done
exit "$EP_URGENCY_OPT_FAILED"
