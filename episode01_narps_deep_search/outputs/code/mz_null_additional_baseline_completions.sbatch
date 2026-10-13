#!/bin/bash -l
#SBATCH --job-name=ep01-null-complete
#SBATCH --partition=normal
#SBATCH --constraint=CPU_SKU:7502
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:50:00
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/behavioral_mechanism_kill_test/20261005-M0-null-additional-baseline-completions-%j.out
#SBATCH --error=outputs/runs/behavioral_mechanism_kill_test/20261005-M0-null-additional-baseline-completions-%j.err

# Exactly the two diagnosed reverse-MZ timeouts108/176. No retry loop,
# resimulation, scheduler calls or changes to converged prefix fits.
# Tests130 + ASTRA40 + parallel completions2710 =2880<3000s.
set -euo pipefail
EP_NULL_EXTRA_HOST=$(hostname)
printf 'HOST=%s SLURM_JOB_ID=%s NTASKS=%s CPUS=%s\n' \
  "$EP_NULL_EXTRA_HOST" "${SLURM_JOB_ID-}" "${SLURM_NTASKS-}" "${SLURM_CPUS_PER_TASK-}"
if [[ -z "${SLURM_JOB_ID-}" || "$EP_NULL_EXTRA_HOST" == *-ln* || "${SLURM_CPUS_PER_TASK-}" != 1 ]]; then
  printf 'An actual compute allocation with one CPU per task is required\n' >&2
  exit 2
fi
cd "$SLURM_SUBMIT_DIR"
if [[ "${1-}" == --worker ]]; then
  if (( $# != 2 )) || [[ "${SLURM_NTASKS-}" != 1 ]]; then
    printf 'Worker requires one Slurm task and one selected index\n' >&2
    exit 2
  fi
  case "$2" in 108|176) ;; *) printf 'Only108/176 are selected\n' >&2; exit 2 ;; esac
  exec timeout --signal=TERM --kill-after=10s 2700s "$EP_NULL_EXTRA_PYTHON" -B \
    outputs/code/run_mz_null_additional_baseline_completion.py \
    --replicate-index "$2" --implementation outputs/code/behavioral_implementation.json \
    --output-dir "outputs/runs/behavioral_mechanism_kill_test/20261005-M0-null-baseline${2}-completion-${SLURM_JOB_ID}" \
    --max-seconds 2700
fi
if (( $# != 0 )) || [[ "${SLURM_NTASKS-}" != 2 ]]; then
  printf 'Main entry requires exactly two Slurm tasks and no arguments\n' >&2
  exit 2
fi
module purge
module load devel python/3.12.1 math py-numpy/1.26.3 py-scikit-learn/1.5.1_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
EP_NULL_EXTRA_PYTHON=$(command -v python3.12)
export EP_NULL_EXTRA_PYTHON
"$EP_NULL_EXTRA_PYTHON" --version
bash -n outputs/code/mz_null_additional_baseline_completions.sbatch
EP_NULL_EXTRA_CHECKS="outputs/runs/behavioral_mechanism_kill_test/20261005-M0-null-additional-baseline-checks-${SLURM_JOB_ID}"
mkdir "$EP_NULL_EXTRA_CHECKS"
timeout --signal=TERM --kill-after=10s 120s "$EP_NULL_EXTRA_PYTHON" -B \
  -m unittest discover -s outputs/code -p test_mz_null_additional_baseline_completion.py -v \
  > "$EP_NULL_EXTRA_CHECKS/tests.txt" 2>&1
printf 'NULL_ADDITIONAL_BASELINE_TESTS_PASS\n'
EP_NULL_EXTRA_ASTRA="$GROUP_HOME/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv/bin/astra-tools"
if timeout --signal=TERM --kill-after=10s 30s "$EP_NULL_EXTRA_ASTRA" validate \
    outputs/astra/v0.0.14/astra.yaml --json \
    > "outputs/astra/v0.0.14/validation_20261005_null_additional_baselines_${SLURM_JOB_ID}.json"; then
  printf 'NULL_ADDITIONAL_BASELINE_ASTRA_PASS\n'
else
  printf 'NULL_ADDITIONAL_BASELINE_ASTRA_OPEN: coordinator must inspect output\n' >&2
fi
EP_NULL_EXTRA_PIDS=()
for EP_NULL_EXTRA_INDEX in 108 176; do
  srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=2G \
    bash outputs/code/mz_null_additional_baseline_completions.sbatch --worker "$EP_NULL_EXTRA_INDEX" \
    > "outputs/runs/behavioral_mechanism_kill_test/20261005-M0-null-baseline${EP_NULL_EXTRA_INDEX}-completion-${SLURM_JOB_ID}.out" \
    2> "outputs/runs/behavioral_mechanism_kill_test/20261005-M0-null-baseline${EP_NULL_EXTRA_INDEX}-completion-${SLURM_JOB_ID}.err" &
  EP_NULL_EXTRA_PIDS+=("$!")
done
EP_NULL_EXTRA_EXIT=0
for EP_NULL_EXTRA_PID in "${EP_NULL_EXTRA_PIDS[@]}"; do
  if ! wait "$EP_NULL_EXTRA_PID"; then EP_NULL_EXTRA_EXIT=1; fi
done
exit "$EP_NULL_EXTRA_EXIT"
