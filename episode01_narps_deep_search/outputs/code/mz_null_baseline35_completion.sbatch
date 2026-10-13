#!/bin/bash -l
#SBATCH --job-name=ep01-null35-complete
#SBATCH --partition=normal
#SBATCH --constraint=CPU_SKU:7502
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:50:00
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/behavioral_mechanism_kill_test/20261005-M0-null-baseline35-completion-%j.out
#SBATCH --error=outputs/runs/behavioral_mechanism_kill_test/20261005-M0-null-baseline35-completion-%j.err

# One selected completion after diagnosed2700s interruption of baseline35.
# Preserve and rescore three converged original500 fits, freshly fit only
# reverse MZ500. No simulation, source copy, polling or automatic retry.
# Tests120+10 + ASTRA30+10 + whole process2700+10 =2880<3000s.
set -euo pipefail
EP_NULL35_HOST=$(hostname)
printf 'HOST=%s SLURM_JOB_ID=%s CPUS=%s INDEX=35 REUSED_FITS=3 FRESH_MODEL=MZ SWEEPS=500\n' \
  "$EP_NULL35_HOST" "${SLURM_JOB_ID-}" "${SLURM_CPUS_PER_TASK-}"
if [[ -z "${SLURM_JOB_ID-}" || "$EP_NULL35_HOST" == *-ln* || \
      "${SLURM_NTASKS-}" != 1 || "${SLURM_CPUS_PER_TASK-}" != 1 ]]; then
  printf 'Requires the selected one-CPU compute allocation\n' >&2
  exit 2
fi
cd "$SLURM_SUBMIT_DIR"
module purge
module load devel python/3.12.1 math py-numpy/1.26.3 py-scikit-learn/1.5.1_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
EP_NULL35_PYTHON=$(command -v python3.12)
"$EP_NULL35_PYTHON" --version
bash -n outputs/code/mz_null_baseline35_completion.sbatch
EP_NULL35_CHECKS="outputs/runs/behavioral_mechanism_kill_test/20261005-M0-null-baseline35-completion-checks-${SLURM_JOB_ID}"
mkdir "$EP_NULL35_CHECKS"
timeout --signal=TERM --kill-after=10s 120s "$EP_NULL35_PYTHON" -B \
  -m unittest discover -s outputs/code -p test_mz_null_baseline_completion.py -v \
  > "$EP_NULL35_CHECKS/tests.txt" 2>&1
printf 'NULL_BASELINE35_COMPLETION_TESTS_PASS\n'
EP_NULL35_ASTRA="$GROUP_HOME/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv/bin/astra-tools"
if timeout --signal=TERM --kill-after=10s 30s "$EP_NULL35_ASTRA" validate \
    outputs/astra/v0.0.14/astra.yaml --json \
    > "outputs/astra/v0.0.14/validation_20261005_null_baseline35_completion_${SLURM_JOB_ID}.json"; then
  printf 'NULL_BASELINE35_COMPLETION_ASTRA_PASS\n'
else
  printf 'NULL_BASELINE35_COMPLETION_ASTRA_OPEN: coordinator must inspect validation output\n' >&2
fi
exec timeout --signal=TERM --kill-after=10s 2700s "$EP_NULL35_PYTHON" -B \
  outputs/code/run_mz_null_baseline_completion.py \
  --implementation outputs/code/behavioral_implementation.json \
  --output-dir "outputs/runs/behavioral_mechanism_kill_test/20261005-M0-null-baseline35-completion-${SLURM_JOB_ID}" \
  --max-seconds 2700
