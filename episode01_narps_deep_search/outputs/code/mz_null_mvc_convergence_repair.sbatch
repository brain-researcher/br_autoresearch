#!/bin/bash -l
#SBATCH --job-name=ep01-null-mvc-repair
#SBATCH --partition=normal
#SBATCH --constraint=CPU_SKU:7502
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=01:35:00
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/behavioral_mechanism_kill_test/20261005-M0-null-MVC-convergence-%j.out
#SBATCH --error=outputs/runs/behavioral_mechanism_kill_test/20261005-M0-null-MVC-convergence-%j.err

# One diagnosed reverse-half numerical repair. No submission, retry or polling.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load devel python/3.12.1 math py-numpy/1.26.3 py-scikit-learn/1.5.1_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
EP_NULL_MVC_PYTHON=$(command -v python3.12)
hostname
printf 'SLURM_JOB_ID=%s CPUS=%s INDEX=0 MODEL=MVC REUSE_FIRST=1 SECOND_RUNTIME_SWEEPS=1000\n' \
  "$SLURM_JOB_ID" "$SLURM_CPUS_PER_TASK"
"$EP_NULL_MVC_PYTHON" --version
bash -n outputs/code/mz_null_mvc_convergence_repair.sbatch
EP_NULL_MVC_CHECKS="outputs/runs/behavioral_mechanism_kill_test/20261005-M0-null-MVC-convergence-checks-${SLURM_JOB_ID}"
mkdir "$EP_NULL_MVC_CHECKS"
timeout --signal=TERM --kill-after=10s 120s "$EP_NULL_MVC_PYTHON" -B \
  -m unittest discover -s outputs/code -p test_mz_null_mvc_convergence_repair.py -v \
  > "$EP_NULL_MVC_CHECKS/tests.txt" 2>&1
printf 'NULL_MVC_CONVERGENCE_TESTS_PASS\n'
EP_NULL_MVC_ASTRA="$GROUP_HOME/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv/bin/astra-tools"
if timeout --signal=TERM --kill-after=10s 30s "$EP_NULL_MVC_ASTRA" validate \
    outputs/astra/v0.0.14/astra.yaml --json \
    > "outputs/astra/v0.0.14/validation_20261005_null_mvc_repair_${SLURM_JOB_ID}.json"; then
  printf 'NULL_MVC_CONVERGENCE_ASTRA_PASS\n'
else
  printf 'NULL_MVC_CONVERGENCE_ASTRA_OPEN: coordinator must inspect validation output\n' >&2
fi
exec timeout --signal=TERM --kill-after=10s 5400s "$EP_NULL_MVC_PYTHON" -B \
  outputs/code/run_mz_null_mvc_convergence_repair.py \
  --implementation outputs/code/behavioral_implementation.json \
  --output-dir "outputs/runs/behavioral_mechanism_kill_test/20261005-M0-null-MVC-convergence-${SLURM_JOB_ID}" \
  --max-seconds 5400
