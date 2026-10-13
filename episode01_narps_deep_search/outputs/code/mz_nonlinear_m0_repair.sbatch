#!/bin/bash -l
#SBATCH --job-name=ep01-nonlinear-m0-repair
#SBATCH --partition=normal
#SBATCH --constraint=CPU_SKU:7502
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=04:30:00
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-nonlinear-M0-repair-%j.out
#SBATCH --error=outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-nonlinear-M0-repair-%j.err

# Changed checks190 + ASTRA40 + fresh two-half M0 repair15610 =15840<16200s.
# The coordinator separately establishes terminal MZ/M0 before mixed pooling.
# No failed fit is reused, and no follow-on job or automatic retry is dispatched.
set -euo pipefail
EP_NONLINEAR_M0_HOST=$(hostname)
printf 'HOST=%s SLURM_JOB_ID=%s CPUS=%s\n' "$EP_NONLINEAR_M0_HOST" "${SLURM_JOB_ID-}" "${SLURM_CPUS_PER_TASK-}"
if [[ -z "${SLURM_JOB_ID-}" || "$EP_NONLINEAR_M0_HOST" == *-ln* || "${SLURM_CPUS_PER_TASK-}" != 1 ]]; then exit 2; fi
cd "$SLURM_SUBMIT_DIR"
module purge
module load devel python/3.12.1 math py-numpy/1.26.3 py-scikit-learn/1.5.1_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
EP_NONLINEAR_M0_PYTHON=$(command -v python3.12)
"$EP_NONLINEAR_M0_PYTHON" --version
bash -n outputs/code/mz_nonlinear_m0_repair.sbatch
EP_NONLINEAR_M0_CHECKS="outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-nonlinear-M0-repair-checks-${SLURM_JOB_ID}"
mkdir "$EP_NONLINEAR_M0_CHECKS"
PYTHONPATH="outputs/code${PYTHONPATH:+:$PYTHONPATH}" \
  timeout --signal=TERM --kill-after=10s 180s "$EP_NONLINEAR_M0_PYTHON" -B \
  -m unittest -v test_mz_nonlinear_m0_repair test_mz_nonlinear_repaired_pool \
  > "$EP_NONLINEAR_M0_CHECKS/tests.txt" 2>&1
printf 'NONLINEAR_M0_REPAIR_AND_POOL_TESTS_PASS\n'
EP_NONLINEAR_M0_ASTRA="$GROUP_HOME/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv/bin/astra-tools"
if timeout --signal=TERM --kill-after=10s 30s "$EP_NONLINEAR_M0_ASTRA" validate outputs/astra/v0.0.14/astra.yaml --json \
    > "$EP_NONLINEAR_M0_CHECKS/astra_validation.json"; then
  printf 'NONLINEAR_M0_REPAIR_ASTRA_PASS\n'
else
  printf 'NONLINEAR_M0_REPAIR_ASTRA_OPEN: coordinator must inspect retained output\n' >&2
fi
exec timeout --signal=TERM --kill-after=10s 15600s "$EP_NONLINEAR_M0_PYTHON" -B \
  outputs/code/run_mz_nonlinear_m0_repair.py \
  --source-dir outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-SESOI-replicate000-attempt001 \
  --implementation outputs/code/behavioral_implementation.json \
  --output-dir "outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-nonlinear-M0-repair-${SLURM_JOB_ID}" \
  --max-seconds 15600
