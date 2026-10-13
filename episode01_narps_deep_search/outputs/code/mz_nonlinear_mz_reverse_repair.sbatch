#!/bin/bash -l
#SBATCH --job-name=ep01-nonlinear-mz-reverse
#SBATCH --partition=normal
#SBATCH --constraint=CPU_SKU:7502
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=02:20:00
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-nonlinear-MZ-reverse-repair-%j.out
#SBATCH --error=outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-nonlinear-MZ-reverse-repair-%j.err

# New checks190 + ASTRA40 + reverse-only repair7810 =8040<8400s.
# The coordinator separately establishes both repaired components terminal.
# No failed optimizer state, automatic retry, actual pool or follow-on launch.
set -euo pipefail
EP_NONLINEAR_MZ_HOST=$(hostname)
printf 'HOST=%s SLURM_JOB_ID=%s CPUS=%s\n' "$EP_NONLINEAR_MZ_HOST" "${SLURM_JOB_ID-}" "${SLURM_CPUS_PER_TASK-}"
if [[ -z "${SLURM_JOB_ID-}" || "$EP_NONLINEAR_MZ_HOST" == *-ln* || "${SLURM_CPUS_PER_TASK-}" != 1 ]]; then exit 2; fi
cd "$SLURM_SUBMIT_DIR"
module purge
module load devel python/3.12.1 math py-numpy/1.26.3 py-scikit-learn/1.5.1_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
EP_NONLINEAR_MZ_PYTHON=$(command -v python3.12)
"$EP_NONLINEAR_MZ_PYTHON" --version
bash -n outputs/code/mz_nonlinear_mz_reverse_repair.sbatch
EP_NONLINEAR_MZ_CHECKS="outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-nonlinear-MZ-reverse-repair-checks-${SLURM_JOB_ID}"
mkdir "$EP_NONLINEAR_MZ_CHECKS"
PYTHONPATH="outputs/code${PYTHONPATH:+:$PYTHONPATH}" \
  timeout --signal=TERM --kill-after=10s 180s "$EP_NONLINEAR_MZ_PYTHON" -B \
  -m unittest -v test_mz_nonlinear_mz_reverse_repair test_mz_nonlinear_both_repaired_pool \
  > "$EP_NONLINEAR_MZ_CHECKS/tests.txt" 2>&1
printf 'NONLINEAR_MZ_REVERSE_AND_BOTH_POOL_TESTS_PASS\n'
EP_NONLINEAR_MZ_ASTRA="$GROUP_HOME/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv/bin/astra-tools"
if timeout --signal=TERM --kill-after=10s 30s "$EP_NONLINEAR_MZ_ASTRA" validate outputs/astra/v0.0.14/astra.yaml --json \
    > "$EP_NONLINEAR_MZ_CHECKS/astra_validation.json"; then
  printf 'NONLINEAR_MZ_REVERSE_ASTRA_PASS\n'
else
  printf 'NONLINEAR_MZ_REVERSE_ASTRA_OPEN: coordinator must inspect retained output\n' >&2
fi
exec timeout --signal=TERM --kill-after=10s 7800s "$EP_NONLINEAR_MZ_PYTHON" -B \
  outputs/code/run_mz_nonlinear_mz_reverse_repair.py \
  --source-dir outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-SESOI-replicate000-attempt001 \
  --implementation outputs/code/behavioral_implementation.json \
  --output-dir "outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-nonlinear-MZ-reverse-repair-${SLURM_JOB_ID}" \
  --max-seconds 7800
