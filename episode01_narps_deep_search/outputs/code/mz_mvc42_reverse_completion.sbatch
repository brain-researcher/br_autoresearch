#!/bin/bash -l
#SBATCH --job-name=ep01-mvc42-reverse
#SBATCH --partition=normal
#SBATCH --constraint=CPU_SKU:7502
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=01:35:00
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/behavioral_mechanism_kill_test/20261005-MZ42-MVC-reverse-completion-%j.out
#SBATCH --error=outputs/runs/behavioral_mechanism_kill_test/20261005-MZ42-MVC-reverse-completion-%j.err

# New single reverse-only numerical repair after diagnosed1000-sweep exhaustion.
# Reuse/rescore the first666-sweep fit; original500/first1000/reverse1500
# provenance stays explicit. No source changes, polling or automatic retries.
# Tests120+10 + ASTRA30+10 + process5400+10 =5580<5700s.
set -euo pipefail
EP_MVC42_HOST=$(hostname)
printf 'HOST=%s SLURM_JOB_ID=%s CPUS=%s INDEX=42 MODEL=MVC FIRST_VALIDATION_SWEEPS=1000 REVERSE_SWEEPS=1500\n' \
  "$EP_MVC42_HOST" "${SLURM_JOB_ID-}" "${SLURM_CPUS_PER_TASK-}"
if [[ -z "${SLURM_JOB_ID-}" || "$EP_MVC42_HOST" == *-ln* || \
      "${SLURM_NTASKS-}" != 1 || "${SLURM_CPUS_PER_TASK-}" != 1 ]]; then
  printf 'Requires the selected one-CPU compute allocation\n' >&2
  exit 2
fi
cd "$SLURM_SUBMIT_DIR"
module purge
module load devel python/3.12.1 math py-numpy/1.26.3 py-scikit-learn/1.5.1_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
EP_MVC42_PYTHON=$(command -v python3.12)
"$EP_MVC42_PYTHON" --version
bash -n outputs/code/mz_mvc42_reverse_completion.sbatch
EP_MVC42_CHECKS="outputs/runs/behavioral_mechanism_kill_test/20261005-MZ42-MVC-reverse-completion-checks-${SLURM_JOB_ID}"
mkdir "$EP_MVC42_CHECKS"
timeout --signal=TERM --kill-after=10s 120s "$EP_MVC42_PYTHON" -B \
  -m unittest discover -s outputs/code -p test_mz_mvc_reverse_completion.py -v \
  > "$EP_MVC42_CHECKS/tests.txt" 2>&1
printf 'MVC42_REVERSE_COMPLETION_TESTS_PASS\n'
EP_MVC42_ASTRA="$GROUP_HOME/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv/bin/astra-tools"
if timeout --signal=TERM --kill-after=10s 30s "$EP_MVC42_ASTRA" validate \
    outputs/astra/v0.0.14/astra.yaml --json \
    > "outputs/astra/v0.0.14/validation_20261005_mvc42_reverse_completion_${SLURM_JOB_ID}.json"; then
  printf 'MVC42_REVERSE_COMPLETION_ASTRA_PASS\n'
else
  printf 'MVC42_REVERSE_COMPLETION_ASTRA_OPEN: coordinator must inspect validation output\n' >&2
fi
exec timeout --signal=TERM --kill-after=10s 5400s "$EP_MVC42_PYTHON" -B \
  outputs/code/run_mz_mvc_reverse_completion.py \
  --implementation outputs/code/behavioral_implementation.json \
  --output-dir "outputs/runs/behavioral_mechanism_kill_test/20261005-MZ42-MVC-reverse-completion-${SLURM_JOB_ID}" \
  --max-seconds 5400
