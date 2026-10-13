#!/bin/bash -l
#SBATCH --job-name=ep01-mz-menu
#SBATCH --partition=normal
#SBATCH --constraint=CPU_SKU:7502
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=01:35:00
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-complex-rival-%j.out
#SBATCH --error=outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-complex-rival-%j.err

# One selected model, two directions, no submission or retry from this script.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
EP_COMPLEX_MODEL="${1:?Supply MVZ or MVC}"
case "$EP_COMPLEX_MODEL" in
  MVZ|MVC) ;;
  *) printf 'Unsupported complex rival\n' >&2; exit 2 ;;
esac
module purge
module load devel python/3.12.1 math py-numpy/1.26.3 py-scikit-learn/1.5.1_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
EP_PYTHON=$(command -v python3.12)
hostname
printf 'SLURM_JOB_ID=%s MODEL=%s CPUS=%s\n' "$SLURM_JOB_ID" "$EP_COMPLEX_MODEL" "$SLURM_CPUS_PER_TASK"
"$EP_PYTHON" --version
exec timeout --signal=TERM --kill-after=10s 5400s "$EP_PYTHON" -B \
  outputs/code/run_mz_complex_rival.py --model "$EP_COMPLEX_MODEL" \
  --implementation outputs/code/behavioral_implementation.json \
  --source-dir outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-SESOI-replicate000-attempt001 \
  --output-dir "outputs/runs/behavioral_mechanism_kill_test/20261005-MZ0-${EP_COMPLEX_MODEL}-rival-${SLURM_JOB_ID}" \
  --max-seconds 5400
