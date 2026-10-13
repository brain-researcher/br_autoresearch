#!/bin/bash -l
#SBATCH --job-name=ep01-mz-half-wave
#SBATCH --partition=normal
#SBATCH --constraint=CPU_SKU:7502
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=01:35:00
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-half-wave-%A_%a.out
#SBATCH --error=outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-half-wave-%A_%a.err

# Root submits index1 with --time=01:40:00; all other indices retain95minutes.
# Index1: new checks190 + ASTRA40 + optional plot130 + fit5410 =5770<6000s.
# Other indices: fit5410<5700s. No polling, automatic retry or later dispatch.
set -euo pipefail
EP_MZ_HALF_HOST=$(hostname)
printf 'HOST=%s SLURM_JOB_ID=%s ARRAY=%s INDEX=%s CPUS=%s\n' "$EP_MZ_HALF_HOST" "${SLURM_JOB_ID-}" "${SLURM_ARRAY_JOB_ID-}" "${SLURM_ARRAY_TASK_ID-}" "${SLURM_CPUS_PER_TASK-}"
if [[ -z "${SLURM_JOB_ID-}" || "$EP_MZ_HALF_HOST" == *-ln* || "${SLURM_CPUS_PER_TASK-}" != 1 ]]; then exit 2; fi
case "${SLURM_ARRAY_JOB_ID-}" in ''|*[!0-9]*) exit 2;; esac
case "${SLURM_ARRAY_TASK_ID-}" in ''|*[!0-9]*) exit 2;; esac
if (( SLURM_ARRAY_TASK_ID < 1 || SLURM_ARRAY_TASK_ID > 199 )); then exit 2; fi
cd "$SLURM_SUBMIT_DIR"
module purge
module load devel python/3.12.1 math py-numpy/1.26.3 py-scikit-learn/1.5.1_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
EP_MZ_HALF_PYTHON=$(command -v python3.12)
"$EP_MZ_HALF_PYTHON" --version
if [[ "$SLURM_ARRAY_TASK_ID" == 1 ]]; then
  bash -n outputs/code/mz_half_wave.sbatch
  EP_MZ_HALF_CHECKS="outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-half-wave-checks-${SLURM_ARRAY_JOB_ID}"
  mkdir "$EP_MZ_HALF_CHECKS"
  PYTHONPATH="outputs/code${PYTHONPATH:+:$PYTHONPATH}" \
    timeout --signal=TERM --kill-after=10s 180s "$EP_MZ_HALF_PYTHON" -B \
    -m unittest -v test_mz_half_wave test_mz_twice_wave_pool \
    > "$EP_MZ_HALF_CHECKS/tests.txt" 2>&1
  printf 'MZ_HALF_AND_TWICE_POOL_TESTS_PASS\n'
  EP_MZ_HALF_ASTRA="$GROUP_HOME/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv/bin/astra-tools"
  if timeout --signal=TERM --kill-after=10s 30s "$EP_MZ_HALF_ASTRA" validate outputs/astra/v0.0.14/astra.yaml --json \
      > "$EP_MZ_HALF_CHECKS/astra_validation.json"; then
    printf 'MZ_HALF_WAVE_ASTRA_PASS\n'
  else
    printf 'MZ_HALF_WAVE_ASTRA_OPEN: coordinator must inspect retained output\n' >&2
  fi
  # The coordinator verified this site module. Its environment changes stay
  # inside the optional rendering subshell, leaving the fitter stack unchanged.
  if (
      module load viz py-matplotlib/3.8.3_py312 &&
      timeout --signal=TERM --kill-after=10s 120s "$EP_MZ_HALF_PYTHON" -B \
        outputs/code/plot_mz0_run_sensitivity.py --output-dir outputs/figures
    ) > "$EP_MZ_HALF_CHECKS/plot.txt" 2>&1; then
    printf 'MZ0_RUN_SENSITIVITY_PLOT_COMPLETED\n'
  else
    printf 'MZ0_RUN_SENSITIVITY_PLOT_OPEN: retained plot.txt; half recovery continues\n' >&2
  fi
fi
exec timeout --signal=TERM --kill-after=10s 5400s "$EP_MZ_HALF_PYTHON" -B \
  outputs/code/run_mz_half_wave.py --replicate-index "$SLURM_ARRAY_TASK_ID" \
  --design-events outputs/runs/behavioral_mechanism_kill_test/20261003-MC-SESOI-replicate000-attempt003/simulated_events.csv.gz \
  --implementation outputs/code/behavioral_implementation.json \
  --output-dir "outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-half-SESOI-wave-${SLURM_ARRAY_JOB_ID}/replicate${SLURM_ARRAY_TASK_ID}" \
  --max-seconds 5400
