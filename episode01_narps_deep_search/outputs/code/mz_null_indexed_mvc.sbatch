#!/bin/bash -l
#SBATCH --job-name=ep01-null-mvc-wave
#SBATCH --partition=normal
#SBATCH --constraint=CPU_SKU:7502
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=02:30:00
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/behavioral_mechanism_kill_test/20261005-M0-null-MVC-indexed-%A_%a.out
#SBATCH --error=outputs/runs/behavioral_mechanism_kill_test/20261005-M0-null-MVC-indexed-%A_%a.err

# Root selects an explicit array of terminal, complete source cohorts. Index1
# runs the changed checks, actual repaired null0 menu and its own substantive
# MVC component. Remaining indices are selected after those checks and actual
# baseline replay pass. No scheduler calls, automatic retries or resimulation.
# Qualification bound: 130+70+40+70+8410=8720s <9000s allocation.
set -euo pipefail
EP_NULL_INDEXED_MVC_HOST=$(hostname)
printf 'HOST=%s SLURM_JOB_ID=%s ARRAY=%s INDEX=%s NTASKS=%s CPUS=%s\n' \
  "$EP_NULL_INDEXED_MVC_HOST" "${SLURM_JOB_ID-}" "${SLURM_ARRAY_JOB_ID-}" \
  "${SLURM_ARRAY_TASK_ID-}" "${SLURM_NTASKS-}" "${SLURM_CPUS_PER_TASK-}"
if [[ -z "${SLURM_JOB_ID-}" || "$EP_NULL_INDEXED_MVC_HOST" == *-ln* || \
      "${SLURM_NTASKS-}" != 1 || "${SLURM_CPUS_PER_TASK-}" != 1 || \
      -z "${SLURM_ARRAY_JOB_ID-}" ]]; then
  printf 'An actual one-CPU compute array allocation is required\n' >&2
  exit 2
fi
EP_NULL_INDEXED_MVC_INDEX="${SLURM_ARRAY_TASK_ID-}"
case "$EP_NULL_INDEXED_MVC_INDEX" in
  ''|*[!0-9]*) printf 'Expected index1 through199\n' >&2; exit 2 ;;
esac
if (( EP_NULL_INDEXED_MVC_INDEX < 1 || EP_NULL_INDEXED_MVC_INDEX > 199 )); then
  printf 'Preserve pilot0 and the fixed200 cohort set\n' >&2
  exit 2
fi
cd "$SLURM_SUBMIT_DIR"
module purge
module load devel python/3.12.1 math py-numpy/1.26.3 py-scikit-learn/1.5.1_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
EP_NULL_INDEXED_MVC_PYTHON=$(command -v python3.12)
"$EP_NULL_INDEXED_MVC_PYTHON" --version
if (( EP_NULL_INDEXED_MVC_INDEX == 1 )); then
  bash -n outputs/code/mz_null_indexed_mvc.sbatch
  EP_NULL_INDEXED_MVC_CHECKS="outputs/runs/behavioral_mechanism_kill_test/20261005-M0-null-MVC-indexed-checks-${SLURM_ARRAY_JOB_ID}"
  mkdir "$EP_NULL_INDEXED_MVC_CHECKS"
  timeout --signal=TERM --kill-after=10s 120s "$EP_NULL_INDEXED_MVC_PYTHON" -B \
    -m unittest discover -s outputs/code -p test_mz_null_indexed_mvc.py -v \
    > "$EP_NULL_INDEXED_MVC_CHECKS/indexed_tests.txt" 2>&1
  printf 'NULL_INDEXED_MVC_TESTS_PASS\n'
  timeout --signal=TERM --kill-after=10s 60s "$EP_NULL_INDEXED_MVC_PYTHON" -B \
    -m unittest discover -s outputs/code -p test_mz_null_repaired_menu.py -v \
    > "$EP_NULL_INDEXED_MVC_CHECKS/menu_tests.txt" 2>&1
  printf 'NULL_REPAIRED_MENU_TESTS_PASS\n'
  EP_NULL_INDEXED_MVC_ASTRA="$GROUP_HOME/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv/bin/astra-tools"
  if timeout --signal=TERM --kill-after=10s 30s "$EP_NULL_INDEXED_MVC_ASTRA" validate \
      outputs/astra/v0.0.14/astra.yaml --json \
      > "outputs/astra/v0.0.14/validation_20261005_null_indexed_mvc_${SLURM_ARRAY_JOB_ID}.json"; then
    printf 'NULL_INDEXED_MVC_ASTRA_PASS\n'
  else
    printf 'NULL_INDEXED_MVC_ASTRA_OPEN: coordinator must inspect output\n' >&2
  fi
  timeout --signal=TERM --kill-after=10s 60s "$EP_NULL_INDEXED_MVC_PYTHON" -B \
    outputs/code/summarize_mz_null_repaired_menu.py \
    --implementation outputs/code/behavioral_implementation.json \
    --output "outputs/runs/behavioral_mechanism_kill_test/20261005-M0-null0-repaired-menu-${SLURM_ARRAY_JOB_ID}.json"
  printf 'NULL0_REPAIRED_MENU_COMPLETED\n'
fi
exec timeout --signal=TERM --kill-after=10s 8400s "$EP_NULL_INDEXED_MVC_PYTHON" -B \
  outputs/code/run_mz_null_indexed_mvc.py \
  --replicate-index "$EP_NULL_INDEXED_MVC_INDEX" \
  --implementation outputs/code/behavioral_implementation.json \
  --output-dir "outputs/runs/behavioral_mechanism_kill_test/20261005-M0-null-MVC-indexed-array-${SLURM_ARRAY_JOB_ID}/replicate${EP_NULL_INDEXED_MVC_INDEX}" \
  --max-seconds 8400
