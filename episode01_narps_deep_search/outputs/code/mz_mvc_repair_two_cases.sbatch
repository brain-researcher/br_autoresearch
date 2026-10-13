#!/bin/bash -l
#SBATCH --job-name=ep01-mvc-repair-cases
#SBATCH --partition=normal
#SBATCH --constraint=CPU_SKU:7502
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=03:40:00
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-MVC-repair-cases-%j.out
#SBATCH --error=outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-MVC-repair-cases-%j.err

# Selected after index42 completion46587905 succeeds at1067 reverse sweeps.
# Root records these two diagnosed cases and costs before submitting.
# Index2 exhausted the first half; index3 exhausted only its second half.
# Fresh failed halves use1500; retain index3's completed original500 first fit.
# Maximum130+70+70+40+70+100+12610=13090s<13200s, including timeout graces.
# Preserve all sources, completed42 and the unresolved MCindex1 hold.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
EP_MVC_CASE_HOST=$(hostname)
printf 'HOST=%s SLURM_JOB_ID=%s NTASKS=%s CPUS_PER_TASK=%s INDICES=2,3\n' \
  "$EP_MVC_CASE_HOST" "${SLURM_JOB_ID-}" "${SLURM_NTASKS-}" "${SLURM_CPUS_PER_TASK-}"
if [[ -z "${SLURM_JOB_ID-}" || "$EP_MVC_CASE_HOST" == *-ln* || \
      "${SLURM_NTASKS-}" != 2 || "${SLURM_CPUS_PER_TASK-}" != 1 ]]; then
  printf 'Requires the selected two-CPU compute allocation\n' >&2
  exit 2
fi
module purge
module load devel python/3.12.1 math py-numpy/1.26.3 py-scikit-learn/1.5.1_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
EP_MVC_CASE_PYTHON=$(command -v python3.12)
"$EP_MVC_CASE_PYTHON" --version
bash -n outputs/code/mz_mvc_repair_two_cases.sbatch
EP_MVC_CASE_CHECKS="outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-MVC-repair-cases-checks-${SLURM_JOB_ID}"
mkdir "$EP_MVC_CASE_CHECKS"
timeout --signal=TERM --kill-after=10s 120s "$EP_MVC_CASE_PYTHON" -B \
  -m unittest discover -s outputs/code -p test_mz_mvc_convergence_repair.py -v \
  > "$EP_MVC_CASE_CHECKS/repair_tests.txt" 2>&1
printf 'MVC_REPAIR_CASES_TESTS_PASS\n'
EP_MVC_CASE_EXIT=0
EP_MVC_CASE_MENU_OK=1
if timeout --signal=TERM --kill-after=10s 60s "$EP_MVC_CASE_PYTHON" -B \
    -m unittest discover -s outputs/code -p test_mz_mvc_reverse_completion.py -v \
    > "$EP_MVC_CASE_CHECKS/reverse_tests.txt" 2>&1; then
  printf 'MVC_REVERSE_COMPLETION_TESTS_PASS\n'
else
  EP_MVC_CASE_EXIT=1
  EP_MVC_CASE_MENU_OK=0
fi
if timeout --signal=TERM --kill-after=10s 60s "$EP_MVC_CASE_PYTHON" -B \
    -m unittest discover -s outputs/code -p test_mz_mvc_repaired_menu.py -v \
    > "$EP_MVC_CASE_CHECKS/menu_tests.txt" 2>&1; then
  printf 'MVC_REPAIRED_MENU_TESTS_PASS\n'
else
  EP_MVC_CASE_EXIT=1
  EP_MVC_CASE_MENU_OK=0
fi
EP_MVC_CASE_ASTRA="$GROUP_HOME/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv/bin/astra-tools"
if timeout --signal=TERM --kill-after=10s 30s "$EP_MVC_CASE_ASTRA" validate \
    outputs/astra/v0.0.14/astra.yaml --json \
    > "outputs/astra/v0.0.14/validation_20261005_mvc_repair_cases_${SLURM_JOB_ID}.json"; then
  printf 'MVC_REPAIR_CASES_ASTRA_PASS\n'
else
  printf 'MVC_REPAIR_CASES_ASTRA_OPEN: coordinator must inspect validation output\n' >&2
fi
if [[ "$EP_MVC_CASE_MENU_OK" == 1 ]]; then
  if timeout --signal=TERM --kill-after=10s 60s "$EP_MVC_CASE_PYTHON" -B \
    outputs/code/summarize_mz_mvc_repaired_menu.py \
    --implementation outputs/code/behavioral_implementation.json \
    --output "outputs/runs/behavioral_mechanism_kill_test/20261005-MZ42-repaired-menu-${SLURM_JOB_ID}.json"; then
    printf 'MVC42_REPAIRED_MENU_COMPLETED\n'
  else
    EP_MVC_CASE_EXIT=1
    printf 'MVC42_REPAIRED_MENU_INCOMPLETE: preserve error; independent selected cases continue\n' >&2
  fi
else
  printf 'MVC42_REPAIRED_MENU_TESTS_FAILED: preserve errors; independent selected cases continue\n' >&2
fi
# Twelve indexed-null menu checks already passed in46596399. Root fixed this
# set from terminal component parents at08:19:48UTC; no source discovery here.
if timeout --signal=TERM --kill-after=10s 90s "$EP_MVC_CASE_PYTHON" -B \
    outputs/code/summarize_mz_null_indexed_menu.py \
    --terminal-indices 0,4,45,54,78,90,115,122,127,128,141,199 \
    --additional-mvc-array-job-id 46596399 \
    --implementation outputs/code/behavioral_implementation.json \
    --output "outputs/runs/behavioral_mechanism_kill_test/20261005-M0-null-indexed-menu-expanded-${SLURM_JOB_ID}.json"; then
  printf 'EXPANDED_INDEXED_NULL_MENU_COMPLETED\n'
else
  EP_MVC_CASE_EXIT=1
  printf 'EXPANDED_INDEXED_NULL_MENU_INCOMPLETE: preserve error; independent selected cases continue\n' >&2
fi
EP_MVC_CASE_PIDS=()
EP_MVC_CASE_INDICES=(2 3)
for EP_MVC_CASE_INDEX in "${EP_MVC_CASE_INDICES[@]}"; do
  case "$EP_MVC_CASE_INDEX" in
    2) EP_MVC_CASE_SECONDS=12600 ;;
    3) EP_MVC_CASE_SECONDS=6600 ;;
    *) exit 2 ;;
  esac
  srun --nodes=1 --ntasks=1 --cpus-per-task=1 --exclusive --exact --mem=2G --cpu-bind=cores \
    --output="outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-MVC-repair-${SLURM_JOB_ID}-${EP_MVC_CASE_INDEX}.out" \
    --error="outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-MVC-repair-${SLURM_JOB_ID}-${EP_MVC_CASE_INDEX}.err" \
    timeout --signal=TERM --kill-after=10s "${EP_MVC_CASE_SECONDS}s" "$EP_MVC_CASE_PYTHON" -B \
    outputs/code/run_mz_mvc_convergence_repair.py --replicate-index "$EP_MVC_CASE_INDEX" \
    --implementation outputs/code/behavioral_implementation.json \
    --output-dir "outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-MVC-repair-cases-${SLURM_JOB_ID}/replicate${EP_MVC_CASE_INDEX}" \
    --max-seconds "$EP_MVC_CASE_SECONDS" &
  EP_MVC_CASE_PIDS+=("$!")
done
for EP_MVC_CASE_SLOT in "${!EP_MVC_CASE_PIDS[@]}"; do
  if wait "${EP_MVC_CASE_PIDS[$EP_MVC_CASE_SLOT]}"; then
    printf 'MVC_REPAIR_CASE_COMPLETED index=%s\n' "${EP_MVC_CASE_INDICES[$EP_MVC_CASE_SLOT]}"
  else
    EP_MVC_CASE_EXIT=1
    printf 'MVC_REPAIR_CASE_FAILED index=%s: preserve attempt as unknown; no retry\n' \
      "${EP_MVC_CASE_INDICES[$EP_MVC_CASE_SLOT]}" >&2
  fi
done
exit "$EP_MVC_CASE_EXIT"
