#!/bin/bash -l
#SBATCH --job-name=ep01-twice-MVC-repair
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=100
#SBATCH --no-requeue

# Root selects exact terminal indices after successful index0 qualification.
# The separately selected first task aggregates new checks and summaries.
set -euo pipefail
EP_INDEXED_CHECKS=${1:?Pass 1 for the selected first check task, otherwise 0}
[[ "$EP_INDEXED_CHECKS" == 0 || "$EP_INDEXED_CHECKS" == 1 ]]
cd "$SLURM_SUBMIT_DIR"
EP_INDEXED_HOST=$(hostname)
printf 'HOST=%s JOB=%s ARRAY=%s INDEX=%s CPUS=%s CHECKS=%s\n' "$EP_INDEXED_HOST" "$SLURM_JOB_ID" "$SLURM_ARRAY_JOB_ID" "$SLURM_ARRAY_TASK_ID" "$SLURM_CPUS_PER_TASK" "$EP_INDEXED_CHECKS"
[[ "$SLURM_JOB_ID" =~ ^[0-9]+$ && "$EP_INDEXED_HOST" != *-ln* && "$SLURM_CPUS_PER_TASK" == 4 ]]
[[ "$SLURM_ARRAY_TASK_ID" =~ ^[0-9]+$ && "$SLURM_ARRAY_TASK_ID" -ge 1 && "$SLURM_ARRAY_TASK_ID" -le 199 ]]
module purge
module load devel python/3.12.1 math py-numpy/1.26.3 py-scikit-learn/1.5.1_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="outputs/code:outputs/tests${PYTHONPATH:+:$PYTHONPATH}"
EP_INDEXED_ROOT="outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-twice-MVC-indexed-repair-${SLURM_ARRAY_JOB_ID}"
EP_INDEXED_TASK="$EP_INDEXED_ROOT/task${SLURM_ARRAY_TASK_ID}"
mkdir -p "$EP_INDEXED_ROOT"
mkdir "$EP_INDEXED_TASK"

if [[ "$EP_INDEXED_CHECKS" == 1 ]]; then
  # A120-minute parent is required for this first task's full bounded envelope.
  if timeout --signal=TERM --kill-after=10s 120s python3.12 -B -m unittest discover -s outputs/code -p test_twice_mz_mvc_repair_component.py -v >"$EP_INDEXED_TASK/component-tests.log" 2>&1; then
    printf 'INDEXED_COMPONENT_TEST_EXIT=0\n'
  else
    EP_INDEXED_TEST_EXIT=$?
    printf 'INDEXED_COMPONENT_TEST_EXIT=%s; scientific component not entered\n' "$EP_INDEXED_TEST_EXIT"
    exit "$EP_INDEXED_TEST_EXIT"
  fi
  # The independent summaries do not gate the qualified component fitter.
  if timeout --signal=TERM --kill-after=10s 240s python3.12 -B -m unittest -v test_resolved_fixed_menu_counts >"$EP_INDEXED_TASK/resolved-count-tests.log" 2>&1; then
    printf 'RESOLVED_COUNT_TEST_EXIT=0\n'
    if timeout --signal=TERM --kill-after=10s 260s python3.12 -B outputs/code/summarize_resolved_fixed_menu_counts.py --implementation outputs/code/behavioral_implementation.json --output "$EP_INDEXED_TASK/resolved-fixed-menu-counts.json" --max-seconds 240 >"$EP_INDEXED_TASK/resolved-count-summary.log" 2>&1; then
      printf 'RESOLVED_COUNT_REPORT_EXIT=0\n'
    else
      EP_INDEXED_COUNT_EXIT=$?
      printf 'RESOLVED_COUNT_REPORT_EXIT=%s; derived counts remain unresolved\n' "$EP_INDEXED_COUNT_EXIT"
    fi
  else
    EP_INDEXED_COUNT_TEST_EXIT=$?
    printf 'RESOLVED_COUNT_TEST_EXIT=%s; derived counts not entered\n' "$EP_INDEXED_COUNT_TEST_EXIT"
  fi
  if timeout --signal=TERM --kill-after=10s 240s python3.12 -B -m unittest -v test_twice_mz_repaired_pilot >"$EP_INDEXED_TASK/repaired-menu-tests.log" 2>&1; then
    printf 'REPAIRED_MENU_TEST_EXIT=0\n'
    if timeout --signal=TERM --kill-after=10s 260s python3.12 -B outputs/code/summarize_twice_mz_repaired_pilot.py --repair-report outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-twice-MVC-repair-46642546-attempt001/pilot/twice_mz_mvc_repair_pilot.json --implementation outputs/code/behavioral_implementation.json --output "$EP_INDEXED_TASK/twice0-repaired-menu.json" --max-seconds 240 >"$EP_INDEXED_TASK/repaired-menu-summary.log" 2>&1; then
      printf 'REPAIRED_MENU_REPORT_EXIT=0\n'
    else
      EP_INDEXED_MENU_EXIT=$?
      printf 'REPAIRED_MENU_REPORT_EXIT=%s; repaired menu remains unresolved\n' "$EP_INDEXED_MENU_EXIT"
    fi
  else
    EP_INDEXED_MENU_TEST_EXIT=$?
    printf 'REPAIRED_MENU_TEST_EXIT=%s; repaired menu not entered\n' "$EP_INDEXED_MENU_TEST_EXIT"
  fi
  EP_INDEXED_ASTRA="$GROUP_HOME/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv/bin/astra-tools"
  if timeout --signal=TERM --kill-after=10s 40s "$EP_INDEXED_ASTRA" validate outputs/astra/v0.0.14/astra.yaml --json >"$EP_INDEXED_TASK/astra-validation.json" 2>&1; then
    printf 'ASTRA_CHECK_EXIT=0\n'
  else
    EP_INDEXED_ASTRA_EXIT=$?
    printf 'ASTRA_CHECK_EXIT=%s; changed record validation remains open\n' "$EP_INDEXED_ASTRA_EXIT"
  fi
fi

srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=8G \
  timeout --signal=TERM --kill-after=20s 5420s python3.12 -B \
  outputs/code/run_twice_mz_mvc_repair_component.py \
  --replicate-index "$SLURM_ARRAY_TASK_ID" \
  --implementation outputs/code/behavioral_implementation.json \
  --output-dir "$EP_INDEXED_ROOT/replicate${SLURM_ARRAY_TASK_ID}" --max-seconds 5400
