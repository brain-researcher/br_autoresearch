#!/bin/bash -l
#SBATCH --job-name=ep01-null-extra-mvc
#SBATCH --partition=normal
#SBATCH --constraint=CPU_SKU:7502
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=02:30:00
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/behavioral_mechanism_kill_test/20261005-M0-null-additional-MVC-%A_%a.out
#SBATCH --error=outputs/runs/behavioral_mechanism_kill_test/20261005-M0-null-additional-MVC-%A_%a.err

# Selected after passing additional-adapter tests, successful baseline
# completions and actual source replay in companion rival46595106.
# Exact selected array108,176 only. Index108 bundles changed menu-pool
# admission checks and ASTRA; these independent checks do not block its fit.
# Maximum130+40+8410=8580s<9000s. No adapter-test repetition or automatic retry.
set -euo pipefail
EP_NULL_EXTRA_MVC_HOST=$(hostname)
printf 'HOST=%s SLURM_JOB_ID=%s ARRAY=%s INDEX=%s NTASKS=%s CPUS=%s\n' \
  "$EP_NULL_EXTRA_MVC_HOST" "${SLURM_JOB_ID-}" "${SLURM_ARRAY_JOB_ID-}" \
  "${SLURM_ARRAY_TASK_ID-}" "${SLURM_NTASKS-}" "${SLURM_CPUS_PER_TASK-}"
if [[ -z "${SLURM_JOB_ID-}" || "$EP_NULL_EXTRA_MVC_HOST" == *-ln* || \
      "${SLURM_NTASKS-}" != 1 || "${SLURM_CPUS_PER_TASK-}" != 1 || -z "${SLURM_ARRAY_JOB_ID-}" ]]; then exit 2; fi
case "${SLURM_ARRAY_TASK_ID-}" in 108|176) ;; *) exit 2 ;; esac
cd "$SLURM_SUBMIT_DIR"
module purge
module load devel python/3.12.1 math py-numpy/1.26.3 py-scikit-learn/1.5.1_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
EP_NULL_EXTRA_MVC_PYTHON=$(command -v python3.12)
"$EP_NULL_EXTRA_MVC_PYTHON" --version
EP_NULL_EXTRA_MVC_AUX_EXIT=0
if [[ "$SLURM_ARRAY_TASK_ID" == 108 ]]; then
  bash -n outputs/code/mz_null_additional_completed_mvc.sbatch
  EP_NULL_EXTRA_MVC_CHECKS="outputs/runs/behavioral_mechanism_kill_test/20261005-M0-null-additional-menu-checks-${SLURM_ARRAY_JOB_ID}"
  mkdir "$EP_NULL_EXTRA_MVC_CHECKS"
  if timeout --signal=TERM --kill-after=10s 120s "$EP_NULL_EXTRA_MVC_PYTHON" -B \
      -m unittest discover -s outputs/code -p test_mz_null_indexed_menu.py -v \
      > "$EP_NULL_EXTRA_MVC_CHECKS/tests.txt" 2>&1; then
    printf 'ADDITIONAL_INDEXED_MENU_TESTS_PASS\n'
  else
    EP_NULL_EXTRA_MVC_AUX_EXIT=1
    printf 'ADDITIONAL_INDEXED_MENU_TESTS_FAILED: preserve result; independently qualified MVC fit continues\n' >&2
  fi
  EP_NULL_EXTRA_MVC_ASTRA="$GROUP_HOME/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv/bin/astra-tools"
  if timeout --signal=TERM --kill-after=10s 30s "$EP_NULL_EXTRA_MVC_ASTRA" validate \
      outputs/astra/v0.0.14/astra.yaml --json \
      > "outputs/astra/v0.0.14/validation_20261005_null_completed_menu_${SLURM_ARRAY_JOB_ID}.json"; then
    printf 'ADDITIONAL_INDEXED_MENU_ASTRA_PASS\n'
  else
    printf 'ADDITIONAL_INDEXED_MENU_ASTRA_OPEN: coordinator must inspect output\n' >&2
  fi
fi
if timeout --signal=TERM --kill-after=10s 8400s "$EP_NULL_EXTRA_MVC_PYTHON" -B \
    outputs/code/run_mz_null_additional_completed_rival.py \
    --replicate-index "$SLURM_ARRAY_TASK_ID" --model MVC \
    --implementation outputs/code/behavioral_implementation.json \
    --output-dir "outputs/runs/behavioral_mechanism_kill_test/20261005-M0-null-MVC-additional-array-${SLURM_ARRAY_JOB_ID}/replicate${SLURM_ARRAY_TASK_ID}" \
    --max-seconds 8400; then
  exit "$EP_NULL_EXTRA_MVC_AUX_EXIT"
else
  EP_NULL_EXTRA_MVC_FIT_EXIT=$?
  exit "$EP_NULL_EXTRA_MVC_FIT_EXIT"
fi
