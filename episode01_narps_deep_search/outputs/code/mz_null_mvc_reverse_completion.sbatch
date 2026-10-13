#!/bin/bash -l
#SBATCH --job-name=ep01-null-mvc-reverse
#SBATCH --partition=normal
#SBATCH --constraint=CPU_SKU:7502
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=02:05:00
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/behavioral_mechanism_kill_test/20261005-M0-null-MVC-reverse-%A_%a.out
#SBATCH --error=outputs/runs/behavioral_mechanism_kill_test/20261005-M0-null-MVC-reverse-%A_%a.err

# Exact terminal null12/31/37/109/135/140/159/164/183/188 second-half1000 exhaustion only. Preserve each first
# under its original1000 runtime and rescore; one fresh reverse1500 fit.
# Tests130 + ASTRA40 + fit6610 = 6780s; reserved auxiliary work460s gives
# 7240s < 7500s. Root owns later auxiliary commands and execution selection.
# Qualification183 runs checks first;31 is fit-only after observed checks/replay,
# with root's --time=01:55:00 override. No dispatch or retry occurs here.
set -euo pipefail
EP_NULL_REVERSE_HOST=$(hostname)
printf 'HOST=%s SLURM_JOB_ID=%s ARRAY=%s INDEX=%s CPUS=%s\n' "$EP_NULL_REVERSE_HOST" "${SLURM_JOB_ID-}" \
  "${SLURM_ARRAY_JOB_ID-}" "${SLURM_ARRAY_TASK_ID-}" "${SLURM_CPUS_PER_TASK-}"
if [[ -z "${SLURM_JOB_ID-}" || "$EP_NULL_REVERSE_HOST" == *-ln* || -z "${SLURM_ARRAY_JOB_ID-}" || \
      "${SLURM_NTASKS-}" != 1 || "${SLURM_CPUS_PER_TASK-}" != 1 ]]; then exit 2; fi
case "${SLURM_ARRAY_TASK_ID-}" in 12|31|37|109|135|140|159|164|183|188) ;; *) exit 2 ;; esac
cd "$SLURM_SUBMIT_DIR"
module purge
module load devel python/3.12.1 math py-numpy/1.26.3 py-scikit-learn/1.5.1_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
EP_NULL_REVERSE_PYTHON=$(command -v python3.12)
"$EP_NULL_REVERSE_PYTHON" --version
bash -n outputs/code/mz_null_mvc_reverse_completion.sbatch
if [[ "$SLURM_ARRAY_TASK_ID" == 183 ]]; then
EP_NULL_REVERSE_CHECKS="outputs/runs/behavioral_mechanism_kill_test/20261005-M0-null-MVC-reverse-checks-${SLURM_ARRAY_JOB_ID}"
mkdir "$EP_NULL_REVERSE_CHECKS"
timeout --signal=TERM --kill-after=10s 120s "$EP_NULL_REVERSE_PYTHON" -B \
  -m unittest discover -s outputs/code -p test_mz_null_indexed_mvc_reverse_completion.py -v \
  > "$EP_NULL_REVERSE_CHECKS/tests.txt" 2>&1
printf 'NULL_MVC_REVERSE_TESTS_PASS\n'
EP_NULL_REVERSE_ASTRA="$GROUP_HOME/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv/bin/astra-tools"
if timeout --signal=TERM --kill-after=10s 30s "$EP_NULL_REVERSE_ASTRA" validate \
    outputs/astra/v0.0.14/astra.yaml --json \
    > "outputs/astra/v0.0.14/validation_20261005_null183_mvc_reverse_${SLURM_ARRAY_JOB_ID}.json"; then
  printf 'NULL_MVC_REVERSE_ASTRA_PASS\n'
else
  printf 'NULL_MVC_REVERSE_ASTRA_OPEN: coordinator must inspect output\n' >&2
fi
# These auxiliary diagnostics never block the already-qualified scientific fit.
# Exact terminal selection observed by root at2026-10-05 09:11:18UTC.
EP_NULL_REVERSE_POOL_READY=true
if timeout --signal=TERM --kill-after=10s 60s "$EP_NULL_REVERSE_PYTHON" -B \
    -m unittest discover -s outputs/code -p test_mz_null_mvc_repair_admission.py -v \
    > "$EP_NULL_REVERSE_CHECKS/repair_admission_tests.txt" 2>&1; then
  printf 'NULL_MVC_REPAIR_ADMISSION_TESTS_PASS\n'
else
  EP_NULL_REVERSE_POOL_READY=false
  printf 'NULL_MVC_REPAIR_ADMISSION_TESTS_FAILED: retain output; continue selected reverse fit\n' >&2
fi
if timeout --signal=TERM --kill-after=10s 60s "$EP_NULL_REVERSE_PYTHON" -B \
    -m unittest discover -s outputs/code -p test_mz_null_indexed_menu.py -v \
    > "$EP_NULL_REVERSE_CHECKS/indexed_pool_tests.txt" 2>&1; then
  printf 'NULL_INDEXED_POOL_TESTS_PASS\n'
else
  EP_NULL_REVERSE_POOL_READY=false
  printf 'NULL_INDEXED_POOL_TESTS_FAILED: retain output; continue selected reverse fit\n' >&2
fi
if [[ "$EP_NULL_REVERSE_POOL_READY" == true ]]; then
  if timeout --signal=TERM --kill-after=10s 240s "$EP_NULL_REVERSE_PYTHON" -B \
      outputs/code/summarize_mz_null_indexed_menu.py \
      --terminal-indices 0,1,2,3,4,6,7,8,9,11,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,32,33,35,36,38,39,40,41,42,44,45,46,47,48,49,50,51,52,53,54,55,56,57,59,60,63,64,65,66,67,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,85,86,87,88,89,90,92,93,94,95,96,97,98,99,100,101,102,103,105,106,107,108,110,111,112,113,114,115,116,117,118,119,120,122,123,124,125,126,127,128,129,130,131,133,136,137,138,139,141,142,143,144,145,146,147,149,150,152,153,154,155,156,158,160,161,162,163,165,166,167,169,170,171,172,173,174,177,178,180,181,184,185,187,189,190,192,193,194,195,196,197,198,199 \
      --additional-mvc-array-job-id 46596399 \
      --mvc-repair-arrays '43:46599566,61:46599991,168:46599991' \
      --implementation outputs/code/behavioral_implementation.json \
      --output "$EP_NULL_REVERSE_CHECKS/null_indexed_menu.json" \
      > "$EP_NULL_REVERSE_CHECKS/null_stdout.txt" 2>&1; then
    printf 'NULL_166_MENU_POOL_DONE\n'
  else
    printf 'NULL_MENU_POOL_FAILED: retain output; continue selected reverse fit\n' >&2
  fi
fi
if timeout --signal=TERM --kill-after=10s 60s "$EP_NULL_REVERSE_PYTHON" -B \
    outputs/code/summarize_mz_repaired_menu_wave.py \
    --terminal-indices 0,36,42,73,83,86,92,105,163,196 --fresh-array-job-id 46597540 --reverse-array-job-id 46597546 \
    --implementation outputs/code/behavioral_implementation.json \
    --output "$EP_NULL_REVERSE_CHECKS/mz_repaired_menu.json" \
    > "$EP_NULL_REVERSE_CHECKS/mz_stdout.txt" 2>&1; then
  printf 'MZ_10_MENU_POOL_DONE\n'
else
  printf 'MZ_MENU_POOL_FAILED: retain output; continue selected reverse fit\n' >&2
fi
fi
if [[ "$SLURM_ARRAY_TASK_ID" == 12 ]]; then
EP_NULL_REVERSE_POOL_CHECKS="outputs/runs/behavioral_mechanism_kill_test/20261005-M0-null-MVC-reverse-pool-checks-${SLURM_ARRAY_JOB_ID}"
# Qualified numerical code is unchanged; these checks govern pool admission.
# New9 checks70 + affected20 checks70 + ASTRA40 + fit6610 = 6790 < 6900s.
if mkdir "$EP_NULL_REVERSE_POOL_CHECKS"; then
  if timeout --signal=TERM --kill-after=10s 60s "$EP_NULL_REVERSE_PYTHON" -B \
      -m unittest discover -s outputs/code -p test_mz_null_mvc_reverse_admission.py -v \
      > "$EP_NULL_REVERSE_POOL_CHECKS/reverse_admission_tests.txt" 2>&1; then
    printf 'NULL_MVC_REVERSE_ADMISSION_TESTS_PASS\n'
  else
    printf 'NULL_MVC_REVERSE_ADMISSION_TESTS_FAILED: retain output; continue qualified fit\n' >&2
  fi
  if timeout --signal=TERM --kill-after=10s 60s env \
      PYTHONPATH="$PWD/outputs/code${PYTHONPATH:+:$PYTHONPATH}" "$EP_NULL_REVERSE_PYTHON" -B \
      -m unittest test_mz_null_indexed_menu test_mz_null_mvc_repair_admission -v \
      > "$EP_NULL_REVERSE_POOL_CHECKS/affected_pool_tests.txt" 2>&1; then
    printf 'NULL_MVC_AFFECTED_POOL_TESTS_PASS\n'
  else
    printf 'NULL_MVC_AFFECTED_POOL_TESTS_FAILED: retain output; continue qualified fit\n' >&2
  fi
  EP_NULL_REVERSE_POOL_ASTRA="$GROUP_HOME/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv/bin/astra-tools"
  if timeout --signal=TERM --kill-after=10s 30s "$EP_NULL_REVERSE_POOL_ASTRA" validate \
      outputs/astra/v0.0.14/astra.yaml --json > "$EP_NULL_REVERSE_POOL_CHECKS/astra_validation.json"; then
    printf 'NULL_MVC_REVERSE_POOL_ASTRA_PASS\n'
  else
    printf 'NULL_MVC_REVERSE_POOL_ASTRA_OPEN: retain output; continue qualified fit\n' >&2
  fi
else
  printf 'NULL_MVC_REVERSE_POOL_CHECK_DIRECTORY_FAILED: continue qualified fit\n' >&2
fi
fi
if [[ "$SLURM_ARRAY_TASK_ID" == 140 ]]; then
# Metadata-only ASTRA update; no repeated numerical or pool test suite.
# ASTRA40 + fit6610 = 6650 < 6900s.
EP_NULL_REVERSE_EXTRA_ASTRA="$GROUP_HOME/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv/bin/astra-tools"
if timeout --signal=TERM --kill-after=10s 30s "$EP_NULL_REVERSE_EXTRA_ASTRA" validate \
    outputs/astra/v0.0.14/astra.yaml --json \
    > "outputs/astra/v0.0.14/validation_20261005_null140_mvc_reverse_${SLURM_ARRAY_JOB_ID}.json"; then
  printf 'NULL_MVC_REVERSE_EXTRA_ASTRA_PASS\n'
else
  printf 'NULL_MVC_REVERSE_EXTRA_ASTRA_OPEN: retain output; continue qualified fit\n' >&2
fi
fi
exec timeout --signal=TERM --kill-after=10s 6600s "$EP_NULL_REVERSE_PYTHON" -B \
  outputs/code/run_mz_null_indexed_mvc_reverse_completion.py \
  --replicate-index "$SLURM_ARRAY_TASK_ID" \
  --implementation outputs/code/behavioral_implementation.json \
  --output-dir "outputs/runs/behavioral_mechanism_kill_test/20261005-M0-null-MVC-reverse-completion-array-${SLURM_ARRAY_JOB_ID}/replicate${SLURM_ARRAY_TASK_ID}" \
  --max-seconds 6600
