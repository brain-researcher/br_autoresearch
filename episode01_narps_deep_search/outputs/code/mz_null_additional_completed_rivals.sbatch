#!/bin/bash -l
#SBATCH --job-name=ep01-null-extra-rivals
#SBATCH --partition=normal
#SBATCH --constraint=CPU_SKU:7502
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:30:00
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/behavioral_mechanism_kill_test/20261005-M0-null-additional-completed-rivals-%j.out
#SBATCH --error=outputs/runs/behavioral_mechanism_kill_test/20261005-M0-null-additional-completed-rivals-%j.err

# PREPARED: root must establish successful108/176 baselines and no competing
# valid-source component fit before selection. Old immutable incomplete-source
# components can only reject before fitting; preserve their parent jobs.
# Four workers:
# MC then MV for each index, plus MVZ for each index. MVC is separately selected.
# Changed adapter and menu-pool checks run inside this substantive allocation.
# A first real pool uses only terminal menus0/127; other indices remain unknown.
# Bound130+130+40+70+max(2*2410,3310)=5190s<5400s. No retry or polling.
set -euo pipefail
EP_NULL_EXTRA_RIVAL_HOST=$(hostname)
printf 'HOST=%s SLURM_JOB_ID=%s NTASKS=%s CPUS=%s\n' \
  "$EP_NULL_EXTRA_RIVAL_HOST" "${SLURM_JOB_ID-}" "${SLURM_NTASKS-}" "${SLURM_CPUS_PER_TASK-}"
if [[ -z "${SLURM_JOB_ID-}" || "$EP_NULL_EXTRA_RIVAL_HOST" == *-ln* || "${SLURM_CPUS_PER_TASK-}" != 1 ]]; then
  printf 'An actual compute allocation with one CPU per task is required\n' >&2
  exit 2
fi
cd "$SLURM_SUBMIT_DIR"
if [[ "${1-}" == --worker ]]; then
  if (( $# != 3 )) || [[ "${SLURM_NTASKS-}" != 1 ]]; then exit 2; fi
  case "$2" in 108|176) ;; *) exit 2 ;; esac
  EP_NULL_EXTRA_RIVAL_INDEX="$2"
  case "$3" in
    MC_MV) EP_NULL_EXTRA_RIVAL_MODELS=(MC MV); EP_NULL_EXTRA_RIVAL_SECONDS=2400 ;;
    MVZ) EP_NULL_EXTRA_RIVAL_MODELS=(MVZ); EP_NULL_EXTRA_RIVAL_SECONDS=3300 ;;
    *) exit 2 ;;
  esac
  EP_NULL_EXTRA_RIVAL_EXIT=0
  for EP_NULL_EXTRA_RIVAL_MODEL in "${EP_NULL_EXTRA_RIVAL_MODELS[@]}"; do
    if timeout --signal=TERM --kill-after=10s "${EP_NULL_EXTRA_RIVAL_SECONDS}s" "$EP_NULL_EXTRA_RIVAL_PYTHON" -B \
        outputs/code/run_mz_null_additional_completed_rival.py \
        --replicate-index "$EP_NULL_EXTRA_RIVAL_INDEX" --model "$EP_NULL_EXTRA_RIVAL_MODEL" \
        --implementation outputs/code/behavioral_implementation.json \
        --output-dir "outputs/runs/behavioral_mechanism_kill_test/20261005-M0-null${EP_NULL_EXTRA_RIVAL_INDEX}-${EP_NULL_EXTRA_RIVAL_MODEL}-completed-baseline-${SLURM_JOB_ID}" \
        --max-seconds "$EP_NULL_EXTRA_RIVAL_SECONDS"; then
      printf 'ADDITIONAL_COMPLETED_RIVAL_PASS index=%s model=%s\n' "$EP_NULL_EXTRA_RIVAL_INDEX" "$EP_NULL_EXTRA_RIVAL_MODEL"
    else
      EP_NULL_EXTRA_RIVAL_EXIT=1
      printf 'ADDITIONAL_COMPLETED_RIVAL_FAILED index=%s model=%s; preserve attempt, no retry\n' \
        "$EP_NULL_EXTRA_RIVAL_INDEX" "$EP_NULL_EXTRA_RIVAL_MODEL" >&2
    fi
  done
  exit "$EP_NULL_EXTRA_RIVAL_EXIT"
fi
if (( $# != 0 )) || [[ "${SLURM_NTASKS-}" != 4 ]]; then exit 2; fi
module purge
module load devel python/3.12.1 math py-numpy/1.26.3 py-scikit-learn/1.5.1_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
EP_NULL_EXTRA_RIVAL_PYTHON=$(command -v python3.12)
export EP_NULL_EXTRA_RIVAL_PYTHON
"$EP_NULL_EXTRA_RIVAL_PYTHON" --version
bash -n outputs/code/mz_null_additional_completed_rivals.sbatch
bash -n outputs/code/mz_null_additional_completed_mvc.sbatch
EP_NULL_EXTRA_RIVAL_CHECKS="outputs/runs/behavioral_mechanism_kill_test/20261005-M0-null-additional-completed-rival-checks-${SLURM_JOB_ID}"
mkdir "$EP_NULL_EXTRA_RIVAL_CHECKS"
timeout --signal=TERM --kill-after=10s 120s "$EP_NULL_EXTRA_RIVAL_PYTHON" -B \
  -m unittest discover -s outputs/code -p test_mz_null_additional_completed_rival.py -v \
  > "$EP_NULL_EXTRA_RIVAL_CHECKS/rival_tests.txt" 2>&1
printf 'ADDITIONAL_COMPLETED_RIVAL_TESTS_PASS\n'
EP_NULL_EXTRA_RIVAL_AUX_EXIT=0
EP_NULL_EXTRA_MENU_READY=1
if timeout --signal=TERM --kill-after=10s 120s "$EP_NULL_EXTRA_RIVAL_PYTHON" -B \
    -m unittest discover -s outputs/code -p test_mz_null_indexed_menu.py -v \
    > "$EP_NULL_EXTRA_RIVAL_CHECKS/menu_tests.txt" 2>&1; then
  printf 'INDEXED_NULL_MENU_TESTS_PASS\n'
else
  EP_NULL_EXTRA_RIVAL_AUX_EXIT=1
  EP_NULL_EXTRA_MENU_READY=0
  printf 'INDEXED_NULL_MENU_TESTS_FAILED: skip pool; independent qualified rival fits continue\n' >&2
fi
EP_NULL_EXTRA_RIVAL_ASTRA="$GROUP_HOME/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv/bin/astra-tools"
if timeout --signal=TERM --kill-after=10s 30s "$EP_NULL_EXTRA_RIVAL_ASTRA" validate \
    outputs/astra/v0.0.14/astra.yaml --json \
    > "outputs/astra/v0.0.14/validation_20261005_null_additional_completed_rivals_${SLURM_JOB_ID}.json"; then
  printf 'ADDITIONAL_COMPLETED_RIVAL_ASTRA_PASS\n'
else
  printf 'ADDITIONAL_COMPLETED_RIVAL_ASTRA_OPEN: coordinator must inspect output\n' >&2
fi
if (( EP_NULL_EXTRA_MENU_READY == 1 )); then
  if timeout --signal=TERM --kill-after=10s 60s "$EP_NULL_EXTRA_RIVAL_PYTHON" -B \
      outputs/code/summarize_mz_null_indexed_menu.py --terminal-indices 0,127 \
      --implementation outputs/code/behavioral_implementation.json \
      --output "outputs/runs/behavioral_mechanism_kill_test/20261005-M0-null-indexed-menu-initial-${SLURM_JOB_ID}.json"; then
    printf 'INITIAL_INDEXED_NULL_MENU_COMPLETED\n'
  else
    EP_NULL_EXTRA_RIVAL_AUX_EXIT=1
    printf 'INITIAL_INDEXED_NULL_MENU_FAILED: preserve result; independent qualified rival fits continue\n' >&2
  fi
fi
EP_NULL_EXTRA_RIVAL_PIDS=()
for EP_NULL_EXTRA_RIVAL_INDEX in 108 176; do
  for EP_NULL_EXTRA_RIVAL_MODE in MC_MV MVZ; do
    srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=2G \
      bash outputs/code/mz_null_additional_completed_rivals.sbatch --worker "$EP_NULL_EXTRA_RIVAL_INDEX" "$EP_NULL_EXTRA_RIVAL_MODE" \
      > "outputs/runs/behavioral_mechanism_kill_test/20261005-M0-null${EP_NULL_EXTRA_RIVAL_INDEX}-${EP_NULL_EXTRA_RIVAL_MODE}-completed-baseline-${SLURM_JOB_ID}.out" \
      2> "outputs/runs/behavioral_mechanism_kill_test/20261005-M0-null${EP_NULL_EXTRA_RIVAL_INDEX}-${EP_NULL_EXTRA_RIVAL_MODE}-completed-baseline-${SLURM_JOB_ID}.err" &
    EP_NULL_EXTRA_RIVAL_PIDS+=("$!")
  done
done
EP_NULL_EXTRA_RIVAL_EXIT="$EP_NULL_EXTRA_RIVAL_AUX_EXIT"
for EP_NULL_EXTRA_RIVAL_PID in "${EP_NULL_EXTRA_RIVAL_PIDS[@]}"; do
  if ! wait "$EP_NULL_EXTRA_RIVAL_PID"; then EP_NULL_EXTRA_RIVAL_EXIT=1; fi
done
exit "$EP_NULL_EXTRA_RIVAL_EXIT"
