#!/bin/bash -l
#SBATCH --job-name=ep01-null-rivals
#SBATCH --partition=normal
#SBATCH --constraint=CPU_SKU:7502
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=01:35:00
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/behavioral_mechanism_kill_test/20261005-M0-null-rivals-%j.out
#SBATCH --error=outputs/runs/behavioral_mechanism_kill_test/20261005-M0-null-rivals-%j.err

# Root submits MVZ first with --time=01:00:00 after the null pilot completes.
# MVC and MC_MV are selected only after that job's startup tests pass.
# MC_MV requires --dependency=afterany:46572753:46575931:46576234 so all
# selected MZ and null complex-rival attempts are terminal before summaries.
# No job submission, polling, automatic retry or new simulation occurs here.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
case "${1-}" in
  MVZ) EP_NULL_MODELS=(MVZ); EP_NULL_CAP=3300 ;;
  MVC) EP_NULL_MODELS=(MVC); EP_NULL_CAP=5400 ;;
  MC_MV) EP_NULL_MODELS=(MC MV); EP_NULL_CAP=2400 ;;
  *) printf 'Expected MVZ, MVC or MC_MV group\n' >&2; exit 2 ;;
esac
EP_NULL_GROUP="$1"
module purge
module load devel python/3.12.1 math py-numpy/1.26.3 py-scikit-learn/1.5.1_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
EP_NULL_PYTHON=$(command -v python3.12)
hostname
printf 'SLURM_JOB_ID=%s CPUS=%s NULL_RIVAL_GROUP=%s\n' "$SLURM_JOB_ID" "$SLURM_CPUS_PER_TASK" "$EP_NULL_GROUP"
"$EP_NULL_PYTHON" --version

if [[ "$EP_NULL_GROUP" == MVZ ]]; then
  bash -n outputs/code/mz_null_rivals.sbatch
  EP_NULL_CHECKS="outputs/runs/behavioral_mechanism_kill_test/20261005-M0-null-rival-checks-${SLURM_JOB_ID}"
  mkdir "$EP_NULL_CHECKS"
  timeout --signal=TERM --kill-after=10s 60s "$EP_NULL_PYTHON" -B \
    outputs/code/test_mz_null_rival.py -v > "$EP_NULL_CHECKS/tests.txt" 2>&1
  printf 'NULL_RIVAL_TESTS_PASS\n'
  EP_NULL_ASTRA="$GROUP_HOME/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv/bin/astra-tools"
  if timeout --signal=TERM --kill-after=10s 30s "$EP_NULL_ASTRA" validate \
      outputs/astra/v0.0.14/astra.yaml --json \
      > "outputs/astra/v0.0.14/validation_20261005_null_rivals_${SLURM_JOB_ID}.json"; then
    printf 'NULL_RIVAL_ASTRA_PASS\n'
  else
    printf 'NULL_RIVAL_ASTRA_OPEN: coordinator must inspect validation output\n' >&2
  fi
fi

if [[ "$EP_NULL_GROUP" == MC_MV ]]; then
  bash -n outputs/code/mz_null_rivals.sbatch
  EP_NULL_CHECKS="outputs/runs/behavioral_mechanism_kill_test/20261005-M0-null-menu-checks-${SLURM_JOB_ID}"
  mkdir "$EP_NULL_CHECKS"
  timeout --signal=TERM --kill-after=10s 60s "$EP_NULL_PYTHON" -B \
    outputs/code/test_mz_null_menu.py -v > "$EP_NULL_CHECKS/null_menu_tests.txt" 2>&1
  timeout --signal=TERM --kill-after=10s 60s "$EP_NULL_PYTHON" -B \
    outputs/code/test_mz_menu_wave.py -v > "$EP_NULL_CHECKS/mz_wave_tests.txt" 2>&1
  printf 'MENU_TESTS_PASS\n'
  EP_NULL_ASTRA="$GROUP_HOME/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv/bin/astra-tools"
  if timeout --signal=TERM --kill-after=10s 30s "$EP_NULL_ASTRA" validate \
      outputs/astra/v0.0.14/astra.yaml --json \
      > "outputs/astra/v0.0.14/validation_20261005_null_menu_${SLURM_JOB_ID}.json"; then
    printf 'NULL_MENU_ASTRA_PASS\n'
  else
    printf 'NULL_MENU_ASTRA_OPEN: coordinator must inspect validation output\n' >&2
  fi
  # This bounded loop builds one explicit argument; it never polls anything.
  EP_MZ_TERMINAL_INDICES=0
  for ((EP_MZ_INDEX=2; EP_MZ_INDEX<=199; EP_MZ_INDEX++)); do
    EP_MZ_TERMINAL_INDICES+=",${EP_MZ_INDEX}"
  done
  if timeout --signal=TERM --kill-after=10s 180s "$EP_NULL_PYTHON" -B \
      outputs/code/summarize_mz_menu_wave.py \
      --terminal-indices "$EP_MZ_TERMINAL_INDICES" \
      --output "outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-menu-wave-${SLURM_JOB_ID}.json"; then
    printf 'MZ_MENU_WAVE_POOL_COMPLETED\n'
  else
    printf 'MZ_MENU_WAVE_POOL_INCOMPLETE: preserve diagnostic error; independent null fits continue\n' >&2
  fi
fi

EP_NULL_GROUP_EXIT=0
for EP_NULL_MODEL in "${EP_NULL_MODELS[@]}"; do
  if timeout --signal=TERM --kill-after=10s "${EP_NULL_CAP}s" "$EP_NULL_PYTHON" -B \
      outputs/code/run_mz_null_rival.py --model "$EP_NULL_MODEL" \
      --source-dir outputs/runs/behavioral_mechanism_kill_test/20261005-M0-null-representative-46574266 \
      --implementation outputs/code/behavioral_implementation.json \
      --output-dir "outputs/runs/behavioral_mechanism_kill_test/20261005-M0-null-${EP_NULL_MODEL}-rival-${SLURM_JOB_ID}" \
      --max-seconds "$EP_NULL_CAP"; then
    printf 'NULL_RIVAL_COMPLETED model=%s\n' "$EP_NULL_MODEL"
  else
    EP_NULL_GROUP_EXIT=1
    printf 'NULL_RIVAL_FAILED model=%s: preserve attempt; no retry\n' "$EP_NULL_MODEL" >&2
  fi
done
if [[ "$EP_NULL_GROUP" == MC_MV && "$EP_NULL_GROUP_EXIT" == 0 ]]; then
  if timeout --signal=TERM --kill-after=10s 120s "$EP_NULL_PYTHON" -B \
      outputs/code/summarize_mz_null_menu.py \
      --base-report outputs/runs/behavioral_mechanism_kill_test/20261005-M0-null-representative-46574266/mz_null_representative.json \
      --mc-report "outputs/runs/behavioral_mechanism_kill_test/20261005-M0-null-MC-rival-${SLURM_JOB_ID}/mz_null_rival.json" \
      --mv-report "outputs/runs/behavioral_mechanism_kill_test/20261005-M0-null-MV-rival-${SLURM_JOB_ID}/mz_null_rival.json" \
      --mvz-report outputs/runs/behavioral_mechanism_kill_test/20261005-M0-null-MVZ-rival-46575931/mz_null_rival.json \
      --mvc-report outputs/runs/behavioral_mechanism_kill_test/20261005-M0-null-MVC-rival-46576234/mz_null_rival.json \
      --implementation outputs/code/behavioral_implementation.json \
      --output "outputs/runs/behavioral_mechanism_kill_test/20261005-M0-null-full-menu-${SLURM_JOB_ID}.json"; then
    printf 'NULL_FULL_MENU_DIAGNOSTICS_COMPLETED\n'
  else
    EP_NULL_GROUP_EXIT=1
    printf 'NULL_FULL_MENU_INCOMPLETE: preserve strict validation failure; no false-selection verdict\n' >&2
  fi
fi
exit "$EP_NULL_GROUP_EXIT"
