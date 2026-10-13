#!/bin/bash -l
#SBATCH --job-name=ep01-mz-mvc-rivals
#SBATCH --partition=normal
#SBATCH --constraint=CPU_SKU:7502
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=01:35:00
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-MVC-saved-rival-%A_%a.out
#SBATCH --error=outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-MVC-saved-rival-%A_%a.err

# Root supplies the unique array indices at submission; this wave fits MVC only.
# No simulation, baseline refit, resubmission, or follow-on work.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
EP_RIVAL_MODEL=MVC
case "${SLURM_ARRAY_TASK_ID-}" in
  ''|*[!0-9]*) printf 'Missing numeric array index\n' >&2; exit 2 ;;
esac
if (( SLURM_ARRAY_TASK_ID < 1 || SLURM_ARRAY_TASK_ID > 199 )); then
  printf 'Expected fixed replicate index 1 through 199\n' >&2
  exit 2
fi
EP_SOURCE_ARRAY=46539590
if (( SLURM_ARRAY_TASK_ID <= 20 )); then
  EP_SOURCE_ARRAY=46538652
fi
module purge
module load devel python/3.12.1 math py-numpy/1.26.3 py-scikit-learn/1.5.1_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
EP_PYTHON=$(command -v python3.12)
hostname
printf 'SLURM_JOB_ID=%s ARRAY=%s INDEX=%s MODEL=%s CPUS=%s\n' \
  "$SLURM_JOB_ID" "$SLURM_ARRAY_JOB_ID" "$SLURM_ARRAY_TASK_ID" "$EP_RIVAL_MODEL" "$SLURM_CPUS_PER_TASK"
"$EP_PYTHON" --version
# Aggregate the completed MZ0 menu once before the substantive MVC1 fit.
if (( SLURM_ARRAY_TASK_ID == 1 )); then
  bash -n outputs/code/mz_saved_mvc_rivals.sbatch
  printf 'MVC_BATCH_SYNTAX_PASS\n'
  EP_AST_BIN="$GROUP_HOME/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv/bin/astra-tools"
  if timeout --signal=TERM --kill-after=10s 30s "$EP_AST_BIN" validate \
      outputs/astra/v0.0.14/astra.yaml --json \
      > "outputs/astra/v0.0.14/validation_20261005_mvc_wave_${SLURM_ARRAY_JOB_ID}.json"; then
    printf 'MVC_ASTRA_VALIDATION_PASS\n'
  else
    printf 'MVC_ASTRA_VALIDATION_OPEN: coordinator must inspect validation output\n' >&2
  fi
  timeout --signal=TERM --kill-after=10s 180s "$EP_PYTHON" -B \
    outputs/code/summarize_mz_fixed_menu.py \
    --base-report outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-SESOI-replicate000-attempt001/single_process_recovery.json \
    --simple-rivals-report outputs/runs/behavioral_mechanism_kill_test/20261005-MZ0-fixed-rivals-attempt001/single_process_rivals.json \
    --mvz-report outputs/runs/behavioral_mechanism_kill_test/20261005-MZ0-MVZ-rival-46545300/mz_complex_rival.json \
    --mvc-report outputs/runs/behavioral_mechanism_kill_test/20261005-MZ0-MVC-rival-46545303/mz_complex_rival.json \
    --implementation outputs/code/behavioral_implementation.json \
    --output "outputs/runs/behavioral_mechanism_kill_test/20261005-MZ0-full-menu-${SLURM_ARRAY_JOB_ID}.json"
fi
exec timeout --signal=TERM --kill-after=10s 5400s "$EP_PYTHON" -B \
  outputs/code/run_mz_saved_replicate_rival.py \
  --model "$EP_RIVAL_MODEL" --replicate-index "$SLURM_ARRAY_TASK_ID" \
  --source-dir "outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-SESOI-array-${EP_SOURCE_ARRAY}/replicate${SLURM_ARRAY_TASK_ID}" \
  --implementation outputs/code/behavioral_implementation.json \
  --output-dir "outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-${EP_RIVAL_MODEL}-saved-array-${SLURM_ARRAY_JOB_ID}/replicate${SLURM_ARRAY_TASK_ID}" \
  --max-seconds 5400
