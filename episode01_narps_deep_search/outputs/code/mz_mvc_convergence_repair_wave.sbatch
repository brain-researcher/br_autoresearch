#!/bin/bash -l
#SBATCH --job-name=ep01-mvc-repair-wave
#SBATCH --partition=normal
#SBATCH --constraint=CPU_SKU:7502
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=03:40:00
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-MVC-repair-wave-%A_%a.out
#SBATCH --error=outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-MVC-repair-wave-%A_%a.err

# Selected in explicit disjoint arrays after cases2/3 in46597025 pass
# eight repair checks and actual saved-cohort validation/replay.
# Required argument: fresh-both or reverse-only. Submit fresh-both with
# --time=03:40:00 (12600+10<13200s), reverse-only with --time=01:55:00
# (6600+10<6900s). Both use one CPU and2GiB; no repeated startup tests.
# Exclude held1, initialcases2/3, completed42, and originalsuccesses92/105.
# The runner independently validates the immutable original500 diagnosis,
# preserves a completed first fit only for second-half exhaustion, and uses
# fresh1500 fits otherwise. No polling, automatic retries or source changes.
set -euo pipefail
EP_MVC_REPAIR_HOST=$(hostname)
printf 'HOST=%s SLURM_JOB_ID=%s ARRAY_JOB_ID=%s ARRAY_TASK_ID=%s NTASKS=%s CPUS_PER_TASK=%s\n' \
  "$EP_MVC_REPAIR_HOST" "${SLURM_JOB_ID-}" "${SLURM_ARRAY_JOB_ID-}" \
  "${SLURM_ARRAY_TASK_ID-}" "${SLURM_NTASKS-}" "${SLURM_CPUS_PER_TASK-}"
if [[ -z "${SLURM_JOB_ID-}" || "$EP_MVC_REPAIR_HOST" == *-ln* || \
      ! "${SLURM_ARRAY_JOB_ID-}" =~ ^[1-9][0-9]*$ || \
      ! "${SLURM_ARRAY_TASK_ID-}" =~ ^[0-9]+$ || \
      "${SLURM_NTASKS-}" != 1 || "${SLURM_CPUS_PER_TASK-}" != 1 ]]; then
  printf 'Requires one selected one-CPU compute array task\n' >&2
  exit 2
fi
if [[ "$#" != 1 || ( "$1" != fresh-both && "$1" != reverse-only ) ]]; then
  printf 'Supply exactly one explicit mode: fresh-both or reverse-only\n' >&2
  exit 2
fi
EP_MVC_REPAIR_INDEX=$((10#$SLURM_ARRAY_TASK_ID))
if (( EP_MVC_REPAIR_INDEX < 4 || EP_MVC_REPAIR_INDEX > 199 )); then
  printf 'Only remaining indices4 through199 are eligible for this wrapper\n' >&2
  exit 2
fi
case "$EP_MVC_REPAIR_INDEX" in
  42|92|105)
    printf 'Preserve the already completed MVC cohort%s\n' "$EP_MVC_REPAIR_INDEX" >&2
    exit 2
    ;;
  11|22|36|61|71|73|83|86|144|149|151|152|155|163|165|168|171|172|181|188|196)
    EP_MVC_REPAIR_MODE=reverse-only
    EP_MVC_REPAIR_SECONDS=6600
    ;;
  *)
    EP_MVC_REPAIR_MODE=fresh-both
    EP_MVC_REPAIR_SECONDS=12600
    ;;
esac
if [[ "$1" != "$EP_MVC_REPAIR_MODE" ]]; then
  printf 'Requested mode%s differs from selected cohort%s mode%s\n' \
    "$1" "$EP_MVC_REPAIR_INDEX" "$EP_MVC_REPAIR_MODE" >&2
  exit 2
fi
cd "$SLURM_SUBMIT_DIR"
module purge
module load devel python/3.12.1 math py-numpy/1.26.3 py-scikit-learn/1.5.1_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
EP_MVC_REPAIR_PYTHON=$(command -v python3.12)
printf 'MVC_REPAIR index=%s mode=%s internal_seconds=%s fresh_maximum_sweeps=1500\n' \
  "$EP_MVC_REPAIR_INDEX" "$EP_MVC_REPAIR_MODE" "$EP_MVC_REPAIR_SECONDS"
exec timeout --signal=TERM --kill-after=10s "${EP_MVC_REPAIR_SECONDS}s" "$EP_MVC_REPAIR_PYTHON" -B \
  outputs/code/run_mz_mvc_convergence_repair.py \
  --replicate-index "$EP_MVC_REPAIR_INDEX" \
  --implementation outputs/code/behavioral_implementation.json \
  --output-dir "outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-MVC-repair-wave-${SLURM_ARRAY_JOB_ID}/replicate${EP_MVC_REPAIR_INDEX}" \
  --max-seconds "$EP_MVC_REPAIR_SECONDS"
