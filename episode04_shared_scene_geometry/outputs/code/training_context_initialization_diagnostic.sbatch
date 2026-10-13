#!/bin/bash
#SBATCH --job-name=ep04_context_init
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/training_neural_context/initialization-%j.out
#SBATCH --error=outputs/runs/training_neural_context/initialization-%j.err

# Prepared only: no submission is selected by this file. The coordinator may
# instead run the same command inside a suitable owned shared allocation.
# One diagnosis, at most120s+10s grace; no padding, retries or scientific work.
set -euo pipefail
EP04_HOST=$(hostname)
printf 'hostname=%s SLURM_JOB_ID=%s SLURM_CPUS_PER_TASK=%s\n' \
    "$EP04_HOST" "${SLURM_JOB_ID:?}" "${SLURM_CPUS_PER_TASK:?}"
if [[ "$EP04_HOST" == *-ln* || "$SLURM_CPUS_PER_TASK" != 1 ]]; then
    printf 'Use an actual one-CPU compute allocation.\n' >&2
    exit 2
fi
cd "${OAK:?}/users/zijiao/br_autoresearch/episode04_shared_scene_geometry"
module purge
module load devel python/3.12.1 math py-numpy/1.26.3
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1

EP04_DIAGNOSTIC_DIR="outputs/runs/training_neural_context/20261005-initialization-diagnostic-${SLURM_JOB_ID}"
mkdir "$EP04_DIAGNOSTIC_DIR"
if timeout --signal=TERM --kill-after=10s 120s \
    python3.12 -u outputs/code/diagnose_training_context_initialization.py \
    >"$EP04_DIAGNOSTIC_DIR/initialization.jsonl" \
    2>"$EP04_DIAGNOSTIC_DIR/traceback.txt"; then
    EP04_DIAGNOSTIC_EXIT=0
else
    EP04_DIAGNOSTIC_EXIT=$?
fi
printf '%s\n' "$EP04_DIAGNOSTIC_EXIT" >"$EP04_DIAGNOSTIC_DIR/exit_status.txt"
printf 'diagnostic_exit_code=%s output_dir=%s\n' "$EP04_DIAGNOSTIC_EXIT" "$EP04_DIAGNOSTIC_DIR"
exit "$EP04_DIAGNOSTIC_EXIT"
