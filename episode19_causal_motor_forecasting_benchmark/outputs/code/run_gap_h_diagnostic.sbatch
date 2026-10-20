#!/bin/bash
#SBATCH --job-name=ep19_gap_h
#SBATCH --partition=russpold
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --chdir=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode19_causal_motor_forecasting_benchmark
#SBATCH --output=outputs/runs/source_views_and_timing/gap_h_diagnostic_%j.out
#SBATCH --constraint=CPU_GEN:SKX|CPU_GEN:SIE

# Before submission, the orchestrator must record the declaration, read
# /etc/agents/AGENTS.md and relevant topic files, and run bash -n.
#
# Health smoke:
# env SMOKE=1 PARTICIPANTS=sub-02,sub-15,sub-22 \
#   sbatch --export=ALL --output=<scratch>/gap_h_smoke_%j.out <this-file>
#
# Full run:
# sbatch outputs/code/run_gap_h_diagnostic.sbatch
#
# No nested allocations, automatic renewals, keepalives or background services.

set -euo pipefail
set -o noclobber

EP19_HOST="$(hostname)"
if [[ -z "${SLURM_JOB_ID:-}" || "${EP19_HOST,,}" == *-ln* ]]; then
    echo "An actual Slurm compute shell is required" >&2
    exit 2
fi
if [[ "${SLURM_CPUS_PER_TASK:-0}" -lt 8 ]]; then
    echo "Eight allocated CPUs are required" >&2
    exit 2
fi
printf 'host=%s SLURM_JOB_ID=%s\n' "$EP19_HOST" "$SLURM_JOB_ID"

EP19_ROOT=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode19_causal_motor_forecasting_benchmark
EP19_CODE="$EP19_ROOT/outputs/code"
EP19_VIEWS="$EP19_ROOT/outputs/runs/source_views_and_timing"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1

scope_args=()
case "${SMOKE:-0}" in
    1)
        : "${SCRATCH:?SCRATCH is required}"
        : "${PARTICIPANTS:?Set PARTICIPANTS for a smoke run}"
        EP19_WORK="$SCRATCH/br_autoresearch/episode19_causal_motor_forecasting_benchmark/gap_h_work"
        if [[ ! -d "$EP19_WORK" ]]; then
            echo "The existing gap_h_work scratch directory is required" >&2
            exit 2
        fi
        EP19_SMOKE="$EP19_WORK/smoke_${SLURM_JOB_ID}"
        mkdir "$EP19_SMOKE"
        exec > "$EP19_SMOKE/slurm.out" 2>&1
        RUN="$EP19_SMOKE/gap_h_diagnostic"
        scope_args=(--health-only --participants "$PARTICIPANTS")
        ;;
    0)
        if [[ -n "${PARTICIPANTS:-}" ]]; then
            echo "PARTICIPANTS is allowed only with SMOKE=1" >&2
            exit 2
        fi
        RUN="$EP19_VIEWS/gap_h_diagnostic_job_${SLURM_JOB_ID}"
        ;;
    *)
        echo "SMOKE must be 0 or 1" >&2
        exit 2
        ;;
esac

ml python/3.12.1 py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 > /dev/null 2>&1

python3 -P "$EP19_CODE/gap_h_diagnostic.py" \
    --stage known-positive --output-dir "$RUN" --workers 8 \
    ${scope_args[@]+"${scope_args[@]}"}

python3 -P "$EP19_CODE/gap_h_diagnostic.py" \
    --stage features --output-dir "$RUN" --workers 8 \
    ${scope_args[@]+"${scope_args[@]}"}

if [[ "${SMOKE:-0}" == "1" ]]; then
    echo "Health report: $RUN/known_positive/health.json"
    echo "Health report: $RUN/features/health.json"
else
    python3 -P "$EP19_CODE/gap_h_diagnostic.py" \
        --stage decision --output-dir "$RUN" --workers 8

    python3 -P "$EP19_CODE/gap_h_diagnostic.py" \
        --stage provider --output-dir "$RUN" --workers 8

    module purge
    ml python/3.12.1 py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 viz py-matplotlib/3.8.3_py312 > /dev/null 2>&1

    python3 -P "$EP19_CODE/plot_gap_h_diagnostic.py" \
        --run-dir "$RUN" --output-dir "$RUN/figure"
fi
