#!/bin/bash
#SBATCH --job-name=ep19_v5_diagnostics
#SBATCH --partition=russpold
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --chdir=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode19_causal_motor_forecasting_benchmark
#SBATCH --output=outputs/runs/source_views_and_timing/v5_diagnostics_%j.out
#SBATCH --constraint=CPU_GEN:SKX|CPU_GEN:SIE

# AVX-512 nodes only: v5's control rows (job 48261284, CPU_GEN:SIE) reproduce
# bitwise on SKX/SIE, but an AVX2-only RME node changed the last bits of the
# normalized random directions (smoke 48280333), failing the exact first-32 regression.
# Before submission, the orchestrator records the declaration, reads
# /etc/agents/AGENTS.md and relevant topic files, and runs bash -n.
# No nested allocations, renewals, keepalives or background services.

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
        EP19_SMOKE="$SCRATCH/br_autoresearch/episode19_causal_motor_forecasting_benchmark/v5_diagnostics_work/smoke_${SLURM_JOB_ID}"
        mkdir "$EP19_SMOKE"
        exec > "$EP19_SMOKE/slurm.out" 2>&1
        PRD="$EP19_SMOKE/premature_r_diagnostic"
        CPC="$EP19_SMOKE/control_precision"
        scope_args=(--health-only --participants "$PARTICIPANTS")
        ;;
    0)
        if [[ -n "${PARTICIPANTS:-}" ]]; then
            echo "PARTICIPANTS is allowed only with SMOKE=1" >&2
            exit 2
        fi
        PRD="$EP19_VIEWS/premature_r_diagnostic_job_${SLURM_JOB_ID}"
        CPC="$EP19_VIEWS/control_precision_job_${SLURM_JOB_ID}"
        ;;
    *)
        echo "SMOKE must be 0 or 1" >&2
        exit 2
        ;;
esac

ml python/3.12.1 py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 > /dev/null 2>&1

python3 -P "$EP19_CODE/premature_r_diagnostic.py" \
    --stage known-positive --output-dir "$PRD" --workers 8 \
    ${scope_args[@]+"${scope_args[@]}"}

python3 -P "$EP19_CODE/premature_r_diagnostic.py" \
    --stage features --output-dir "$PRD" --workers 8 \
    ${scope_args[@]+"${scope_args[@]}"}

python3 -P "$EP19_CODE/control_precision_check.py" \
    --stage strata --output-dir "$CPC" --prd-dir "$PRD" --workers 8 \
    ${scope_args[@]+"${scope_args[@]}"}

python3 -P "$EP19_CODE/control_precision_check.py" \
    --stage controls --output-dir "$CPC" --prd-dir "$PRD" --workers 8 \
    ${scope_args[@]+"${scope_args[@]}"}

if [[ "${SMOKE:-0}" == "1" ]]; then
    echo "Health report: $PRD/features/health.json"
    echo "Health report: $CPC/controls/health.json"
else
    python3 -P "$EP19_CODE/premature_r_diagnostic.py" \
        --stage provider --output-dir "$PRD" --cpc-dir "$CPC" --workers 8

    module purge; ml python/3.12.1 py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 viz py-matplotlib/3.8.3_py312 > /dev/null 2>&1

    python3 -P "$EP19_CODE/plot_v5_diagnostics.py" \
        --prd-dir "$PRD" --cpc-dir "$CPC" --output-dir "$CPC/figure"
fi
