#!/bin/bash
#SBATCH --job-name=ep19_detector_v2
#SBATCH --partition=russpold
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --chdir=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode19_causal_motor_forecasting_benchmark
#SBATCH --output=outputs/runs/source_views_and_timing/detector_v2_acceptance_%j.out

# EP19 self-paced detector v2 + raw-onset test v5 (development only). Rules are pre-declared in
# outputs/component_details.md section 1.3 and the timing handoff before any v2 output.
# Smoke: env SMOKE=1 PARTICIPANTS=sub-02,sub-15,sub-22 sbatch --export=ALL --output=<scratch>/smoke_%j.slurm.out <this>

set -euo pipefail
set -o noclobber

if [[ -z "${SLURM_JOB_ID:-}" || "$(hostname)" == *-ln* ]]; then
    echo "Actual Slurm compute shell required" >&2
    exit 2
fi
if [[ "${SLURM_CPUS_PER_TASK:-0}" -lt 8 ]]; then
    echo "This script requires eight allocated CPUs" >&2
    exit 2
fi

EP19_ROOT=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode19_causal_motor_forecasting_benchmark
EP19_CODE="$EP19_ROOT/outputs/code"
EP19_VIEWS="$EP19_ROOT/outputs/runs/source_views_and_timing"
FREEZE="$EP19_VIEWS/detector_freeze_job_48219799"
V4RUN="$EP19_VIEWS/raw_onset_test_job_48230171"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1

scope_args=()
health_args=()
if [[ "${SMOKE:-0}" == "1" ]]; then
    : "${SCRATCH:?SCRATCH is required}"
    : "${PARTICIPANTS:?Set PARTICIPANTS for a smoke run}"
    EP19_SMOKE="$SCRATCH/br_autoresearch/episode19_causal_motor_forecasting_benchmark/detector_v2_work/smoke_${SLURM_JOB_ID}"
    mkdir "$EP19_SMOKE"
    exec > "$EP19_SMOKE/slurm.out" 2>&1
    RUN="$EP19_SMOKE/detector_v2"
    TEST="$EP19_SMOKE/raw_onset_test_v5"
    REG_A="$EP19_SMOKE/regression_detector_v1"
    REG_B="$EP19_SMOKE/regression_estimator_v4"
    scope_args=(--participants "$PARTICIPANTS" --health-only)
    health_args=(--health-only)
else
    if [[ -n "${PARTICIPANTS:-}" ]]; then
        echo "PARTICIPANTS requires SMOKE=1 in this batch driver" >&2
        exit 2
    fi
    RUN="$EP19_VIEWS/detector_v2_job_${SLURM_JOB_ID}"
    TEST="$EP19_VIEWS/raw_onset_test_v5_job_${SLURM_JOB_ID}"
    REG_A="$EP19_VIEWS/detector_v2_regression_job_${SLURM_JOB_ID}"
    REG_B="$EP19_VIEWS/raw_onset_v5_regression_job_${SLURM_JOB_ID}"
fi

ml python/3.12.1 py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 > /dev/null 2>&1

python3 -P "$EP19_CODE/run_detector_v2.py" \
    --stage regression --freeze-dir "$FREEZE" \
    --output-dir "$REG_A" --workers 8 ${health_args[@]+"${health_args[@]}"}

python3 -P "$EP19_CODE/self_paced_raw_onset_test_v5.py" \
    --stage regression-v4 --freeze-dir "$FREEZE" --v4-dir "$V4RUN" \
    --output-dir "$REG_B" --workers 8 ${health_args[@]+"${health_args[@]}"}

python3 -P "$EP19_CODE/run_detector_v2.py" \
    --stage settings --output-dir "$RUN" --workers 8 ${health_args[@]+"${health_args[@]}"}

python3 -P "$EP19_CODE/self_paced_raw_onset_test_v5.py" \
    --stage controls --detector-dir "$RUN" --output-dir "$TEST" \
    --workers 8 ${scope_args[@]+"${scope_args[@]}"}

python3 -P "$EP19_CODE/run_detector_v2.py" \
    --stage labels --output-dir "$RUN" --test-dir "$TEST" \
    --workers 8 ${scope_args[@]+"${scope_args[@]}"}

python3 -P "$EP19_CODE/self_paced_raw_onset_test_v5.py" \
    --stage events --detector-dir "$RUN" --output-dir "$TEST" \
    --freeze-dir "$FREEZE" --workers 8 ${scope_args[@]+"${scope_args[@]}"}

if [[ "${SMOKE:-0}" == "1" ]]; then
    echo "Health report: $TEST/events/health.json"
else
    module purge
    ml python/3.12.1 viz py-matplotlib/3.8.3_py312 > /dev/null 2>&1
    python3 -P "$EP19_CODE/plot_detector_v2_acceptance.py" \
        --detector-dir "$RUN" --test-dir "$TEST" --v4-dir "$V4RUN" \
        --output-dir "$TEST/figure"
fi
