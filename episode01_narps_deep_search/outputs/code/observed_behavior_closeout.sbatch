#!/bin/bash -l
#SBATCH --job-name=ep01-real-report
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=10
#SBATCH --no-requeue
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
EP_OBS_HOST=$(hostname)
printf 'HOST=%s JOB=%s CPUS=%s\n' "$EP_OBS_HOST" "$SLURM_JOB_ID" "$SLURM_CPUS_PER_TASK"
[[ "$SLURM_JOB_ID" =~ ^[0-9]+$ && "$EP_OBS_HOST" != *-ln* && "$SLURM_CPUS_PER_TASK" == 1 ]]
module purge
module load devel python/3.12.1 math py-numpy/1.26.3 viz py-matplotlib/3.8.3_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
EP_OBS_RUNS="outputs/runs/exploratory_observed_behavior"
EP_OBS_OUT="$EP_OBS_RUNS/20261005-closeout-${SLURM_JOB_ID}"
mkdir "$EP_OBS_OUT"
export MPLCONFIGDIR="${L_SCRATCH_JOB:?}/ep01-observed-matplotlib"
mkdir -p "$MPLCONFIGDIR"
EP_OBS_REPORTS=("$EP_OBS_RUNS/20261005-job46650095/fit/observed_behavior_fit.json")
for EP_OBS_PAIR in M0-direction1 MV-direction0 MV-direction1 MZ-direction0 MZ-direction1; do
  EP_OBS_REPORTS+=("$EP_OBS_RUNS/20261005-pack-46651310/$EP_OBS_PAIR/observed_behavior_fit.json")
done
for EP_OBS_PAIR in MC-direction0 MC-direction1 MVZ-direction0 MVZ-direction1; do
  EP_OBS_REPORTS+=("$EP_OBS_RUNS/20261005-pack-46651320/$EP_OBS_PAIR/observed_behavior_fit.json")
done
for EP_OBS_PAIR in MVC-direction0 MVC-direction1; do
  EP_OBS_REPORTS+=("$EP_OBS_RUNS/20261005-pack-46651326/$EP_OBS_PAIR/observed_behavior_fit.json")
done
EP_OBS_ARGS=()
for EP_OBS_REPORT in "${EP_OBS_REPORTS[@]}"; do
  if [[ -f "$EP_OBS_REPORT" ]]; then
    EP_OBS_ARGS+=(--fit-report "$EP_OBS_REPORT")
  else
    printf '%s\n' "$EP_OBS_REPORT" >>"$EP_OBS_OUT/missing_reports.txt"
  fi
done
EP_OBS_FAILURE=0
if timeout --signal=TERM --kill-after=10s 120s python3.12 -B \
  outputs/code/summarize_observed_behavior.py "${EP_OBS_ARGS[@]}" \
  --output "$EP_OBS_OUT/observed_behavior_summary.json" >"$EP_OBS_OUT/summary.log" 2>&1; then
  printf 'summary\t0\n' >>"$EP_OBS_OUT/action_exits.tsv"
else
  EP_OBS_EXIT=$?; EP_OBS_FAILURE=1
  printf 'summary\t%s\n' "$EP_OBS_EXIT" >>"$EP_OBS_OUT/action_exits.tsv"
fi
if timeout --signal=TERM --kill-after=10s 120s python3.12 -B \
  outputs/code/plot_observed_behavior.py \
  --descriptive-report "$EP_OBS_RUNS/20261005-job46650095/descriptions/observed_behavior_descriptive.json" \
  --output-dir outputs/figures >"$EP_OBS_OUT/plot.log" 2>&1; then
  printf 'plot\t0\n' >>"$EP_OBS_OUT/action_exits.tsv"
else
  EP_OBS_EXIT=$?; EP_OBS_FAILURE=1
  printf 'plot\t%s\n' "$EP_OBS_EXIT" >>"$EP_OBS_OUT/action_exits.tsv"
fi
EP_OBS_EP04="$OAK/users/zijiao/br_autoresearch/episode04_shared_scene_geometry"
EP_OBS_EP04_OUT="$EP_OBS_EP04/outputs/runs/training_neural_context/20261005-chunk-diagnostic-${SLURM_JOB_ID}"
if srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=4G \
  --chdir="$EP_OBS_EP04" env SLURM_CPUS_PER_TASK=1 \
  timeout --signal=TERM --kill-after=10s 190s python3.12 -u \
  outputs/code/diagnose_training_context_chunk.py --output-dir "$EP_OBS_EP04_OUT" \
  >"$EP_OBS_OUT/ep04-chunk.log" 2>"$EP_OBS_OUT/ep04-chunk.err"; then
  printf 'ep04_chunk\t0\n' >>"$EP_OBS_OUT/action_exits.tsv"
else
  EP_OBS_EXIT=$?; EP_OBS_FAILURE=1
  printf 'ep04_chunk\t%s\n' "$EP_OBS_EXIT" >>"$EP_OBS_OUT/action_exits.tsv"
fi
exit "$EP_OBS_FAILURE"
