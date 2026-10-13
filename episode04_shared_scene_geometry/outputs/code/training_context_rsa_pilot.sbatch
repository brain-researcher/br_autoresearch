#!/bin/bash -l
#SBATCH --job-name=ep04-real-rsa
#SBATCH --partition=normal,russpold
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --time=40
#SBATCH --no-requeue
set -euo pipefail
EP_SHARED_ROOT="${OAK:?}/users/zijiao/br_autoresearch"
EP_SHARED_EP04="$EP_SHARED_ROOT/episode04_shared_scene_geometry"
EP_SHARED_EP03="$EP_SHARED_ROOT/episode03_condition_semantics_effect_maps"
EP_SHARED_EP19="$EP_SHARED_ROOT/episode19_causal_motor_forecasting_benchmark"
cd "$EP_SHARED_EP04"
EP_SHARED_HOST=$(hostname)
printf 'HOST=%s JOB=%s CPUS=%s\n' "$EP_SHARED_HOST" "${SLURM_JOB_ID:?}" "${SLURM_CPUS_PER_TASK:?}"
[[ "$SLURM_JOB_ID" =~ ^[0-9]+$ && "$EP_SHARED_HOST" != *-ln* && "$SLURM_CPUS_PER_TASK" == 1 ]]
module purge
module load devel python/3.12.1 math py-numpy/1.26.3 py-scikit-learn/1.5.1_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
mkdir -p outputs/runs/training_context_rsa
EP_SHARED_OUT="outputs/runs/training_context_rsa/20261005-rsa-pilot-${SLURM_JOB_ID}"
mkdir "$EP_SHARED_OUT"
python3.12 -B - "$EP_SHARED_EP03/outputs/code/join_first_group_map_route.py" \
  "$EP_SHARED_EP19/outputs/code/run_way_p2_support_trace.py" \
  "$EP_SHARED_EP04/outputs/code/run_training_context_rsa.py" <<'PY'
import ast
from pathlib import Path
import sys
for name in sys.argv[1:]:
    path = Path(name)
    ast.parse(path.read_text(), filename=str(path))
    print(f'Syntax PASS: {path.name}', flush=True)
PY
EP_SHARED_FAILURE=0
printf 'action\texit_status\n' >"$EP_SHARED_OUT/action_exits.tsv"
if srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=2G \
  --chdir="$EP_SHARED_EP03" \
  timeout --signal=TERM --kill-after=10s 100s python3.12 -u -B \
  outputs/code/join_first_group_map_route.py \
  --output-dir "$EP_SHARED_EP03/outputs/runs/source_handoff_qualification/20261005-first-route-${SLURM_JOB_ID}" \
  >"$EP_SHARED_OUT/ep03-first-route.out" 2>"$EP_SHARED_OUT/ep03-first-route.err"; then
  printf 'ep03_first_route\t0\n' >>"$EP_SHARED_OUT/action_exits.tsv"
else
  EP_SHARED_STATUS=$?; EP_SHARED_FAILURE=1
  printf 'ep03_first_route\t%s\n' "$EP_SHARED_STATUS" >>"$EP_SHARED_OUT/action_exits.tsv"
fi
if srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=4G \
  --chdir="$EP_SHARED_EP19" \
  timeout --signal=TERM --kill-after=10s 190s python3.12 -u -B \
  outputs/code/run_way_p2_support_trace.py \
  --development-view "${SCRATCH:?}/br_autoresearch/episode19_causal_motor_forecasting_benchmark/development_view_job_46425841" \
  --output-dir "$EP_SHARED_EP19/outputs/runs/source_views_and_timing/20261005-P2-support-${SLURM_JOB_ID}" \
  --max-seconds 180 \
  >"$EP_SHARED_OUT/ep19-P2-support.out" 2>"$EP_SHARED_OUT/ep19-P2-support.err"; then
  printf 'ep19_P2_support\t0\n' >>"$EP_SHARED_OUT/action_exits.tsv"
else
  EP_SHARED_STATUS=$?; EP_SHARED_FAILURE=1
  printf 'ep19_P2_support\t%s\n' "$EP_SHARED_STATUS" >>"$EP_SHARED_OUT/action_exits.tsv"
fi
if timeout --signal=TERM --kill-after=10s 1560s python3.12 -u -B \
  outputs/code/run_training_context_rsa.py --participant sub-01 \
  --output-dir "$EP_SHARED_OUT/sub-01" --max-seconds 1500 \
  >"$EP_SHARED_OUT/ep04-training-rsa.out" 2>"$EP_SHARED_OUT/ep04-training-rsa.err"; then
  printf 'ep04_training_rsa\t0\n' >>"$EP_SHARED_OUT/action_exits.tsv"
else
  EP_SHARED_STATUS=$?; EP_SHARED_FAILURE=1
  printf 'ep04_training_rsa\t%s\n' "$EP_SHARED_STATUS" >>"$EP_SHARED_OUT/action_exits.tsv"
fi
exit "$EP_SHARED_FAILURE"
