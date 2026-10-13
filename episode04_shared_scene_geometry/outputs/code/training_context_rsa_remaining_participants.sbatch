#!/bin/bash -l
#SBATCH --job-name=ep04-rsa-rest
#SBATCH --partition=normal,russpold
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=28G
#SBATCH --time=45
#SBATCH --no-requeue

# Preparation only: coordinator selects submission after the complete sub-01
# pilot and its measured cost and all shared-step reviews are checked.
# One finite pack; no retry/resume or later dispatch. Shared cost belongs to EP04.
set -euo pipefail
[[ $# == 0 ]]
EP04_RSA_SHARED_ROOT="${OAK:?}/users/zijiao/br_autoresearch"
EP04_RSA_EP03="$EP04_RSA_SHARED_ROOT/episode03_condition_semantics_effect_maps"
EP04_RSA_EP04="$EP04_RSA_SHARED_ROOT/episode04_shared_scene_geometry"
EP04_RSA_EP19="$EP04_RSA_SHARED_ROOT/episode19_causal_motor_forecasting_benchmark"
cd "$EP04_RSA_EP04"
EP04_RSA_REST_HOST=$(hostname)
printf 'HOST=%s JOB=%s NTASKS=%s CPUS_PER_TASK=%s\n' \
  "$EP04_RSA_REST_HOST" "${SLURM_JOB_ID:?}" "${SLURM_NTASKS:?}" "${SLURM_CPUS_PER_TASK:?}"
[[ "$SLURM_JOB_ID" =~ ^[0-9]+$ && "$EP04_RSA_REST_HOST" != *-ln* \
   && "$SLURM_NTASKS" == 4 && "$SLURM_CPUS_PER_TASK" == 1 ]]
module purge
module load devel python/3.12.1 math py-numpy/1.26.3 py-scikit-learn/1.5.1_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
EP04_RSA_REST_ROOT="outputs/runs/training_context_rsa/20261005-rsa-remaining-${SLURM_JOB_ID}"
mkdir "$EP04_RSA_REST_ROOT"
python3.12 -B - "$EP04_RSA_EP03/outputs/code/join_first_group_map_route.py" \
  "$EP04_RSA_EP19/outputs/code/run_way_remaining_s1_support.py" \
  "$EP04_RSA_EP04/outputs/code/run_training_context_rsa.py" \
  >"$EP04_RSA_REST_ROOT/driver_syntax.out" 2>"$EP04_RSA_REST_ROOT/driver_syntax.err" <<'PY'
import ast
from pathlib import Path
import sys
for name in sys.argv[1:]:
    path = Path(name)
    ast.parse(path.read_text(), filename=str(path))
    print(f'Syntax PASS: {path.name}', flush=True)
PY
EP04_RSA_ASTRA="${GROUP_HOME:-}/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv/bin/astra-tools"
printf 'episode\texit_status\n' >"$EP04_RSA_REST_ROOT/astra_exit_status.tsv"
EP04_RSA_ASTRA_EPISODES=("$EP04_RSA_EP03" "$EP04_RSA_EP04" "$EP04_RSA_EP19")
EP04_RSA_ASTRA_LABELS=(ep03 ep04 ep19)
for EP04_RSA_ASTRA_INDEX in 0 1 2; do
  EP04_RSA_ASTRA_LABEL="${EP04_RSA_ASTRA_LABELS[$EP04_RSA_ASTRA_INDEX]}"
  if timeout --signal=TERM --kill-after=10s 30s "$EP04_RSA_ASTRA" validate \
    "${EP04_RSA_ASTRA_EPISODES[$EP04_RSA_ASTRA_INDEX]}/outputs/astra/v0.0.14/astra.yaml" --json \
    >"$EP04_RSA_REST_ROOT/${EP04_RSA_ASTRA_LABEL}_astra_validation.json" \
    2>"$EP04_RSA_REST_ROOT/${EP04_RSA_ASTRA_LABEL}_astra_validation.err"; then
    EP04_RSA_ASTRA_STATUS=0
  else
    EP04_RSA_ASTRA_STATUS=$?
  fi
  printf '%s\t%s\n' "$EP04_RSA_ASTRA_LABEL" "$EP04_RSA_ASTRA_STATUS" \
    >>"$EP04_RSA_REST_ROOT/astra_exit_status.tsv"
done
EP04_RSA_REST_PARTICIPANTS=(sub-03 sub-05 sub-06 sub-07)
EP04_RSA_REST_PIDS=()
printf 'participant\texit_status\n' >"$EP04_RSA_REST_ROOT/participant_exit_status.tsv"
for EP04_RSA_REST_PARTICIPANT in "${EP04_RSA_REST_PARTICIPANTS[@]}"; do
  srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=7G \
    timeout --signal=TERM --kill-after=10s 1560s python3.12 -u -B \
    outputs/code/run_training_context_rsa.py \
    --participant "$EP04_RSA_REST_PARTICIPANT" \
    --output-dir "$EP04_RSA_REST_ROOT/$EP04_RSA_REST_PARTICIPANT" \
    --max-seconds 1500 \
    >"$EP04_RSA_REST_ROOT/${EP04_RSA_REST_PARTICIPANT}.out" \
    2>"$EP04_RSA_REST_ROOT/${EP04_RSA_REST_PARTICIPANT}.err" &
  EP04_RSA_REST_PIDS+=("$!")
done
EP04_RSA_REST_ANY_FAILURE=0
for EP04_RSA_REST_INDEX in 0 1 2 3; do
  if wait "${EP04_RSA_REST_PIDS[$EP04_RSA_REST_INDEX]}"; then
    EP04_RSA_REST_STATUS=0
  else
    EP04_RSA_REST_STATUS=$?
    EP04_RSA_REST_ANY_FAILURE=1
  fi
  printf '%s\t%s\n' "${EP04_RSA_REST_PARTICIPANTS[$EP04_RSA_REST_INDEX]}" "$EP04_RSA_REST_STATUS" \
    >>"$EP04_RSA_REST_ROOT/participant_exit_status.tsv"
done
printf 'action\texit_status\n' >"$EP04_RSA_REST_ROOT/shared_action_exit_status.tsv"
# All four scientific children have settled before these independent steps.
# Each action writes only its own episode's scientific output directory.
if srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=2G \
  --chdir="$EP04_RSA_EP03" \
  timeout --signal=TERM --kill-after=10s 100s python3.12 -u -B \
  outputs/code/join_first_group_map_route.py \
  --output-dir "$EP04_RSA_EP03/outputs/runs/source_handoff_qualification/20261005-first-route-repair-${SLURM_JOB_ID}" \
  >"$EP04_RSA_REST_ROOT/ep03-first-route-repair.out" \
  2>"$EP04_RSA_REST_ROOT/ep03-first-route-repair.err"; then
  printf 'ep03_first_route_repair\t0\n' >>"$EP04_RSA_REST_ROOT/shared_action_exit_status.tsv"
else
  EP04_RSA_REST_STATUS=$?; EP04_RSA_REST_ANY_FAILURE=1
  printf 'ep03_first_route_repair\t%s\n' "$EP04_RSA_REST_STATUS" \
    >>"$EP04_RSA_REST_ROOT/shared_action_exit_status.tsv"
fi
if srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=4G \
  --chdir="$EP04_RSA_EP19" \
  timeout --signal=TERM --kill-after=10s 620s python3.12 -u -B \
  outputs/code/run_way_remaining_s1_support.py \
  --development-view "${SCRATCH:?}/br_autoresearch/episode19_causal_motor_forecasting_benchmark/development_view_job_46425841" \
  --output-dir "$EP04_RSA_EP19/outputs/runs/source_views_and_timing/20261005-remaining-S1-${SLURM_JOB_ID}" \
  --max-seconds 600 \
  >"$EP04_RSA_REST_ROOT/ep19-remaining-S1-support.out" \
  2>"$EP04_RSA_REST_ROOT/ep19-remaining-S1-support.err"; then
  printf 'ep19_remaining_S1_support\t0\n' >>"$EP04_RSA_REST_ROOT/shared_action_exit_status.tsv"
else
  EP04_RSA_REST_STATUS=$?; EP04_RSA_REST_ANY_FAILURE=1
  printf 'ep19_remaining_S1_support\t%s\n' "$EP04_RSA_REST_STATUS" \
    >>"$EP04_RSA_REST_ROOT/shared_action_exit_status.tsv"
fi
exit "$EP04_RSA_REST_ANY_FAILURE"
