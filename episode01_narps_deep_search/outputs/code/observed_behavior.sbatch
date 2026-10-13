#!/bin/bash -l
#SBATCH --job-name=ep01-real-behavior
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=35
#SBATCH --no-requeue

set -euo pipefail
EP_OBS_MODE=${1:?Pass pilot or component}
[[ "$EP_OBS_MODE" == pilot || "$EP_OBS_MODE" == component ]]
cd "$SLURM_SUBMIT_DIR"
EP_OBS_HOST=$(hostname)
printf 'HOST=%s JOB=%s CPUS=%s MODE=%s\n' "$EP_OBS_HOST" "$SLURM_JOB_ID" "$SLURM_CPUS_PER_TASK" "$EP_OBS_MODE"
[[ "$SLURM_JOB_ID" =~ ^[0-9]+$ && "$EP_OBS_HOST" != *-ln* && "$SLURM_CPUS_PER_TASK" == 4 ]]
module purge
module load devel python/3.12.1 math py-numpy/1.26.3 py-scikit-learn/1.5.1_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="outputs/code:outputs/tests${PYTHONPATH:+:$PYTHONPATH}"
EP_OBS_ROOT="outputs/runs/exploratory_observed_behavior/20261005-job${SLURM_JOB_ID}"
mkdir "$EP_OBS_ROOT"

if [[ "$EP_OBS_MODE" == pilot ]]; then
  EP_OBS_INDEX=0
  timeout --signal=TERM --kill-after=10s 90s python3.12 -B -m unittest discover -s outputs/code -p 'test_observed_behavior*.py' -v >"$EP_OBS_ROOT/tests.log" 2>&1
  EP_OBS_ASTRA="$GROUP_HOME/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv/bin/astra-tools"
  if timeout --signal=TERM --kill-after=10s 30s "$EP_OBS_ASTRA" validate outputs/astra/v0.0.14/astra.yaml --json >"$EP_OBS_ROOT/astra-validation.json" 2>&1; then
    printf 'ASTRA_CHECK_EXIT=0\n'
  else
    printf 'ASTRA validation unresolved; optional record check does not open scientific claims\n'
  fi
  timeout --signal=TERM --kill-after=10s 130s python3.12 -B outputs/code/run_observed_behavior.py describe --output-dir "$EP_OBS_ROOT/descriptions" --max-seconds 120 >"$EP_OBS_ROOT/descriptions.log" 2>&1
  # Independent, selected EP04 source diagnosis shares this parent allocation.
  # It finishes before the four-worker EP01 fit; failure does not change EP01.
  EP_OBS_EP04="$OAK/users/zijiao/br_autoresearch/episode04_shared_scene_geometry"
  EP_OBS_EP04_OUT="$EP_OBS_EP04/outputs/runs/training_neural_context/20261005-initialization-diagnostic-${SLURM_JOB_ID}"
  mkdir "$EP_OBS_EP04_OUT"
  if srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=4G \
    --chdir="$EP_OBS_EP04" env SLURM_CPUS_PER_TASK=1 \
    timeout --signal=TERM --kill-after=10s 120s python3.12 -u \
    outputs/code/diagnose_training_context_initialization.py \
    >"$EP_OBS_EP04_OUT/initialization.jsonl" 2>"$EP_OBS_EP04_OUT/traceback.txt"; then
    printf '0\n' >"$EP_OBS_EP04_OUT/exit_status.txt"
  else
    EP_OBS_EP04_EXIT=$?
    printf '%s\n' "$EP_OBS_EP04_EXIT" >"$EP_OBS_EP04_OUT/exit_status.txt"
  fi
else
  EP_OBS_INDEX=${SLURM_ARRAY_TASK_ID:?Array index 1 through 11 is required}
  [[ "$EP_OBS_INDEX" =~ ^[0-9]+$ && "$EP_OBS_INDEX" -ge 1 && "$EP_OBS_INDEX" -le 11 ]]
fi
EP_OBS_MODELS=(M0 MV MZ MC MVZ MVC)
EP_OBS_MODEL=${EP_OBS_MODELS[$((EP_OBS_INDEX / 2))]}
EP_OBS_DIRECTION=$((EP_OBS_INDEX % 2))
printf 'MODEL=%s DIRECTION=%s\n' "$EP_OBS_MODEL" "$EP_OBS_DIRECTION"
srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=8G \
  timeout --signal=TERM --kill-after=20s 1820s python3.12 -B \
  outputs/code/run_observed_behavior.py fit --model "$EP_OBS_MODEL" \
  --direction "$EP_OBS_DIRECTION" --implementation outputs/code/behavioral_implementation.json \
  --output-dir "$EP_OBS_ROOT/fit" --max-seconds 1800
