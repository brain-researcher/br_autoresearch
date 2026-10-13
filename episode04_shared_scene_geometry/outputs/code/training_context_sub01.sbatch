#!/bin/bash -l
#SBATCH --job-name=ep04-real-context
#SBATCH --partition=normal,russpold
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=45
#SBATCH --no-requeue
set -euo pipefail
cd "${OAK:?}/users/zijiao/br_autoresearch/episode04_shared_scene_geometry"
EP04_CONTEXT_HOST=$(hostname)
printf 'HOST=%s JOB=%s CPUS=%s\n' "$EP04_CONTEXT_HOST" "${SLURM_JOB_ID:?}" "${SLURM_CPUS_PER_TASK:?}"
[[ "$SLURM_JOB_ID" =~ ^[0-9]+$ && "$EP04_CONTEXT_HOST" != *-ln* && "$SLURM_CPUS_PER_TASK" == 1 ]]
module purge
module load devel python/3.12.1 math py-numpy/1.26.3
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
python3.12 -B - <<'PY'
import ast
from pathlib import Path
path = Path('outputs/code/run_training_context_sub01.py')
ast.parse(path.read_text(), filename=str(path))
print('Continuation driver syntax: PASS', flush=True)
PY
EP04_CONTEXT_OUT="outputs/runs/training_neural_context/20261005-sub01-context-${SLURM_JOB_ID}"
if timeout --signal=TERM --kill-after=10s 2460s python3.12 -u -B \
    outputs/code/run_training_context_sub01.py --output-dir "$EP04_CONTEXT_OUT"; then
  printf 'training_context_exit=0\n'
else
  EP04_CONTEXT_EXIT=$?
  printf 'training_context_exit=%s\n' "$EP04_CONTEXT_EXIT"
  exit "$EP04_CONTEXT_EXIT"
fi
