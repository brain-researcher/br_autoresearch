#!/bin/bash -l
#SBATCH --job-name=ep01-real-fixed
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=155
#SBATCH --no-requeue
set -euo pipefail
EP_OBS_PACK=${1:?Pass pack0, pack1 or pack2}
case "$EP_OBS_PACK" in
  pack0) EP_OBS_INDICES=(1 2 3 4 5);;
  pack1) EP_OBS_INDICES=(6 7 8 9);;
  pack2) EP_OBS_INDICES=(10 11);;
  *) exit 2;;
esac
cd "$SLURM_SUBMIT_DIR"
EP_OBS_HOST=$(hostname)
printf 'HOST=%s JOB=%s CPUS=%s PACK=%s\n' "$EP_OBS_HOST" "$SLURM_JOB_ID" "$SLURM_CPUS_PER_TASK" "$EP_OBS_PACK"
[[ "$SLURM_JOB_ID" =~ ^[0-9]+$ && "$EP_OBS_HOST" != *-ln* && "$SLURM_CPUS_PER_TASK" == 4 ]]
module purge
module load devel python/3.12.1 math py-numpy/1.26.3 py-scikit-learn/1.5.1_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="outputs/code:outputs/tests${PYTHONPATH:+:$PYTHONPATH}"
EP_OBS_ROOT="outputs/runs/exploratory_observed_behavior/20261005-pack-${SLURM_JOB_ID}"
mkdir "$EP_OBS_ROOT"
EP_OBS_MODELS=(M0 MV MZ MC MVZ MVC)
EP_OBS_ANY_FAILURE=0
for EP_OBS_INDEX in "${EP_OBS_INDICES[@]}"; do
  EP_OBS_MODEL=${EP_OBS_MODELS[$((EP_OBS_INDEX / 2))]}
  EP_OBS_DIRECTION=$((EP_OBS_INDEX % 2))
  printf 'MODEL=%s DIRECTION=%s\n' "$EP_OBS_MODEL" "$EP_OBS_DIRECTION"
  if srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=8G \
    timeout --signal=TERM --kill-after=20s 1820s python3.12 -B \
    outputs/code/run_observed_behavior.py fit --model "$EP_OBS_MODEL" \
    --direction "$EP_OBS_DIRECTION" --implementation outputs/code/behavioral_implementation.json \
    --output-dir "$EP_OBS_ROOT/${EP_OBS_MODEL}-direction${EP_OBS_DIRECTION}" --max-seconds 1800 \
    >"$EP_OBS_ROOT/${EP_OBS_MODEL}-direction${EP_OBS_DIRECTION}.log" 2>&1; then
    printf '%s\t%s\t0\n' "$EP_OBS_MODEL" "$EP_OBS_DIRECTION" >>"$EP_OBS_ROOT/component_exits.tsv"
  else
    EP_OBS_EXIT=$?
    EP_OBS_ANY_FAILURE=1
    printf '%s\t%s\t%s\n' "$EP_OBS_MODEL" "$EP_OBS_DIRECTION" "$EP_OBS_EXIT" >>"$EP_OBS_ROOT/component_exits.tsv"
  fi
done
exit "$EP_OBS_ANY_FAILURE"
