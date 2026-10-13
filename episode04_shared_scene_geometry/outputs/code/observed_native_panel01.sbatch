#!/bin/bash
# Complete fixed observed panel; submit after successful full low-level extraction.
#SBATCH --job-name=ep04-observed-panel
#SBATCH --partition=russpold
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:45:00
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/observed_native_panel/20261005-observed01-%j.out
#SBATCH --error=outputs/runs/observed_native_panel/20261005-observed01-%j.err
set -euo pipefail
[[ $# -eq 0 ]] || exit 2
ep04_panel_host=$(hostname)
printf 'HOST=%s JOB=%s CPUS=%s\n' "$ep04_panel_host" "${SLURM_JOB_ID:?}" "${SLURM_CPUS_PER_TASK:?}"
[[ "$ep04_panel_host" != *-ln* && "$SLURM_CPUS_PER_TASK" == 1 ]]
cd "${OAK:?}/users/zijiao/br_autoresearch/episode04_shared_scene_geometry"
source /share/software/user/open/lmod/lmod/init/bash
module purge
module load devel python/3.12.1 math py-numpy/1.26.3 viz py-pillow/10.2.0_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
bash -n outputs/code/observed_native_panel01.sbatch
python3.12 -B -c 'import ast; from pathlib import Path; ast.parse(Path("outputs/code/run_observed_native_panel.py").read_text())'
ep04_astra="${GROUP_HOME:?}/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv/bin/astra-tools"
set +e
timeout --signal=TERM --kill-after=10s 30s "$ep04_astra" validate \
  outputs/astra/v0.0.14/astra.yaml --json \
  > "outputs/runs/observed_native_panel/20261005-astra-${SLURM_JOB_ID}.json" 2>&1
ep04_astra_exit=$?
set -e
printf 'astra_validation_exit=%s\n' "$ep04_astra_exit"
exec timeout --signal=TERM --kill-after=10s 2460s python3.12 -u -B \
  outputs/code/run_observed_native_panel.py \
  --low-level-dir outputs/runs/source_readiness/20261005-low_level_full01 \
  --output-dir outputs/runs/observed_native_panel/20261005-observed01 \
  --max-seconds 2400
