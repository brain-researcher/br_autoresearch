#!/bin/bash
# One bounded description of the already-open real outgoing development data.
#SBATCH --job-name=ep12-real-description
#SBATCH --partition=russpold
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/observed_outgoing_description/20261005-description01-%j.out
#SBATCH --error=outputs/runs/observed_outgoing_description/20261005-description01-%j.err
set -euo pipefail
[[ $# -eq 0 ]] || exit 2
ep12_description_host=$(hostname)
printf 'HOST=%s JOB=%s CPUS=%s\n' "$ep12_description_host" "${SLURM_JOB_ID:?}" "${SLURM_CPUS_PER_TASK:?}"
[[ "$ep12_description_host" != *-ln* && "$SLURM_CPUS_PER_TASK" == 1 ]]
cd "${OAK:?}/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency"
source /share/software/user/open/lmod/lmod/init/bash
module purge
module load devel python/3.12.1 py-pyarrow/18.1.0_py312 math py-numpy/1.26.3 py-scipy/1.12.0_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
bash -n outputs/tools/observed_outgoing_description01.sbatch
python3.12 -B -c 'import ast; from pathlib import Path; ast.parse(Path("outputs/tools/summarize_observed_outgoing.py").read_text())'
timeout --signal=TERM --kill-after=10s 30s python3.12 -u -B \
  outputs/tools/summarize_observed_outgoing.py --check-equations-only \
  > "outputs/runs/observed_outgoing_description/20261005-equations-${SLURM_JOB_ID}.json" 2>&1
ep12_astra="${GROUP_HOME:?}/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv/bin/astra-tools"
set +e
timeout --signal=TERM --kill-after=10s 30s "$ep12_astra" validate \
  outputs/astra/v0.0.14/astra.yaml --json \
  > "outputs/runs/observed_outgoing_description/20261005-astra-${SLURM_JOB_ID}.json" 2>&1
ep12_astra_exit=$?
set -e
printf 'astra_validation_exit=%s\n' "$ep12_astra_exit"
exec timeout --signal=TERM --kill-after=10s 1690s python3.12 -u -B \
  outputs/tools/summarize_observed_outgoing.py \
  --output-dir outputs/runs/observed_outgoing_description/20261005-description01 \
  --max-seconds 1680
