#!/bin/bash
#SBATCH --job-name=ep14-provider-pca
#SBATCH --partition=normal,russpold
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/provider_group_compression/logs/compression-%j.out
#SBATCH --error=outputs/runs/provider_group_compression/logs/compression-%j.err

set -euo pipefail
hostname
printf 'SLURM_JOB_ID=%s\n' "${SLURM_JOB_ID:?}"
case "$(hostname)" in *-ln*) exit 2;; esac
cd "${OAK:?}/users/zijiao/br_autoresearch/episode14_openbhb_roi_site_generalization"
source /share/software/user/open/lmod/lmod/init/bash
unset PYTHONPATH PYTHONHOME
module purge
module load devel math python/3.12.1 py-numpy/1.26.3_py312
module load py-scikit-learn/1.5.1_py312 viz py-matplotlib/3.8.3_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export MPLCONFIGDIR="${L_SCRATCH_JOB:?}/ep14-matplotlib"
mkdir -p "$MPLCONFIGDIR"
PYTHON=/share/software/user/open/python/3.12.1/bin/python3

timeout --signal=TERM --kill-after=5s 90s "$PYTHON" \
  outputs/tools/test_provider_compression.py -v
timeout --signal=TERM --kill-after=15s 3450s "$PYTHON" \
  outputs/tools/run_provider_compression.py \
  --out outputs/runs/provider_group_compression/attempt001 --workers 3 --max-seconds 3300
timeout --signal=TERM --kill-after=5s 30s "$PYTHON" \
  outputs/tools/plot_provider_compression.py \
  --report outputs/runs/provider_group_compression/attempt001/provider_compression_report.json \
  --out outputs/figures/provider_group_compression/attempt001

# Documentary validation of the changed program; not a science gate.
ep14_astra="${GROUP_HOME:?}/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv"
ep14_deps="$GROUP_HOME/users/zijiao/ep18_model_runtime_cpu241_py312_46842156"
env PYTHONPATH="$ep14_astra/lib/python3.12/site-packages:$ep14_deps/lib/python3.12/site-packages" timeout --signal=TERM --kill-after=5s 30s "$ep14_astra/bin/astra-tools" validate outputs/astra/v0.0.14/astra.yaml --json > "outputs/runs/provider_group_compression/logs/astra-${SLURM_JOB_ID}.txt" 2>&1 || true
