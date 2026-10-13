#!/bin/bash
#SBATCH --job-name=ep05-pls2
#SBATCH --partition=normal
#SBATCH --constraint=CPU_SKU:7502
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:35:00
#SBATCH --no-requeue

set -eo pipefail
hostname
printf 'SLURM_JOB_ID=%s\n' "${SLURM_JOB_ID:-}"
if [[ -z "${SLURM_JOB_ID:-}" || "$(hostname -s)" == *-ln* ]]; then
    exit 2
fi
source /etc/profile >/dev/null
set -u
module purge
module load devel python/3.12.1 math py-numpy/1.26.3 py-scipy/1.12.0_py312 py-scikit-learn/1.5.1_py312 py-h5py/3.10.0_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
EP_PYTHON=$(command -v python3.12)
printf 'selected_python=%s\n' "$EP_PYTHON"
cd "$OAK/users/zijiao/br_autoresearch/episode05_sensorimotor_lfp"
timeout --signal=TERM --kill-after=10s 2040s \
    "$EP_PYTHON" -B outputs/tools/run_primary_history_pls_panel.py remainder \
    --source "$OAK/data/br_autoresearch_data/sensorimotor_lfp/dryad-xd2547dkt-v5-payload/source" \
    --trial-table outputs/runs/source_support/attempt002/trial_identity.csv \
    --scalar-bank-root outputs/runs/adaptive_search/primary_reference_attempt001 \
    --hw-bank-root outputs/runs/adaptive_search/primary_history_repeatability_weight_control_attempt002 \
    --k-bank-root outputs/runs/adaptive_search/primary_capacity_attempt001 \
    --blend-bank-root outputs/runs/adaptive_search/primary_history_blend_attempt001 \
    --svd-bank-root outputs/runs/adaptive_search/primary_covariance_svd_control_attempt002 \
    --rbf-bank-root outputs/runs/adaptive_search/primary_history_rbf_attempt001 \
    --hw-summary outputs/runs/adaptive_search/primary_history_repeatability_weight_summary_attempt002/primary_history_repeatability_weight.json \
    --k-summary outputs/runs/adaptive_search/primary_capacity_summary_attempt001/primary_capacity.json \
    --blend-summary outputs/runs/adaptive_search/primary_history_blend_summary_attempt001/primary_history_blend.json \
    --svd-summary outputs/runs/adaptive_search/primary_covariance_svd_summary_attempt001/primary_covariance_svd.json \
    --rbf-summary outputs/runs/adaptive_search/primary_history_rbf_summary_attempt001/primary_history_rbf.json \
    --bank-root outputs/runs/adaptive_search/primary_history_pls_attempt001 \
    --summary-out outputs/runs/adaptive_search/primary_history_pls_summary_attempt001 \
    --out outputs/runs/adaptive_search/primary_history_pls_remainder_attempt001 \
    --workers 1 --max-seconds 1980
