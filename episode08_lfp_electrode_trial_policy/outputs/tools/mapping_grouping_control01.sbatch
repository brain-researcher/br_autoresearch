#!/bin/bash
#SBATCH --job-name=ep08_mapping_group_control
#SBATCH --partition=normal
#SBATCH --constraint=CPU_SKU:7502
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:20:00
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/development_search/mapping_grouping_control_logs01/slurm-%j.out
#SBATCH --error=outputs/runs/development_search/mapping_grouping_control_logs01/slurm-%j.err

set -euo pipefail
hostname
printf 'SLURM_JOB_ID=%s SLURM_CPUS_PER_TASK=%s\n' "${SLURM_JOB_ID:?}" "${SLURM_CPUS_PER_TASK:?}"
cd "${OAK:?}/users/zijiao/br_autoresearch/episode08_lfp_electrode_trial_policy"
module purge
module load devel python/3.12.1 math py-numpy/1.26.3 py-scikit-learn/1.5.1_py312 py-h5py/3.10.0_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
# Root reuses the successful focused tests and ASTRA validation from its PTY.
# This finite batch runs only the selected substantive comparison.
exec timeout 1200s python3.12 outputs/tools/development_mapping_grouping.py \
  --source-root "${OAK}/data/br_autoresearch_data/sensorimotor_lfp/dryad-xd2547dkt-v5-payload/source" \
  --metadata-report outputs/runs/source_support/attempt002/metadata_support.json \
  --trial-roles outputs/runs/source_support/development_trial_roles_attempt001/development_trial_roles.json \
  --mapping-report outputs/runs/development_search/mapping_variance_attempt001/mapping_variance.json \
  --original-report outputs/runs/development_search/trial_allocation_attempt001/trial_allocation.json \
  --arrival-report outputs/runs/development_search/arrival_order_attempt001/arrival_order.json \
  --max-seconds 1140 \
  --output-dir outputs/runs/development_search/mapping_grouping_control_attempt001
