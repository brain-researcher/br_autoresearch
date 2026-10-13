#!/bin/bash
#SBATCH --job-name=ep04_neural_handoff
#SBATCH --partition=normal
#SBATCH --constraint=CPU_SKU:7502
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --array=0-4%3
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/source_readiness/20261003-development_neural_full01/logs/slurm-%A_%a.out
#SBATCH --error=outputs/runs/source_readiness/20261003-development_neural_full01/logs/slurm-%A_%a.err

set -euo pipefail
hostname
printf 'SLURM_JOB_ID=%s SLURM_ARRAY_TASK_ID=%s\n' "${SLURM_JOB_ID:?}" "${SLURM_ARRAY_TASK_ID:?}"
case "${SLURM_ARRAY_TASK_ID}" in 0|1|2|3|4) ;; *) exit 2 ;; esac
cd "${OAK:?}/users/zijiao/br_autoresearch/episode04_shared_scene_geometry"
module purge
module load devel python/3.12.1 math py-numpy/1.26.3
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
exec python3.12 outputs/code/prepare_development_neural_participant.py \
  --participant-index "${SLURM_ARRAY_TASK_ID}" \
  --source-root "${OAK}/users/zijiao/br_autoresearch_data/steward_acquisition/laion_fmri_mixed_role_20260922" \
  --metadata-dir outputs/runs/source_readiness/20261003-metadata_folds04 \
  --array-metadata outputs/runs/source_readiness/20261003-array_metadata02/array_metadata.json \
  --roi-mapping outputs/config/roi_mapping.json \
  --pilot-report outputs/runs/source_readiness/20261003-development_neural_pilot01/neural_handoff.json \
  --payload-root "${SCRATCH:?}/br_autoresearch/episode04_shared_scene_geometry/neural_full01" \
  --output-root outputs/runs/source_readiness/20261003-development_neural_full01 \
  --max-seconds 1740
