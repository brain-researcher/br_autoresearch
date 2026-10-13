#!/bin/bash
#SBATCH --job-name=ep21-model-assets
#SBATCH --partition=normal
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=outputs/source_qualification/attempt-014/slurm-%j.out

set -euo pipefail

episode_dir="/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode21_wang_neural_code_conversion_reproduction"
source_archive="/scratch/users/zijiao/br_autoresearch/episode21_wang_neural_code_conversion_reproduction/source_qualification/attempt-001/KamitaniLab/InterSiteNeuralCodeConversion-V1.0.0.zip"
asset_cache="${SCRATCH}/br_autoresearch/episode21_wang_neural_code_conversion_reproduction/public_model_assets"

cd "$episode_dir"
export PYTHONDONTWRITEBYTECODE=1
python outputs/source_qualification/tools/audit_public_model_payloads.py \
  --source-archive "$source_archive" \
  --cache-dir "$asset_cache" \
  --output outputs/source_qualification/attempt-014/public_model_payload_audit.json \
  --attempt 14
