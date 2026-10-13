#!/bin/bash
#SBATCH --job-name=ep21-source-qual
#SBATCH --partition=normal
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --chdir=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode21_wang_neural_code_conversion_reproduction
#SBATCH --output=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode21_wang_neural_code_conversion_reproduction/outputs/source_qualification/attempt-016/slurm-%j.out

set -euo pipefail

export EP21_LEDGER_ATTEMPT=16
export PYTHONDONTWRITEBYTECODE=1

validator_root="${SCRATCH}/br_autoresearch/episode21_wang_neural_code_conversion_reproduction/astra-validator"
mkdir -p "${validator_root}/tmp" "${validator_root}/cache"
TMPDIR="${validator_root}/tmp" \
UV_CACHE_DIR="${validator_root}/cache" \
uvx --python 3.12 \
  --from astra-tools==0.2.18 \
  --with astra-spec==0.0.14 \
  astra validate outputs/astra/v0.0.14/astra.yaml --skip-evidence

python outputs/source_qualification/tools/qualify_output.py \
  --structural-manifest inputs/conversion_structural_manifest.json \
  --asset-inventory inputs/upstream_pipeline_asset_inventory.json \
  --output outputs/source_qualification/attempt-016/qualification_report.json \
  --attempt "$EP21_LEDGER_ATTEMPT"

python outputs/source_qualification/tools/render_branch.py \
  --branch laion \
  --qualification outputs/source_qualification/attempt-016/qualification_report.json \
  --output outputs/source_qualification/attempt-016/laion_branch_qualification.json

python outputs/source_qualification/tools/render_branch.py \
  --branch external \
  --qualification outputs/source_qualification/attempt-016/qualification_report.json \
  --output outputs/source_qualification/attempt-016/external_branch_qualification.json
