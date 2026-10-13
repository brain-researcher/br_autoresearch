#!/bin/bash
#SBATCH --job-name=ep21-source-qual
#SBATCH --partition=normal
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --chdir=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode21_wang_neural_code_conversion_reproduction
#SBATCH --output=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode21_wang_neural_code_conversion_reproduction/outputs/source_qualification/attempt-017/slurm-%j.out

set -euo pipefail

export EP21_LEDGER_ATTEMPT=17
export PYTHONDONTWRITEBYTECODE=1

module purge
module load devel uv/0.8.4 python/3.12.1
uvx_path="$(command -v uvx)"
python_path="$(command -v python3.12)"
printf 'uvx_path=%s\n' "$uvx_path"
"$uvx_path" --version
printf 'python_path=%s\n' "$python_path"
"$python_path" --version

validator_root="${SCRATCH}/br_autoresearch/episode21_wang_neural_code_conversion_reproduction/astra-validator"
mkdir -p "${validator_root}/tmp" "${validator_root}/cache" "${validator_root}/tools"
export TMPDIR="${validator_root}/tmp"
export UV_CACHE_DIR="${validator_root}/cache"
export UV_TOOL_DIR="${validator_root}/tools"
export UV_PYTHON_DOWNLOADS=never

"$uvx_path" \
  --no-managed-python \
  --no-python-downloads \
  --python "$python_path" \
  --from astra-tools==0.2.18 \
  --with astra-spec==0.0.14 \
  astra validate outputs/astra/v0.0.14/astra.yaml --skip-evidence

"$python_path" outputs/source_qualification/tools/qualify_output.py \
  --goal GOAL.md \
  --datasets DATASETS.md \
  --protocol-table PROTOCOL_TABLE.yaml \
  --input-contract inputs/README.md \
  --structural-manifest inputs/conversion_structural_manifest.json \
  --asset-inventory inputs/upstream_pipeline_asset_inventory.json \
  --output outputs/source_qualification/attempt-017/qualification_report.json \
  --attempt "$EP21_LEDGER_ATTEMPT"

"$python_path" outputs/source_qualification/tools/render_branch.py \
  --branch laion \
  --qualification outputs/source_qualification/attempt-017/qualification_report.json \
  --output outputs/source_qualification/attempt-017/laion_branch_qualification.json

"$python_path" outputs/source_qualification/tools/render_branch.py \
  --branch external \
  --qualification outputs/source_qualification/attempt-017/qualification_report.json \
  --output outputs/source_qualification/attempt-017/external_branch_qualification.json
