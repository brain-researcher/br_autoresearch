#!/bin/bash
#SBATCH --job-name=ep03-astra-validate
#SBATCH --partition=russpold,normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --chdir=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode03_condition_semantics_effect_maps
#SBATCH --output=/scratch/users/zijiao/br_autoresearch/episode03_condition_semantics_effect_maps/astra-validator/slurm-%j.out

# Pinned ASTRA 0.0.14 schema and semantic validation of the EP03 program.
set -uo pipefail
module purge
module load devel uv/0.8.4 python/3.12.1
root=/scratch/users/zijiao/br_autoresearch/episode03_condition_semantics_effect_maps/astra-validator
mkdir -p "$root/tmp" "$root/cache" "$root/tools"
export TMPDIR="$root/tmp" UV_CACHE_DIR="$root/cache" UV_TOOL_DIR="$root/tools" UV_PYTHON_DOWNLOADS=never
echo "host=$(hostname) job=$SLURM_JOB_ID"
"$(command -v uvx)" --no-managed-python --no-python-downloads \
  --python "$(command -v python3.12)" \
  --from astra-tools==0.2.18 --with astra-spec==0.0.14 \
  astra validate outputs/astra/v0.0.14/astra.yaml --skip-evidence \
  > "outputs/astra/v0.0.14/validation_20261016_job${SLURM_JOB_ID}.txt" 2>&1
status=$?
cat "outputs/astra/v0.0.14/validation_20261016_job${SLURM_JOB_ID}.txt"
echo "validate_exit=$status"
exit $status
