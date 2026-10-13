#!/bin/bash
#SBATCH --job-name=ep12-attest36
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --output=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency/outputs/development_run002/slurm-observed-attestation-%j.out

# Submit only after all 18+9+9 observed trials have durable v1 manifests.
# This job reads source artifacts, extracts structural/status metadata only,
# and writes a separate attestation root.  It never modifies a trial directory,
# compares objectives, submits successor work, or accesses final connectivity.
set -euo pipefail

EP12_ROOT=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency
EP12_RUN_ROOT="${EP12_ROOT}/outputs/development_run002"

cd "${EP12_ROOT}"
module purge
module load python/3.12.1

export PYTHONPATH="${EP12_ROOT}/outputs/executor:${PYTHONPATH:-}"

python3 -m ep12_executor.attest_observed_trials \
  --run-root "${EP12_RUN_ROOT}" \
  --output-dir "${EP12_RUN_ROOT}/observed_attestation" \
  --policy "${EP12_ROOT}/SEARCH_POLICY.yaml"
