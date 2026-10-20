#!/bin/bash
#SBATCH --job-name=ep03-astra-validate
#SBATCH --partition=russpold,normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --chdir=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode03_condition_semantics_effect_maps
#SBATCH --output=/scratch/users/zijiao/br_autoresearch/episode03_condition_semantics_effect_maps/astra-validator/slurm-%j.out

# ASTRA 0.0.14 schema and semantic validation with the prebuilt brain-researcher-astra venv.
module purge
module load devel python/3.12.1
ASTRA=$GROUP_HOME/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv/bin/astra-tools
echo "host=$(hostname) job=$SLURM_JOB_ID"
timeout --signal=TERM --kill-after=10s 120s "$ASTRA" validate outputs/astra/v0.0.14/astra.yaml --json \
  > "outputs/astra/v0.0.14/validation_20261016_job${SLURM_JOB_ID}.json"
status=$?
cat "outputs/astra/v0.0.14/validation_20261016_job${SLURM_JOB_ID}.json"
echo "astra_exit=$status"
exit $status
