#!/usr/bin/env bash
#SBATCH --job-name=ep10-mop-targets
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --partition=russpold,normal
#SBATCH --output=/scratch/users/zijiao/br_autoresearch/episode10_single_cell_coprojection/ep10-mop-targets-%j.out
#SBATCH --error=/scratch/users/zijiao/br_autoresearch/episode10_single_cell_coprojection/ep10-mop-targets-%j.err

set -euo pipefail
umask 007

python_bin=/scratch/users/zijiao/br_autoresearch/episode11_projection_types_vs_gradients/ccf_env_py311/bin/python
script=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode10_single_cell_coprojection/outputs/stage0/tools/construct_mop_terminal_targets.py
stage0=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode10_single_cell_coprojection/outputs/stage0
assets=/scratch/users/zijiao/br_autoresearch/episode11_projection_types_vs_gradients/allen_outcome_assets
development=/scratch/users/zijiao/br_autoresearch/episode10_single_cell_coprojection/gao_mop_development_v1
output=/scratch/users/zijiao/br_autoresearch/episode10_single_cell_coprojection/gao_mop_development_targets_v1

"$python_bin" "$script" \
  --execute \
  --contract "$stage0/projection_observation_contract.yaml" \
  --vocabulary "$stage0/allen_target_vocabulary.csv" \
  --annotation "$assets/annotation_25_ccf_2017.nrrd" \
  --ontology "$assets/structure_graph_1.json" \
  --manifest "$stage0/gao_mop_development_manifest.csv" \
  --swc-root "$development/swc" \
  --output-root "$output"
