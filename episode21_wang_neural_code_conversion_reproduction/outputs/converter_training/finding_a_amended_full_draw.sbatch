#!/bin/bash
# Coordinator binds afterok to the new amended decode job. All63 batches and
# original1000 synchronized image draws remain; array elements are distinct jobs.
#SBATCH --job-name=ep21-fa-amended-draw
#SBATCH --partition=russpold
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-62%4
#SBATCH --kill-on-invalid-dep=yes
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/finding_a_feature_decoding/amended_full001/draws/slurm-%A_%a.out
#SBATCH --error=outputs/runs/finding_a_feature_decoding/amended_full001/draws/slurm-%A_%a.err
set -euo pipefail
[[ $# -eq 0 ]] || exit 2
source "${OAK:?}/users/zijiao/br_autoresearch/episode21_wang_neural_code_conversion_reproduction/outputs/converter_training/finding_a_amended_full_common.sh" draw
