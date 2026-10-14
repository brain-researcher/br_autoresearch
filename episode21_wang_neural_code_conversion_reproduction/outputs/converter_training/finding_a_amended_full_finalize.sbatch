#!/bin/bash
# Coordinator binds afterok to the entire new amended draw array. Every63
# closed successful result is mandatory; this is the only final publication.
#SBATCH --job-name=ep21-fa-amended-finalize
#SBATCH --partition=russpold
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --time=02:00:00
#SBATCH --kill-on-invalid-dep=yes
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/finding_a_feature_decoding/amended_full001/finalize/slurm-%j.out
#SBATCH --error=outputs/runs/finding_a_feature_decoding/amended_full001/finalize/slurm-%j.err
set -euo pipefail
[[ $# -eq 0 ]] || exit 2
source "${OAK:?}/users/zijiao/br_autoresearch/episode21_wang_neural_code_conversion_reproduction/outputs/converter_training/finding_a_amended_full_common.sh" finalize
