#!/bin/bash
# New amended readout only after a separately selected grant and forty exact
# completed/active model parents. Coordinator supplies actual afterok bindings.
# Original held decode and original sources remain unchanged.
#SBATCH --job-name=ep21-fa-amended-decode
#SBATCH --partition=russpold
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=16G
#SBATCH --time=3-00:00:00
#SBATCH --kill-on-invalid-dep=yes
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/finding_a_feature_decoding/amended_full001/decode/slurm-%j.out
#SBATCH --error=outputs/runs/finding_a_feature_decoding/amended_full001/decode/slurm-%j.err
set -euo pipefail
[[ $# -eq 0 ]] || exit 2
source "${OAK:?}/users/zijiao/br_autoresearch/episode21_wang_neural_code_conversion_reproduction/outputs/converter_training/finding_a_amended_full_common.sh" decode
