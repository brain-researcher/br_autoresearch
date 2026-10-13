#!/bin/bash
# Selected original full readout. The coordinator supplies afterok only for model parents
# still active, and binds verified completed parents separately (32 in total).
# Resources, the profile and procedural job bindings must first be selected.
#SBATCH --job-name=ep21-fa-full-decode
#SBATCH --partition=russpold
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=16G
#SBATCH --time=3-00:00:00
#SBATCH --kill-on-invalid-dep=yes
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/finding_a_feature_decoding/full001/decode/slurm-%j.out
#SBATCH --error=outputs/runs/finding_a_feature_decoding/full001/decode/slurm-%j.err
set -euo pipefail
[[ $# -eq 0 ]] || exit 2
source "${OAK:?}/users/zijiao/br_autoresearch/episode21_wang_neural_code_conversion_reproduction/outputs/converter_training/finding_a_full_common.sh" decode
