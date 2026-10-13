#!/bin/bash
# Selected original full readout. The coordinator supplies afterok:<whole_draw_array_id>.
# No individual element, point-only stage or partial draw set can substitute.
#SBATCH --job-name=ep21-fa-full-finalize
#SBATCH --partition=russpold
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --time=02:00:00
#SBATCH --kill-on-invalid-dep=yes
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/finding_a_feature_decoding/full001/finalize/slurm-%j.out
#SBATCH --error=outputs/runs/finding_a_feature_decoding/full001/finalize/slurm-%j.err
set -euo pipefail
[[ $# -eq 0 ]] || exit 2
source "${OAK:?}/users/zijiao/br_autoresearch/episode21_wang_neural_code_conversion_reproduction/outputs/converter_training/finding_a_full_common.sh" finalize
