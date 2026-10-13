#!/bin/bash
# Selected original full readout. The coordinator supplies afterok:<decode_job_id>.
# Each of the 63 elements is a separate bounded read-only draw batch.
#SBATCH --job-name=ep21-fa-full-draw
#SBATCH --partition=russpold
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-62%4
#SBATCH --kill-on-invalid-dep=yes
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/finding_a_feature_decoding/full001/draws/slurm-%A_%a.out
#SBATCH --error=outputs/runs/finding_a_feature_decoding/full001/draws/slurm-%A_%a.err
set -euo pipefail
[[ $# -eq 0 ]] || exit 2
source "${OAK:?}/users/zijiao/br_autoresearch/episode21_wang_neural_code_conversion_reproduction/outputs/converter_training/finding_a_full_common.sh" draw
