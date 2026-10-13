#!/bin/bash
# One original brain pair; fixed grant, no retry or automatic successor.
#SBATCH --job-name=ep21-brain-p017
#SBATCH --partition=russpold
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/brain_pair_training/sub07_to_sub03_001/slurm-%j.out
#SBATCH --error=outputs/runs/brain_pair_training/sub07_to_sub03_001/slurm-%j.err
set -euo pipefail
[[ $# -eq 0 ]] || exit 2
exec bash "${OAK:?}/users/zijiao/br_autoresearch/episode21_wang_neural_code_conversion_reproduction/outputs/converter_training/brain_pair_remaining_common.sh" remaining-017
