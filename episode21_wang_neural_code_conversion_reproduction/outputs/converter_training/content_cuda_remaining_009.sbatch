#!/bin/bash
# One separately granted original full content pair; no automatic successor.
#SBATCH --job-name=ep21-content-p009
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gpus=1
#SBATCH --constraint="GPU_SKU:V100_SXM2&GPU_MEM:32GB"
#SBATCH --time=2-00:00:00
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/content_pair_training/sub05_to_sub03_cuda001/slurm-%j.out
#SBATCH --error=outputs/runs/content_pair_training/sub05_to_sub03_cuda001/slurm-%j.err
set -euo pipefail
[[ $# -eq 0 ]] || exit 2
exec bash "${OAK:?}/users/zijiao/br_autoresearch/episode21_wang_neural_code_conversion_reproduction/outputs/converter_training/content_cuda_remaining_cpuqualified_common.sh" remaining-009
