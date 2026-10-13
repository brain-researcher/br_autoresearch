#!/bin/bash
# Full approved unique-development control, sized from the completed 60-image run.
#SBATCH --job-name=ep04-lowlevel-full
#SBATCH --partition=russpold
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=03:00:00
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/source_readiness/20261005-low_level_full01-%j.out
#SBATCH --error=outputs/runs/source_readiness/20261005-low_level_full01-%j.err
set -euo pipefail
[[ $# -eq 0 ]] || exit 2
source /share/software/user/open/lmod/lmod/init/bash
cd "${OAK:?}/users/zijiao/br_autoresearch/episode04_shared_scene_geometry"
bash -n outputs/code/low_level_full01.sbatch
exec bash outputs/code/run_low_level_control.sh full 10680 \
  outputs/runs/source_readiness/20261005-low_level_full01
