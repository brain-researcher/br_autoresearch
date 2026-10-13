#!/bin/bash -l
#SBATCH --job-name=ep01-null-single-wave
#SBATCH --partition=normal
#SBATCH --constraint=CPU_SKU:7502
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:35:00
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/behavioral_mechanism_kill_test/20261005-M0-null-single-group-%A_%a.out
#SBATCH --error=outputs/runs/behavioral_mechanism_kill_test/20261005-M0-null-single-group-%A_%a.err

# Root selects exactly MC or MV after that model's representative qualifies.
# Reuse the finite grouped implementation inside this existing allocation;
# its3h SBATCH header is only a comment when invoked by bash. No resubmission.
set -euo pipefail
if (( $# < 1 || $# > 2 )); then printf 'Select MC or MV and optional qualification group1 or8\n' >&2; exit 2; fi
case "$1" in
  MC|MV) ;;
  *) printf 'Single-component95m entry permits only MC or MV\n' >&2; exit 2 ;;
esac
cd "$SLURM_SUBMIT_DIR"
exec bash outputs/code/mz_null_indexed_simple_groups.sbatch "$@"
