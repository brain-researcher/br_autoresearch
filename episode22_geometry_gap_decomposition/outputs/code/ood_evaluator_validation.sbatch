#!/bin/bash
#SBATCH --job-name=ep22-ood-validate
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --chdir=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode22_geometry_gap_decomposition
#SBATCH --output=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode22_geometry_gap_decomposition/outputs/runs/ood_evaluator_validation/20261012-attempt02/slurm-%j.out
# EP22 pre-lock validation of the OOD evaluator. No OOD response value is read:
# synthetic known-answer tests, outcome-blind coverage (trial metadata), and the
# full pipeline on pseudo-OOD development images with a bit-for-bit reader check.
set -uo pipefail
attempt=outputs/runs/ood_evaluator_validation/20261012-attempt02
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} start=$(date -u +%FT%TZ)"
module purge
module load devel python/3.12.1 py-h5py/3.10.0_py312 math py-numpy/1.26.3 py-scipy/1.12.0_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
timeout 900s python3.12 -u -B outputs/tests/test_ood_confirmation.py > "$attempt/synthetic_tests.txt" 2>&1
echo "tests_exit=$?"
(cd outputs/code && timeout 900s python3.12 -u -B ood_coverage_check.py "../../$attempt/ood_coverage_check.json")
echo "coverage_exit=$?"
timeout 4200s python3.12 -u -B outputs/code/ood_confirmation.py --mode dev-test --workers 5 --permutations 1000 \
  --out-dir "$attempt"
echo "devtest_exit=$? end=$(date -u +%FT%TZ)"
