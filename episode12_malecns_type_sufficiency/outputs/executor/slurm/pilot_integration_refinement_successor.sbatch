#!/bin/bash
#SBATCH --job-name=ep12-int-refine2
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency/integration-refinement-successor-%j.out

# Generated-only successor to the 512-versus-1,024 numerical pilot.  The
# tolerance and exact selected-model identity rule remain unchanged.
# shellcheck source=/etc/profile
. /etc/profile >/dev/null
set -euo pipefail

EP12_ROOT=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency
EP12_TMP_ROOT=/scratch/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency
EP12_OUTPUT=${1:-${EP12_ROOT}/outputs/scientific_qualification/pilots/integration_refinement_common_random_1024_2048}

cd "${EP12_ROOT}"
module purge
module load python/3.12.1 \
  py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 \
  py-scikit-learn/1.5.1_py312 py-pandas/2.2.1_py312 \
  py-pyarrow/18.1.0_py312
export PYTHONPATH="${EP12_ROOT}/outputs/executor${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p "${EP12_TMP_ROOT}"
export TMPDIR="${EP12_TMP_ROOT}"

python3 -m ep12_executor.integration_refinement \
  --output-dir "${EP12_OUTPUT}" \
  --draw-grid 512 1024 2048 \
  --candidate-draws 1024 \
  --high-draws 2048
