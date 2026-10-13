#!/bin/bash
#SBATCH --job-name=ep11-nested-frontier
#SBATCH --partition=russpold
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/scratch/users/zijiao/br_autoresearch/episode11_projection_types_vs_gradients/nested-frontier-%j.out

set -euo pipefail

EP11_REPO_ROOT=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch
EP11_EPISODE_ROOT=${EP11_REPO_ROOT}/episode11_projection_types_vs_gradients
EP11_RUN_ROOT=${SCRATCH}/br_autoresearch/episode11_projection_types_vs_gradients/nested_frontier
EP11_RUN_DIR=${EP11_RUN_ROOT}/run_${SLURM_JOB_ID}
EP11_GROUP_ENV=/home/groups/russpold/zijiao/envs/ep11-geometry-py312
EP11_PIP_CACHE=${SCRATCH}/br_autoresearch/episode11_projection_types_vs_gradients/pip-cache

ml purge
ml load devel math
ml load python/3.12.1 \
  py-numpy/1.26.3_py312 \
  py-scipy/1.12.0_py312 \
  py-h5py/3.10.0_py312 \
  py-pandas/2.2.1_py312 \
  py-scikit-image/0.24.0_py312

mkdir -p "${EP11_RUN_DIR}/results" "${EP11_PIP_CACHE}"
export PIP_CACHE_DIR=${EP11_PIP_CACHE}
export XDG_CACHE_HOME=${EP11_PIP_CACHE}/xdg
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# ccf-streamlines and pynrrd are not Sherlock modules. Create or repair the
# small persistent group environment only inside the compute job; numerical
# dependencies continue to come from the loaded modules.
if [[ ! -x "${EP11_GROUP_ENV}/bin/python" ]]; then
  python3 -m venv --system-site-packages "${EP11_GROUP_ENV}"
fi
if ! "${EP11_GROUP_ENV}/bin/python" -c \
  'from importlib.metadata import version; from ccf_streamlines.linestring3d import LineString3D; import nrrd; assert version("ccf-streamlines") == "1.2.0"; assert version("pynrrd") == "1.1.3"' \
  >/dev/null 2>&1; then
  "${EP11_GROUP_ENV}/bin/python" -m pip install \
    --disable-pip-version-check \
    --no-deps \
    'ccf-streamlines==1.2.0' \
    'pynrrd==1.1.3' \
    'tqdm==4.67.1' \
    'typing_extensions==4.15.0'
fi

cp \
  "${EP11_EPISODE_ROOT}/outputs/stage0/tools/build_whole_cortex_ssptr_geometry_support.py" \
  "${EP11_RUN_DIR}/"
cp \
  "${EP11_EPISODE_ROOT}/outputs/stage0/tools/run_whole_cortex_ssptr_nested_role_sensitivity.py" \
  "${EP11_RUN_DIR}/"
cp \
  "${EP11_EPISODE_ROOT}/outputs/stage0/tools/run_whole_cortex_ssptr_nested_frontier.py" \
  "${EP11_RUN_DIR}/"
cp \
  "${EP11_EPISODE_ROOT}/outputs/stage0/whole_cortex_ssptr_inventory.csv" \
  "${EP11_RUN_DIR}/"

"${EP11_GROUP_ENV}/bin/python" - <<'PY'
import sys
import ccf_streamlines
import h5py
import nrrd
import numpy
import scipy

print(
    "versions",
    {
        "python": sys.version.split()[0],
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "h5py": h5py.__version__,
        "ccf_streamlines": ccf_streamlines.__version__,
        "pynrrd": nrrd.__version__,
    },
)
PY

"${EP11_GROUP_ENV}/bin/python" \
  "${EP11_RUN_DIR}/run_whole_cortex_ssptr_nested_frontier.py" \
  --self-test

"${EP11_GROUP_ENV}/bin/python" \
  "${EP11_RUN_DIR}/run_whole_cortex_ssptr_nested_frontier.py" \
  --inventory "${EP11_RUN_DIR}/whole_cortex_ssptr_inventory.csv" \
  --output-csv \
    "${EP11_RUN_DIR}/results/whole_cortex_ssptr_nested_frontier_sensitivity.csv" \
  --output-md \
    "${EP11_RUN_DIR}/results/whole_cortex_ssptr_nested_frontier_sensitivity.md" \
  --milp-time-limit-seconds 120

echo "EP11_NESTED_FRONTIER_RUN_DIR=${EP11_RUN_DIR}"
