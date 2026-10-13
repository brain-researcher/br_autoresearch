#!/bin/bash
#SBATCH --job-name=ep12-oracle-look
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency/oracle-look-%j.out

# Execute one complete predeclared oracle look serially.  Grouping all cells in
# one allocation avoids dozens of two-minute jobs and makes the look-level
# aggregate an exact fail-closed successor of every cell shard.
. /etc/profile >/dev/null
set -euo pipefail

EP12_ROOT=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency
EP12_TIER=${1:?tier is required}
EP12_LOOK=${2:?look index is required}
EP12_BANK_ROOT=${3:?bank root or auto is required}
EP12_OUTPUT=${4:?aggregate output or auto is required}

if [[ "${EP12_BANK_ROOT}" == auto ]]; then
  EP12_BANK_ROOT=${EP12_ROOT}/outputs/scientific_qualification/stochastic_oracle_v3/${EP12_TIER}_attempt-${SLURM_JOB_ID:?auto paths require Slurm}
fi
if [[ "${EP12_OUTPUT}" == auto ]]; then
  EP12_OUTPUT=${EP12_BANK_ROOT}/${EP12_TIER}_look$(printf '%02d' "${EP12_LOOK}").json
fi

cd "${EP12_ROOT}"
module purge
module load python/3.12.1 \
  py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 \
  py-scikit-learn/1.5.1_py312 py-pandas/2.2.1_py312 \
  py-pyarrow/18.1.0_py312
export PYTHONPATH="${EP12_ROOT}/outputs/executor${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

EP12_CELL_COUNT=$(python3 - "${EP12_TIER}" <<'PY'
import sys
from pathlib import Path

from ep12_executor.scientific_qualification import qualification_cells
from ep12_executor.yaml_subset import load_yaml_subset

tier = sys.argv[1]
root = Path.cwd()
spec = load_yaml_subset(
    root / "outputs" / "executor" / "SYNTHETIC_QUALIFICATION_SPEC_POST36_V3.yaml"
)
tier_spec = spec["stochastic_oracle"]["tiers"][tier]
cells = qualification_cells(str(tier_spec["qualification_cell_tier"]), spec)
print(min(len(cells), int(tier_spec.get("cell_limit", len(cells)))))
PY
)

EP12_LOOK_DIR=${EP12_BANK_ROOT}/look_$(printf '%02d' "${EP12_LOOK}")
if compgen -G "${EP12_LOOK_DIR}/cell_*.json" >/dev/null; then
  echo "refusing to mix a new look with existing cell shards: ${EP12_LOOK_DIR}" >&2
  exit 1
fi

for ((EP12_CELL_INDEX=0; EP12_CELL_INDEX<EP12_CELL_COUNT; EP12_CELL_INDEX++)); do
  python3 -m ep12_executor run-oracle-shard \
    --tier "${EP12_TIER}" \
    --look-index "${EP12_LOOK}" \
    --cell-index "${EP12_CELL_INDEX}" \
    --output-dir "${EP12_LOOK_DIR}" \
    --contract "${EP12_ROOT}/outputs/executor/SCIENTIFIC_CONTRACT_POST36_V3.yaml" \
    --qualification-spec "${EP12_ROOT}/outputs/executor/SYNTHETIC_QUALIFICATION_SPEC_POST36_V3.yaml"
done

python3 -m ep12_executor aggregate-oracle-look \
  --tier "${EP12_TIER}" \
  --look-index "${EP12_LOOK}" \
  --bank-root "${EP12_BANK_ROOT}" \
  --output "${EP12_OUTPUT}" \
  --qualification-spec "${EP12_ROOT}/outputs/executor/SYNTHETIC_QUALIFICATION_SPEC_POST36_V3.yaml"
