#!/bin/bash
#SBATCH --job-name=ep12-oracle-refine
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency/oracle-refine-%j.out

# Follow only the next_action already written by a completed oracle aggregate.
# This wrapper adds no stopping threshold: it either executes the next
# predeclared look or stops when the aggregate says precision is complete or
# the bounded schedule is unresolved.
. /etc/profile >/dev/null
set -euo pipefail

EP12_ROOT=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency
EP12_FIRST_LOOK=${1:?first look index is required}
EP12_LAST_LOOK=${2:?last look index is required}
EP12_BANK_ROOT=${3:?bank root is required}

if (( EP12_FIRST_LOOK < 1 || EP12_LAST_LOOK < EP12_FIRST_LOOK )); then
  echo "invalid refinement look range" >&2
  exit 1
fi

cd "${EP12_ROOT}"
module purge
module load python/3.12.1 \
  py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 \
  py-scikit-learn/1.5.1_py312 py-pandas/2.2.1_py312 \
  py-pyarrow/18.1.0_py312
export PYTHONPATH="${EP12_ROOT}/outputs/executor${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

for ((EP12_LOOK=EP12_FIRST_LOOK; EP12_LOOK<=EP12_LAST_LOOK; EP12_LOOK++)); do
  EP12_PREVIOUS_LOOK=$((EP12_LOOK - 1))
  EP12_PREVIOUS_AGGREGATE=${EP12_BANK_ROOT}/qualification_look$(printf '%02d' "${EP12_PREVIOUS_LOOK}").json
  if [[ ! -f "${EP12_PREVIOUS_AGGREGATE}" ]]; then
    echo "missing prior aggregate: ${EP12_PREVIOUS_AGGREGATE}" >&2
    exit 1
  fi

  EP12_ACTION=$(python3 - "${EP12_PREVIOUS_AGGREGATE}" "${EP12_LOOK}" "${EP12_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

from ep12_executor.yaml_subset import load_yaml_subset

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
look_index = int(sys.argv[2])
root = Path(sys.argv[3])
spec = load_yaml_subset(
    root / "outputs" / "executor" / "SYNTHETIC_QUALIFICATION_SPEC_POST36_V3.yaml"
)
looks = [
    int(value)
    for value in spec["stochastic_oracle"]["tiers"]["qualification"][
        "planned_oracle_replicates_per_cell"
    ]
]
if not 1 <= look_index < len(looks):
    raise SystemExit("refinement look index is outside the frozen schedule")
if report.get("real_connectivity_accessed") is not False:
    raise SystemExit("oracle aggregate real-connectivity guard failed")
if report.get("final_connectivity_accessed") is not False:
    raise SystemExit("oracle aggregate final-connectivity guard failed")
action = report.get("next_action", {}).get("action")
if action not in {
    "oracle_precision_complete",
    "refine_oracle_bank",
    "unresolved_at_bounded_oracle_cost",
}:
    raise SystemExit("oracle aggregate next_action is invalid")
if int(report.get("completed_oracle_replicates_per_cell", -1)) != looks[look_index - 1]:
    raise SystemExit("oracle aggregate rows do not match the frozen prior look")
if action == "refine_oracle_bank" and int(
    report.get("next_action", {}).get("next_rows", -1)
) != looks[look_index]:
    raise SystemExit("oracle aggregate next_rows does not match the frozen schedule")
print(action)
PY
)

  case "${EP12_ACTION}" in
    oracle_precision_complete|unresolved_at_bounded_oracle_cost)
      echo "stopping refinement before look ${EP12_LOOK}: ${EP12_ACTION}"
      exit 0
      ;;
    refine_oracle_bank)
      EP12_OUTPUT=${EP12_BANK_ROOT}/qualification_look$(printf '%02d' "${EP12_LOOK}").json
      bash "${EP12_ROOT}/outputs/executor/slurm/run_stochastic_oracle_look.sbatch" \
        qualification "${EP12_LOOK}" "${EP12_BANK_ROOT}" "${EP12_OUTPUT}"
      ;;
  esac
done
