#!/bin/bash
#SBATCH --job-name=ep12-qualify-v3
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency/qualify-v3-%j.out

# Release full generated-method qualification only from an oracle aggregate
# that has reached the frozen precision rule.  No real-connectivity reader is
# reachable from this wrapper or either child command.
. /etc/profile >/dev/null
set -euo pipefail

EP12_ROOT=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency
EP12_BANK_ROOT=${1:?oracle bank root is required}
EP12_OUTPUT=${2:-${EP12_ROOT}/outputs/scientific_qualification/v3_qualification_attempt-${SLURM_JOB_ID}}

cd "${EP12_ROOT}"
module purge
module load python/3.12.1 \
  py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 \
  py-scikit-learn/1.5.1_py312 py-pandas/2.2.1_py312 \
  py-pyarrow/18.1.0_py312

EP12_ORACLE_AGGREGATE=$(python3 - "${EP12_BANK_ROOT}" <<'PY'
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
pattern = re.compile(r"qualification_look(\d+)\.json$")
candidates = []
for path in root.glob("qualification_look*.json"):
    match = pattern.fullmatch(path.name)
    if match:
        candidates.append((int(match.group(1)), path))
if not candidates:
    raise SystemExit("no qualification oracle aggregate exists")
look_index, path = max(candidates)
report = json.loads(path.read_text(encoding="utf-8"))
if report.get("scientific_gate") is not True:
    raise SystemExit("latest oracle aggregate is not a scientific gate")
if report.get("all_cells_precision_met") is not True:
    raise SystemExit("latest oracle aggregate has not reached precision")
if report.get("next_action", {}).get("action") != "oracle_precision_complete":
    raise SystemExit("latest oracle aggregate does not release qualification")
if report.get("real_connectivity_accessed") is not False:
    raise SystemExit("oracle aggregate real-connectivity guard failed")
if report.get("final_connectivity_accessed") is not False:
    raise SystemExit("oracle aggregate final-connectivity guard failed")
print(path)
PY
)

bash "${EP12_ROOT}/outputs/executor/slurm/qualify_executor_conformance.sbatch"
bash "${EP12_ROOT}/outputs/executor/slurm/qualify_methods.sbatch" \
  qualification "${EP12_OUTPUT}" "${EP12_ORACLE_AGGREGATE}"
