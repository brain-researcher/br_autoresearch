#!/bin/bash
#SBATCH --job-name=ep12-conformance
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency/conformance-%j.out

# Generated-data and source-code conformance only. This job has no real-data
# reader and cannot open development or final connectivity outcomes.
. /etc/profile >/dev/null
set -euo pipefail

EP12_ROOT=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency
EP12_SCRATCH=/scratch/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency
EP12_ATTEMPT=${EP12_ROOT}/outputs/executor_conformance/attempt-${SLURM_JOB_ID}
EP12_TMP=${EP12_SCRATCH}/executor_conformance/attempt-${SLURM_JOB_ID}

cd "${EP12_ROOT}"
module purge
module load python/3.12.1 \
  py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 \
  py-scikit-learn/1.5.1_py312 py-pandas/2.2.1_py312 \
  py-pyarrow/18.1.0_py312
export PYTHONPATH="${EP12_ROOT}/outputs/executor${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

mkdir -p "${EP12_TMP}" "${EP12_ATTEMPT}"
set +e
python3 -m unittest -v \
  outputs.executor.tests.test_scientific \
  outputs.executor.tests.test_scientific_oracle \
  outputs.executor.tests.test_fitted_null \
  outputs.executor.tests.test_mode_support \
  outputs.executor.tests.test_pre_null_materializer \
  outputs.executor.tests.test_coverage_successor \
  outputs.executor.tests.test_integration_refinement \
  outputs.executor.tests.test_production_readiness \
  outputs.executor.tests.test_resource_authorization \
  >"${EP12_TMP}/conformance_report.txt" 2>&1
EP12_STATUS=$?
set -e

install -m 0444 "${EP12_TMP}/conformance_report.txt" \
  "${EP12_ATTEMPT}/conformance_report.txt"
python3 - "${EP12_ATTEMPT}" "${EP12_STATUS}" "${SLURM_JOB_ID}" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

attempt = Path(sys.argv[1])
status = int(sys.argv[2])
job_id = sys.argv[3]
report = attempt / "conformance_report.txt"
payload = {
    "schema_version": "ep12.executor_conformance.v1",
    "astra_analysis_id": "executor_conformance",
    "job_id": job_id,
    "exit_code": status,
    "passed": status == 0,
    "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
    "real_connectivity_accessed": False,
    "final_connectivity_accessed": False,
}
temporary = attempt / f".conformance_manifest.json.tmp-{os.getpid()}"
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, attempt / "conformance_manifest.json")
PY

exit "${EP12_STATUS}"
