#!/bin/bash
#SBATCH --job-name=ep12-procedure-lock
#SBATCH --partition=normal
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency/procedure-lock-%j.out

# Build the development-only pre-null inputs, freeze the one scientific
# procedure lock, materialize its runtime, and dispatch only within the explicit
# resource authorization. This wrapper never opens final connectivity.
set -euo pipefail

EP12_ROOT=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency
EP12_SCRATCH=/scratch/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency
EP12_RUN_ROOT=${EP12_ROOT}/outputs/development_run002
EP12_PRE_NULL=${EP12_RUN_ROOT}/pre_null_inputs
EP12_LOCK_DIR=${EP12_RUN_ROOT}/procedure_lock
EP12_RUNTIME_BUNDLE=${EP12_RUN_ROOT}/full_search_runtime_bundle
EP12_RUNTIME_SCRATCH=${EP12_SCRATCH}/development_run002/full_search_runtime_bundle
EP12_NULL_WORK=${EP12_SCRATCH}/development_run002/full_search_null
EP12_NULL_OUTPUT=${EP12_RUN_ROOT}/full_search_null
EP12_AUTHORIZATION=${1:-${EP12_ROOT}/outputs/executor/DEVELOPMENT_BUDGET_OVERRIDE.yaml}

cd "${EP12_ROOT}"
module purge
module load python/3.12.1 \
  py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 \
  py-scikit-learn/1.5.1_py312 py-pandas/2.2.1_py312 \
  py-pyarrow/18.1.0_py312
export PYTHONPATH="${EP12_ROOT}/outputs/executor${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

python3 -m ep12_executor.pre_null_materializer \
  --output-dir "${EP12_PRE_NULL}" \
  --run-root "${EP12_RUN_ROOT}" \
  --observed-finalization-dir "${EP12_RUN_ROOT}/observed_finalization" \
  --tail-root "${EP12_RUN_ROOT}/post36_tail" \
  --snapshot-dir "${EP12_SCRATCH}/development_snapshot_v1" \
  --materialization-dir "${EP12_ROOT}/outputs/development_run001/materialization" \
  --preparation-dir "${EP12_RUN_ROOT}/preparation" \
  --policy "${EP12_ROOT}/SEARCH_POLICY.yaml" \
  --prefix-contract "${EP12_ROOT}/outputs/executor/SCIENTIFIC_CONTRACT.yaml" \
  --contract "${EP12_ROOT}/outputs/executor/SCIENTIFIC_CONTRACT_POST36.yaml" \
  --post36-plan "${EP12_ROOT}/outputs/executor/OBSERVED_POST36_TAIL_PLAN.json"

python3 -m ep12_executor.procedure_lock \
  --policy "${EP12_ROOT}/SEARCH_POLICY.yaml" \
  --prefix-contract "${EP12_ROOT}/outputs/executor/SCIENTIFIC_CONTRACT.yaml" \
  --active-contract "${EP12_ROOT}/outputs/executor/SCIENTIFIC_CONTRACT_POST36.yaml" \
  --comparison-archive "${EP12_PRE_NULL}/complete_comparison_archive.json" \
  --mode-support "${EP12_PRE_NULL}/mode_support.json" \
  --realized-schedule "${EP12_RUN_ROOT}/post36_tail/realized_schedule.json" \
  --control-results "${EP12_PRE_NULL}/required_control_results.json" \
  --null-generator-selection "${EP12_PRE_NULL}/null_generator_selection.json" \
  --null-controller-program "${EP12_PRE_NULL}/null_controller_program.json" \
  --pre-null-manifest "${EP12_PRE_NULL}/manifest.json" \
  --runtime-source-binding "${EP12_PRE_NULL}/runtime_source_binding.json" \
  --output-dir "${EP12_LOCK_DIR}"

EP12_LOCK=${EP12_LOCK_DIR}/procedure_lock.json
EP12_NULL_ELIGIBLE=$(python3 -c 'import json,sys; print("1" if json.load(open(sys.argv[1]))["null_eligible"] else "0")' "${EP12_LOCK}")
if [[ "${EP12_NULL_ELIGIBLE}" == 1 ]]; then
  EP12_MAX_TIMING_REPORT=$(python3 -c '
from pathlib import Path
import sys
from ep12_executor.policy import EpisodePolicy
from ep12_executor.pre_null_materializer import load_observed_trial_status
root=Path(sys.argv[1]); policy=EpisodePolicy.load(Path(sys.argv[2]))
status=load_observed_trial_status(run_root=root, policy=policy)
row=min(status["rows_by_id"].values(), key=lambda value:(-float(value["wall_seconds"]), str(value["trial_id"])))
print(row["trial_report_path"])
' "${EP12_RUN_ROOT}" "${EP12_ROOT}/SEARCH_POLICY.yaml")

  python3 -m ep12_executor.post_lock_runtime_materializer \
    --pre-null-dir "${EP12_PRE_NULL}" \
    --procedure-lock "${EP12_LOCK}" \
    --output-dir "${EP12_RUNTIME_BUNDLE}" \
    --scratch-bundle-dir "${EP12_RUNTIME_SCRATCH}" \
    --full-null-work-root "${EP12_NULL_WORK}" \
    --policy "${EP12_ROOT}/SEARCH_POLICY.yaml" \
    --development-role-manifest "${EP12_ROOT}/outputs/prelaunch/source_preflight/annotation_role_manifest.json" \
    --max-timing-report "${EP12_MAX_TIMING_REPORT}"

  python3 -m ep12_executor.procedure_lock_dispatch \
    --policy "${EP12_ROOT}/SEARCH_POLICY.yaml" \
    --procedure-lock "${EP12_LOCK}" \
    --output-dir "${EP12_RUN_ROOT}/pre_null_dispatch" \
    --full-search-null-script "${EP12_ROOT}/outputs/executor/slurm/run_full_search_null.sbatch" \
    --full-search-null-aggregate-script "${EP12_ROOT}/outputs/executor/slurm/run_full_search_null_aggregate.sbatch" \
    --runtime-bundle "${EP12_RUNTIME_SCRATCH}" \
    --full-null-work-root "${EP12_NULL_WORK}" \
    --full-null-durable-root "${EP12_NULL_OUTPUT}" \
    --resource-authorization "${EP12_AUTHORIZATION}"
else
  python3 -m ep12_executor.procedure_lock_dispatch \
    --policy "${EP12_ROOT}/SEARCH_POLICY.yaml" \
    --procedure-lock "${EP12_LOCK}" \
    --output-dir "${EP12_RUN_ROOT}/pre_null_dispatch" \
    --full-search-null-script "${EP12_ROOT}/outputs/executor/slurm/run_full_search_null.sbatch" \
    --full-search-null-aggregate-script "${EP12_ROOT}/outputs/executor/slurm/run_full_search_null_aggregate.sbatch" \
    --full-null-work-root "${EP12_NULL_WORK}" \
    --full-null-durable-root "${EP12_NULL_OUTPUT}"
fi
