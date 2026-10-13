#!/bin/bash
# Prospective fixed full001/cost002 -> fit001. Submit only after coordinator
# admission and the exact family001/dry001 evidence below has completed.
# 4 model workers, one numerical thread each; no retry, requeue or successor.
# Measured cost projection: ~10.66 serial CPUh /4 ideal workers =2.66h before
# unmeasured full-copy I/O, scoring/variance work and concurrency overhead.
# 8h is a bounded allocation ceiling, not a promised completion duration.
# Retained array estimate 4.39 GB before native workspaces/runtime; 16 GiB Slurm,
# 12 GiB observed RSS guard; serialized prediction checkpoint cap 4 GiB.
#SBATCH --job-name=ep17-fit001
#SBATCH --partition=russpold,normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/ep17-fit001-%j.out
#SBATCH --error=outputs/runs/ep17-fit001-%j.err
set -euo pipefail
umask 077
[[ $# -eq 0 ]] || exit 2
ep17_fit_host=$(hostname)
printf 'hostname=%s job_id=%s cpus=%s memory_mib=%s\n' "$ep17_fit_host" \
  "${SLURM_JOB_ID:?}" "${SLURM_CPUS_PER_TASK:?}" "${SLURM_MEM_PER_NODE:?}"
[[ "$ep17_fit_host" != *-ln* && "$SLURM_CPUS_PER_TASK" == 4 && "$SLURM_MEM_PER_NODE" == 16384 ]]
[[ "${SLURM_JOB_END_TIME:?}" =~ ^[0-9]+$ ]]
cd "${OAK:?}/users/zijiao/br_autoresearch/episode17_cneuromod_model_ranking_stability"

# Recorded earliest substantive execution +240h: Oct13 18:29:09UTC.
# Coordinator verifies this selected epoch before submission. Pending time
# cannot extend it. The actual allocation end always bounds the process.
ep17_episode_deadline=1791916149
ep17_fit_started=$(date -u +%s)
ep17_fit_deadline=$(( ep17_fit_started + 8*3600 - 60 ))
if (( SLURM_JOB_END_TIME - 60 < ep17_fit_deadline )); then
  ep17_fit_deadline=$(( SLURM_JOB_END_TIME - 60 ))
fi
if (( ep17_episode_deadline < ep17_fit_deadline )); then
  ep17_fit_deadline=$ep17_episode_deadline
fi
# Leave the full selected short-check envelope before admitting any fitting.
(( ep17_fit_deadline > ep17_fit_started + 150 ))
ep17_family_record="$PWD/outputs/runs/prediction_qualification/endpoint_family/family001/family_report.json"
ep17_dry_record="$PWD/outputs/runs/prediction_qualification/evaluator_dry_run/dry001/dry_run_report.json"
[[ -f "$ep17_family_record" && -f "$ep17_dry_record" ]]
[[ ! -e outputs/runs/nested_prediction/fit001 && ! -L outputs/runs/nested_prediction/fit001 ]]
[[ ! -e outputs/cache/private/nested_prediction/fit001 && ! -L outputs/cache/private/nested_prediction/fit001 ]]

ep17_fit_python="${GROUP_HOME:?}/zijiao/miniforge3/envs/encoding_model/bin/python"
ep17_fit_h5py="$GROUP_HOME/users/zijiao/ep17_h5py_3_10_0_cp311"
[[ -x "$ep17_fit_python" && -d "$ep17_fit_h5py" ]]
ep17_fit_local="${L_SCRATCH_JOB:?}/ep17_fit001"
mkdir "$ep17_fit_local"
ep17_fit_ld_path=$(printenv LD_LIBRARY_PATH || true)
ep17_fit_ops="$PWD/outputs/runs/nested_fit001_job_${SLURM_JOB_ID}"
mkdir "$ep17_fit_ops"
trap 'ep17_fit_exit=$?; printf "exit=%s\nfinished_utc=%s\n" "$ep17_fit_exit" "$(date -u +%FT%TZ)" >>"$ep17_fit_ops/status.txt"' EXIT
printf 'job_id=%s\nhostname=%s\ncpus=4\nmemory_mib=16384\nworkers=4\nnumerical_threads_per_worker=1\nstarted_epoch=%s\ndeadline_epoch=%s\nslurm_end_epoch=%s\nparent_charged_once_to=EP17\n' \
  "$SLURM_JOB_ID" "$ep17_fit_host" "$ep17_fit_started" "$ep17_fit_deadline" \
  "$SLURM_JOB_END_TIME" >"$ep17_fit_ops/status.txt"

ep17_fit_clean() {
  env -i "PATH=$PATH" "HOME=$HOME" "USER=$USER" \
    "GROUP_HOME=$GROUP_HOME" "OAK=$OAK" "LD_LIBRARY_PATH=$ep17_fit_ld_path" \
    "SLURM_JOB_ID=$SLURM_JOB_ID" "SLURM_STEP_ID=${SLURM_STEP_ID-}" \
    SLURM_CPUS_PER_TASK=4 SLURM_MEM_PER_NODE=16384 "SLURM_JOB_END_TIME=$SLURM_JOB_END_TIME" \
    "L_SCRATCH_JOB=$L_SCRATCH_JOB" "TMPDIR=$ep17_fit_local" \
    "XDG_CACHE_HOME=$ep17_fit_local/cache" "HF_HOME=$ep17_fit_local/hf" \
    "TORCH_HOME=$ep17_fit_local/torch" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= LANG=C LC_ALL=C TZ=UTC "$@"
}

# These are the two concrete pre-score records selected by the coordinator.
# No extra receipts, hashes, payload reads or audit evidence are introduced.
ep17_fit_clean timeout --signal=TERM --kill-after=5s 30s "$ep17_fit_python" -B - \
  "$ep17_family_record" "$ep17_dry_record" "$PWD" >"$ep17_fit_ops/pre_score_records.txt" 2>&1 <<'PY'
import json
from pathlib import Path
import sys

family_path, dry_path, episode = map(Path, sys.argv[1:])
def read(path):
    with path.open("rb") as stream:
        payload = stream.read(65537)
    assert len(payload) <= 65536
    return json.loads(payload)

family, dry = read(family_path), read(dry_path)
assert family["analysis_id"] == "prediction_qualification"
assert family["status"] == "qualified_calibration_endpoint_family"
assert all(family[key] is False for key in
           ("candidate_scores_read", "raw_calibration_responses_read", "audit_values_read"))
assert family["source_private_operator_attempt"] == str(episode / "outputs/cache/private/calibration_operators/attempt001")
assert family["component_ids"] == family["primary_endpoint_ids"] + family["h_component_ids"]
assert len(family["component_ids"]) == len(family["score_scales"])
assert isinstance(family["directed_edges"], list) and isinstance(family["calibration_inapplicability"], dict)
assert set(family["group_widths"]) == {f"{subject}_{roi}" for subject in
        ("sub-01", "sub-02", "sub-03", "sub-06") for roi in ("left", "right")}
assert all(type(value) is int and value >= 256 for value in family["group_widths"].values())
assert sum(family["group_widths"].values()) == 8558
assert dry["analysis_id"] == "prediction_qualification" and dry["status"] == "completed"
assert dry["stage"] == "outcome_free_evaluator_dry_run" and dry["attempt"] == "dry001"
assert dry["study_payloads_opened"] is False and dry["scientific_audit_openings"] == 0
scope = dry["composition"]
assert all(scope[key] is True for key in
           ("frozen_prediction_reference_matched", "complete_bounds_computed", "adjudication_completed"))
assert (scope["images"], scope["concepts"], scope["concept_blocks"], scope["participant_ROI_groups"]) == (606, 120, 12, 8)
assert (scope["bootstrap_draws"], scope["bootstrap_seed"], scope["bootstrap_order_statistic"]) == (10000, 1729, 9501)
print("selected_pre_score_family_and_outcome_free_evaluator_records_passed")
PY
printf 'pre_score_records_exit=0\n' >>"$ep17_fit_ops/status.txt"

# Run only the overlap method previously skipped in the 1 CPU parent. Reuse all
# passed serial/numerical/binding checks. Refuse a skipped or incomplete method.
ep17_fit_clean "PYTHONPATH=$ep17_fit_h5py:$PWD/outputs/tests" \
  timeout --signal=TERM --kill-after=5s 60s "$ep17_fit_python" -B - \
  >"$ep17_fit_ops/concurrency_check.txt" 2>&1 <<'PY'
import unittest
from test_parallel_nested_prediction import ParallelPredictionTests
suite = unittest.TestSuite([ParallelPredictionTests(
    "test_two_worker_overlap_canonical_collection_and_synchronized_guard")])
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(0 if result.wasSuccessful() and result.testsRun == 1 and not result.skipped else 1)
PY
printf 'concurrency_check_exit=0\n' >>"$ep17_fit_ops/status.txt"
ep17_fit_remaining=$(( ep17_fit_deadline - $(date -u +%s) ))
(( ep17_fit_remaining > 0 ))
set +e
ep17_fit_clean "PYTHONPATH=$ep17_fit_h5py" \
  timeout --signal=TERM --kill-after=10s "$(( ep17_fit_remaining + 10 ))s" \
  "$ep17_fit_python" -B outputs/tools/run_complete_development_prediction.py \
  --feature-attempt full001 --cost-attempt cost002 --attempt fit001 --workers 4 \
  --deadline-epoch "$ep17_fit_deadline" --rss-limit-kib 12582912 \
  --metadata-byte-cap 67108864 --feature-byte-cap 268435456 --response-byte-cap 536870912 \
  --checkpoint-byte-cap 4294967296 --report-byte-cap 131072 \
  >"$ep17_fit_ops/full_fit.txt" 2>&1
ep17_fit_exit=$?
set -e
printf 'full_fit_exit=%s\n' "$ep17_fit_exit" >>"$ep17_fit_ops/status.txt"
exit "$ep17_fit_exit"
