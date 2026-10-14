#!/bin/bash
# Selected complete private endpoint panel; coordinator supplies afterok:47577068.
# Reuse the two passing caller checks and existing bootstrap qualification.
# No tests, fitting, image inference, secondary scoring, audit or successor.
# One numerical worker; 7GiB Slurm memory and 6GiB observed process-RSS guard.
# The2h reservation is a ceiling, not a measured endpoint runtime projection.
#SBATCH --job-name=ep17-panel001
#SBATCH --partition=russpold,normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/ep17-panel001-%j.out
#SBATCH --error=outputs/runs/ep17-panel001-%j.err
set -euo pipefail
umask 077
[[ $# -eq 0 ]] || exit 2
ep17_panel_host=$(hostname)
[[ -n "${SLURM_JOB_ID:-}" && "$ep17_panel_host" != *-ln* ]]
[[ "${SLURM_CPUS_PER_TASK:-}" == 1 && "${SLURM_MEM_PER_NODE:-}" == 7168 ]]
[[ "${SLURM_JOB_NUM_NODES:-}" == 1 && "${SLURM_JOB_CPUS_PER_NODE:-}" == 1 ]]
[[ -z "${SLURM_ARRAY_TASK_ID:-}" && -z "${SLURM_JOB_GPUS:-}" ]]
[[ "${SLURM_JOB_START_TIME:?}" =~ ^[0-9]+$ && "${SLURM_JOB_END_TIME:?}" =~ ^[0-9]+$ ]]
ep17_panel_started=$(date -u +%s)
(( SLURM_JOB_START_TIME <= ep17_panel_started && SLURM_JOB_END_TIME > SLURM_JOB_START_TIME ))
ep17_panel_deadline=$((SLURM_JOB_START_TIME + 7200 - 60))
if (( SLURM_JOB_END_TIME - 60 < ep17_panel_deadline )); then
  ep17_panel_deadline=$((SLURM_JOB_END_TIME - 60))
fi
# Original240h deadline; reserve external10s timeout offset,10s kill grace,
# and10s dispatch margin. No queue delay or predecessor resets this clock.
if (( 1791916149 - 30 < ep17_panel_deadline )); then
  ep17_panel_deadline=$((1791916149 - 30))
fi
(( ep17_panel_deadline > ep17_panel_started + 600 ))
cd "${OAK:?}/users/zijiao/br_autoresearch/episode17_cneuromod_model_ranking_stability"
for ep17_panel_destination in outputs/runs/development_endpoints/panel001 \
    outputs/cache/private/development_endpoints/panel001; do
  [[ ! -e "$ep17_panel_destination" && ! -L "$ep17_panel_destination" ]]
done
ep17_panel_python="${GROUP_HOME:?}/zijiao/miniforge3/envs/encoding_model/bin/python"
ep17_panel_h5py="$GROUP_HOME/users/zijiao/ep17_h5py_3_10_0_cp311"
[[ -x "$ep17_panel_python" && -d "$ep17_panel_h5py" ]]
ep17_panel_local="${L_SCRATCH_JOB:?}/ep17_panel001"
mkdir -m 700 "$ep17_panel_local"
ep17_panel_ld_path=$(printenv LD_LIBRARY_PATH || true)
ep17_panel_ops="$PWD/outputs/runs/endpoint_panel001_job_${SLURM_JOB_ID}"
mkdir -m 700 "$ep17_panel_ops"
ep17_panel_stage=completed_fit_binding
trap 'ep17_panel_exit=$?; printf "stage=%s\nexit=%s\nfinished_utc=%s\n" "$ep17_panel_stage" "$ep17_panel_exit" "$(date -u +%FT%TZ)" >>"$ep17_panel_ops/status.txt"' EXIT
printf 'job_id=%s\nhostname=%s\ncpus=1\nmemory_mib=7168\nworkers=1\nnumerical_threads=1\nreserved_CPU_hours=2\nallocation_start_epoch=%s\nallocation_end_epoch=%s\ndeadline_epoch=%s\noriginal_episode_deadline_epoch=1791916149\nsource_fit=fit002\nendpoint_attempt=panel001\npassing_caller_checks_reused=true\nautomatic_retry=false\naudit_access=false\n' \
  "$SLURM_JOB_ID" "$ep17_panel_host" "$SLURM_JOB_START_TIME" "$SLURM_JOB_END_TIME" \
  "$ep17_panel_deadline" >"$ep17_panel_ops/status.txt"
ulimit -c 0

# Identical qualified scientific interpreter/addon and clean environment as
# fit002; no module or tooling environment can leak into the scientific child.
ep17_panel_clean() {
  env -i "PATH=$PATH" "HOME=$HOME" "USER=$USER" \
    "GROUP_HOME=$GROUP_HOME" "OAK=$OAK" "LD_LIBRARY_PATH=$ep17_panel_ld_path" \
    "SLURM_JOB_ID=$SLURM_JOB_ID" "SLURM_STEP_ID=${SLURM_STEP_ID-}" \
    SLURM_CPUS_PER_TASK=1 SLURM_MEM_PER_NODE=7168 "SLURM_JOB_END_TIME=$SLURM_JOB_END_TIME" \
    "L_SCRATCH_JOB=$L_SCRATCH_JOB" "TMPDIR=$ep17_panel_local" \
    "XDG_CACHE_HOME=$ep17_panel_local/cache" "HF_HOME=$ep17_panel_local/hf" \
    "TORCH_HOME=$ep17_panel_local/torch" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= LANG=C LC_ALL=C TZ=UTC "$@"
}

# Operational completion record and exact named file only; do not open private
# arrays here. The existing caller subsequently enforces full checkpoint and
# frozen family/response bindings before endpoint computation.
ep17_panel_clean timeout --signal=TERM --kill-after=5s 30s "$ep17_panel_python" -B - \
  "$PWD" >"$ep17_panel_ops/completed_fit_binding.txt" 2>&1 <<'PY'
import json
from pathlib import Path
import sys

episode = Path(sys.argv[1])
report = episode / "outputs/runs/nested_prediction/fit002/fit_report.json"
checkpoint = episode / "outputs/cache/private/nested_prediction/fit002/complete_prediction.zip"
with report.open("rb") as stream:
    raw = stream.read(131073)
assert len(raw) <= 131072
fit = json.loads(raw)
assert fit["analysis_id"] == "prediction_qualification"
assert fit["stage"] == "complete_nested_prediction" and fit["attempt"] == "fit002"
assert fit["feature_attempt"] == "full001" and fit["cost_attempt"] == "cost002"
assert fit["status"] == "completed" and fit["checkpoint_published"] is True
assert fit["checkpoint_publisher_returned"] is True and fit["checkpoint_path_present"] is True
assert fit["checkpoint_path"] == str(checkpoint) and fit["audit_access"] is False
assert fit["endpoint_evaluation"] is False
assert checkpoint.is_file() and not checkpoint.is_symlink()
assert 0 < checkpoint.stat().st_size <= 4294967296
print("completed_fit002_record_and_named_checkpoint_present")
PY
printf 'completed_fit_binding_exit=0\n' >>"$ep17_panel_ops/status.txt"

ep17_panel_stage=complete_private_endpoint_panel
ep17_panel_remaining=$((ep17_panel_deadline - $(date -u +%s)))
(( ep17_panel_remaining > 0 ))
# The operator producer's public report records2,455,772bytes total private
# output; reuse the existing family001 release's32MiB operator-array cap.
set +e
ep17_panel_clean "PYTHONPATH=$ep17_panel_h5py" \
  timeout --signal=TERM --kill-after=10s "$((ep17_panel_remaining + 10))s" \
  "$ep17_panel_python" -B outputs/tools/run_development_endpoint_panel.py \
  --attempt panel001 --fit-attempt fit002 --feature-attempt full001 --family-attempt family001 \
  --deadline-epoch "$ep17_panel_deadline" --rss-limit-kib 6291456 \
  --metadata-byte-cap 67108864 --feature-byte-cap 268435456 --response-byte-cap 536870912 \
  --operator-array-byte-cap 33554432 --checkpoint-byte-cap 4294967296 \
  --science-byte-cap 1048576 --report-byte-cap 131072 \
  >"$ep17_panel_ops/endpoint_panel.txt" 2>&1
ep17_panel_exit=$?
set -e
printf 'endpoint_panel_exit=%s\n' "$ep17_panel_exit" >>"$ep17_panel_ops/status.txt"
if (( ep17_panel_exit == 0 )); then ep17_panel_stage=completed_private_endpoint_panel; fi
exit "$ep17_panel_exit"
