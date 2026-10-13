#!/bin/bash
# Submit from the EP18 directory. The coordinator supplies the discovered
# partition and phase-specific --time/--array/--dependency overrides.
# Operations root below is precreated by the coordinator; scientific run/cache
# directories must remain absent until the trusted builder creates them.
#SBATCH --job-name=ep18-full-images
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --time=06:00:00
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/feature_preparation/image_full_operations_20261012_001/%x-%A_%a.out
#SBATCH --error=outputs/runs/feature_preparation/image_full_operations_20261012_001/%x-%A_%a.err
set -euo pipefail
umask 077

ep18_phase=${1:?Specify exactly builder, worker or assembly}
[[ $# == 1 ]]
[[ -n "${SLURM_JOB_ID:-}" && "$(hostname)" != *-ln* ]]
[[ "${SLURM_CPUS_PER_TASK:-}" == 1 && "${SLURM_MEM_PER_NODE:-}" == 7168 && -z "${SLURM_JOB_GPUS:-}" ]]
: "${OAK:?}" "${GROUP_HOME:?}" "${L_SCRATCH_JOB:?}"
ep18_episode="$OAK/users/zijiao/br_autoresearch/episode18_things_eeg_concept_stability"
ep18_config="$ep18_episode/outputs/config/full_image_extraction_20261012_001.json"
ep18_code="$ep18_episode/outputs/code"
ep18_operations_root="$ep18_episode/outputs/runs/feature_preparation/image_full_operations_20261012_001"
ep18_task=single
case "$ep18_phase" in
  builder) [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; ep18_phase_cap=21600 ;;
  assembly) [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; ep18_phase_cap=3600 ;;
  worker)
    ep18_task=${SLURM_ARRAY_TASK_ID:?Worker requires its selected array task index}
    [[ "$ep18_task" =~ ^[0-9]+$ && "$ep18_task" -le 25 ]]
    if (( ep18_task >= 6 && ep18_task <= 21 )); then
      ep18_phase_cap=14400
    else
      ep18_phase_cap=7200
    fi
    ;;
  *) exit 2 ;;
esac
[[ -d "$ep18_operations_root" ]]
ep18_operations="$ep18_operations_root/${ep18_phase}_${ep18_task}_${SLURM_JOB_ID}"
mkdir -m 700 "$ep18_operations"
ep18_stage=allocation_admission
ep18_started=$SECONDS
ep18_deadline=0
finish() {
  ep18_exit=$?
  trap - EXIT
  printf '{"phase":"%s","stage":"%s","exit_code":%s,"elapsed_seconds":%s,"job_id":"%s","task":"%s","deadline_unix":%s,"automatic_retry":false}\n' \
    "$ep18_phase" "$ep18_stage" "$ep18_exit" "$((SECONDS-ep18_started))" "$SLURM_JOB_ID" "$ep18_task" "$ep18_deadline" \
    > "$ep18_operations/exit.json"
  exit "$ep18_exit"
}
trap finish EXIT

# Slurm exports the allocation's start/projected-end epochs. The coordinator
# verifies requested TimeLimits at submission. No per-array scheduler query or
# current-time clock reset; absent/nonchronological scheduler clocks refuse.
# https://slurm.schedmd.com/sbatch.html#SECTION_OUTPUT-ENVIRONMENT-VARIABLES
ep18_start=${SLURM_JOB_START_TIME:?Scheduler start epoch is required}
ep18_end=${SLURM_JOB_END_TIME:?Scheduler end epoch is required}
[[ "$ep18_start" =~ ^[0-9]+$ && "$ep18_end" =~ ^[0-9]+$ ]]
[[ "${SLURM_JOB_NUM_NODES:-}" == 1 && "${SLURM_JOB_CPUS_PER_NODE:-}" == 1 ]]
ep18_observed_now=$(date +%s)
(( ep18_start > 0 && ep18_start <= ep18_observed_now && ep18_end > ep18_start ))
ep18_deadline=$((ep18_start+ep18_phase_cap-60))
if (( ep18_end-60 < ep18_deadline )); then ep18_deadline=$((ep18_end-60)); fi
if (( 1791927000 < ep18_deadline )); then ep18_deadline=1791927000; fi
remaining_seconds() {
  local ep18_now ep18_left
  ep18_now=$(date +%s)
  ep18_left=$((ep18_deadline-ep18_now))
  (( ep18_left > 0 )) || return 2
  printf '%s\n' "$ep18_left"
}
ep18_remaining=$(remaining_seconds)
printf 'hostname=%s job_id=%s phase=%s task=%s start=%s end=%s deadline=%s cpus=1 memory_mib=7168\n' \
  "$(hostname)" "$SLURM_JOB_ID" "$ep18_phase" "$ep18_task" "$ep18_start" "$ep18_end" "$ep18_deadline"

ep18_stage=selected_runtime
source /share/software/user/open/lmod/lmod/init/bash
module purge
module load devel python/3.12.1
unset PYTHONPATH PYTHONHOME
ep18_python="$GROUP_HOME/users/zijiao/ep18_model_runtime_cpu241_py312_46842156/bin/python"
[[ -x "$ep18_python" ]]
ep18_local="$L_SCRATCH_JOB/ep18-full-wrapper-${ep18_phase}-${ep18_task}"
mkdir -m 700 "$ep18_local"
mkdir -m 700 "$ep18_local/tmp" "$ep18_local/cache" "$ep18_local/hf"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES='' PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export TMPDIR="$ep18_local/tmp" XDG_CACHE_HOME="$ep18_local/cache" HF_HOME="$ep18_local/hf"
cd "$ep18_episode"

if [[ "$ep18_phase" == builder ]]; then
  ep18_stage=configuration_coverage
  ep18_remaining=$(remaining_seconds)
  ep18_check_seconds=60
  if (( ep18_remaining < ep18_check_seconds )); then ep18_check_seconds=$ep18_remaining; fi
  timeout --signal=TERM --kill-after=5s "${ep18_check_seconds}s" "$ep18_python" -B - "$ep18_config" "$ep18_code" \
    > "$ep18_operations/configuration_admission.json" <<'PY'
import json
import sys
sys.path.insert(0, sys.argv[2])
from run_image_feature_shard import load_config, planned_tasks
config = load_config(sys.argv[1])
tasks = planned_tasks(config)
if len(tasks) != 26 or config["max_parallel_tasks"] != 22 or config["run_cpu_hour_cap"] != 91:
    raise ValueError("Selected task/concurrency/reservation binding differs")
for task in tasks:
    expected_seconds = 14400 if 6 <= task["task_index"] <= 21 else 7200
    if config["worker_max_seconds"][task["family"]] != expected_seconds:
        raise ValueError("Selected scheduler group and family deadline differ")
    array_bytes = (task["stop"]-task["start"])*__import__("feature_extractors").DIMENSIONS[task["family"]]*4+128
    if array_bytes+config["metadata_byte_cap"] >= config["worker_artifact_byte_cap"][task["family"]]:
        raise ValueError("Selected task cannot store its required feature matrix")
print(json.dumps({"configuration_admission": "passed", "tasks": len(tasks), "images_per_family": 22448,
                  "families": 6, "reserved_cpu_hours": 91, "maximum_concurrent_workers": 22}))
PY
  ep18_stage=astra_validation
  ep18_remaining=$(remaining_seconds)
  ep18_check_seconds=60
  if (( ep18_remaining < ep18_check_seconds )); then ep18_check_seconds=$ep18_remaining; fi
  ep18_astra_env="$GROUP_HOME/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv"
  ep18_astra="$ep18_astra_env/bin/astra-tools"
  env PYTHONPATH="$ep18_astra_env/lib/python3.12/site-packages:$GROUP_HOME/users/zijiao/ep18_model_runtime_cpu241_py312_46842156/lib/python3.12/site-packages" \
    timeout --signal=TERM --kill-after=5s "${ep18_check_seconds}s" "$ep18_astra" validate \
    outputs/astra/v0.0.14/astra.yaml --json > "$ep18_operations/astra_validation.json"

  # Independent EP17 pilot: actual 4-GiB step; the parent reservation and
  # original deadline include all companion time. EP17 failure is recorded
  # separately and does not invalidate independent EP18 image preparation.
  ep18_stage=ep17_private_image_companion
  ep18_remaining=$(remaining_seconds)
  ep18_companion_seconds=900
  if (( ep18_remaining < ep18_companion_seconds )); then ep18_companion_seconds=$ep18_remaining; fi
  set +e
  timeout --signal=TERM --kill-after=10s "${ep18_companion_seconds}s" \
    srun --exact --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=4G --time=00:15:00 \
    bash "$OAK/users/zijiao/br_autoresearch/episode17_cneuromod_model_ranking_stability/outputs/tools/run_private_pilot_companion_001.sh" \
    > "$ep18_operations/ep17_companion_stdout.txt" 2> "$ep18_operations/ep17_companion_stderr.txt"
  ep18_ep17_exit=$?
  set -e
  printf 'ep17_companion_exit=%s\n' "$ep18_ep17_exit" > "$ep18_operations/ep17_companion_status.txt"

  ep18_stage=trusted_pixel_builder
  ep18_remaining=$(remaining_seconds)
  timeout --signal=TERM --kill-after=5s "${ep18_remaining}s" "$ep18_python" -B \
    "$ep18_code/prepare_image_feature_shards.py" --config "$ep18_config" \
    --image-archive "$OAK/users/zijiao/br_autoresearch_data/steward_acquisition/things_images_osf_jum2f_snapshot_20260922/archives/images_THINGS.zip" \
    --password-file "$OAK/users/zijiao/br_autoresearch_data/steward_acquisition/things_images_osf_jum2f_snapshot_20260922/terms/password_images.txt" \
    --deadline-unix "$ep18_deadline" > "$ep18_operations/execution.txt" 2>/dev/null
elif [[ "$ep18_phase" == worker ]]; then
  ep18_stage=pixel_only_worker
  ep18_remaining=$(remaining_seconds)
  cd "$ep18_local"
  # Only numerical/runtime/scheduler fields survive. No inherited credentials,
  # source routing variables, PYTHONPATH, source image path or credential path.
  timeout --signal=TERM --kill-after=5s "${ep18_remaining}s" env -i \
    PATH="$PATH" LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" LANG=C LC_ALL=C PYTHONUTF8=1 \
    SLURM_JOB_ID="$SLURM_JOB_ID" SLURM_CPUS_PER_TASK=1 SLURM_MEM_PER_NODE=7168 L_SCRATCH_JOB="$L_SCRATCH_JOB" \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 \
    CUDA_VISIBLE_DEVICES='' PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false \
    TMPDIR="$ep18_local/tmp" XDG_CACHE_HOME="$ep18_local/cache" HF_HOME="$ep18_local/hf" \
    "$ep18_python" -B "$ep18_code/run_image_feature_shard.py" \
    --config "$ep18_config" --task-index "$ep18_task" --deadline-unix "$ep18_deadline" \
    --local-dir "$L_SCRATCH_JOB/ep18-full-worker-${ep18_task}" > "$ep18_operations/execution.txt" 2>/dev/null
else
  ep18_stage=complete_image_assembly
  ep18_remaining=$(remaining_seconds)
  timeout --signal=TERM --kill-after=5s "${ep18_remaining}s" "$ep18_python" -B \
    "$ep18_code/assemble_image_features.py" --config "$ep18_config" \
    --deadline-unix "$ep18_deadline" > "$ep18_operations/execution.txt" 2>/dev/null
fi
ep18_stage=completed
