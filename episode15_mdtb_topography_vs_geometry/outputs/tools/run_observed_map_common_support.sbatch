#!/bin/bash
#SBATCH --job-name=ep15-common-support
#SBATCH --partition=russpold,normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/observed_map_audit/common_support_operations001/%x-%j.out
#SBATCH --error=outputs/runs/observed_map_audit/common_support_operations001/%x-%j.err
set -euo pipefail
umask 077
[[ $# == 0 ]]
[[ -n "${SLURM_JOB_ID:-}" && "$(hostname)" != *-ln* ]]
[[ "${SLURM_CPUS_PER_TASK:-}" == 1 && "${SLURM_MEM_PER_NODE:-}" == 7168 ]]
[[ "${SLURM_JOB_NUM_NODES:-}" == 1 ]]
[[ -z "${SLURM_JOB_GPUS:-}" && -z "${SLURM_ARRAY_TASK_ID:-}" ]]
: "${OAK:?}" "${GROUP_HOME:?}" "${L_SCRATCH_JOB:?}"
ep15_episode="$OAK/users/zijiao/br_autoresearch/episode15_mdtb_topography_vs_geometry"
ep15_run="$ep15_episode/outputs/runs/observed_map_audit"
ep15_operations="$ep15_run/common_support_operations001/initial_${SLURM_JOB_ID}"
[[ -d "$ep15_run/common_support_operations001" ]]
[[ -d "$ep15_run/predictions" && -f "$ep15_run/audit_execution.json" ]]
[[ -f "$ep15_run/manifests001/view/evaluation_manifest.json" ]]
[[ ! -e "$ep15_run/common_support001" ]]
mkdir -m 700 "$ep15_operations"
ep15_stage=allocation_admission
ep15_started=$SECONDS
ep15_deadline=0
finish() {
  ep15_exit=$?
  trap - EXIT
  printf '{"stage":"%s","exit_code":%s,"job_id":"%s","elapsed_seconds":%s,"deadline_epoch":%s,"allocated_cpus":1,"automatic_retry":false}\n' \
    "$ep15_stage" "$ep15_exit" "$SLURM_JOB_ID" "$((SECONDS-ep15_started))" "$ep15_deadline" > "$ep15_operations/exit.json"
  exit "$ep15_exit"
}
trap finish EXIT
ep15_start=${SLURM_JOB_START_TIME:?}
ep15_end=${SLURM_JOB_END_TIME:?}
[[ "$ep15_start" =~ ^[0-9]+$ && "$ep15_end" =~ ^[0-9]+$ ]]
ep15_deadline=$((ep15_start+14400-90))
if (( ep15_end-90 < ep15_deadline )); then ep15_deadline=$((ep15_end-90)); fi
(( ep15_deadline > $(date +%s)+180 ))
printf 'hostname=%s job_id=%s start=%s end=%s deadline=%s cpus=1 memory_mib=7168\n' \
  "$(hostname)" "$SLURM_JOB_ID" "$ep15_start" "$ep15_end" "$ep15_deadline"
ep15_free_kib=$(df -Pk "$L_SCRATCH_JOB" | awk 'NR==2 {print $4}')
[[ "$ep15_free_kib" =~ ^[0-9]+$ && "$ep15_free_kib" -ge 262144 ]]
ep15_local="$L_SCRATCH_JOB/ep15-common-support-${SLURM_JOB_ID}"
mkdir -m 700 "$ep15_local"
export TMPDIR="$ep15_local" XDG_CACHE_HOME="$ep15_local" MPLCONFIGDIR="$ep15_local/matplotlib"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES='' PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH PYTHONHOME
ulimit -c 0
ulimit -S -f 262144
cd "$ep15_episode"
source /share/software/user/open/lmod/lmod/init/bash
module purge
module load devel python/3.12.1 math py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 viz py-matplotlib/3.8.3_py312
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$GROUP_HOME/users/zijiao/ep18_model_runtime_cpu241_py312_46842156/lib/python3.12/site-packages"
ep15_python=/share/software/user/open/python/3.12.1/bin/python3.12
run_stage() {
  local ep15_remaining=$((ep15_deadline-$(date +%s)))
  (( ep15_remaining > 10 ))
  timeout --signal=TERM --kill-after=5s "${ep15_remaining}s" "$ep15_python" -B "$@"
}
# Reuse the established audit plotting runtime before any renewed B access.
ep15_stage=runtime_admission
run_stage -c 'import yaml, numpy, scipy, matplotlib; print("report_runtime_ready", numpy.__version__, scipy.__version__, matplotlib.__version__)' \
  > "$ep15_operations/runtime.txt" 2>&1
ep15_stage=astra_validation
ep15_astra="$GROUP_HOME/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv"
env PYTHONPATH="$ep15_astra/lib/python3.12/site-packages:$PYTHONPATH" \
  timeout --signal=TERM --kill-after=5s 60s "$ep15_astra/bin/astra-tools" validate \
  "$ep15_episode/outputs/astra/v0.0.14/astra.yaml" --json \
  > "$ep15_operations/astra_validation.json" 2>&1 || printf '%s\n' 'ASTRA validation remains open; see evidence.' > "$ep15_operations/astra_unresolved.txt"
ep15_stage=complete_common_support_scan_and_evaluation
run_stage outputs/tools/run_observed_map_common_support.py \
  --evaluation-manifest "$ep15_run/manifests001/view/evaluation_manifest.json" \
  --deadline-epoch "$ep15_deadline" --rss-limit-kib 6291456 --durable-byte-cap 134217728 \
  > "$ep15_operations/evaluate.txt" 2>&1
ep15_stage=complete_exploratory_common_support
