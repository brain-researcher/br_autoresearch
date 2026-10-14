#!/bin/bash
# Full adopted development route, preserving original clock and cost001.
#SBATCH --job-name=ep15-cert-fit
#SBATCH --partition=russpold,normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --time=24:00:00
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/observed_beta_development/operations003/%x-%j.out
#SBATCH --error=outputs/runs/observed_beta_development/operations003/%x-%j.err
set -euo pipefail
umask 077
[[ $# == 0 ]]
[[ -n "${SLURM_JOB_ID:-}" && "$(hostname)" != *-ln* ]]
[[ "${SLURM_CPUS_PER_TASK:-}" == 1 && "${SLURM_MEM_PER_NODE:-}" == 7168 ]]
[[ "${SLURM_JOB_NUM_NODES:-}" == 1 && "${SLURM_JOB_CPUS_PER_NODE:-}" == 1 ]]
[[ -z "${SLURM_JOB_GPUS:-}" && -z "${SLURM_ARRAY_TASK_ID:-}" ]]
: "${OAK:?}" "${GROUP_HOME:?}" "${L_SCRATCH_JOB:?}"
ep15_episode="$OAK/users/zijiao/br_autoresearch/episode15_mdtb_topography_vs_geometry"
ep15_run="$ep15_episode/outputs/runs/observed_beta_development"
ep15_operations="$ep15_run/operations003/initial_${SLURM_JOB_ID}"
[[ -d "$ep15_run/operations003" && -f "$ep15_run/finite_support001/selection.npz" && ! -e "$ep15_run/cost003" ]]
mkdir -m 700 "$ep15_operations"
ep15_stage=allocation_admission
ep15_started=$SECONDS
ep15_deadline=0
finish() {
  ep15_exit=$?
  trap - EXIT
  printf '{"stage":"%s","exit_code":%s,"job_id":"%s","elapsed_seconds":%s,"deadline_epoch":%s,"allocated_cpus":1,"reserved_cpu_hours":24,"automatic_retry":false}\n' \
    "$ep15_stage" "$ep15_exit" "$SLURM_JOB_ID" "$((SECONDS-ep15_started))" "$ep15_deadline" > "$ep15_operations/exit.json"
  exit "$ep15_exit"
}
trap finish EXIT
ep15_start=${SLURM_JOB_START_TIME:?}
ep15_end=${SLURM_JOB_END_TIME:?}
[[ "$ep15_start" =~ ^[0-9]+$ && "$ep15_end" =~ ^[0-9]+$ ]]
ep15_deadline=$((ep15_start+86400-60))
if (( ep15_end-60 < ep15_deadline )); then ep15_deadline=$((ep15_end-60)); fi
if (( 1792277107 < ep15_deadline )); then ep15_deadline=1792277107; fi
# Reuse completed mask and ten passing certificate checks.
(( ep15_deadline > $(date +%s)+180 ))
printf 'hostname=%s job_id=%s start=%s end=%s deadline=%s cpus=1 memory_mib=7168 search_start=1791845107 search_deadline=1792277107\n' \
  "$(hostname)" "$SLURM_JOB_ID" "$ep15_start" "$ep15_end" "$ep15_deadline"
ep15_free_kib=$(df -Pk "$L_SCRATCH_JOB" | awk 'NR==2 {print $4}')
[[ "$ep15_free_kib" =~ ^[0-9]+$ && "$ep15_free_kib" -ge 262144 ]]
ep15_local="$L_SCRATCH_JOB/ep15-cost003-${SLURM_JOB_ID}"
mkdir -m 700 "$ep15_local"
export TMPDIR="$ep15_local" XDG_CACHE_HOME="$ep15_local"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES='' PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH PYTHONHOME
ulimit -c 0
ulimit -S -f 65536
ep15_python="$GROUP_HOME/zijiao/miniforge3/envs/encoding_model/bin/python"
[[ -x "$ep15_python" ]]
cd "$ep15_episode"

ep15_stage=adopted_policy_admission
timeout --signal=TERM --kill-after=5s 30s "$ep15_python" -B -c \
  'import yaml; p=yaml.safe_load(open("SEARCH_POLICY.yaml")); assert p["scientist_amendment_20261013_m4_certificate"]["status"] == "adopted"' \
  > "$ep15_operations/policy_parse.txt" 2>&1

ep15_stage=astra_validation
(
  source /share/software/user/open/lmod/lmod/init/bash
  module purge
  module load devel python/3.12.1
  ep15_astra="$GROUP_HOME/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv"
  ep15_astra_dependencies="$GROUP_HOME/users/zijiao/ep18_model_runtime_cpu241_py312_46842156"
  env PYTHONPATH="$ep15_astra/lib/python3.12/site-packages:$ep15_astra_dependencies/lib/python3.12/site-packages" \
    timeout --signal=TERM --kill-after=5s 60s "$ep15_astra/bin/astra-tools" validate \
    "$ep15_episode/outputs/astra/v0.0.14/astra.yaml" --json
) > "$ep15_operations/astra_validation.json" 2>&1 || printf '%s\n' 'ASTRA validation remains open; see evidence.' > "$ep15_operations/astra_unresolved.txt"

ep15_stage=complete_observed_beta_cost_panel
ep15_remaining=$((ep15_deadline-$(date +%s)))
(( ep15_remaining > 60 ))
timeout --signal=TERM --kill-after=5s "${ep15_remaining}s" "$ep15_python" -B outputs/tools/run_development_point_panel.py \
  --manifest "$ep15_episode/outputs/runs/development_release/attempt001/view/point_panel_manifest.json" \
  --derived-support "$ep15_run/finite_support001/selection.npz" \
  --output-dir "$ep15_run/cost003" --deadline-epoch "$((ep15_deadline-10))" \
  --rss-limit-kib 6291456 --durable-byte-cap 33554432 \
  --analysis-scope conditional_observed_beta --publication-mode withheld_cost_measurement \
  > "$ep15_operations/cost.txt" 2>&1
ep15_stage=completed_private_cost_panel
