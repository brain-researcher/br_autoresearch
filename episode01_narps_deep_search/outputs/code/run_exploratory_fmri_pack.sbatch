#!/bin/bash
# Submit only after root selects measured-cost packs from complete preparation.
#SBATCH --job-name=ep01-real-fmri
#SBATCH --partition=russpold,normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=15G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/exploratory_fmri_specification_sensitivity/cohort_20261013_001/slurm-%A_%a.out
#SBATCH --error=outputs/runs/exploratory_fmri_specification_sensitivity/cohort_20261013_001/slurm-%A_%a.err
set -euo pipefail
umask 077
[[ -n "${SLURM_JOB_ID:-}" && "$(hostname)" != *-ln* ]]
[[ "${SLURM_CPUS_PER_TASK:-}" == 2 && "${SLURM_MEM_PER_NODE:-}" == 15360 ]]
[[ "${SLURM_ARRAY_TASK_ID:-}" =~ ^[0-9]+$ && "${SLURM_RESTART_COUNT:-0}" == 0 ]]
[[ -z "${SLURM_JOB_GPUS:-}" ]]
: "${OAK:?}" "${GROUP_HOME:?}" "${L_SCRATCH_JOB:?}"
ep01_base="$OAK/users/zijiao/br_autoresearch/episode01_narps_deep_search"
ep01_prepared="$ep01_base/outputs/runs/exploratory_fmri_specification_sensitivity/prepared_20261013_002"
ep01_cohort="$ep01_base/outputs/runs/exploratory_fmri_specification_sensitivity/cohort_20261013_001"
[[ -s "$ep01_prepared/prepared.json" && -s "$ep01_cohort/participant_packs.tsv" ]]
ep01_line=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$ep01_cohort/participant_packs.tsv")
read -r -a ep01_people <<< "$ep01_line"
(( ${#ep01_people[@]} > 0 ))
ep01_args=()
for ep01_person in "${ep01_people[@]}"; do
  [[ "$ep01_person" =~ ^sub-[A-Za-z0-9]+$ ]]
  ep01_args+=(--participant "$ep01_person")
done
ep01_ops="$ep01_cohort/pack_${SLURM_ARRAY_TASK_ID}_${SLURM_JOB_ID}"
mkdir -m 700 "$ep01_ops"
ep01_begin=$SECONDS
finish() {
  ep01_exit=$?
  trap - EXIT
  printf '{"job_id":"%s","array_task":"%s","exit_code":%s,"elapsed_seconds":%s,"allocated_cpus":2,"memory_mib":15360,"automatic_retry":false}\n' \
    "$SLURM_JOB_ID" "$SLURM_ARRAY_TASK_ID" "$ep01_exit" "$((SECONDS-ep01_begin))" > "$ep01_ops/exit.json"
  exit "$ep01_exit"
}
trap finish EXIT
ulimit -c 0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH PYTHONHOME
ep01_tmp="$L_SCRATCH_JOB/ep01-pack-tmp"
mkdir -m 700 "$ep01_tmp"
export TMPDIR="$ep01_tmp"
ep01_python="$GROUP_HOME/zijiao/miniforge3/envs/encoding_model/bin/python"
[[ -x "$ep01_python" ]]
ep01_now=$(date +%s)
ep01_seconds=$((SLURM_JOB_END_TIME-ep01_now-60))
if (( ep01_seconds > 3540 )); then ep01_seconds=3540; fi
(( ep01_seconds > 60 ))
cd "$ep01_base/outputs/code"
timeout --signal=TERM --kill-after=10s "$((ep01_seconds+10))s" "$ep01_python" -B \
  exploratory_fmri_runner.py fit --prepared-dir "$ep01_prepared" \
  --output-dir "$ep01_cohort/models" "${ep01_args[@]}" \
  --max-seconds "$ep01_seconds" --max-rss-gib 12 --max-local-gib 12 \
  > "$ep01_ops/fit.log" 2>&1
