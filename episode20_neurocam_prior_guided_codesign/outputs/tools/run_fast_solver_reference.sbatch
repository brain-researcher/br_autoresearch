#!/bin/bash
# Stage 1 (development test): record the UNMODIFIED reference on a prefix of cycle 0.
# Usage: sbatch [--time=...] run_fast_solver_reference.sbatch RUN_SUBDIR SLOTS TOLERANCE [OFFSETS_JSON]
# One finite allocation, no requeue, no automatic resubmission.
#SBATCH --job-name=ep20-fs-ref
#SBATCH --partition=russpold,normal
#SBATCH --constraint=CPU_GEN:SIE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --no-requeue
#SBATCH --output=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode20_neurocam_prior_guided_codesign/outputs/runs/fast_forward_stage1/solver/slurm-%j.out
#SBATCH --error=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode20_neurocam_prior_guided_codesign/outputs/runs/fast_forward_stage1/solver/slurm-%j.err
set -uo pipefail
ep20_host=$(hostname)
printf 'HOST=%s JOB=%s CPUS=%s MEM_MIB=%s START=%s\n' "$ep20_host" "${SLURM_JOB_ID:?}" \
  "${SLURM_CPUS_PER_TASK:?}" "${SLURM_MEM_PER_NODE:-unknown}" "$(date -Is)"
[[ "$ep20_host" != *-ln* ]] || exit 2
[[ $# -ge 3 ]] || exit 2
ep20_root=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode20_neurocam_prior_guided_codesign
ep20_run="$ep20_root/outputs/runs/fast_forward_stage1/solver/$1"
ep20_slots="$2"
ep20_tolerance="$3"
ep20_offsets="${4:-}"
mkdir "$ep20_run" || exit 2
source /share/software/user/open/lmod/lmod/init/bash
module purge
unset PYTHONPATH PYTHONHOME
module load devel python/3.12.1 math py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 py-h5py/3.10.0_py312
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK" NUMEXPR_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export VECLIB_MAXIMUM_THREADS="$SLURM_CPUS_PER_TASK" BLIS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export TMPDIR="${L_SCRATCH_JOB:-/tmp}"
# Internal deadline at 90% of the allocation's time limit.
ep20_limit=$(squeue -h -j "$SLURM_JOB_ID" -o %l)
ep20_seconds=$(python3 -c "import sys;p=sys.argv[1].replace('-',':').split(':');p=[int(x) for x in p];print(sum(v*m for v,m in zip(reversed(p),(1,60,3600,86400))))" "$ep20_limit")
ep20_deadline=$(( ep20_seconds * 9 / 10 ))
ep20_checkpoints="2,$(seq -s, 25 25 1600)"
printf 'job=%s\nhost=%s\nlimit_seconds=%s\ndeadline_seconds=%s\nslots=%s\ntolerance=%s\noffsets=%s\n' \
  "$SLURM_JOB_ID" "$ep20_host" "$ep20_seconds" "$ep20_deadline" "$ep20_slots" "$ep20_tolerance" \
  "$ep20_offsets" > "$ep20_run/status.txt"
ep20_extra=()
[[ -n "$ep20_offsets" ]] && ep20_extra=(--offsets-json "$ep20_offsets")
cd "$ep20_root/outputs/tools" || exit 2
/usr/bin/time -v -o "$ep20_run/resources.txt" \
  timeout --signal=TERM --kill-after=30s "$(( ep20_seconds - 120 ))s" \
  python3.12 -u -B fast_solver_reference.py --run-dir "$ep20_run" --slots "$ep20_slots" \
    --deadline-seconds "$ep20_deadline" --tolerance "$ep20_tolerance" \
    --checkpoint-slots "$ep20_checkpoints" ${ep20_extra[@]+"${ep20_extra[@]}"} > "$ep20_run/stdout.txt" 2>&1
ep20_exit=$?
printf 'exit=%s\nend=%s\n' "$ep20_exit" "$(date -Is)" >> "$ep20_run/status.txt"
if [[ ! -f "$ep20_run/reference_report.json" ]]; then
  printf '{"status": "failed", "failure": "no terminal report from python (exit %s)", "slurm_job_id": "%s"}\n' \
    "$ep20_exit" "$SLURM_JOB_ID" > "$ep20_run/reference_report_fallback.json"
fi
exit "$ep20_exit"
