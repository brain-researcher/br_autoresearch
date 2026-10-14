#!/bin/bash
# Combined substantive finalization and reporting; coordinator reviews first.
# Replace the pending finalizer only after checking descendants, bind the new
# actual job ID in dependency_bindings.json, and retain the whole draw dependency.
# Existing model, decode, draw and finalizer sources are unchanged.
#SBATCH --job-name=ep21-fa-final-report
#SBATCH --partition=russpold
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --time=02:00:00
#SBATCH --dependency=afterok:47586235
#SBATCH --kill-on-invalid-dep=yes
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/finding_a_feature_decoding/amended_full001/finalize/slurm-%j.out
#SBATCH --error=outputs/runs/finding_a_feature_decoding/amended_full001/finalize/slurm-%j.err
set -euo pipefail
[[ $# -eq 0 ]] || exit 2
ep21_figure_host=$(hostname)
printf 'HOST=%s JOB=%s\n' "$ep21_figure_host" "${SLURM_JOB_ID:-}"
[[ "$ep21_figure_host" != *-ln* && -n "${SLURM_JOB_ID:-}" ]] || exit 2
cd "${OAK:?}/users/zijiao/br_autoresearch/episode21_wang_neural_code_conversion_reproduction"
# The original common wrapper exits explicitly. A child shell retains its
# admission, finite clock, traps, report-last publication and closed status,
# while allowing this parent to render only after its successful return.
bash outputs/converter_training/finding_a_amended_full_finalize.sbatch
ep21_figure_run="$PWD/outputs/runs/finding_a_feature_decoding/amended_full001/finalize"
printf 'finalizer_exit=0\nrender_started=false\n' > "$ep21_figure_run/render_status.txt"
ep21_figure_finish() {
  ep21_figure_exit=$?
  trap - EXIT
  printf 'render_exit=%s\nscientific_finalizer_remains_complete=true\n' "$ep21_figure_exit" \
    >> "$ep21_figure_run/render_status.txt"
  exit "$ep21_figure_exit"
}
trap ep21_figure_finish EXIT
# Use the original captured allocation; no extra scheduler poll. Reporting is
# limited to four minutes and ends at least 30 seconds before Slurm EndTime.
ep21_figure_allocation=$(< "$ep21_figure_run/allocation.txt")
[[ "$ep21_figure_allocation" =~ EndTime=([^[:space:]]+) ]] || exit 2
ep21_figure_end=$(date -u -d "${BASH_REMATCH[1]}" +%s)
source /share/software/user/open/lmod/lmod/init/bash
module purge
unset PYTHONPATH PYTHONHOME
module load devel python/3.12.1 math py-numpy/1.26.3_py312 viz py-matplotlib/3.8.3_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=''
export MPLCONFIGDIR="${L_SCRATCH_JOB:?}/ep21_finding_a_matplotlib"
export TMPDIR="${L_SCRATCH_JOB}/ep21_finding_a_render"
mkdir -p "$MPLCONFIGDIR" "$TMPDIR"
ep21_figure_remaining=$((ep21_figure_end - $(date -u +%s) - 30))
(( ep21_figure_remaining > 0 )) || exit 124
if (( ep21_figure_remaining > 240 )); then ep21_figure_remaining=240; fi
printf 'render_started=true\nrender_timeout_seconds=%s\n' "$ep21_figure_remaining" \
  >> "$ep21_figure_run/render_status.txt"
timeout --signal=TERM --kill-after=5s "${ep21_figure_remaining}s" python3.12 -B \
  outputs/converter_training/tools/render_finding_a_result.py \
  > "$ep21_figure_run/render_execution.txt" 2>&1
