#!/bin/bash
# Finite CRAM16 full-frame comparison only; no records or calibration outcomes.
# Run from the episode root. Coordinator owns submission and source freeze.
# Existing production writers continue independently on unchanged imports.
#SBATCH --job-name=ep20-s5-cram16-check
#SBATCH --partition=russpold,normal
#SBATCH --qos=normal
#SBATCH --constraint=CPU_GEN:SIE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/reference_s5/cram16-compare-%j.out
#SBATCH --error=outputs/runs/reference_s5/cram16-compare-%j.err
set -euo pipefail
umask 077
[[ $# -eq 0 ]]
[[ -n "${SLURM_JOB_ID:-}" && "$(hostname)" != *-ln* ]]
[[ "${SLURM_CPUS_PER_TASK:-}" == 1 && "${SLURM_MEM_PER_NODE:-}" == 2048 ]]
[[ "${SLURM_JOB_CPUS_PER_NODE:-}" =~ ^1(\(x1\))?$ ]]
[[ -z "${SLURM_JOB_GPUS:-}" && -z "${SLURM_STEP_GPUS:-}" && -z "${SLURM_ARRAY_JOB_ID:-}" ]]
ep20_cram_root="${OAK:?}/users/zijiao/br_autoresearch/episode20_neurocam_prior_guided_codesign"
ep20_cram_out="$ep20_cram_root/outputs/runs/reference_s5/cram16_frame_job_${SLURM_JOB_ID}"
mkdir -m 700 "$ep20_cram_out"
ep20_cram_finish() {
  ep20_cram_exit=$?
  trap - EXIT
  printf '{"job_id":"%s","exit_code":%s,"finished_utc":"%s","held_aside_access":false,"record_writer_created":false,"automatic_successor":false}\n' \
    "$SLURM_JOB_ID" "$ep20_cram_exit" "$(date -u +%FT%TZ)" > "$ep20_cram_out/terminal.json"
  exit "$ep20_cram_exit"
}
trap ep20_cram_finish EXIT
source /share/software/user/open/lmod/lmod/init/bash
module purge
unset PYTHONPATH PYTHONHOME
module load devel python/3.12.1 math py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 py-h5py/3.10.0_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 TMPDIR="${L_SCRATCH_JOB:?}"
cd "$ep20_cram_root/outputs/tools"
ep20_cram_seconds=$(( ${SLURM_JOB_END_TIME:?} - $(date +%s) - 120 ))
(( ep20_cram_seconds > 60 ))
if (( ep20_cram_seconds > 3300 )); then ep20_cram_seconds=3300; fi
printf 'hostname=%s\njob_id=%s\ncpus=%s\nmemory_mib=%s\ninternal_seconds=%s\n' \
  "$(hostname)" "$SLURM_JOB_ID" "$SLURM_CPUS_PER_TASK" "$SLURM_MEM_PER_NODE" "$ep20_cram_seconds" \
  > "$ep20_cram_out/execution.txt"
/usr/bin/time -v -o "$ep20_cram_out/resources.txt" \
  timeout --signal=TERM --kill-after=10s "$((ep20_cram_seconds+20))s" \
  python3.12 -u -B compare_reference_s5_cram16_frame.py \
  --total-seconds "$ep20_cram_seconds" --output "$ep20_cram_out/comparison.json" \
  >> "$ep20_cram_out/execution.txt" 2>&1
