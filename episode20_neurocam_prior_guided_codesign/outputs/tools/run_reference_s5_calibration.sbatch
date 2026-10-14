#!/bin/bash
#SBATCH --job-name=ep20-s5-calibration
#SBATCH --partition=russpold,normal
#SBATCH --constraint=CPU_GEN:SIE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/reference_s5/slurm-%j.out
#SBATCH --error=outputs/runs/reference_s5/slurm-%j.err
set -euo pipefail
umask 077
[[ -n "${SLURM_JOB_ID:-}" && "$(hostname)" != *-ln* ]]
[[ "${SLURM_JOB_CPUS_PER_NODE:-}" =~ ^1(\(x1\))?$ ]]
s5_root="${OAK:?}/users/zijiao/br_autoresearch/episode20_neurocam_prior_guided_codesign"
s5_base="$s5_root/outputs/runs/reference_s5"
s5_run="$s5_base/calibration_job_${SLURM_JOB_ID}"
s5_manifest="$s5_base/conditional_family.json"
# One declared nominal R1 point only. Extend from the measured record/checkpoint
# manually; no distant comparison, optimization successor or self-submission.
s5_record="$s5_base/records/R1/point_01"
mkdir "$s5_run"
s5_phase=setup
trap 's5_exit=$?; printf "{\"phase\":\"%s\",\"exit_code\":%s,\"finished_utc\":\"%s\",\"target_frames\":4000,\"held_aside_access\":false,\"automatic_successor\":false}\n" "$s5_phase" "$s5_exit" "$(date -u +%FT%TZ)" > "$s5_run/terminal.json"' EXIT
source /share/software/user/open/lmod/lmod/init/bash
module purge
unset PYTHONPATH PYTHONHOME
module load devel python/3.12.1 math py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 py-h5py/3.10.0_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export TMPDIR="${L_SCRATCH_JOB:?}"
cd "$s5_root/outputs/tools"
s5_deadline=$(( ${SLURM_JOB_END_TIME:?} - 120 ))
s5_phase=focused_new_operator_checks
timeout --signal=TERM --kill-after=10s 300s \
 python3.12 -B -m unittest -v test_reference_s5 > "$s5_run/checks.txt" 2>&1
s5_phase=prefit_family_freeze
if [[ ! -e "$s5_manifest" ]]; then
 python3.12 -B reference_s5_prediction.py prepare --output "$s5_manifest" > "$s5_run/prepare.txt" 2>&1
fi
if [[ -e "$s5_record/frame_checkpoint.npz" ]]; then
 set -- --resume
else
 set --
fi
s5_left=$((s5_deadline-$(date +%s)))
(( s5_left > 60 ))
s5_phase=full_record_nearest_only_calibration_point
/usr/bin/time -v -o "$s5_run/resources.txt" \
 timeout --signal=TERM --kill-after=30s "$((s5_left+30))s" \
 python3.12 -u -B reference_s5_prediction.py run \
 --manifest "$s5_manifest" --structure R1 --point 1 \
 --record-dir "$s5_record" --run-dir "$s5_run/record_operation" \
 --stop-after-frame 4000 --deadline-seconds "$s5_left" --memory-mib 6656 \
 "$@" > "$s5_run/record_stdout.txt" 2>&1
s5_phase=bounded_calibration_complete_or_checkpointed_no_prediction
