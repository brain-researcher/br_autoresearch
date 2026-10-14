#!/bin/bash
#SBATCH --job-name=ep20-restart-seam
#SBATCH --partition=russpold,normal
#SBATCH --constraint=CPU_GEN:SIE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/fast_forward_stage1/solver/slurm-%j.out
#SBATCH --error=outputs/runs/fast_forward_stage1/solver/slurm-%j.err
set -euo pipefail
umask 077
[[ -n "${SLURM_JOB_ID:-}" && "$(hostname)" != *-ln* ]]
[[ "${SLURM_JOB_CPUS_PER_NODE:-}" =~ ^1(\(x1\))?$ ]]
ep20_root="${OAK:?}/users/zijiao/br_autoresearch/episode20_neurocam_prior_guided_codesign"
ep20_label="${1:-restart_seam001}"
[[ "$ep20_label" =~ ^restart_seam[0-9]+$ ]]
ep20_run="$ep20_root/outputs/runs/fast_forward_stage1/solver/$ep20_label"
mkdir "$ep20_run"
ep20_phase=setup
trap 'ep20_exit=$?; printf "{\"phase\":\"%s\",\"exit_code\":%s,\"finished_utc\":\"%s\",\"scientific_scoring\":false,\"automatic_successor\":false}\n" "$ep20_phase" "$ep20_exit" "$(date -u +%FT%TZ)" > "$ep20_run/terminal.json"' EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
printf 'host=%s\njob=%s\nallocated_cpus=%s\nmemory_mib=%s\n' "$(hostname)" "$SLURM_JOB_ID" "$SLURM_JOB_CPUS_PER_NODE" "$SLURM_MEM_PER_NODE" > "$ep20_run/status.txt"
source /share/software/user/open/lmod/lmod/init/bash
module purge
unset PYTHONPATH PYTHONHOME
module load devel python/3.12.1 math py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 py-h5py/3.10.0_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export TMPDIR="${L_SCRATCH_JOB:?}"
cd "$ep20_root/outputs/tools"
ep20_deadline=$((${SLURM_JOB_START_TIME:?}+1620))
(( ep20_deadline <= ${SLURM_JOB_END_TIME:?}-60 )) || ep20_deadline=$((${SLURM_JOB_END_TIME:?}-60))
ep20_left=$((ep20_deadline-$(date +%s)))
(( ep20_left > 60 ))
export EP20_RESTART_TEST_DIR="$ep20_run/seam"
export EP20_RESTART_DEADLINE_SECONDS="$ep20_left" EP20_RESTART_MEMORY_MIB=6656
mkdir "$EP20_RESTART_TEST_DIR"
ep20_phase=two_continuous_frames_then_one_restored_frame
/usr/bin/time -v -o "$ep20_run/resources.txt" \
 timeout --signal=TERM --kill-after=30s "$((ep20_left+20))s" \
 python3.12 -u -B -m unittest -v \
 test_fast_solver_restart.RestartIntegrationTests.test_two_frame_restart_matches_uninterrupted \
 > "$ep20_run/test_stdout.txt" 2>&1
ep20_phase=restart_seam_complete_no_scoring
