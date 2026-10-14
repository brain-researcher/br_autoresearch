#!/bin/bash
#SBATCH --job-name=ep06-lmp-followup
#SBATCH --partition=russpold,normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --time=00:35:00
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/lmp_support_followup/%x-%j.out
#SBATCH --error=outputs/runs/lmp_support_followup/%x-%j.err
set -euo pipefail
umask 077
[[ -n "${SLURM_JOB_ID:-}" && "$(hostname)" != *-ln* ]]
: "${OAK:?}" "${GROUP_HOME:?}" "${L_SCRATCH_JOB:?}"
ep06_root="$OAK/users/zijiao/br_autoresearch/episode06_lfp_regional_fingerprints"
ep06_ops="$ep06_root/outputs/runs/lmp_support_followup/job_${SLURM_JOB_ID}"
mkdir "$ep06_ops"
ep06_phase=startup
trap 'ep06_exit=$?; printf "phase=%s\nexit=%s\nfinished_utc=%s\n" "$ep06_phase" "$ep06_exit" "$(date -u +%FT%TZ)" >> "$ep06_ops/status.txt"' EXIT
printf 'hostname=%s\njob=%s\nrequested_cpus=1\nallocated_cpus=%s\nmemory_mib=%s\n' "$(hostname)" "$SLURM_JOB_ID" "${SLURM_JOB_CPUS_PER_NODE:-unknown}" "${SLURM_MEM_PER_NODE:-unknown}" > "$ep06_ops/status.txt"
cd "$ep06_root"
ep06_phase=substantive_lmp_analysis
set +e
bash outputs/tools/lmp_support_followup.sh attempt001 > "$ep06_ops/ep06.txt" 2>&1
ep06_science_exit=$?
set -e
printf 'ep06_exit=%s\n' "$ep06_science_exit" >> "$ep06_ops/status.txt"
# Separately authorized EP17 saved-result closeout; no fitting or audit access.
ep06_phase=ep17_saved_result_companion
source /share/software/user/open/lmod/lmod/init/bash
module purge
unset PYTHONPATH PYTHONHOME
module load devel python/3.12.1 math py-numpy/1.26.3_py312 viz py-matplotlib/3.8.3_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=''
export TMPDIR="$L_SCRATCH_JOB/ep17_saved_panel_${SLURM_JOB_ID}" MPLCONFIGDIR="$L_SCRATCH_JOB/ep17_mpl_${SLURM_JOB_ID}"
mkdir "$TMPDIR" "$MPLCONFIGDIR"
ep17_root="$OAK/users/zijiao/br_autoresearch/episode17_cneuromod_model_ranking_stability"
set +e
timeout --signal=TERM --kill-after=5s 120s python3.12 -B "$ep17_root/outputs/tools/report_saved_fixed_panel.py" > "$ep06_ops/ep17_report.txt" 2>&1
ep17_report_exit=$?
set -e
printf 'ep17_report_exit=%s\n' "$ep17_report_exit" >> "$ep06_ops/status.txt"
ep06_phase=changed_astra_validation
ep06_astra="$GROUP_HOME/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv"
ep06_deps="$GROUP_HOME/users/zijiao/ep18_model_runtime_cpu241_py312_46842156"
for ep06_program in "$ep06_root" "$ep17_root" "$OAK/users/zijiao/br_autoresearch/episode20_neurocam_prior_guided_codesign"; do
  ep06_name=${ep06_program##*/}
  env PYTHONPATH="$ep06_astra/lib/python3.12/site-packages:$ep06_deps/lib/python3.12/site-packages" timeout --signal=TERM --kill-after=5s 30s "$ep06_astra/bin/astra-tools" validate "$ep06_program/outputs/astra/v0.0.14/astra.yaml" --json > "$ep06_ops/${ep06_name}_astra.txt" 2>&1 || true
done
ep06_phase=complete
exit "$ep06_science_exit"
