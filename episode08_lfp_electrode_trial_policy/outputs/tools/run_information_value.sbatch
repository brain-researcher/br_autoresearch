#!/bin/bash
#SBATCH --job-name=ep08-info-value
#SBATCH --partition=russpold,normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:35:00
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/offline_information_value/%x-%j.out
#SBATCH --error=outputs/runs/offline_information_value/%x-%j.err
set -euo pipefail
umask 077
[[ -n "${SLURM_JOB_ID:-}" && "$(hostname)" != *-ln* ]]
ep08_root="${OAK:?}/users/zijiao/br_autoresearch/episode08_lfp_electrode_trial_policy"
cd "$ep08_root"
ep08_ops="$PWD/outputs/runs/offline_information_value/job_${SLURM_JOB_ID}"
mkdir "$ep08_ops"
ep08_phase=startup
trap 'ep08_exit=$?; printf "phase=%s\nexit=%s\nfinished_utc=%s\n" "$ep08_phase" "$ep08_exit" "$(date -u +%FT%TZ)" >> "$ep08_ops/status.txt"' EXIT
printf 'host=%s\njob=%s\nallocated_cpus=%s\nmemory_mib=%s\n' "$(hostname)" "$SLURM_JOB_ID" "${SLURM_JOB_CPUS_PER_NODE:-unknown}" "${SLURM_MEM_PER_NODE:-unknown}" > "$ep08_ops/status.txt"
source /share/software/user/open/lmod/lmod/init/bash
module purge
unset PYTHONPATH PYTHONHOME
module load devel python/3.12.1 math py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 py-h5py/3.10.0_py312 viz py-matplotlib/3.8.3_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=''
export TMPDIR="${L_SCRATCH_JOB:?}/ep08_info_${SLURM_JOB_ID}" MPLCONFIGDIR="$L_SCRATCH_JOB/ep08_mpl_${SLURM_JOB_ID}"
mkdir "$TMPDIR" "$MPLCONFIGDIR"
ep08_phase=focused_checks
timeout --signal=TERM --kill-after=5s 120s python3.12 -B -m unittest discover -s outputs/tools -p test_electrode_information_value.py -v > "$ep08_ops/checks.txt" 2>&1
ep08_phase=real_data_analysis
timeout --signal=TERM --kill-after=10s 1860s python3.12 -u -B outputs/tools/development_information_value.py \
 --source-root "$OAK/data/br_autoresearch_data/sensorimotor_lfp/dryad-xd2547dkt-v5-payload/source" \
 --metadata-report outputs/runs/source_support/attempt002/metadata_support.json \
 --trial-roles outputs/runs/source_support/development_trial_roles_attempt001/development_trial_roles.json \
 --output-dir outputs/runs/offline_information_value/attempt001 --max-seconds 1800 --max-rss-gib 5 > "$ep08_ops/run.txt" 2>&1
ep08_phase=changed_astra_validation
ep08_astra="${GROUP_HOME:?}/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv"
ep08_deps="$GROUP_HOME/users/zijiao/ep18_model_runtime_cpu241_py312_46842156"
for ep08_program in episode06_lfp_regional_fingerprints episode08_lfp_electrode_trial_policy episode17_cneuromod_model_ranking_stability episode20_neurocam_prior_guided_codesign; do
 env PYTHONPATH="$ep08_astra/lib/python3.12/site-packages:$ep08_deps/lib/python3.12/site-packages" timeout --signal=TERM --kill-after=5s 25s "$ep08_astra/bin/astra-tools" validate "$OAK/users/zijiao/br_autoresearch/$ep08_program/outputs/astra/v0.0.14/astra.yaml" --json > "$ep08_ops/${ep08_program}_astra.txt" 2>&1 || true
done
ep08_phase=complete
