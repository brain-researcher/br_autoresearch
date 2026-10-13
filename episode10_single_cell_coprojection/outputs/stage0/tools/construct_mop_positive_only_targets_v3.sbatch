#!/usr/bin/env bash
#SBATCH --job-name=ep10-v3-targets
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --partition=russpold,normal

set -euo pipefail
umask 007

: "${SCRATCH:?SCRATCH is required}"
: "${OAK:?OAK is required}"
: "${GROUP_HOME:?GROUP_HOME is required}"
: "${SLURM_JOB_ID:?SLURM_JOB_ID is required}"

ml purge
ml load devel python/3.12.1
export PYTHONNOUSERSITE=1

python_bin="$GROUP_HOME/users/zijiao/envs/ep10_targets_py312/bin/python"
project="$OAK/users/zijiao/br_autoresearch/episode10_single_cell_coprojection"
stage0="$project/outputs/stage0"
constructor="$stage0/tools/construct_mop_terminal_targets.py"
validator="$stage0/tools/validate_mop_positive_only_v3_outputs.py"
base_contract="$stage0/projection_observation_contract.yaml"
v2_contract="$stage0/projection_observation_contract_v2.yaml"
v3_contract="$stage0/projection_observation_contract_v3.yaml"
authorization="$stage0/v3_user_authorization.yaml"
overlap="$stage0/gao_mop_development_structural_qc_overlap_v2.json"
vocabulary="$stage0/allen_target_vocabulary.csv"
assets="$SCRATCH/br_autoresearch/episode11_projection_types_vs_gradients/allen_outcome_assets"
development="$SCRATCH/br_autoresearch/episode10_single_cell_coprojection/gao_mop_development_v1"
output="$SCRATCH/br_autoresearch/episode10_single_cell_coprojection/gao_mop_development_targets_v3"

for required in \
  "$python_bin" \
  "$constructor" \
  "$validator" \
  "$base_contract" \
  "$v2_contract" \
  "$v3_contract" \
  "$authorization" \
  "$overlap" \
  "$vocabulary" \
  "$stage0/gao_mop_development_manifest.csv" \
  "$assets/annotation_25_ccf_2017.nrrd" \
  "$assets/structure_graph_1.json" \
  "$development/download_log.csv"; do
  test -f "$required"
done

if [[ -e "$output" ]]; then
  printf 'ERROR: refusing to overwrite existing v3 output: %s\n' "$output" >&2
  exit 3
fi

"$python_bin" --version
"$python_bin" - <<'PY'
from importlib.metadata import version

for distribution in ("numpy", "PyYAML", "pynrrd", "typing_extensions"):
    print(f"{distribution}={version(distribution)}")
PY

# Both checks are synthetic or target-blind. They run before the vocabulary is
# passed to the constructor.
"$python_bin" "$constructor" --self-test
"$python_bin" "$validator" --self-test
"$python_bin" "$validator" \
  --preflight-only \
  --contract "$v3_contract" \
  --authorization "$authorization" \
  --overlap-summary "$overlap"

"$python_bin" "$constructor" \
  --execute \
  --contract "$base_contract" \
  --revision-contract "$v2_contract" \
  --positive-only-contract "$v3_contract" \
  --vocabulary "$vocabulary" \
  --annotation "$assets/annotation_25_ccf_2017.nrrd" \
  --ontology "$assets/structure_graph_1.json" \
  --manifest "$stage0/gao_mop_development_manifest.csv" \
  --swc-root "$development/swc" \
  --output-root "$output"

"$python_bin" "$validator" \
  --execute \
  --output-root "$output" \
  --contract "$v3_contract" \
  --authorization "$authorization" \
  --overlap-summary "$overlap"
