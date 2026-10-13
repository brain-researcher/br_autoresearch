#!/bin/bash
#SBATCH --job-name=ep22-d0
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --chdir=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode22_geometry_gap_decomposition
#SBATCH --output=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode22_geometry_gap_decomposition/outputs/runs/d0_outcome_blind_spectra/20261012-attempt01/slurm-%j.out
# EP22 D0: outcome-blind embedding spectra. Reads embeddings and stimulus
# identity metadata only; no beta, ROI or RSA value is opened.
set -uo pipefail

attempt=outputs/runs/d0_outcome_blind_spectra/20261012-attempt01
figdir=outputs/figures/d0_outcome_blind_spectra/20261012-attempt01
corrected=inputs/laion_ood_embeddings_corrected
source_root="${OAK:?}/users/zijiao/br_autoresearch_data/steward_acquisition/laion_fmri_mixed_role_20260922"
dev_handoff=../episode04_shared_scene_geometry/outputs/runs/source_readiness/20261003-development_embeddings01
bucket=https://laion-fmri.s3.us-west-2.amazonaws.com/stimuli
models="CLIP DINOv2 PEcore SigLIP2"

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} start=$(date -u +%FT%TZ)"
module purge
module load devel python/3.12.1 py-h5py/3.10.0_py312 math py-numpy/1.26.3 viz py-matplotlib/3.8.3_py312
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK \
       MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK PYTHONDONTWRITEBYTECODE=1

# 1. Acquire the provider's 2026-09-28 corrected embedding files (public CC0), once.
if [ ! -f "$corrected/download_record.json" ]; then
  tmp="$SCRATCH/br_autoresearch/episode22_geometry_gap_decomposition/d0_download_${SLURM_JOB_ID}"
  mkdir -p "$tmp"
  ok=1
  for m in $models; do
    f="task-images_desc-${m}_embeddings.h5"
    curl -sS --fail --retry 3 --retry-delay 10 --max-time 900 -D "$tmp/$m.headers" -o "$tmp/$f" "$bucket/$f" || ok=0
  done
  if [ "$ok" = 1 ]; then
    python3.12 - "$tmp" "$bucket" $models <<'PY'
import json, os, sys, datetime
tmp, bucket, models = sys.argv[1], sys.argv[2], sys.argv[3:]
record = {"downloaded_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
          "job_id": os.environ.get("SLURM_JOB_ID"), "files": {}}
for m in models:
    name = f"task-images_desc-{m}_embeddings.h5"
    headers = {}
    for line in open(os.path.join(tmp, f"{m}.headers")):
        if ":" in line:
            k, v = line.split(":", 1)
            headers[k.strip().lower()] = v.strip()
    size = os.path.getsize(os.path.join(tmp, name))
    if str(size) != headers.get("content-length"):
        raise SystemExit(f"{name}: size {size} != Content-Length {headers.get('content-length')}")
    record["files"][m] = {"url": f"{bucket}/{name}", "etag": headers.get("etag"),
                          "last_modified": headers.get("last-modified"), "bytes": size}
json.dump(record, open(os.path.join(tmp, "download_record.json"), "w"), indent=1)
PY
    [ $? -eq 0 ] || ok=0
  fi
  if [ "$ok" = 1 ]; then
    mkdir -p "$corrected"
    for m in $models; do mv "$tmp/task-images_desc-${m}_embeddings.h5" "$corrected/"; done
    mv "$tmp/download_record.json" "$corrected/"
    chmod a-w "$corrected"/*
    rm -rf "$tmp"
  else
    echo "corrected embedding download failed; D0 continues with development strata only" | tee "$attempt/download_failure.txt"
  fi
fi

# 2. D0 analysis (embeddings and identity metadata only).
timeout --signal=TERM --kill-after=10s 1500s python3.12 -u -B outputs/code/d0_outcome_blind_spectra.py \
  --dev-embeddings "$dev_handoff" \
  --stimulus-metadata "$source_root/dataset/stimuli/task-images_metadata.csv" \
  --original-embeddings "$source_root/dataset/stimuli" \
  --corrected-embeddings "$corrected" \
  --out-dir "$attempt/results"
d0_exit=$?
echo "d0_exit=$d0_exit"

# 3. Figure from the saved summary only.
if [ -f "$attempt/results/d0_spectra_summary.json" ]; then
  python3.12 -B outputs/code/plot_d0_spectra.py --summary "$attempt/results/d0_spectra_summary.json" --out-dir "$figdir"
  echo "plot_exit=$?"
fi

# 4. Program record checks (not execution gates).
astra="${GROUP_HOME:?}/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv/bin/astra-tools"
timeout --signal=TERM --kill-after=10s 60s "$astra" validate outputs/astra/v0.0.14/astra.yaml --json > "$attempt/astra_validation.json" 2>&1
echo "astra_exit=$?"
venv_py="$(dirname "$astra")/python"
timeout 60s "$venv_py" -I -c 'import sys, yaml; [yaml.safe_load(open(p)) for p in sys.argv[1:]]; print("yaml ok", len(sys.argv) - 1)' \
  SEARCH_POLICY.yaml ../EPISODE_REGISTRY.yaml > "$attempt/yaml_parse_check.txt" 2>&1
echo "yaml_exit=$?"
echo "end=$(date -u +%FT%TZ)"
exit $d0_exit
