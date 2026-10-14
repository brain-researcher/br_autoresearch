#!/bin/bash
#SBATCH --job-name=ep23_opfeas_dl
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=/scratch/users/zijiao/br_autoresearch/episode23_spectral_operator_convergence/feasibility/logs/download_%j.out
# EP23 operator feasibility: fetch public annotation files (no images) into
# transient scratch. Each source gets its own new directory.
set -euo pipefail
D=$SCRATCH/br_autoresearch/episode23_spectral_operator_convergence/feasibility/downloads
REC=$D/download_record.tsv
mkdir -p "$D"
printf 'source\turl\tbytes\tlast_modified\tfetched_utc\n' > "$REC"
fetch() {
  local src=$1 url=$2 out
  mkdir -p "$D/$src"
  out="$D/$src/$(basename "$url")"
  curl -fsSL --retry 3 --max-time 1800 -o "$out" "$url"
  lm=$(curl -sIL --max-time 30 "$url" | grep -i '^last-modified' | tail -1 | cut -d' ' -f2- | tr -d '\r')
  printf '%s\t%s\t%s\t%s\t%s\n' "$src" "$url" "$(stat -c %s "$out")" "$lm" "$(date -u +%FT%TZ)" >> "$REC"
}
fetch visual_genome https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/objects.json.zip
fetch visual_genome https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/object_synsets.json.zip
fetch lvis https://dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip
fetch lvis https://dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip
fetch coco http://images.cocodataset.org/annotations/annotations_trainval2017.zip
fetch open_images https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions.csv
chmod -R a-w "$D"/visual_genome "$D"/lvis "$D"/coco "$D"/open_images
cat "$REC"
