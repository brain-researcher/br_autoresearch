"""Export only approved regular image bytes; never decode or score images."""
import csv
import hashlib
import json
import os
import shutil
from pathlib import Path

import h5py
import numpy as np

episode = Path(__file__).resolve().parents[3]
records = episode / "outputs/steward_provisioning"
source = Path(os.environ["OAK"]) / (
    "users/zijiao/br_autoresearch_data/steward_acquisition/"
    "laion_fmri_raw_stimuli_dua_20260922/payload")
final = records / "regular_stimuli"
job_id = os.environ["SLURM_JOB_ID"]
stage = Path(os.environ["SCRATCH"]) / "br_autoresearch" / episode.name / f"steward-export-{job_id}"
durable_partial = records / f"regular_stimuli.partial-{job_id}"
if final.exists() or stage.exists() or durable_partial.exists():
    raise FileExistsError("Do not duplicate or overwrite an export")
manifest = json.loads((records / "source_structural_manifest.json").read_text())
with (records / "regular_image_index.csv").open() as handle:
    index = list(csv.DictReader(handle))
with (source / "task-images_metadata.csv").open() as handle:
    metadata = list(csv.DictReader(handle))
regular_ids = set().union(*[set(x) for x in manifest["unique_image_ids"].values()],
    set(manifest["provider_split"]["variant"]["train_ids"]),
    set(manifest["provider_split"]["variant"]["test_ids"]))
source_indices = [int(row["source_hdf5_index"]) for row in index]
if (len(index) != 24681 or len(regular_ids) != len(index)
        or source_indices != sorted(set(source_indices))):
    raise ValueError("Incomplete or duplicated approved regular support")
for row, source_index in zip(index, source_indices):
    entry = metadata[source_index]
    if (entry["dataset"] == "OOD" or entry["image_name"] != row["image_id"]
            or row["image_id"] not in regular_ids):
        raise ValueError("Forbidden image index or metadata mismatch")

stage.mkdir(parents=True)
image_file = stage / "task-images_stimuli.h5"
encoded_bytes = 0
with h5py.File(source / image_file.name, "r") as src, h5py.File(image_file, "w") as dst:
    if src["images"].shape != (len(metadata),):
        raise ValueError("Source image/metadata row alignment changed")
    images = dst.create_dataset("images", shape=(len(index),), dtype=h5py.vlen_dtype(np.dtype("uint8")))
    dst.attrs["ep21_roles"] = "regular_subject_unique_and_shared_tau_only"
    dst.attrs["ood_excluded"] = True
    for output_index, source_index in enumerate(source_indices):
        # Each dereferenced vlen row has an authenticated regular image ID.
        data = src["images"][source_index]
        encoded_bytes += data.nbytes
        if encoded_bytes > 4_900_000_000:
            raise ValueError("Approved durable-output limit would be exceeded")
        images[output_index] = data
        if (output_index + 1) % 5000 == 0:
            print(f"Exported {output_index + 1}/{len(index)} regular images", flush=True)

with (stage / "task-images_metadata.csv").open("w") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(metadata[0]))
    writer.writeheader()
    writer.writerows(metadata[i] for i in source_indices)
shutil.copyfile(records / "regular_image_index.csv", stage / "source_index.csv")
if sum(p.stat().st_size for p in stage.iterdir()) > 5_000_000_000:
    raise ValueError("Approved 5 GB durable-output limit exceeded")

durable_partial.mkdir()
hashes = {}
for path in stage.iterdir():
    digest = hashlib.sha256()
    with path.open("rb") as src, (durable_partial / path.name).open("wb") as dst:
        while chunk := src.read(8 * 1024 * 1024):
            digest.update(chunk)
            dst.write(chunk)
    hashes[path.name] = digest.hexdigest()
report = dict(status="complete_regular_only_stimulus_export", slurm_job_id=job_id,
    image_count=len(index), regular_image_ids=sorted(regular_ids), excluded_ood_images=371,
    source=str(source), output=str(final), encoded_bytes=encoded_bytes,
    artifact_bytes=sum(p.stat().st_size for p in durable_partial.iterdir()), sha256=hashes,
    terms_version="2026-05-12", risk="Low_user_attested",
    neural_values_read=False, ood_image_values_read=False, images_decoded=False,
    visual_features_extracted=False, models_fitted=False, scores_computed=False,
    source_integrity="reuse_completed_steward_acquisition_provider_manifest_and_receipt",
    approved_resources=dict(cpus=1, memory_gib=4, minutes=30,
        durable_bytes=5_000_000_000, scratch_bytes=10_000_000_000))
(durable_partial / "export_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
if sum(p.stat().st_size for p in durable_partial.iterdir()) > 5_000_000_000:
    raise ValueError("Final export manifest would exceed approved output limit")
for path in durable_partial.iterdir():
    path.chmod(0o400)
durable_partial.chmod(0o500)
durable_partial.rename(final)
shutil.rmtree(stage)
print(json.dumps({k: v for k, v in report.items() if k not in ("regular_image_ids", "sha256")}), flush=True)
