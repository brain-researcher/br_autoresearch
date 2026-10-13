"""Prepare EP21 role metadata from real provider files; read no payload values."""
import csv
import gzip
import itertools
import json
import os
import struct
import tarfile
from collections import Counter
from pathlib import Path
from trial_binding import session_volume_indices

episode = Path(__file__).resolve().parents[3]
out = episode / "outputs/steward_provisioning"
sources = Path(os.environ["OAK"]) / "users/zijiao/br_autoresearch_data/steward_acquisition"
mixed = sources / "laion_fmri_mixed_role_20260922"
raw = sources / "laion_fmri_raw_stimuli_dua_20260922"
meta = mixed / "dataset/stimuli/task-images_metadata.csv"
if meta.read_bytes() != (raw / "payload/task-images_metadata.csv").read_bytes():
    raise ValueError("Image metadata differs between beta and stimulus acquisitions")
with meta.open() as handle:
    images = list(csv.DictReader(handle))
by_id = {row["image_name"]: row for row in images}
if len(by_id) != len(images):
    raise ValueError("Duplicate image identity")
regular = {key for key, row in by_id.items() if row["dataset"] != "OOD"}
excluded = set(by_id) - regular
shared = {key for key in regular if by_id[key]["unique_or_shared"] == "shared"}
subjects = sorted({row["participant"] for row in images
                   if row["unique_or_shared"] == "unique"})
unique = {sub: {key for key in regular if by_id[key]["participant"] == sub
               and by_id[key]["unique_or_shared"] == "unique"} for sub in subjects}
archive = mixed / "provider_code/LAION-fMRI-ffb6abef8f787ca37f9a7faa9dd4a97119b6d603.tar.gz"
with tarfile.open(archive) as source:
    member = next(name for name in source.getnames()
                  if name.endswith("/splits/data/shared/tau.json"))
    split = json.load(source.extractfile(member))
variant = next(v for v in split["variants"] if v["variant_id"] == 0)
train, test = set(variant["train_ids"]), set(variant["test_ids"])
if (len(subjects) != 5 or any(len(x) != 4712 for x in unique.values())
        or len(train) != 897 or len(test) != 224 or train & test
        or train | test != shared or len(excluded) != 371
        or len(train) != len(variant["train_ids"])
        or len(test) != len(variant["test_ids"])):
    raise ValueError("Provider role/split metadata does not match the existing design")
pair_table = []
for source, target in itertools.permutations(subjects, 2):
    overlap = unique[source] & unique[target]
    if overlap or unique[source] & test or unique[target] & test:
        raise ValueError("Forbidden exact training/evaluation image overlap")
    pair_table.append(dict(source=source, target=target,
                           source_target_unique_overlap=len(overlap)))

def nifti_header(path):
    with gzip.open(path, "rb") as handle:
        header = handle.read(348)
    endian = "<" if struct.unpack("<i", header[:4])[0] == 348 else ">"
    if len(header) != 348 or struct.unpack(endian + "i", header[:4])[0] != 348:
        raise ValueError(f"Unsupported NIfTI header: {path.name}")
    dims = struct.unpack_from(endian + "8h", header, 40)
    if dims[0] != 4:
        raise ValueError("Expected four-dimensional single-trial beta file")
    return dict(shape=list(dims[1:5]),
                datatype_code=struct.unpack_from(endian + "h", header, 70)[0],
                bits_per_value=struct.unpack_from(endian + "h", header, 72)[0],
                voxel_spacing=list(struct.unpack_from(endian + "8f", header, 76)[1:4]))

out.mkdir(parents=True, exist_ok=True)
trial_path = out / "regular_trial_index.tsv"
rows_by_subject, headers, sessions = {}, [], []
full_unmasked_regular_bytes = 0
with trial_path.open("w") as handle:
    fields = ["participant", "session", "run", "beta_index", "beta_volume_index", "image_id", "role"]
    writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
    writer.writeheader()
    for sub in subjects:
        repetitions, role_rows, removed = Counter(), Counter(), 0
        for number in range(1, 31):
            session = f"ses-{number:02d}"
            prefix = f"{sub}_{session}_task-images"
            folder = mixed / f"dataset/derivatives/glmsingle-tedana/{sub}/{session}/func"
            trials = folder / f"{prefix}_desc-SingletrialBetas_trials.tsv"
            beta = folder / f"{prefix}_space-T1w_stat-effect_desc-SingletrialBetas_statmap.nii.gz"
            header = nifti_header(beta)
            with trials.open() as table:
                records = list(csv.DictReader(table, delimiter="\t"))
            indices = session_volume_indices(records, session, header["shape"][3])
            kept = 0
            for volume_index, row in zip(indices, records):
                key = row["label"]
                if key in excluded:
                    removed += 1
                    continue
                if key not in regular:
                    raise ValueError(f"Unknown trial image ID in {sub}/{session}")
                if key in unique[sub]:
                    role = "subject_unique"
                elif key in train:
                    role = "tau_train"
                elif key in test:
                    role = "tau_test"
                else:
                    raise ValueError(f"Participant/image role mismatch: {sub}/{session}")
                writer.writerow(dict(participant=sub, session=session, run=row["run"],
                    beta_index=row["beta_index"], beta_volume_index=volume_index,
                    image_id=key, role=role))
                repetitions[key] += 1
                role_rows[role] += 1
                kept += 1
            full_unmasked_regular_bytes += (kept * header["bits_per_value"] // 8
                * header["shape"][0] * header["shape"][1] * header["shape"][2])
            sessions.append(dict(participant=sub, session=session,
                beta_path=str(beta), trials_path=str(trials), header=header,
                regular_rows=kept, excluded_ood_rows=len(records)-kept))
        expected = unique[sub] | shared
        if set(repetitions) != expected:
            raise ValueError(f"Incomplete observed regular image support: {sub}")
        rows_by_subject[sub] = dict(unique_images=len(unique[sub]),
            tau_train_images=len(train), tau_test_images=len(test),
            trial_rows=dict(role_rows), excluded_ood_rows=removed,
            repetition_histogram=dict(sorted(Counter(repetitions.values()).items())))

with (out / "regular_image_index.csv").open("w") as handle:
    writer = csv.DictWriter(handle, fieldnames=["source_hdf5_index", "image_id", "dataset",
        "participant", "role", "provider_repetitions"])
    writer.writeheader()
    for index, row in enumerate(images):
        key = row["image_name"]
        if key not in regular:
            continue
        role = "subject_unique" if row["unique_or_shared"] == "unique" else (
            "tau_train" if key in train else "tau_test")
        writer.writerow(dict(source_hdf5_index=index, image_id=key, dataset=row["dataset"],
            participant=row["participant"], role=role, provider_repetitions=row["n_reps"]))

manifest = dict(schema_version="ep21.steward_structural.v1", contains_neural_values=False,
    contains_stimulus_values=False, contains_outcome_summaries=False,
    provider_code_commit="ffb6abef8f787ca37f9a7faa9dd4a97119b6d603",
    source_acquisition="2026-09-22 selected provider object snapshot",
    release_metadata_status="provider_snapshot_bound_dataset_DOI_and_semantic_release_unspecified",
    provider_split=dict(archive=str(archive), member=member, name=split["name"],
        pool=split["pool"], splitter=split["splitter"], variant=variant),
    participants=subjects, unique_image_ids={s: sorted(v) for s, v in unique.items()},
    subject_role_support=rows_by_subject, directed_pairs=pair_table,
    role_image_counts=dict(subject_unique_total=sum(map(len, unique.values())),
        regular_shared=len(shared), tau_train=len(train), tau_test=len(test),
        excluded_ood=len(excluded), regular_union=len(regular)),
    observed_sessions=sessions, regular_trial_index=str(trial_path),
    regular_image_index=str(out / "regular_image_index.csv"),
    unmasked_regular_beta_bytes=full_unmasked_regular_bytes,
    authorization=dict(terms_version="2026-05-12", risk="Low_user_attested",
        steward_provisioning=True, empirical_fitting=False, neural_scoring=False),
    unresolved=dict(eligible_whole_visual_cortex_mask_provenance=True,
        neural_finite_nan_and_mask_coverage=True, physically_exported_regular_payloads=True,
        large_export_resource_allocation=True, exact_pipeline_assets_and_protocol_lock=True))
(out / "source_structural_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(dict(status="real_provider_structural_metadata_prepared",
    participants=subjects, counts=manifest["role_image_counts"],
    subject_role_support=rows_by_subject, directed_pairs=len(pair_table),
    sessions=len(sessions), unmasked_regular_beta_bytes=full_unmasked_regular_bytes)))
