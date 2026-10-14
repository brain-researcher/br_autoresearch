"""Build EP23 A0 direct-count and shared-context operators."""

import argparse
import csv
import io
import json
import re
import time
import zipfile
from array import array
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import numpy as np
from scipy import sparse


PARTICIPANTS = ("sub-01", "sub-03", "sub-05", "sub-06", "sub-07")
STRATA = ("laion", "things", "thingsplus")
META_COLUMNS = ("uniqueID", "Word", "WordNet ID", "Wordnet ID4")
IMAGE_PATTERN = re.compile(r"^unique_THINGS\+?_(.+)_p\d+\.jpg$")
THRESHOLDS = (1, 5, 10, 50)
VG_DROP_KEYS = {"x", "y", "w", "h", "merged_object_ids", "object_id"}
INT64_MAX = int(np.iinfo(np.int64).max)


def require_fields(obj, fields, where):
    """Require an object and its named fields."""
    if not isinstance(obj, dict):
        raise ValueError(f"{where}: expected a JSON object")
    missing = [field for field in fields if field not in obj]
    if missing:
        raise ValueError(f"{where}: missing fields {missing}")


def require_string(value, where):
    """Require a nonempty string."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{where}: expected a nonempty string")
    return value


def require_integer(value, where):
    """Require a nonnegative int64-compatible integer."""
    if type(value) is not int or not 0 <= value <= INT64_MAX:
        raise ValueError(f"{where}: expected a nonnegative int64 integer")
    return value


def read_things(path):
    """Read THINGS metadata while preserving its concept order."""
    rows = []
    seen_uids = set()
    with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        headers = reader.fieldnames or []
        missing = sorted(set(META_COLUMNS) - set(headers))
        if missing:
            raise ValueError(f"{path}: missing columns {missing}")
        if len(headers) != len(set(headers)):
            raise ValueError(f"{path}: duplicate column names")
        for index, row in enumerate(reader):
            where = f"{path}: concept row {index}"
            if None in row or any(row[name] is None for name in META_COLUMNS):
                raise ValueError(f"{where}: malformed TSV row")
            record = {name: row[name].strip() for name in META_COLUMNS}
            uid = require_string(record["uniqueID"], f"{where}: uniqueID")
            if uid in seen_uids:
                raise ValueError(f"{where}: duplicate uniqueID {uid!r}")
            seen_uids.add(uid)
            rows.append(record)
    if len(rows) != 1854:
        raise ValueError(f"{path}: expected 1854 concepts, found {len(rows)}")
    return rows


def jsonl_rows(path):
    """Yield numbered JSON objects, rejecting blank or malformed lines."""
    with path.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, 1):
            where = f"{path}:{line_number}"
            if not line.strip():
                raise ValueError(f"{where}: blank JSONL line")
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise ValueError(f"{where}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{where}: expected a JSON object")
            yield line_number, row


def read_development(path, uid_to_index):
    """Collect mapped development concepts and unmapped-row counts."""
    concepts = {participant: set() for participant in PARTICIPANTS}
    row_counts = Counter({participant: 0 for participant in PARTICIPANTS})
    unmapped_participant = Counter(
        {participant: 0 for participant in PARTICIPANTS}
    )
    unmapped_stratum = Counter({"things": 0, "thingsplus": 0})
    unmapped_reasons = Counter()
    seen_output_rows = set()
    seen_images = set()

    for line_number, row in jsonl_rows(path):
        where = f"{path}:{line_number}"
        require_fields(
            row, ("participant", "image_id", "source_stratum", "output_row"),
            where,
        )
        participant = require_string(row["participant"], f"{where}: participant")
        image_id = require_string(row["image_id"], f"{where}: image_id")
        stratum = require_string(row["source_stratum"], f"{where}: source_stratum")
        output_row = require_integer(row["output_row"], f"{where}: output_row")
        if participant not in PARTICIPANTS:
            raise ValueError(f"{where}: unexpected participant {participant!r}")
        if stratum not in STRATA:
            raise ValueError(f"{where}: unexpected source_stratum {stratum!r}")

        output_key = (participant, output_row)
        image_key = (participant, image_id)
        if output_key in seen_output_rows:
            raise ValueError(f"{where}: duplicate participant/output_row {output_key}")
        if image_key in seen_images:
            raise ValueError(f"{where}: duplicate participant/image_id {image_key}")
        seen_output_rows.add(output_key)
        seen_images.add(image_key)
        row_counts[participant] += 1

        if stratum == "laion":
            continue
        match = IMAGE_PATTERN.match(image_id)
        concept = uid_to_index.get(match.group(1)) if match else None
        if concept is None:
            unmapped_participant[participant] += 1
            unmapped_stratum[stratum] += 1
            reason = "unknown_uniqueID" if match else "image_id_pattern_mismatch"
            unmapped_reasons[reason] += 1
        else:
            concepts[participant].add(concept)

    pooled = set().union(*concepts.values())
    report = {
        "rows": int(sum(row_counts.values())),
        "rows_by_participant": dict(row_counts),
        "pooled_dev_concept_count": len(pooled),
        "expected_pooled_dev_concept_count": 1526,
        "matches_expected_pooled_dev_count": len(pooled) == 1526,
        "mapped_concepts_by_participant": {
            participant: len(indices) for participant, indices in concepts.items()
        },
        "unmapped_rows": int(sum(unmapped_participant.values())),
        "unmapped_rows_by_participant": dict(unmapped_participant),
        "unmapped_rows_by_stratum": dict(unmapped_stratum),
        "unmapped_rows_by_reason": dict(unmapped_reasons),
    }
    return pooled, concepts, report


def unique_zip_member(archive, name=None, basename=None):
    """Require exactly one matching ZIP member without extracting it."""
    matches = [
        info for info in archive.infolist()
        if not info.is_dir()
        and (
            info.filename == name if name is not None
            else PurePosixPath(info.filename).name == basename
        )
    ]
    label = name if name is not None else f"basename {basename!r}"
    if len(matches) != 1:
        raise ValueError(
            f"{archive.filename}: expected exactly one {label}, found {len(matches)}"
        )
    return matches[0]


def read_wordnet_index(path):
    """Parse noun lemmas and sense-ordered WordNet offsets."""
    index = {}
    with zipfile.ZipFile(path) as archive:
        member = unique_zip_member(archive, basename="index.noun")
        with archive.open(member) as raw:
            with io.TextIOWrapper(raw, encoding="utf-8") as fh:
                for line_number, line in enumerate(fh, 1):
                    if line.startswith(" ") or not line.strip():
                        continue
                    where = f"{path}:{member.filename}:{line_number}"
                    fields = line.split()
                    if len(fields) < 6 or fields[1] != "n":
                        raise ValueError(f"{where}: malformed noun index entry")
                    try:
                        synset_count = int(fields[2])
                        pointer_count = int(fields[3])
                    except ValueError as exc:
                        raise ValueError(f"{where}: invalid entry counts") from exc
                    if synset_count < 1 or pointer_count < 0:
                        raise ValueError(f"{where}: invalid entry counts")
                    expected = 6 + pointer_count + synset_count
                    if len(fields) != expected:
                        raise ValueError(
                            f"{where}: expected {expected} fields, found {len(fields)}"
                        )
                    try:
                        sense_count = int(fields[4 + pointer_count])
                        tagged_count = int(fields[5 + pointer_count])
                    except ValueError as exc:
                        raise ValueError(f"{where}: invalid sense counts") from exc
                    if sense_count < 0 or not 0 <= tagged_count <= sense_count:
                        raise ValueError(f"{where}: invalid sense counts")

                    offsets = fields[-synset_count:]
                    if any(
                        re.fullmatch(r"[0-9]{1,8}", offset) is None
                        for offset in offsets
                    ):
                        raise ValueError(f"{where}: invalid noun offset")
                    lemma = fields[0].lower()
                    if lemma in index:
                        raise ValueError(f"{where}: duplicate lemma {lemma!r}")
                    index[lemma] = [offset.zfill(8) for offset in offsets]
    if not index:
        raise ValueError(f"{path}: index.noun contains no noun entries")
    return index


def decode_synset(name, index):
    """Decode a noun synset name, returning None when undecodable."""
    if not isinstance(name, str):
        return None
    parts = name.rsplit(".", 2)
    if len(parts) != 3:
        return None
    lemma, pos, sense = parts
    if pos != "n" or re.fullmatch(r"[0-9]+", sense) is None:
        return None
    # Avoid parsing an arbitrarily long integer from an untrusted label.
    sense = sense.lstrip("0")
    offsets = index.get(lemma.lower())
    if not sense or offsets is None or len(sense) > len(str(len(offsets))):
        return None
    position = int(sense) - 1
    if not 0 <= position < len(offsets):
        return None
    return offsets[position]


def column_offset(value, where):
    """Parse the optional n-prefixed THINGS offset column."""
    if not value or value.upper() == "NA":
        return None
    match = re.fullmatch(r"n([0-9]{8})", value)
    if match is None:
        raise ValueError(f"{where}: invalid WordNet ID {value!r}")
    return match.group(1)


def concept_record(index, things):
    """Return an auditable THINGS concept identity."""
    index = int(index)
    return {
        "concept_index": index,
        "uniqueID": things[index]["uniqueID"],
        "Word": things[index]["Word"],
    }


def make_crosswalk(things, noun_index):
    """Prefer decoded ID4 offsets and retain all offset collisions."""
    offsets = []
    offset_to_concepts = defaultdict(list)
    sources = Counter({"id4": 0, "wordnet_id_column": 0, "none": 0})
    records = []
    agreements = 0
    comparisons = 0
    disagreements = []
    nonempty_undecodable_id4 = 0

    for index, concept in enumerate(things):
        id4 = concept["Wordnet ID4"]
        decoded = decode_synset(id4, noun_index)
        column = column_offset(
            concept["WordNet ID"], f"THINGS concept {index} ({concept['uniqueID']})"
        )
        if decoded is not None:
            offset, source = decoded, "id4"
        elif column is not None:
            offset, source = column, "wordnet_id_column"
        else:
            offset, source = None, "none"

        if id4 and id4.upper() != "NA" and decoded is None:
            nonempty_undecodable_id4 += 1
        record = {
            **concept_record(index, things),
            "id4": id4,
            "decoded_id4_offset": decoded,
            "wordnet_id_column_offset": column,
            "offset": offset,
            "offset_source": source,
        }
        records.append(record)
        sources[source] += 1
        offsets.append(offset)
        if offset is not None:
            offset_to_concepts[offset].append(index)
        if decoded is not None and column is not None:
            comparisons += 1
            agreements += int(decoded == column)
            if decoded != column:
                disagreements.append(record.copy())

    shared = {
        offset: indices
        for offset, indices in offset_to_concepts.items()
        if len(indices) > 1
    }
    report = {
        "wordnet_noun_lemma_count": len(noun_index),
        "things_offsets_by_source": dict(sources),
        "nonempty_undecodable_id4_count": nonempty_undecodable_id4,
        "id4_column_agreement": {
            "both_available": comparisons,
            "agree": agreements,
            "disagree": comparisons - agreements,
            "agreement_fraction": agreements / comparisons if comparisons else None,
            "disagreements": disagreements,
        },
        "offsets_mapping_to_multiple_things_concepts": len(shared),
        "things_concepts_on_shared_offsets": sum(map(len, shared.values())),
        "shared_offset_mappings": [
            {
                "offset": offset,
                "concepts": [concept_record(index, things) for index in shared[offset]],
            }
            for offset in sorted(shared)
        ],
        "concept_offsets": records,
    }
    return offsets, dict(offset_to_concepts), report


class IncidenceBuilder:
    """Accumulate binary image incidences in compact coordinate buffers."""

    def __init__(self, n_concepts):
        """Initialize buffers and the context vocabulary."""
        self.n_concepts = n_concepts
        self.n_images = 0
        self.context_columns = {}
        self.concept_rows = array("q")
        self.concept_columns = array("q")
        self.context_rows = array("q")
        self.context_indices = array("q")

    def add(self, concepts, contexts):
        """Add one image with deduplicated concept and context labels."""
        image_row = self.n_images
        for concept in sorted(set(concepts)):
            self.concept_rows.append(image_row)
            self.concept_columns.append(concept)
        for context in sorted(set(contexts)):
            if context not in self.context_columns:
                self.context_columns[context] = len(self.context_columns)
            self.context_rows.append(image_row)
            self.context_indices.append(self.context_columns[context])
        self.n_images += 1

    def finish(self):
        """Return integer CSR incidence matrices and context labels."""
        if self.n_images == 0:
            raise ValueError("Source contains no images")

        def matrix(rows, columns, width):
            """Convert coordinate buffers to an integer CSR matrix."""
            return sparse.csr_matrix(
                (
                    np.ones(len(rows), dtype=np.int64),
                    (
                        np.asarray(rows, dtype=np.int64),
                        np.asarray(columns, dtype=np.int64),
                    ),
                ),
                shape=(self.n_images, width),
                dtype=np.int64,
            )

        return (
            matrix(self.concept_rows, self.concept_columns, self.n_concepts),
            matrix(
                self.context_rows, self.context_indices, len(self.context_columns)
            ),
            self.context_columns,
        )


def image_key(value, where):
    """Accept an integer or nonempty string image identity."""
    if type(value) is int:
        return value
    if isinstance(value, str) and value:
        return value
    raise ValueError(f"{where}: image_id must be an integer or nonempty string")


def vg_object_hook(pairs):
    """Drop unused VG geometry and reject duplicate retained keys."""
    result = {}
    for key, value in pairs:
        if key in VG_DROP_KEYS:
            continue
        if key in result:
            raise ValueError(f"VG JSON object has duplicate key {key!r}")
        result[key] = value
    return result


def read_zip_json(archive, name, hook=None):
    """Load a unique ZIP JSON member without extracting it."""
    member = unique_zip_member(archive, name=name)
    with archive.open(member) as fh:
        try:
            return json.load(fh, object_pairs_hook=hook)
        except ValueError as exc:
            raise ValueError(f"{archive.filename}:{name}: {exc}") from exc


def read_vg(path, noun_index, offset_to_concepts, n_concepts):
    """Build VG incidences from each image's union of synsets."""
    with zipfile.ZipFile(path) as archive:
        payload = read_zip_json(archive, "objects.json", hook=vg_object_hook)
    if not isinstance(payload, list):
        raise ValueError(f"{path}:objects.json: expected a list")

    builder = IncidenceBuilder(n_concepts)
    seen_images = set()
    decoded_names = {}
    for image_number, record in enumerate(payload):
        where = f"{path}:objects.json: image record {image_number}"
        require_fields(record, ("image_id", "objects"), where)
        key = image_key(record["image_id"], where)
        if key in seen_images:
            raise ValueError(f"{where}: duplicate image_id {key!r}")
        seen_images.add(key)
        if not isinstance(record["objects"], list):
            raise ValueError(f"{where}: objects must be a list")
        names = set()
        for object_number, obj in enumerate(record["objects"]):
            object_where = f"{where}: object {object_number}"
            require_fields(obj, ("synsets",), object_where)
            if not isinstance(obj["synsets"], list):
                raise ValueError(f"{object_where}: synsets must be a list")
            for name in obj["synsets"]:
                if not isinstance(name, str):
                    raise ValueError(f"{object_where}: synset names must be strings")
                names.add(name)

        concepts = set()
        contexts = set()
        for name in sorted(names):
            if name not in decoded_names:
                decoded_names[name] = decode_synset(name, noun_index)
            offset = decoded_names[name]
            if offset is None:
                contexts.add("name:" + name)
            else:
                contexts.add(offset)
                concepts.update(offset_to_concepts.get(offset, ()))
        builder.add(concepts, contexts)
    return (*builder.finish(), decoded_names)


def normalize_name(name):
    """Normalize a caption name using the prescribed character set."""
    text = name.lower().replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9 ]", "", text)
    return " ".join(text.split())


def name_variants(name):
    """Generate a base name and the specified plural variants."""
    if not name:
        return set()
    variants = {name, name + "s"}
    if name.endswith(("s", "x", "z", "ch", "sh")):
        variants.add(name + "es")
    last_word = name.rsplit(" ", 1)[-1]
    consonants = "bcdfghjklmnpqrstvwxyz"
    if (
        len(last_word) >= 2
        and last_word.endswith("y")
        and last_word[-2] in consonants
    ):
        variants.add(name[:-1] + "ies")
    if name.endswith("fe"):
        variants.add(name[:-2] + "ves")
    elif name.endswith("f"):
        variants.add(name[:-1] + "ves")
    return variants


def make_caption_lexicon(things, excluded=()):
    """Build an unambiguous lexicon from Word and uniqueID only.

    Audit-excluded concepts (dominant caption sense differs from the THINGS
    sense) contribute no names, so they are neither targets nor contexts.
    """
    candidates = defaultdict(set)
    for index, concept in enumerate(things):
        if concept["uniqueID"] in excluded:
            continue
        for raw_name in (concept["Word"], concept["uniqueID"]):
            for variant in name_variants(normalize_name(raw_name)):
                candidates[variant].add(index)

    ambiguous = {
        name: indices for name, indices in candidates.items() if len(indices) > 1
    }
    lexicon = {
        tuple(name.split()): next(iter(indices))
        for name, indices in candidates.items()
        if len(indices) == 1
    }
    report = {
        "name_fields": ["Word", "uniqueID"],
        "audit_excluded_concepts": sorted(excluded),
        "ambiguous_caption_variants_dropped": len(ambiguous),
        "ambiguous_variants": [
            {
                "variant": name,
                "concepts": [
                    concept_record(index, things)
                    for index in sorted(ambiguous[name])
                ],
            }
            for name in sorted(ambiguous)
        ],
        "unambiguous_variants": len(lexicon),
        "unambiguous_variants_longer_than_four_tokens": sum(
            len(tokens) > 4 for tokens in lexicon
        ),
    }
    return lexicon, report


def match_caption(caption, lexicon):
    """Scan caption tokens with greedy longest-first matching."""
    tokens = re.findall(r"[a-z0-9]+", caption.lower())
    concepts = set()
    position = 0
    while position < len(tokens):
        for length in (4, 3, 2, 1):
            if position + length > len(tokens):
                continue
            concept = lexicon.get(tuple(tokens[position:position + length]))
            if concept is not None:
                concepts.add(concept)
                position += length
                break
        else:
            position += 1
    return concepts


def read_captions(path, lexicon, n_concepts):
    """Union caption matches by image across the two COCO members."""
    images = {}
    members = (
        "annotations/captions_train2017.json",
        "annotations/captions_val2017.json",
    )
    with zipfile.ZipFile(path) as archive:
        for member in members:
            payload = read_zip_json(archive, member)
            require_fields(payload, ("annotations",), f"{path}:{member}")
            if not isinstance(payload["annotations"], list):
                raise ValueError(f"{path}:{member}: annotations must be a list")
            for annotation_number, annotation in enumerate(payload["annotations"]):
                where = f"{path}:{member}: annotation {annotation_number}"
                require_fields(annotation, ("image_id", "caption"), where)
                key = image_key(annotation["image_id"], where)
                caption = annotation["caption"]
                if not isinstance(caption, str):
                    raise ValueError(f"{where}: caption must be a string")
                images.setdefault(key, set()).update(
                    match_caption(caption, lexicon)
                )
            del payload

    builder = IncidenceBuilder(n_concepts)
    for concepts in images.values():
        builder.add(concepts, concepts)
    return builder.finish()


def build_operator(
    source, x, y, context_columns, own_labels,
    pooled, participant_concepts, things, min_images,
):
    """Compute direct counts and uncentered normalized PPMI kernels."""
    n_images = x.shape[0]
    if y.shape[0] != n_images or x.shape[1] != len(things):
        raise ValueError(f"{source}: incidence matrix shape mismatch")
    counts = np.asarray(x.sum(axis=0)).ravel().astype(np.int64, copy=False)
    context_counts = np.asarray(y.sum(axis=0)).ravel().astype(np.int64, copy=False)
    pooled_indices = np.asarray(sorted(pooled), dtype=np.int64)
    candidates = pooled_indices[counts[pooled_indices] >= min_images]

    cooccurrence = (x[:, candidates].T @ y).tocoo()
    own_columns = np.asarray(
        [context_columns.get(own_labels[int(index)], -1) for index in candidates],
        dtype=np.int64,
    )
    keep_entry = cooccurrence.col != own_columns[cooccurrence.row]
    rows = cooccurrence.row[keep_entry]
    columns = cooccurrence.col[keep_entry]
    values = cooccurrence.data[keep_entry].astype(np.float64)
    if values.size:
        denominator = (
            counts[candidates[rows]].astype(np.float64)
            * context_counts[columns].astype(np.float64)
        )
        values = np.maximum(0.0, np.log(values * float(n_images) / denominator))
    positive = values > 0.0
    ppmi = sparse.csr_matrix(
        (values[positive], (rows[positive], columns[positive])),
        shape=(len(candidates), y.shape[1]),
        dtype=np.float64,
    )
    if not np.all(np.isfinite(ppmi.data)):
        raise ValueError(f"{source}: nonfinite PPMI values")
    norms = np.sqrt(np.asarray(ppmi.multiply(ppmi).sum(axis=1)).ravel())
    if not np.all(np.isfinite(norms)):
        raise ValueError(f"{source}: nonfinite PPMI row norms")
    keep_row = norms > 0.0
    concepts = candidates[keep_row].astype(np.int64, copy=False)
    dropped = candidates[~keep_row]

    if concepts.size:
        normalized = ppmi[keep_row].multiply(
            (1.0 / norms[keep_row])[:, None]
        ).tocsr()
        kernel = np.asarray(
            (normalized @ normalized.T).toarray(), dtype=np.float64
        )
        selected_incidence = x[:, concepts]
        direct = np.asarray(
            (selected_incidence.T @ selected_incidence).toarray(),
            dtype=np.float64,
        )
        np.fill_diagonal(direct, 0.0)
    else:
        kernel = np.empty((0, 0), dtype=np.float64)
        direct = np.empty((0, 0), dtype=np.float64)

    def audit(indices):
        """Describe concepts using source-wide image frequencies."""
        return [
            {**concept_record(index, things), "n_c": int(counts[index])}
            for index in indices
        ]

    participant_indices = {
        participant: np.asarray(sorted(indices), dtype=np.int64)
        for participant, indices in participant_concepts.items()
    }
    coverage = {
        str(threshold): {
            "all_concepts": int(np.count_nonzero(counts >= threshold)),
            "pooled_dev": int(
                np.count_nonzero(counts[pooled_indices] >= threshold)
            ),
            "participants": {
                participant: int(np.count_nonzero(counts[indices] >= threshold))
                for participant, indices in participant_indices.items()
            },
        }
        for threshold in THRESHOLDS
    }
    final_indices = concepts.tolist()
    report = {
        "N": int(n_images),
        "images_with_at_least_one_things_concept": int(
            np.count_nonzero(x.getnnz(axis=1))
        ),
        "vocabulary_size": int(y.shape[1]),
        "S_size_before_zero_row_drop": len(candidates),
        "S_size": len(concepts),
        "dropped_zero_row_count": len(dropped),
        "dropped_zero_rows": audit(dropped.tolist()),
        "coverage_counts_before_zero_row_drop": coverage,
        "audit_most_frequent_40": audit(
            sorted(final_indices, key=lambda index: (-int(counts[index]), index))[:40]
        ),
        "audit_least_frequent_40": audit(
            sorted(final_indices, key=lambda index: (int(counts[index]), index))[:40]
        ),
    }
    arrays = {
        f"{source}_concepts": concepts,
        f"{source}_counts": counts[concepts].astype(np.int64, copy=False),
        f"{source}_kernel": kernel,
        f"{source}_direct": direct,
    }
    return arrays, report


def main():
    """Read source data, construct both operators, and write artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "things-meta", "dev-rows", "wordnet-zip",
        "vg-objects", "coco-annotations", "out-dir",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--min-images", type=int, default=10)
    parser.add_argument("--exclude-caption-concepts", default="",
                        help="comma-separated THINGS uniqueIDs removed from the caption lexicon")
    args = parser.parse_args()
    if args.min_images < 1:
        parser.error("--min-images must be at least 1")

    started = time.perf_counter()
    paths = {
        name: getattr(args, name).expanduser().resolve()
        for name in (
            "things_meta", "dev_rows", "wordnet_zip",
            "vg_objects", "coco_annotations", "out_dir",
        )
    }
    things = read_things(paths["things_meta"])
    uid_to_index = {row["uniqueID"]: index for index, row in enumerate(things)}
    pooled, participant_concepts, development_report = read_development(
        paths["dev_rows"], uid_to_index
    )
    noun_index = read_wordnet_index(paths["wordnet_zip"])
    offsets, offset_to_concepts, crosswalk_report = make_crosswalk(
        things, noun_index
    )
    output_arrays = {
        "things_uids": np.asarray(
            [row["uniqueID"] for row in things], dtype=np.str_
        )
    }
    source_reports = {}

    x, y, contexts, vg_names = read_vg(
        paths["vg_objects"], noun_index, offset_to_concepts, len(things)
    )
    decoded_vg_offsets = {
        offset for offset in vg_names.values() if offset is not None
    }
    reached = {
        index
        for offset in decoded_vg_offsets
        for index in offset_to_concepts.get(offset, ())
    }
    exact_name_reached = {
        index for index, row in enumerate(things)
        if row["Wordnet ID4"] in vg_names
    }
    newly_reached = sorted(reached - exact_name_reached)
    crosswalk_report.update({
        "distinct_vg_synsets": len(vg_names),
        "decoded_vg_synsets": sum(
            offset is not None for offset in vg_names.values()
        ),
        "distinct_decoded_vg_offsets": len(decoded_vg_offsets),
        "things_concepts_reachable_via_vg_offsets": len(reached),
        "things_concepts_reachable_via_exact_synset_name": len(exact_name_reached),
        "things_concepts_newly_reached_count": len(newly_reached),
        "things_concepts_newly_reached_by_offset": [
            {
                **concept_record(index, things),
                "id4": things[index]["Wordnet ID4"],
                "offset": offsets[index],
            }
            for index in newly_reached
        ],
    })
    arrays, source_reports["vg"] = build_operator(
        "vg", x, y, contexts, offsets, pooled, participant_concepts,
        things, args.min_images,
    )
    output_arrays.update(arrays)
    del x, y, contexts, vg_names, arrays

    excluded = {name for name in args.exclude_caption_concepts.split(",") if name}
    unknown = excluded - set(uid_to_index)
    if unknown:
        raise ValueError(f"Unknown caption exclusions: {sorted(unknown)}")
    lexicon, caption_report = make_caption_lexicon(things, excluded)
    x, y, contexts = read_captions(
        paths["coco_annotations"], lexicon, len(things)
    )
    arrays, source_reports["captions"] = build_operator(
        "captions", x, y, contexts, list(range(len(things))),
        pooled, participant_concepts, things, args.min_images,
    )
    output_arrays.update(arrays)
    del x, y, contexts, arrays

    paths["out_dir"].mkdir(parents=True, exist_ok=True)
    npz_path = paths["out_dir"] / "operators.npz"
    report_path = paths["out_dir"] / "operators_report.json"
    np.savez(npz_path, **output_arrays)
    report = {
        "settings": {
            "min_images": args.min_images,
            "coverage_thresholds": list(THRESHOLDS),
            "caption_match_lengths": [4, 3, 2, 1],
            "caption_plural_variants": (
                "base and +s, plus applicable +es, consonant-y/ies, f/ves, fe/ves"
            ),
            "caption_N_definition": "distinct image_ids in caption annotations",
            "operator": "uncentered kernel of L2-normalized PPMI rows",
            "arithmetic_dtype": "float64",
        },
        "input_paths": {
            name: str(path) for name, path in paths.items() if name != "out_dir"
        },
        "output_directory": str(paths["out_dir"]),
        "development": development_report,
        "crosswalk": crosswalk_report,
        "caption_matching": caption_report,
        "ambiguous_caption_variants_dropped": (
            caption_report["ambiguous_caption_variants_dropped"]
        ),
        "sources": source_reports,
        "elapsed_seconds_before_report_write": time.perf_counter() - started,
    }
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False, allow_nan=False)
        fh.write("\n")

    print(
        f"Pooled dev: {len(pooled)} concepts (expected 1526); "
        f"unmapped THINGS/THINGSplus rows: {development_report['unmapped_rows']}"
    )
    agreement = crosswalk_report["id4_column_agreement"]
    print(
        f"ID4/column agreement: {agreement['agree']}/{agreement['both_available']}; "
        f"ambiguous caption variants dropped: "
        f"{caption_report['ambiguous_caption_variants_dropped']}"
    )
    for source, source_report in source_reports.items():
        print(
            f"{source}: N={source_report['N']}, S={source_report['S_size']}, "
            f"zero rows dropped={source_report['dropped_zero_row_count']}"
        )
    print(f"Wrote {npz_path} and {report_path}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, TypeError, zipfile.BadZipFile) as exc:
        raise SystemExit(f"a0_operators.py: {exc}") from exc
