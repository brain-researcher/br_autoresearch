#!/usr/bin/env python3
"""Assess THINGS co-occurrence operators without model or neural data."""

import argparse
import csv
import itertools
import json
import re
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import sparse, stats
from scipy.sparse.csgraph import connected_components

DROP_KEYS = {"segmentation", "bbox", "area", "coco_url", "flickr_url", "license"}
KINDS = ("direct", "shared_context")
DEV_PATTERN = re.compile(r"^unique_THINGS\+?_(.+)_p\d+\.jpg$")


def compact_object(pairs):
    return {key: value for key, value in pairs if key not in DROP_KEYS}


def load_member(path, member):
    """Read JSON directly from its required archive member."""
    with zipfile.ZipFile(path) as archive:
        if member not in archive.namelist():
            raise ValueError(f"{path}: missing required member {member!r}")
        with archive.open(member) as fh:
            return json.load(fh, object_pairs_hook=compact_object)


def table(path, required, delimiter=",", errors="strict"):
    with open(path, encoding="utf-8", errors=errors, newline="") as fh:
        reader = csv.DictReader(fh, delimiter=delimiter)
        missing = set(required) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing required columns {sorted(missing)}")
        for line, row in enumerate(reader, 2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"{path}:{line}: malformed table row")
            yield row


def normalize_name(value):
    return " ".join(re.sub(r"[^a-z0-9 ]", "", value.lower().replace("_", " ")).split())


def name_variants(name):
    variants = {name, name + "s"}
    if name.endswith(("s", "x", "z", "ch", "sh")):
        variants.add(name + "es")
    if re.search(r"[bcdfghjklmnpqrstvwxyz]y$", name):
        variants.add(name[:-1] + "ies")
    if name.endswith("f"):
        variants.add(name[:-1] + "ves")
    if name.endswith("fe"):
        variants.add(name[:-2] + "ves")
    return variants


def load_things(meta_path, category_path):
    """Preserve metadata order and remove ambiguous name variants."""
    columns = ("uniqueID", "Word", "Wordnet ID4", "WordNet Synonyms")
    metadata = list(table(meta_path, columns, "\t", errors="replace"))
    uids = [row["uniqueID"].strip() for row in metadata]
    if not uids or len(set(uids)) != len(uids) or any(not uid for uid in uids):
        raise ValueError(f"{meta_path}: uniqueID must be nonempty and unique")
    with open(category_path, encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh, delimiter="\t")
        categories = next(reader, [])
        values = [[value.strip() for value in row] for row in reader]
    if len(categories) != 27 or len(values) != len(uids):
        raise ValueError(f"{category_path}: expected 27 columns and {len(uids)} data rows")
    if any(len(row) != 27 or any(v not in {"0", "1"} for v in row) for row in values):
        raise ValueError(f"{category_path}: expected rectangular binary category labels")
    taxonomy = np.asarray(values, dtype=np.int8)
    synsets, variants = defaultdict(set), defaultdict(set)
    for index, row in enumerate(metadata):
        synset = row["Wordnet ID4"].strip()
        if synset and synset != "NA":
            synsets[synset].add(index)
        for value in [row["Word"], row["uniqueID"], *row["WordNet Synonyms"].split(",")]:
            name = normalize_name(value)
            if name:
                for variant in name_variants(name):
                    variants[variant].add(index)
    ambiguous = sum(len(indices) > 1 for indices in variants.values())
    variants = {name: next(iter(indices)) for name, indices in variants.items() if len(indices) == 1}
    return uids, taxonomy, categories, synsets, variants, ambiguous


def load_dev(path, uids):
    lookup = {uid: index for index, uid in enumerate(uids)}
    participants, unmatched, selected = defaultdict(set), Counter(), 0
    with open(path, encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, 1):
            try:
                row = json.loads(line, object_pairs_hook=compact_object)
                participant = row["participant"]
                participants[participant]
                if row["source_stratum"] not in {"things", "thingsplus"}:
                    continue
                selected += 1
                match = DEV_PATTERN.match(row["image_id"])
                name = match.group(1) if match else f"<invalid-image-id>:{row['image_id']}"
                if name in lookup:
                    participants[participant].add(lookup[name])
                else:
                    unmatched[name] += 1
            except (KeyError, ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    participants = dict(sorted(participants.items()))
    pool = set().union(*participants.values())
    report = {
        "selected_rows": selected, "matched_rows": selected - sum(unmatched.values()),
        "unmatched_rows": sum(unmatched.values()), "unmatched_name_counts": dict(unmatched),
        "pooled_size": len(pool), "pooled_concept_indices": sorted(pool),
        "participants": {p: {"n_concepts": len(s), "concept_indices": sorted(s)}
                         for p, s in participants.items()},
    }
    return participants, pool, report


def match_caption(caption, variants):
    tokens, found, position = re.findall(r"[a-z0-9]+", caption.lower()), set(), 0
    while position < len(tokens):
        for width in (4, 3, 2, 1):
            if position + width > len(tokens):
                continue
            index = variants.get(" ".join(tokens[position:position + width]))
            if index is not None:
                found.add(index)
                position += width
                break
        else:
            position += 1
    return found


def load_source(name, args, synsets, variants, n_concepts):
    """Return context sets, their concept mapping, and annotation diagnostics."""
    limit = args.limit_images
    if name == "visual_genome":
        records = load_member(args.vg_objects, "objects.json")
        records = records[:limit] if limit else records
        contexts = [{s for obj in row["objects"] for s in obj["synsets"]} for row in records]
        return contexts, synsets, {}
    if name == "lvis":
        contexts, category_synsets, seen = [], {}, set()
        positive_pairs = nonexhaustive_pairs = negative_pairs = 0
        for path, member in ((args.lvis_train, "lvis_v1_train.json"),
                             (args.lvis_val, "lvis_v1_val.json")):
            data = load_member(path, member)
            for category in data["categories"]:
                key, synset = category["id"], category["synset"]
                if key in category_synsets and category_synsets[key] != synset:
                    raise ValueError(f"LVIS: conflicting synsets for category {key}")
                category_synsets[key] = synset
            images = data["images"]
            images = images[:max(0, limit - len(contexts))] if limit else images
            positives = {row["id"]: set() for row in images}
            if len(positives) != len(images) or seen.intersection(positives):
                raise ValueError("LVIS: duplicate selected image IDs")
            seen.update(positives)
            for annotation in data["annotations"]:
                image_id, category_id = annotation["image_id"], annotation["category_id"]
                if image_id in positives:
                    if category_id not in category_synsets:
                        raise ValueError(f"LVIS: undefined category {category_id}")
                    positives[image_id].add(category_id)
            for row in images:
                labels = positives[row["id"]]
                contexts.append(labels)
                positive_pairs += len(labels)
                nonexhaustive_pairs += len(labels.intersection(row["not_exhaustive_category_ids"]))
                negative_pairs += len(set(row["neg_category_ids"]))
            del data
        mapping = {key: synsets.get(synset, ()) for key, synset in category_synsets.items()}
        return contexts, mapping, {
            "declared_category_count": len(category_synsets),
            "positive_image_category_pairs": positive_pairs,
            "nonexhaustive_positive_pair_fraction":
                nonexhaustive_pairs / positive_pairs if positive_pairs else None,
            "mean_verified_negative_categories_per_image":
                negative_pairs / len(contexts) if contexts else None,
        }
    images = {}
    for member in ("annotations/captions_train2017.json", "annotations/captions_val2017.json"):
        data = load_member(args.coco_annotations, member)
        for row in data.get("images", []):
            if not limit or len(images) < limit:
                images.setdefault(row["id"], set())
        for annotation in data["annotations"]:
            image_id = annotation["image_id"]
            if image_id not in images:
                if limit and len(images) >= limit:
                    continue
                images[image_id] = set()
            images[image_id].update(match_caption(annotation["caption"], variants))
        del data
    return list(images.values()), {i: (i,) for i in range(n_concepts)}, {}


def incidence(rows, vocabulary):
    indices = np.fromiter((vocabulary[label] for row in rows for label in row), dtype=np.int32)
    indptr = np.r_[0, np.cumsum([len(row) for row in rows], dtype=np.int64)]
    matrix = sparse.csr_matrix((np.ones(len(indices)), indices, indptr),
                               shape=(len(rows), len(vocabulary)))
    matrix.sort_indices()
    return matrix


def group_counts(mask, pool, participants):
    return {"all": int(mask.sum()), "pooled_dev": int(mask[list(pool)].sum()),
            "participants": {p: int(mask[list(indices)].sum()) for p, indices in participants.items()}}


def prepare_source(contexts, mapping, n, pool, participants, taxonomy, minimum):
    """Retain sparse incidence products and globally estimated PPMI profiles."""
    concepts = [{i for label in row for i in mapping.get(label, ())} for row in contexts]
    x = incidence(concepts, {i: i for i in range(n)})
    counts = np.asarray(x.sum(axis=0)).ravel().astype(np.int64)
    report = {"n_images": len(contexts), "n_images_with_things": sum(bool(row) for row in concepts),
              "concept_image_counts": counts,
              "coverage": {str(t): group_counts(counts >= t, pool, participants) for t in (1, 5, 10, 50)}}
    for label, mask in (("any", taxonomy.sum(axis=1) >= 1), ("exactly_one", taxonomy.sum(axis=1) == 1)):
        report[f"min_images_and_{label}_category27"] = group_counts(mask & (counts >= minimum), pool, participants)
    candidates = np.array(sorted(i for i in pool if counts[i] >= minimum), dtype=int)
    x = x[:, candidates]
    a = (x.T @ x).tocsr()
    a.setdiag(0)
    a.eliminate_zeros()
    keep = np.asarray(a.sum(axis=1)).ravel() > 0
    ids, x, a = candidates[keep], x[:, keep], a[keep][:, keep]
    vocabulary = {label: i for i, label in enumerate(sorted({label for row in contexts for label in row}))}
    y = incidence(contexts, vocabulary)
    frequencies = np.asarray(y.sum(axis=0)).ravel()
    c = (x.T @ y).tocsr()
    own = defaultdict(set)
    for label, mapped in mapping.items():
        if label in vocabulary:
            for concept in mapped:
                own[concept].add(vocabulary[label])
    for row, concept in enumerate(ids):
        start, end = c.indptr[row:row + 2]
        c.data[start:end][np.isin(c.indices[start:end], list(own[concept]))] = 0
    c.eliminate_zeros()
    if c.nnz:
        rows = np.repeat(np.arange(len(ids)), np.diff(c.indptr))
        c.data = np.maximum(np.log(c.data * len(contexts) / (counts[ids][rows] * frequencies[c.indices])), 0)
        c.eliminate_zeros()
    norms = np.sqrt(np.asarray(c.multiply(c).sum(axis=1)).ravel())
    c.data /= np.repeat(norms, np.diff(c.indptr))
    lookup = np.full(n, -1, dtype=int)
    lookup[ids] = np.arange(len(ids))
    report.update(vocabulary_size=len(vocabulary), min_images_dev_size=len(candidates),
                  zero_degree_dropped=len(candidates) - len(ids),
                  zero_degree_dropped_concept_indices=candidates[~keep], S_size=len(ids),
                  S_concept_indices=ids, zero_ppmi_rows_dropped=int(np.sum(norms == 0)),
                  zero_ppmi_concept_indices=ids[norms == 0])
    return {"ids": ids, "lookup": lookup, "a": a, "r": c, "context_valid": norms > 0}, report


def spectrum(source, ids, kind):
    """Recompute degrees or centering after restricting the concept rows."""
    positions = source["lookup"][ids]
    a = source["a"][positions][:, positions].toarray()
    degree = a.sum(axis=1)
    if not len(ids):
        return a, degree, np.empty(0), np.empty((0, 0))
    if kind == "direct":
        inverse = np.divide(1.0, np.sqrt(degree), out=np.zeros_like(degree), where=degree > 0)
        matrix = a * inverse[:, None] * inverse[None, :]
    else:
        r = source["r"][positions]
        matrix = (r @ r.T).toarray()
        matrix -= matrix.mean(axis=0)[None, :] + matrix.mean(axis=1)[:, None] - matrix.mean()
    eigenvalues, vectors = np.linalg.eigh((matrix + matrix.T) / 2)
    eigenvalues, vectors = eigenvalues[::-1], vectors[:, ::-1]
    if kind == "direct" and degree.sum() > 0:
        # Fix the first Perron vector even when disconnected components make it nonunique.
        k = int(np.count_nonzero(np.isclose(eigenvalues, 1.0, atol=1e-10, rtol=0)))
        if k:
            reflection = vectors[:, :k].T @ np.sqrt(degree / degree.sum())
            reflection[0] -= 1
            norm = reflection @ reflection
            if norm > 1e-24:
                vectors[:, :k] -= 2 * np.outer(vectors[:, :k] @ reflection, reflection) / norm
    return a, degree, eigenvalues, vectors


def diagnostics(fit, kind, top=21):
    a, degree, eigenvalues, vectors = fit
    n, skip = len(degree), int(kind == "direct")
    components, labels = connected_components(sparse.csr_matrix(a > 0), directed=False)
    result = {
        "n": n, "zero_degree_concepts": int(np.sum(degree == 0)),
        "off_diagonal_density": np.count_nonzero(a > 0) / (n * (n - 1)) if n > 1 else 0.0,
        "connected_components": int(components),
        "largest_component": int(np.bincount(labels).max()) if n else 0,
        "top_eigenvalues": eigenvalues[:top], "consecutive_gaps": -np.diff(eigenvalues[:top]),
        "ipr": np.sum(vectors[:, skip:skip + 8] ** 4, axis=0),
        "ipr_reference": 1 / n if n else None, "abs_leading_mode_log_degree_spearman": None,
        "spearman_positive_degree_n": int(np.sum(degree > 0)),
    }
    if n > skip and np.sum(degree > 0) >= 2:
        positive = degree > 0
        amplitude, log_degree = np.abs(vectors[positive, skip]), np.log(degree[positive])
        if np.ptp(amplitude) > 0 and np.ptp(log_degree) > 0:
            result["abs_leading_mode_log_degree_spearman"] = stats.spearmanr(amplitude, log_degree).statistic
    return result


def taxonomy_alignment(source, taxonomy, kind, n_perm, rng):
    ids = source["ids"][taxonomy[source["ids"]].sum(axis=1) == 1]
    before = len(ids)
    if kind == "shared_context":
        ids = ids[source["context_valid"][source["lookup"][ids]]]
    zero_context = before - len(ids)
    labels = taxonomy[ids].argmax(axis=1)
    ids = ids[np.bincount(labels, minlength=27)[labels] >= 5]
    category_ids, labels = np.unique(taxonomy[ids].argmax(axis=1), return_inverse=True)
    n, c = len(ids), len(category_ids)
    q = max(0, min(8, c - 1))
    result = dict(n=n, C=c, q=q, concept_indices=ids, category_indices=category_ids,
                  zero_ppmi_rows_dropped=zero_context,
                  chance_level=(c - int(kind == "shared_context")) / n if n else None,
                  omega=None, null_mean=None, null_95th_percentile=None, p_value=None)
    if not q:
        return dict(result, reason="Fewer than two categories with at least five usable concepts")
    _, degree, _, vectors = spectrum(source, ids, kind)
    result["zero_degree_concepts"] = int(np.sum(degree == 0))
    if kind == "direct" and degree.sum() == 0:
        return dict(result, reason="Taxonomy subset has no co-occurrence edges")
    u = vectors[:, 1:1 + q] if kind == "direct" else vectors[:, :q]
    v = np.sqrt(degree / degree.sum()) if kind == "direct" else np.full(n, 1 / np.sqrt(n))

    def overlap(permuted_labels):
        h = np.eye(c)[permuted_labels]
        h -= np.outer(v, v @ h)
        # Keep the full projected taxonomy span: rank C-1 after centering, usually C after degree projection.
        left, singular, _ = np.linalg.svd(h, full_matrices=False)
        basis = left[:, singular > 1e-8 * singular[0]]
        return float(np.sum((u.T @ basis) ** 2) / q)

    observed = overlap(labels)
    null = np.array([overlap(rng.permutation(labels)) for _ in range(n_perm)])
    result.update(omega=observed, null_mean=null.mean(), null_95th_percentile=np.quantile(null, .95),
                  p_value=(1 + np.count_nonzero(null >= observed)) / (1 + n_perm))
    return result


def cross_source_agreement(sources, rng):
    common = np.array(sorted(set.intersection(*(set(s["ids"]) for s in sources.values()))), dtype=int)
    report = {"S_common_size": len(common), "S_common_concept_indices": common}
    for kind in KINDS:
        keep = np.ones(len(common), dtype=bool)
        if kind == "shared_context":
            for source in sources.values():
                keep &= source["context_valid"][source["lookup"][common]]
        ids = common[keep]
        fits = {name: spectrum(source, ids, kind) for name, source in sources.items()}
        entry = {"n": len(ids), "concept_indices": ids,
                 "zero_ppmi_excluded_concept_indices": common[~keep],
                 "top_eigenvalues": {name: fit[2][:11] for name, fit in fits.items()}, "pairs": {}}
        for left, right in itertools.combinations(sources, 2):
            result = dict(overlap=None, null_mean=None, null_95th_percentile=None, n_perm=200, q=8)
            if len(ids) < 9:
                result["reason"] = "Fewer than nine common usable concepts"
            elif kind == "direct" and any(fits[name][1].sum() == 0 for name in (left, right)):
                result["reason"] = "A restricted direct operator has no edges"
            else:
                start = int(kind == "direct")
                u = fits[left][3][:, start:start + 8]
                v = fits[right][3][:, start:start + 8]
                observed = np.sum((u.T @ v) ** 2) / 8
                # Permuting eigenvector rows equals conjugating the operator by that permutation.
                null = np.array([np.sum((u.T @ v[rng.permutation(len(ids))]) ** 2) / 8
                                 for _ in range(200)])
                result.update(overlap=observed, null_mean=null.mean(),
                              null_95th_percentile=np.quantile(null, .95))
            entry["pairs"][f"{left}__{right}"] = result
        report[kind] = entry
    return report


def clean(value):
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    return None if isinstance(value, float) and not np.isfinite(value) else value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("things-meta", "things-c27", "dev-rows", "vg-objects", "lvis-train",
                 "lvis-val", "coco-annotations", "openimages-classes", "out"):
        parser.add_argument("--" + flag, required=True)
    for flag, default in (("min-images", 10), ("n-perm", 1000), ("seed", 0), ("limit-images", 0)):
        parser.add_argument("--" + flag, type=int, default=default)
    args = parser.parse_args()
    if args.min_images < 1 or args.n_perm < 1 or args.limit_images < 0 or args.seed < 0:
        parser.error("min-images and n-perm must be positive; limit-images and seed must be nonnegative")
    started = time.perf_counter()
    rng = np.random.default_rng(args.seed)
    uids, taxonomy, categories, synsets, variants, ambiguous = load_things(args.things_meta, args.things_c27)
    participants, pool, dev = load_dev(args.dev_rows, uids)
    timings = {"metadata_and_dev": time.perf_counter() - started}
    output = {
        "settings": dict(vars(args), cross_source_n_perm=200,
                         zero_degree_restricted_direct_policy="Retain isolated rows with inverse degree zero",
                         shared_context_subset_policy="Drop zero PPMI rows; cross-source comparisons use their common intersection",
                         taxonomy_policy="Filter usable single-label concepts to categories with >=5; basis is the full projected category span (rank C-1 centered, C degree-projected)",
                         frequency_policy="PPMI uses source-wide image counts, including images without THINGS matches",
                         limit_order="Archive order, train before val; COCO image table when present, otherwise annotation order",
                         name_normalization="Delete non-[a-z0-9 ] characters after replacing underscores with spaces"),
        "things": {"n_concepts": len(uids), "unique_ids": uids, "category27_names": categories,
                   "ambiguous_synsets": sum(len(indices) > 1 for indices in synsets.values()),
                   "ambiguous_name_variants_dropped": ambiguous, "retained_name_variants": len(variants)},
        "dev_concepts": dev, "sources": {},
    }
    sources = {}
    for name in ("visual_genome", "lvis", "coco_captions"):
        tick = time.perf_counter()
        contexts, mapping, extra = load_source(name, args, synsets, variants, len(uids))
        source, report = prepare_source(contexts, mapping, len(uids), pool, participants, taxonomy, args.min_images)
        del contexts, mapping
        report.update(extra)
        report["operators"] = {}
        for kind in KINDS:
            ids = source["ids"]
            if kind == "shared_context":
                ids = ids[source["context_valid"]]
            report["operators"][kind] = {
                "concept_indices": ids, "spectral": diagnostics(spectrum(source, ids, kind), kind),
                "taxonomy": taxonomy_alignment(source, taxonomy, kind, args.n_perm, rng),
            }
        report["participants"] = {}
        for participant, concepts in participants.items():
            ids = np.array(sorted(set(source["ids"]).intersection(concepts)), dtype=int)
            diagnostic = diagnostics(spectrum(source, ids, "direct"), "direct", top=11)
            report["participants"][participant] = {
                "S_p_size": len(ids), "concept_indices": ids,
                **{key: diagnostic[key] for key in
                   ("zero_degree_concepts", "connected_components", "largest_component", "top_eigenvalues")},
            }
        sources[name], output["sources"][name] = source, report
        timings[name] = time.perf_counter() - tick
    tick = time.perf_counter()
    output["cross_source"] = cross_source_agreement(sources, rng)
    timings["cross_source"] = time.perf_counter() - tick
    tick = time.perf_counter()
    covered = np.zeros(len(uids), dtype=bool)
    class_rows = matched_rows = 0
    for row in table(args.openimages_classes, ("LabelName", "DisplayName")):
        class_rows += 1
        index = variants.get(normalize_name(row["DisplayName"]))
        if index is not None:
            covered[index] = True
            matched_rows += 1
    output["open_images"] = dict(group_counts(covered, pool, participants),
                                 class_rows=class_rows, matched_class_rows=matched_rows)
    timings.update(open_images=time.perf_counter() - tick, total=time.perf_counter() - started)
    output["timing_seconds"] = timings
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as fh:
        json.dump(clean(output), fh, indent=2, allow_nan=False)
        fh.write("\n")
    print("Coverage thresholds: 1/5/10/50 images; counts shown as all concepts / pooled dev.")
    for name, report in output["sources"].items():
        coverage = report["coverage"]
        counts = ", ".join(f"{coverage[str(t)]['all']}/{coverage[str(t)]['pooled_dev']}" for t in (1, 5, 10, 50))
        print(f"{name}: images={report['n_images']:,}; coverage={counts}; |S|={report['S_size']}")
        for kind, operator in report["operators"].items():
            tax = operator["taxonomy"]
            alignment = (f"omega={tax['omega']:.4f}, null mean={tax['null_mean']:.4f}, "
                         f"null95={tax['null_95th_percentile']:.4f}, p={tax['p_value']:.4g}"
                         if tax["omega"] is not None else tax["reason"])
            gaps = np.round(operator["spectral"]["consecutive_gaps"][:3], 4).tolist()
            print(f"  {kind}: leading gaps={gaps}; taxonomy: {alignment}")
    for kind in KINDS:
        for pair, result in output["cross_source"][kind]["pairs"].items():
            value = (f"{result['overlap']:.4f} (null mean={result['null_mean']:.4f}, "
                     f"null95={result['null_95th_percentile']:.4f})"
                     if result["overlap"] is not None else result["reason"])
            print(f"Cross-source {kind} {pair}: {value}")
    print(f"JSON written to {destination}")


if __name__ == "__main__":
    try:
        main()
    except KeyError as exc:
        raise SystemExit(f"operator_feasibility.py: missing required JSON field {exc}") from exc
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise SystemExit(f"operator_feasibility.py: {exc}") from exc
