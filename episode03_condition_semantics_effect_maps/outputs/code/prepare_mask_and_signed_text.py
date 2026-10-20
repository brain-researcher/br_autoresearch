#!/usr/bin/env python3
"""Prepare an anatomical reference mask and signed methods text; never read target arrays.

The command requires compute and exact metadata inputs. It preserves all methods001
rows/exclusions, writes a new attempt, and cannot release baseline access or scoring.
"""

import argparse
from collections import Counter
from decimal import Decimal
import gzip
import json
import math
import os
from pathlib import Path
import re
import socket
import struct

import numpy as np

from build_route_manifest import duration_regressors, route_roles
from prepare_grouped_methods import map_key, read_tsv


# Exact reviewed phrases, not substring stripping of unreviewed source prose.
QUANTITIES = {
    "pump ordinal within balloon, centred (job 48169324)": "centered within-balloon pump ordinal",
    "control pump ordinal, centred (job 48169324)": "centered control pump ordinal",
    "pumps on the cashed balloon minus run mean": "pumps on the cashed balloon minus run mean",
    "pumps on the exploded balloon minus run mean": "pumps on the exploded balloon minus run mean",
    "potential gain amount (centred, original column)": "centered potential gain amount",
    "potential loss magnitude (centred, original column)": "centered potential loss magnitude",
    "distance of the gamble from the participant's indifference point": "distance of the gamble from the participant's indifference point",
    "pump ordinal within trial minus trial mean (producer code first committed 2025-08-07; columns present in the fitted designs)": "pump ordinal within trial minus trial mean",
    "terminal pumps + 1 minus the cash-out mean (producer code)": "terminal pumps plus one minus the cash-out mean",
    "terminal pumps + 1 minus the explosion mean (producer code)": "terminal pumps plus one minus the explosion mean",
    "delay in days, centred (producer code)": "centered delay in days",
    "delayed/immediate amount ratio, centred (producer code)": "centered delayed-to-immediate amount ratio",
    "immediate amount, centred (producer code)": "centered immediate amount",
    "absolute colour coherence (producer abs of the signed column)": "absolute colour coherence",
    "absolute motion coherence": "absolute motion coherence",
    "colour coherence signed toward the rightward response (lateral pole declared by the column name)": "colour coherence signed toward the rightward response",
    "motion coherence signed toward the rightward response": "motion coherence signed toward the rightward response",
    "potential gain, Demean transformation in the executed model": "centered potential gain",
    "potential loss magnitude, Demean transformation in the executed model": "centered potential loss magnitude",
    "stimulus coherence (original column)": "stimulus coherence",
    "liking rating, Scale(demean) in the executed model": "centered liking rating",
    "normative valence (1-9) centred within extrinsic images": "normative valence on the one-to-nine scale centered within extrinsic images",
    "normative valence centred within extrinsic images": "normative valence centered within extrinsic images",
    "normative valence centred within cued-recall images": "normative valence centered within cued-recall images",
    "normative arousal centred within extrinsic images": "normative arousal centered within extrinsic images",
    "normative arousal centred within cued-recall images": "normative arousal centered within cued-recall images",
    "distance of the feedback marker from the origin (producer sqrt(v^2+a^2))": "Euclidean distance of the feedback marker from the origin",
    "rating value (original column, n/a set to 0)": "rating value with unavailable ratings set to zero",
}


def exact_definition(row, model, source_definitions=None):
    """Match the frozen ordered definition; never recover precision from its text."""
    found = {}
    candidates = ([item for node in (model or {}).get("nodes", [])
                   for item in node.get("contrasts", [])]
                  if source_definitions is None else source_definitions)
    for item in candidates:
        # Exact-map definitions were selected with census_route_provenance.norm_name
        # (strip nonalphanumerics, lowercase). Do not widen matching across maps.
        normalize = lambda name: re.sub(r"[^a-z0-9]", "", str(name).lower())
        names_match = (normalize(item.get("name")) == normalize(row["contrast"])
                       if source_definitions is not None else item.get("name") == row["contrast"])
        if not names_match:
            continue
        conditions, weights = item.get("condition_list", []), item.get("weights", [])
        if len(conditions) != len(weights) or not conditions:
            continue
        if any(isinstance(w, bool) or not isinstance(w, (int, float, Decimal))
               or not math.isfinite(float(w)) for w in weights):
            continue
        rounded = " ".join(f"{float(w):+g}*{c}" for c, w in zip(conditions, weights))
        if rounded == row["definition"]:
            key = (tuple(conditions), tuple(Decimal(str(w)) for w in weights))
            found[key] = (conditions, weights)
    if len(found) != 1:
        raise ValueError("exact_source_definition_missing_or_ambiguous")
    return next(iter(found.values()))


def describe_condition(condition, factors, roles, durations):
    """Only structured adopted roles enter model text; raw names are join keys."""
    role = roles.get(condition, {})
    parts = []
    for name in sorted(k for k in role if not k.startswith("_")):
        factor, level = factors.get(name), role[name]
        if not factor or level not in factor.get("levels", {}):
            raise ValueError("condition_factor_level_unresolved")
        kind, value = factor["type"], factor["levels"][level]
        if kind not in {"control", "ordinal", "valence", "category"}:
            raise ValueError("condition_factor_type_unresolved")
        # Numeric ordinal/valence values are declared orders, not measured effects.
        parts.append(f"{name.replace('_', ' ')}: {level} ({kind}: {value})")
    if condition in durations:
        parts.insert(0, "response-time duration regressor; longer versus shorter")
    elif role.get("_kind") == "pm" or not condition.startswith("trial_type."):
        if role.get("_constr") == "U" or role.get("_dir") not in {-1, 1}:
            raise ValueError("parametric_direction_unresolved")
        quantity = QUANTITIES.get(role.get("_quantity"))
        if quantity is None:
            raise ValueError("parametric_quantity_not_in_reviewed_allowlist")
        direction = "higher versus lower" if role["_dir"] == 1 else "lower versus higher"
        parts.insert(0, f"parametric regressor; {direction} {quantity}")
    else:
        if not parts:
            raise ValueError("condition_meaning_unresolved")
        parts.insert(0, "event regressor")
    return "; ".join(parts)


def prepare_text(preparation, orientation_rows, models, roles, map_census=None):
    orientations = {map_key(row): row for row in orientation_rows}
    if len(orientations) != len(orientation_rows):
        raise ValueError("duplicate orientation identity")
    identities = [row["map_id"] for row in preparation["maps"]]
    if len(set(identities)) != len(identities) or set(identities) != set(orientations):
        raise ValueError("methods001 and frozen orientation identities differ")
    result = []
    source_by_id = ({"|".join((r["route"], r["session"] or "", r["contrast"])): r
                     for r in map_census} if map_census is not None else None)
    if source_by_id is not None and (len(source_by_id) != len(map_census) or set(source_by_id) != set(identities)):
        raise ValueError("per-map definitions do not match methods001 identities")
    for original in preparation["maps"]:
        row = orientations[original["map_id"]]
        # Keep methods001 identity, grouping, exclusions and role exactly as read.
        record = dict(original, model_text=None, text_status="unresolved", text_issues=[], terms=[])
        model = models.get(row["route"])
        expected_sign = int(row["s"]) if row["s"] in {"-1", "1"} else None
        if original["polarity"] != expected_sign:
            raise ValueError("methods001 polarity differs from adopted orientation")
        try:
            source = source_by_id[original["map_id"]] if source_by_id is not None else None
            conditions, weights = exact_definition(row, model, source.get("definitions", []) if source else None)
            record["weight_binding"] = ("map_specific_executed_definition" if source else "embedded_route_model")
        except ValueError as exc:
            record["text_issues"].append(str(exc))
            result.append(record)
            continue
        factors, condition_roles = route_roles(roles, row["route"], conditions)
        durations = duration_regressors(model)
        for condition, weight in zip(conditions, weights):
            exact = Decimal(str(weight))
            signed = exact if expected_sign == 1 else exact.copy_negate() if expected_sign == -1 else None
            term = {"condition_join_key": condition, "source_weight": str(weight),
                    "canonical_weight": str(signed) if signed is not None else None,
                    "description": None}
            try:
                term["description"] = describe_condition(condition, factors, condition_roles, durations)
            except ValueError as exc:
                term["unresolved_reason"] = str(exc)
                # Zero terms remain exact provenance but contribute no semantic term.
                if Decimal(str(weight)) != 0:
                    record["text_issues"].append(str(exc))
            record["terms"].append(term)
        if expected_sign is None:
            record["text_issues"].append("adopted_orientation_unresolved")
        record["text_issues"] = sorted(set(record["text_issues"]))
        if not record["text_issues"]:
            pieces = [f"{t['canonical_weight']} times [{t['description']}]"
                      for t in record["terms"] if Decimal(t["source_weight"]) != 0]
            if pieces:
                baseline = " relative to the implicit baseline" if row.get("class") == "E_event_vs_implicit_baseline" else ""
                record["model_text"] = "Canonical weighted contrast: " + " plus ".join(pieces) + baseline + "."
                record["text_status"] = "described_by_adopted_structured_roles"
            else:
                record["text_issues"].append("zero_contrast")
        result.append(record)
    return result


def validate_grid(header, binding):
    """Header-only grid check; the same prospective rule applies to all sources."""
    if len(header) != 352:
        raise ValueError("truncated NIfTI header")
    endian = "<" if struct.unpack_from("<i", header)[0] == 348 else ">"
    unpack = lambda fmt, at: struct.unpack_from(endian + fmt, header, at)
    if unpack("i", 0)[0] != 348 or header[344:348] != b"n+1\x00":
        raise ValueError("expected single-file NIfTI-1")
    dims = unpack("8h", 40)
    if dims[0] != 3 or list(dims[1:4]) != binding["shape"]:
        raise ValueError("shape differs from prospective binding")
    affine = np.eye(4)
    affine[:3] = np.array(unpack("12f", 280)).reshape(3, 4)
    expected = np.asarray(binding["affine"], dtype=float)
    qcode, scode = unpack("2h", 252)
    if scode <= 0 or not np.allclose(affine, expected, atol=1e-5, rtol=0):
        raise ValueError("coded sform differs from selected anatomical grid")
    pixdim = unpack("8f", 76)
    if not np.allclose(pixdim[1:4], [2, 2, 2], atol=1e-6, rtol=0):
        raise ValueError("voxel spacing differs from selected resolution")
    if (header[123] & 7) not in {0, 2}:
        raise ValueError("spatial units are neither unspecified nor millimeters")
    if qcode > 0:
        b, c, d, x, y, z = unpack("6f", 256)
        norm = b*b + c*c + d*d
        if not math.isfinite(norm) or norm > 1 + 1e-5:
            raise ValueError("invalid quaternion")
        if norm > 1:
            b, c, d = np.array([b, c, d]) / math.sqrt(norm)
        a = math.sqrt(max(0, 1 - b*b - c*c - d*d))
        rotation = np.array([[a*a+b*b-c*c-d*d, 2*b*c-2*a*d, 2*b*d+2*a*c],
                             [2*b*c+2*a*d, a*a+c*c-b*b-d*d, 2*c*d-2*a*b],
                             [2*b*d-2*a*c, 2*c*d+2*a*b, a*a+d*d-c*c-b*b]])
        qaffine = np.eye(4)
        qaffine[:3, :3] = rotation @ np.diag([pixdim[1], pixdim[2],
                                            pixdim[3] * (-1 if pixdim[0] < 0 else 1)])
        qaffine[:3, 3] = [x, y, z]
        if not np.allclose(qaffine, expected, atol=1e-5, rtol=0):
            raise ValueError("qform conflicts with selected sform")
    return {"shape": list(dims[1:4]), "affine": affine.tolist(), "qform_code": qcode,
            "sform_code": scode, "spatial_units_code": header[123] & 7,
            "units_assumption_if_unspecified": "millimeters from source/template metadata"}, endian


def read_reference_mask(path, binding):
    """Bounded array read for the sole prospective anatomical reference."""
    with gzip.open(path, "rb") as handle:
        header = handle.read(352)
        report, endian = validate_grid(header, binding)
        unpack = lambda fmt, at: struct.unpack_from(endian + fmt, header, at)
        kinds = {2: "u1", 4: "i2", 8: "i4", 16: "f4", 64: "f8", 256: "i1",
                 512: "u2", 768: "u4", 1024: "i8", 1280: "u8"}
        datatype, bitpix = unpack("2h", 70)
        if datatype not in kinds:
            raise ValueError("unsupported anatomical mask datatype")
        dtype = np.dtype(endian + kinds[datatype])
        if bitpix != dtype.itemsize * 8:
            raise ValueError("reference datatype/bitpix mismatch")
        offset, slope, intercept = unpack("3f", 108)
        if not math.isfinite(offset) or offset != int(offset) or not 352 <= offset <= 65536:
            raise ValueError("invalid reference voxel offset")
        count = math.prod(binding["shape"])
        if count * dtype.itemsize > 16 * 1024**2:
            raise ValueError("reference exceeds selected bounded read")
        handle.seek(int(offset))
        payload = handle.read(count * dtype.itemsize)
        if len(payload) != count * dtype.itemsize:
            raise ValueError("truncated reference mask")
    values = np.frombuffer(payload, dtype=dtype)
    raw_levels = np.unique(values)
    if slope != 0:
        if not math.isfinite(slope) or not math.isfinite(intercept):
            raise ValueError("invalid reference intensity scaling")
        values = values.astype(np.float64) * slope + intercept
    # NIfTI scaling fields are float32. The authentic uint8 0/255 mask uses
    # float32(1/255), yielding foreground 1.0000000591389835 after float64
    # decoding. Preserve its exact discrete support; never threshold levels.
    levels = np.unique(values)
    rounding_tolerance = float(np.finfo(np.float32).eps)
    if (len(raw_levels) != 2 or len(levels) != 2 or
            not np.all(np.isfinite(levels)) or levels[0] != 0 or
            abs(float(levels[1]) - 1.0) > rounding_tolerance):
        raise ValueError("reference mask is not finite two-level binary support within header-scaling precision")
    mask = (values == levels[1]).reshape(binding["shape"], order="F")
    if not mask.any() or mask.all():
        raise ValueError("reference mask must contain anatomical foreground and background")
    return mask, {**report, "path": str(path), "foreground_voxels": int(mask.sum()),
                  "datatype_code": datatype, "raw_levels": raw_levels.tolist(),
                  "scaling_slope": slope, "scaling_intercept": intercept,
                  "scaled_levels": levels.tolist(),
                  "foreground_rounding_tolerance": rounding_tolerance,
                  "support_normalization": "exact discrete foreground to bool; zero unchanged; no thresholding",
                  "target_arrays_read": 0, "target_headers_read": 0}


def bind_target_paths(preparation, maps, analysis_root):
    """Join exact census paths without discovery or opening any target file."""
    lookup = {}
    for item in maps:
        key = "|".join((item["route"], item["session"] or "", item["contrast"]))
        relative = Path(item["path"])
        accession = item["dataset_id"]
        if (relative.is_absolute() or ".." in relative.parts or
                not accession.startswith("ds") or not accession[2:].isdigit() or
                not item["route"].startswith(accession + "/") or
                not relative.name.endswith("_stat-z_statmap.nii.gz")):
            raise ValueError("unsafe or non-Z census binding")
        if key in lookup:
            raise ValueError("duplicate exact-path map identity")
        lookup[key] = {"map_id": key, "path": str(analysis_root / accession / relative),
                       "map_commit": item["map_commit"], "accession": accession}
    if set(lookup) != {r["map_id"] for r in preparation["maps"]}:
        raise ValueError("exact path census does not cover precisely methods001 identities")
    if len({r["path"] for r in lookup.values()}) != len(lookup):
        raise ValueError("duplicate exact target path")
    return [lookup[r["map_id"]] for r in preparation["maps"]]


def check_target_headers(bindings, mask_binding):
    """Read 352 decompressed header bytes at each enumerated path, no voxel array."""
    records = []
    for entry in bindings:
        record = dict(entry)
        try:
            with gzip.open(entry["path"], "rb") as handle:
                header = handle.read(352)
            info, _ = validate_grid(header, mask_binding)
            record.update(status="compatible_header_only", header=info)
        except (OSError, EOFError, ValueError, struct.error) as exc:
            record.update(status="needs_diagnosis", error=str(exc))
        records.append(record)
    return {"analysis_id": "grouped_evaluator_qualification", "headers": records,
            "status_counts": dict(Counter(r["status"] for r in records)),
            "target_arrays_read": 0, "baseline_target_access": False,
            "scientific_scoring": False,
            "interpretation": "Grid compatibility only, not source eligibility or scientific evidence; no rows dropped or resampled."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--headers-only", action="store_true",
                        help="Separate exact-path header phase; no reference or target arrays read")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise SystemExit("Use a coordinator-owned verified Slurm allocation.")
    episode = Path(__file__).resolve().parents[2]
    spec = json.loads((episode / "outputs/config/methods_preparation.json").read_text())
    fixed = "outputs/runs/source_handoff_qualification/20261016-orientation-proposal/"
    paths = {"orientation": fixed + "orientation_by_map.tsv", "roles": fixed + "condition_roles.json",
             "models": "outputs/runs/source_handoff_qualification/20261016-route-census-48185964/embedded_models.json",
             "methods_preparation": "outputs/runs/grouped_evaluator_qualification/methods001/preparation.json",
             "dataset_maps": "outputs/runs/source_handoff_qualification/20261016-route-census-48185964/dataset_maps.json",
             "route_provenance": "outputs/runs/source_handoff_qualification/20261016-route-census-48185964/route_provenance.json"}
    for key, relative in paths.items():
        if spec["inputs"][key] != relative:
            raise ValueError(f"unexpected metadata binding: {key}")
    binding = spec["reference_mask"]
    prefix = ".cache/templateflow/tpl-MNI152NLin2009cAsym/"
    if binding["home_relative_path"] != prefix + "tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask.nii.gz":
        raise ValueError("unexpected anatomical reference path")
    prep = json.loads((episode / paths["methods_preparation"]).read_text())
    if prep["source_maps"] != 628 or prep["baseline_target_access"] or prep["scientific_scoring"]:
        raise ValueError("unexpected methods001 scope or access state")
    provenance = json.loads((episode / paths["route_provenance"]).read_text())
    analysis_root = Path(os.environ["OAK"]) / "data/OpenNeuro_analyses/openneuro_fitlins/analyses"
    if Path(provenance["inputs"]["analysis_root"]) != analysis_root:
        raise ValueError("source analysis root differs from exact permitted binding")
    map_census = json.loads((episode / paths["dataset_maps"]).read_text(), parse_float=Decimal)
    target_paths = bind_target_paths(prep, map_census, analysis_root)
    destination = args.output_dir.resolve()
    if not destination.is_relative_to(episode / "outputs/runs/grouped_evaluator_qualification"):
        raise ValueError("output must be a new grouped_evaluator_qualification attempt")
    destination.mkdir(parents=True, exist_ok=False)
    if args.headers_only:
        report = check_target_headers(target_paths, binding)
        report["job_id"] = os.environ["SLURM_JOB_ID"]
        (destination / "headers.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps(report["status_counts"]))
        return
    models = json.loads((episode / paths["models"]).read_text(), parse_float=Decimal)
    rows = prepare_text(prep, read_tsv(episode / paths["orientation"]), models,
                        json.loads((episode / paths["roles"]).read_text()), map_census)
    mask, report = read_reference_mask(Path(os.environ["HOME"]) / binding["home_relative_path"], binding)
    np.save(destination / "anatomical_mask.npy", mask, allow_pickle=False)
    # Preserve the required reference copyright alongside the derived mask.
    (destination / "REFERENCE_LICENSE.txt").write_text((Path(os.environ["HOME"]) / prefix / "LICENSE").read_text())
    output = {"analysis_id": "grouped_evaluator_qualification", "job_id": os.environ["SLURM_JOB_ID"],
              "evidence": "anatomical_reference_and_methods_text_only", "reference_mask": report,
              "source_maps": len(rows), "text_status_counts": dict(Counter(r["text_status"] for r in rows)),
              "text_issue_counts": dict(Counter(x for r in rows for x in r["text_issues"])),
              "maps": rows, "text_allowlist": spec["signed_text"],
              "target_path_bindings": target_paths,
              "baseline_target_access": False, "scientific_scoring": False,
              "remaining_dependencies": ["unadopted_negative_control_and_inference_threshold_packet",
                  "separate_exact_628_map_header_only_common_grid_check",
                  "remaining_source_and_text_meaning_qualification",
                  "registered_predictor_composition_and_full_controls"]}
    (destination / "preparation.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: output[key] for key in ("source_maps", "text_status_counts", "text_issue_counts",
                                                 "baseline_target_access", "scientific_scoring")}))


if __name__ == "__main__":
    main()
