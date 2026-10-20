#!/usr/bin/env python3
"""Stage-A v1.2 feasibility precheck using soma positions and the pinned Allen atlas."""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import importlib
import importlib.util
import math
import os
import re
import socket
import stat
import sys
import zipfile
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

# Imported in main(), after checking SLURM_JOB_ID and limiting thread counts.
np = None
cKDTree = None

EXPECTED_CELLS = 1703
EXPECTED_ANIMALS = 43
ATLAS_SHAPE = (528, 320, 456)
VOXEL_UM = 25
HALF_VOXEL_UM = 12.5
MIDLINE_UM = 5700
DISTAL_UM = 1200
DISTAL_SQUARED = DISTAL_UM**2

# A voxel's half-diagonal is about 21.65064 um. This radius includes every
# centre whose voxel box could be closer than 1200 um, with ample padding
# for floating-point coordinate conversion and KD-tree arithmetic.
SEARCH_RADIUS_UM = DISTAL_UM + 22.0

ROOT_ID = 8
CTX_ID = 688
ISOCORTEX_ID = 315
OLF_ID = 698
HPF_ID = 1089
CTXSP_ID = 703
STR_ID = 477
PAL_ID = 803
TH_ID = 549
HY_ID = 1097
MB_ID = 313
P_ID = 771
MY_ID = 354
CB_ID = 512

# This order is normative for all per-division reports, including zero counts.
MAJOR_DIVISION_IDS = (
    ISOCORTEX_ID, OLF_ID, HPF_ID, CTXSP_ID, STR_ID, PAL_ID,
    TH_ID, HY_ID, MB_ID, P_ID, MY_ID, CB_ID,
)

EXCLUDED_IDS = {1009, 73}
REL_SIDES = ("ipsi", "contra")
OUTPUT_NAMES = ("precheck_counts.csv", "precheck_summary.json")

# Inspect only the row prefix to identify an integer type of 1.
# Non-soma rows are never split into fields or numerically parsed.
SOMA_PREFIX = re.compile(
    rb"^[ \t]*[^ \t\r\n]+[ \t]+\+?0*1(?=[ \t\r\n]|$)"
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def file_signature(path):
    s = path.stat()
    require(stat.S_ISREG(s.st_mode), f"Not a regular file: {path}")
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns


class InputLedger:
    """Hash declared inputs and detect changes during the run."""

    def __init__(self, zip_path):
        self.zip_path = zip_path
        self.records = {}
        self.signatures = {}

    def add(self, path, role):
        path = Path(path).resolve(strict=True)
        require(path != self.zip_path, "The ZIP must never be hashed.")

        if path in self.records:
            if role not in self.records[path]["roles"]:
                self.records[path]["roles"].append(role)
            return

        before = file_signature(path)
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        require(
            file_signature(path) == before,
            f"Input changed while being hashed: {path}",
        )
        self.signatures[path] = before
        self.records[path] = {
            "path": str(path),
            "roles": [role],
            "size_bytes": before[2],
            "sha256": digest.hexdigest(),
        }

    def verify_unchanged(self):
        for path, original in self.signatures.items():
            require(
                file_signature(path) == original,
                f"Input changed during the run: {path}",
            )

    def as_list(self):
        return [self.records[p] for p in sorted(self.records, key=str)]


def collect_tool_inputs(tools_dir, ledger):
    """Include imported local tool modules and existing bytecode caches."""
    for module in list(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if not filename:
            continue
        path = Path(filename).absolute()
        if not path.is_relative_to(tools_dir):
            continue
        ledger.add(path, "tool_module")
        cached = getattr(module, "__cached__", None)
        if cached and Path(cached).is_file():
            ledger.add(cached, "tool_bytecode")


def check_self_contained_nrrd(path):
    """Reject detached payloads, whose files are not among the declared inputs."""
    with path.open("rb") as stream:
        first = stream.readline(65537)
        require(first.startswith(b"NRRD000"), "Invalid NRRD magic.")
        total = len(first)
        while True:
            line = stream.readline(65537)
            require(line, "NRRD header ended without a blank separator.")
            total += len(line)
            require(
                len(line) <= 65536 and total <= 1024 * 1024,
                "Unexpectedly large NRRD header.",
            )
            if not line.strip():
                return
            if line.lstrip().startswith(b"#"):
                continue
            key = line.split(b":", 1)[0].strip().lower().replace(b" ", b"")
            require(
                key != b"datafile",
                "Detached NRRD payloads are unsupported; use a self-contained NRRD.",
            )


def valid_component(value):
    return (
        bool(value)
        and value not in {".", ".."}
        and not any(c in value for c in ("/", "\\", "\x00"))
    )


def read_manifest(path):
    required = {
        "cell", "brain", "line", "area", "layer",
        "pool_key", "pool_size", "ep09_role",
    }
    cells = []
    seen = set()
    animal_roles = {}

    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        require(len(fields) == len(set(fields)), "Duplicate CSV column names.")
        require(required <= set(fields), f"Missing CSV columns: {required - set(fields)}")

        for line_number, row in enumerate(reader, start=2):
            require(
                None not in row and all(row.get(k) is not None for k in required),
                f"Malformed manifest row {line_number}.",
            )
            row = {k: v.strip() for k, v in row.items()}
            cell, brain = row["cell"], row["brain"]
            require(valid_component(brain), f"Invalid brain ID at row {line_number}.")
            require(cell not in seen, f"Duplicate manifest cell: {cell}")
            prefix = brain + "_"
            require(cell.startswith(prefix), f"Cell/brain mismatch: {cell}, {brain}")
            suffix = cell[len(prefix):]
            require(valid_component(suffix), f"Invalid cell suffix: {cell}")
            require(row["area"], f"Missing manifest area: {cell}")

            role = row["ep09_role"]
            require(role in {"development", "audit"}, f"Invalid role for {cell}: {role}")
            previous = animal_roles.setdefault(brain, role)
            require(previous == role, f"Animal {brain} has mixed development/audit roles.")

            row["member"] = f"data_v3/{brain}/{suffix}.swc"
            cells.append(row)
            seen.add(cell)

    require(
        len(cells) == EXPECTED_CELLS,
        f"Expected {EXPECTED_CELLS} manifest cells; found {len(cells)}.",
    )
    require(
        len(animal_roles) == EXPECTED_ANIMALS,
        f"Expected {EXPECTED_ANIMALS} animals; found {len(animal_roles)}.",
    )

    brains = list(animal_roles)
    if all(re.fullmatch(r"[0-9]+", b) for b in brains):
        brains.sort(key=lambda b: (int(b), b))
    else:
        brains.sort()
    return cells, brains, animal_roles


class AtlasGraph:
    def __init__(self, nodes, by_acr):
        self.nodes = {int(k): v for k, v in nodes.items()}
        require(len(self.nodes) == len(nodes), "Duplicate normalized graph IDs.")
        self.parent = {}
        self.children = {nid: [] for nid in self.nodes}

        for nid, node in self.nodes.items():
            require(int(node["id"]) == nid, f"Graph ID mismatch at node {nid}.")
            require(
                isinstance(node["name"], str) and isinstance(node["acronym"], str),
                f"Invalid graph metadata at node {nid}.",
            )
            raw_parent = node["parent_structure_id"]
            parent = None if raw_parent is None else int(raw_parent)
            require(
                parent is None or parent in self.nodes,
                f"Missing parent {parent} of node {nid}.",
            )
            self.parent[nid] = parent
            if parent is not None:
                self.children[parent].append(nid)

        for children in self.children.values():
            children.sort()

        # Ancestry tuples run from the node itself to its root.
        self.ancestors = {}
        for start in sorted(self.nodes):
            chain = []
            seen = set()
            current = start
            while current is not None and current not in self.ancestors:
                require(current not in seen, f"Cycle in graph at node {current}.")
                seen.add(current)
                chain.append(current)
                current = self.parent[current]
            tail = () if current is None else self.ancestors[current]
            for nid in reversed(chain):
                tail = (nid,) + tail
                self.ancestors[nid] = tail

        self.ancestor_sets = {
            nid: set(path) for nid, path in self.ancestors.items()
        }

        for nid, node in self.nodes.items():
            path = node["structure_id_path"]
            require(isinstance(path, str), f"Invalid structure_id_path for {nid}.")
            path_ids = tuple(int(x) for x in path.split("/") if x)
            require(
                path.startswith("/") and path.endswith("/")
                and path_ids == tuple(reversed(self.ancestors[nid])),
                f"Parent links and structure_id_path disagree for node {nid}.",
            )

        required_ids = {ROOT_ID, CTX_ID} | set(MAJOR_DIVISION_IDS) | EXCLUDED_IDS
        for nid in required_ids:
            require(nid in self.nodes, f"Required graph node is absent: {nid}")

        self.by_acr = {}
        for acronym, node in by_acr.items():
            nid = int(node["id"])
            require(
                nid in self.nodes and self.nodes[nid]["acronym"] == acronym,
                f"Inconsistent acronym lookup: {acronym}",
            )
            self.by_acr[acronym] = nid

    LAYER_SUFFIXES = ("1", "2", "3", "2/3", "4", "5", "6", "6a", "6b")

    def is_layer(self, nid):
        """R2: CTX layer names, plus parent-acronym suffixes in Isocortex only."""
        node = self.nodes[nid]
        if CTX_ID not in self.ancestor_sets[nid]:
            return False
        if "layer" in node["name"].lower():
            return True
        if ISOCORTEX_ID not in self.ancestor_sets[nid]:
            return False
        parent = self.parent.get(nid)
        if parent is None:
            return False
        prefix = self.nodes[parent]["acronym"]
        acr = node["acronym"]
        return acr.startswith(prefix) and acr[len(prefix):] in self.LAYER_SUFFIXES

    def related(self, a, b):
        return a in self.ancestor_sets[b] or b in self.ancestor_sets[a]


def build_vocabulary(graph, labels, frequencies):
    direct = {
        int(label): int(count)
        for label, count in zip(labels, frequencies)
        if int(label) != 0
    }
    unknown = set(direct) - set(graph.nodes)
    require(not unknown, f"Annotation labels absent from graph: {sorted(unknown)}")

    allowed = {
        nid for nid in graph.nodes
        if ROOT_ID in graph.ancestor_sets[nid]
        and not (EXCLUDED_IDS & graph.ancestor_sets[nid])
    }

    mask_sizes = {}
    for nid in sorted(allowed, key=lambda n: len(graph.ancestors[n]), reverse=True):
        mask_sizes[nid] = direct.get(nid, 0) + sum(
            mask_sizes[child]
            for child in graph.children[nid]
            if child in allowed
        )

    targets = []
    unassigned_parents = []
    pooled_outside_isocortex = []
    pooled_with_direct_voxels = []
    mixed_cortical_nodes = []

    def visit(nid):
        if nid not in allowed or mask_sizes[nid] == 0:
            return
        children = [
            child for child in graph.children[nid]
            if child in allowed and mask_sizes[child] > 0
        ]
        node = graph.nodes[nid]
        in_ctx = CTX_ID in graph.ancestor_sets[nid]
        laminar = [graph.is_layer(child) for child in children]
        direct_count = direct.get(nid, 0)

        if in_ctx and any(laminar) and not all(laminar):
            mixed_cortical_nodes.append({
                "node_id": nid,
                "acronym": node["acronym"],
                "name": node["name"],
                "child_acronyms": [
                    graph.nodes[child]["acronym"] for child in children
                ],
                "laminar_child_acronyms": [
                    graph.nodes[child]["acronym"]
                    for child, is_laminar in zip(children, laminar)
                    if is_laminar
                ],
                "nonlaminar_child_acronyms": [
                    graph.nodes[child]["acronym"]
                    for child, is_laminar in zip(children, laminar)
                    if not is_laminar
                ],
            })

        # R1: cortical areal pooling takes precedence over dropping a
        # traversed parent's direct voxels.
        if in_ctx and children and all(laminar):
            targets.append(nid)
            if ISOCORTEX_ID not in graph.ancestor_sets[nid]:
                pooled_outside_isocortex.append({
                    "node_id": nid,
                    "acronym": node["acronym"],
                    "child_acronyms": [
                        graph.nodes[child]["acronym"] for child in children
                    ],
                })
            if direct_count > 0:
                pooled_with_direct_voxels.append({
                    "node_id": nid,
                    "acronym": node["acronym"],
                    "direct_voxel_count": direct_count,
                })
            return

        if not children:
            targets.append(nid)
            return

        if direct_count > 0:
            unassigned_parents.append({
                "node_id": nid,
                "acronym": node["acronym"],
                "name": node["name"],
                "direct_voxel_count": direct_count,
            })
        for child in children:
            visit(child)

    visit(ROOT_ID)
    targets.sort()
    target_set = set(targets)

    for nid in targets:
        require(
            not (set(graph.ancestors[nid][1:]) & target_set),
            f"Vocabulary is not an antichain at node {nid}.",
        )

    unassigned_ids = {item["node_id"] for item in unassigned_parents}
    require(
        len(unassigned_ids) == len(unassigned_parents),
        "Duplicate unassigned parent records.",
    )
    eligible_labels = set(direct) & allowed
    target_index = {nid: i for i, nid in enumerate(targets)}
    label_owner = {}

    for label in sorted(eligible_labels):
        owners = graph.ancestor_sets[label] & target_set
        require(len(owners) <= 1, f"Overlapping target masks at label {label}.")
        if owners:
            require(
                label not in unassigned_ids,
                f"Annotation label is both owned and unassigned: {label}",
            )
            label_owner[label] = target_index[next(iter(owners))]
        else:
            require(
                label in unassigned_ids,
                f"Eligible annotation label is uncovered and unrecorded: {label}",
            )

    owned_labels = set(label_owner)
    require(
        owned_labels.isdisjoint(unassigned_ids)
        and (owned_labels | unassigned_ids) == eligible_labels,
        "Eligible labels are not exactly partitioned into owned and unassigned labels.",
    )

    for records in (
        unassigned_parents, pooled_outside_isocortex,
        pooled_with_direct_voxels, mixed_cortical_nodes,
    ):
        records.sort(key=lambda item: item["node_id"])

    vocabulary_report = {
        "unassigned_parent_nodes": unassigned_parents,
        "total_unassigned_voxels": sum(
            item["direct_voxel_count"] for item in unassigned_parents
        ),
        "pooled_laminar_areas_outside_isocortex": pooled_outside_isocortex,
        "pooled_cortical_areas_with_direct_voxels": pooled_with_direct_voxels,
        "mixed_cortical_nodes": mixed_cortical_nodes,
    }

    # Each voxel has one label. Unique label ownership therefore verifies
    # pairwise disjoint voxel masks without allocating one full mask per target.
    return targets, label_owner, vocabulary_report


def pack_target_voxels(volume, labels, frequencies, label_owner, target_count):
    """Group flat voxel indices by target using one bounded-chunk volume scan."""
    lookup = np.array(
        [label_owner.get(int(label), -1) for label in labels], dtype=np.int32
    )
    totals = np.zeros(target_count, dtype=np.int64)
    included = lookup >= 0
    np.add.at(totals, lookup[included], frequencies[included])

    offsets = np.zeros(target_count + 1, dtype=np.int64)
    np.cumsum(totals, out=offsets[1:])
    packed = np.empty(int(offsets[-1]), dtype=np.uint32)
    cursor = offsets[:-1].copy()

    flat = volume.ravel(order="C")
    require(flat.size <= 2**32, "Atlas is too large for uint32 flat indices.")
    block_size = 1_000_000

    for start in range(0, flat.size, block_size):
        block = flat[start:start + block_size]
        groups = lookup[np.searchsorted(labels, block)]
        valid = np.flatnonzero(groups >= 0)
        if valid.size == 0:
            continue
        groups = groups[valid]
        order = np.argsort(groups, kind="stable")
        groups = groups[order]
        indices = (valid[order] + start).astype(np.uint32)
        cuts = np.concatenate((
            [0],
            np.flatnonzero(groups[1:] != groups[:-1]) + 1,
            [groups.size],
        ))
        for lo, hi in zip(cuts[:-1], cuts[1:]):
            target = int(groups[lo])
            destination = int(cursor[target])
            amount = int(hi - lo)
            packed[destination:destination + amount] = indices[lo:hi]
            cursor[target] += amount

    require(np.array_equal(cursor, offsets[1:]), "Voxel grouping count mismatch.")
    return packed, offsets


def read_first_soma(archive, info):
    with archive.open(info, "r") as stream:
        for line_number, raw in enumerate(stream, start=1):
            if raw.lstrip().startswith(b"#") or not SOMA_PREFIX.match(raw):
                continue

            # Only the first type-1 row is split. No other row's coordinates,
            # radius, parent, or morphology are parsed.
            fields = raw.split(b"#", 1)[0].split()
            require(
                len(fields) == 7,
                f"Malformed soma row in {info.filename}:{line_number}",
            )
            try:
                xyz = tuple(float(token) for token in fields[2:5])
                require(all(math.isfinite(v) for v in xyz), "Non-finite soma.")
                exact = tuple(
                    Fraction(token.decode("ascii")) for token in fields[2:5]
                )
            except (ValueError, OverflowError, UnicodeError) as exc:
                raise ValueError(
                    f"Invalid soma coordinates in {info.filename}:{line_number}"
                ) from exc
            return np.asarray(xyz, dtype=np.float64), exact

    raise ValueError(f"No type-1 soma found in {info.filename}")


def read_somata(zip_path, cells, volume, graph):
    wanted = {cell["member"] for cell in cells}
    require(len(wanted) == len(cells), "Multiple cells map to the same SWC member.")
    xyz = np.empty((len(cells), 3), dtype=np.float64)
    exact_points = []
    sides = np.empty(len(cells), dtype=np.int8)
    coordinate_areas = []
    found = 0

    with zipfile.ZipFile(zip_path, "r") as archive:
        infos = archive.infolist()
        members = {}
        for info in infos:
            if info.filename not in wanted:
                continue
            require(info.filename not in members, f"Duplicate ZIP member: {info.filename}")
            require(not info.is_dir(), f"Expected SWC file, found directory: {info.filename}")
            require(not (info.flag_bits & 1), f"Encrypted SWC member: {info.filename}")
            members[info.filename] = info

        missing = sorted(wanted - set(members))
        require(not missing, f"Missing SWC members (first 10): {missing[:10]}")

        for i, cell in enumerate(cells):
            point, exact = read_first_soma(archive, members[cell["member"]])
            found += 1
            voxel = tuple(value // VOXEL_UM for value in exact)
            require(
                all(0 <= index < size for index, size in zip(voxel, ATLAS_SHAPE)),
                f"Soma outside atlas: {cell['cell']}",
            )
            label = int(volume[voxel])
            require(
                label == 0 or label in graph.nodes,
                f"Soma has an unknown annotation label: {cell['cell']}",
            )
            # Reviewer edit: a background soma voxel is not fatal; its
            # coordinate-derived area is None and only the manifest area
            # drives source exclusion for that cell.
            if label == 0:
                area = None
            else:
                area = graph.parent[label] if graph.is_layer(label) else label
                require(area is not None, f"Layer node has no parent: {label}")

            xyz[i] = point
            exact_points.append(exact)
            coordinate_areas.append(area)
            z = exact[2]
            sides[i] = -1 if z == MIDLINE_UM else (0 if z < MIDLINE_UM else 1)

    return xyz, exact_points, sides, coordinate_areas, found, len(infos)


def outward_squared_sum(gaps, direction):
    squares = np.maximum(np.nextafter(gaps * gaps, direction), 0.0)
    first_two = np.nextafter(squares[..., 0] + squares[..., 1], direction)
    return np.maximum(
        np.nextafter(first_two + squares[..., 2], direction), 0.0
    )


def box_squared_distance_bounds(points, centres):
    """Directed bounds also enclose conversion of decimal soma values to float."""
    with np.errstate(under="ignore"):
        point_lo = np.nextafter(points, -np.inf)
        point_hi = np.nextafter(points, np.inf)
        box_lo = centres - HALF_VOXEL_UM
        box_hi = centres + HALF_VOXEL_UM

        gap_lo = np.maximum(
            np.maximum(
                np.nextafter(box_lo - point_hi, -np.inf),
                np.nextafter(point_lo - box_hi, -np.inf),
            ),
            0.0,
        )
        gap_hi = np.maximum(
            np.maximum(
                np.nextafter(box_lo - point_lo, np.inf),
                np.nextafter(point_hi - box_hi, np.inf),
            ),
            0.0,
        )
        return (
            outward_squared_sum(gap_lo, -np.inf),
            outward_squared_sum(gap_hi, np.inf),
        )


def exact_box_is_too_close(exact_point, centre):
    """Exact comparison against 1200 um, including equality at the threshold."""
    squared = Fraction(0)
    for coordinate, c in zip(exact_point, centre):
        left = int(c - HALF_VOXEL_UM)  # Integer voxel boundary, exactly represented.
        right = left + VOXEL_UM
        gap = max(Fraction(left) - coordinate, coordinate - Fraction(right), 0)
        squared += gap * gap
    return squared < DISTAL_SQUARED


def distal_flags(tree, points, exact_points):
    """Certify the minimum point-to-box distance is at least 1200 um."""
    distances, nearest = tree.query(
        points,
        k=1,
        p=2,
        eps=0.0,
        distance_upper_bound=SEARCH_RADIUS_UM,
        workers=1,
    )

    # An omitted centre is farther than 1222 um; its box is farther than
    # 1200 um because its half-diagonal is <22 um. The padding greatly
    # exceeds floating-point error throughout this bounded atlas.
    result = np.isinf(distances)
    nearby = np.flatnonzero(np.isfinite(distances))
    if nearby.size == 0:
        return result

    _, upper = box_squared_distance_bounds(
        points[nearby], tree.data[nearest[nearby]]
    )

    # A definitely close nearest-centre voxel already makes the pair negative.
    for i in nearby[upper >= DISTAL_SQUARED]:
        indices = tree.query_ball_point(
            points[i],
            r=SEARCH_RADIUS_UM,
            p=2,
            eps=0.0,
            workers=1,
            return_sorted=False,
        )
        centres = tree.data[np.asarray(indices, dtype=np.intp)]
        lower, upper = box_squared_distance_bounds(points[i], centres)
        if np.any(upper < DISTAL_SQUARED):
            continue

        # All potentially threshold-crossing boxes receive an exact rational
        # comparison. Boxes outside the search radius cannot violate the rule.
        uncertain = np.flatnonzero(lower < DISTAL_SQUARED)
        if any(
            exact_box_is_too_close(exact_points[i], centres[j])
            for j in uncertain
        ):
            continue
        result[i] = True

    return result


def count_pairs(
    targets, graph, packed, offsets, xyz, exact_points, soma_sides,
    manifest_areas, coordinate_areas, animal_indices, animal_count,
):
    counts = np.zeros((2 * len(targets), animal_count), dtype=np.int64)

    for target_index, nid in enumerate(targets):
        flat_indices = packed[offsets[target_index]:offsets[target_index + 1]]
        all_ml = flat_indices % ATLAS_SHAPE[2]
        voxel_is_left = (all_ml.astype(np.float64) + 0.5) * VOXEL_UM < MIDLINE_UM

        ipsi_allowed = np.asarray([
            not (
                graph.related(nid, manifest_area)
                or (coordinate_area is not None and graph.related(nid, coordinate_area))
            )
            for manifest_area, coordinate_area
            in zip(manifest_areas, coordinate_areas)
        ], dtype=bool)

        for physical_side in (0, 1):
            voxel_selection = voxel_is_left if physical_side == 0 else ~voxel_is_left
            selected = flat_indices[voxel_selection]
            if selected.size == 0:
                continue  # Empty physical-side masks are always negative.

            eligible_cells = np.flatnonzero(
                (soma_sides >= 0)
                & ((soma_sides != physical_side) | ipsi_allowed)
            )
            if eligible_cells.size == 0:
                continue

            ap_dv, ml = np.divmod(selected.astype(np.int64), ATLAS_SHAPE[2])
            ap, dv = np.divmod(ap_dv, ATLAS_SHAPE[1])
            centres = np.empty((selected.size, 3), dtype=np.float64)
            centres[:, 0] = (ap + 0.5) * VOXEL_UM
            centres[:, 1] = (dv + 0.5) * VOXEL_UM
            centres[:, 2] = (ml + 0.5) * VOXEL_UM

            tree = cKDTree(
                centres, leafsize=32, compact_nodes=True,
                balanced_tree=True, copy_data=False,
            )
            passed = distal_flags(
                tree,
                xyz[eligible_cells],
                [exact_points[int(i)] for i in eligible_cells],
            )
            good = eligible_cells[passed]
            relative_side = (soma_sides[good] != physical_side).astype(np.intp)
            np.add.at(
                counts,
                (2 * target_index + relative_side, animal_indices[good]),
                1,
            )
            del tree, centres, ap_dv, ap, dv, ml

    return counts


def make_family_map(targets, graph):
    """R3: assign every anatomical target to one fixed major division."""
    family_by_node = {}
    for nid in targets:
        divisions = [
            division for division in MAJOR_DIVISION_IDS
            if division in graph.ancestor_sets[nid]
        ]
        require(
            len(divisions) == 1,
            f"Target {nid} ({graph.nodes[nid]['acronym']}) must map to exactly "
            f"one major division; found {divisions}.",
        )
        family_by_node[nid] = divisions[0]
    return family_by_node


def family_gate(survives, targets, family_by_node, graph):
    """R4: report all divisions; gate only the supported secondary families."""
    survives = np.asarray(survives)
    require(
        survives.dtype.kind == "b" and survives.shape == (2 * len(targets),),
        "Family support requires a boolean survival vector over all sided rows.",
    )

    counts = {division: 0 for division in MAJOR_DIVISION_IDS}
    for row in np.flatnonzero(survives):
        nid = targets[int(row) // 2]
        counts[family_by_node[nid]] += 1

    division_counts = [
        {
            "division_node_id": division,
            "acronym": graph.nodes[division]["acronym"],
            "name": graph.nodes[division]["name"],
            "sided_target_count": counts[division],
        }
        for division in MAJOR_DIVISION_IDS
    ]
    supported = [
        item for item in division_counts if item["sided_target_count"] >= 3
    ]
    return {
        "per_division_sided_counts": division_counts,
        "supported_divisions": supported,
        "supported_division_count": len(supported),
        "passes_secondary_family_gate": len(supported) >= 3,
        "strict_all_12_have_3": all(
            item["sided_target_count"] >= 3 for item in division_counts
        ),
    }


def target_record(row, targets, graph):
    nid = targets[row // 2]
    return {
        "node_id": nid,
        "acronym": graph.nodes[nid]["acronym"],
        "side": REL_SIDES[row % 2],
    }


def role_v3(role, brains, animal_roles, counts, targets, graph, family_by_node):
    columns = [i for i, brain in enumerate(brains) if animal_roles[brain] == role]
    survives = (
        np.all(counts[:, columns] >= 3, axis=1)
        if columns else np.zeros(counts.shape[0], dtype=bool)
    )
    rows = np.flatnonzero(survives).tolist()
    return {
        "defined": bool(columns),
        "animals": [brains[i] for i in columns],
        "animal_count": len(columns),
        "sided_target_count": len(rows),
        "targets": [target_record(row, targets, graph) for row in rows],
        **family_gate(survives, targets, family_by_node, graph),
    }


def write_outputs(out_dir, brains, targets, graph, counts, summary):
    """Exclusive creation; remove only files created by this call on failure."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / name for name in OUTPUT_NAMES]
    created = []

    try:
        with contextlib.ExitStack() as stack:
            streams = []
            for path in paths:
                stream = stack.enter_context(
                    path.open("x", encoding="utf-8", newline="")
                )
                s = os.fstat(stream.fileno())
                created.append((path, (s.st_dev, s.st_ino)))
                streams.append(stream)

            writer = csv.writer(streams[0], lineterminator="\n")
            writer.writerow(["node_id", "acronym", "name", "rel_side", *brains])
            for row in range(counts.shape[0]):
                nid = targets[row // 2]
                node = graph.nodes[nid]
                writer.writerow([
                    nid, node["acronym"], node["name"], REL_SIDES[row % 2],
                    *map(int, counts[row]),
                ])

            json.dump(
                summary, streams[1], indent=2, sort_keys=True,
                ensure_ascii=False, allow_nan=False,
            )
            streams[1].write("\n")

            for stream in streams:
                stream.flush()
                os.fchmod(stream.fileno(), 0o444)
                os.fsync(stream.fileno())
    except BaseException:
        for path, identity in created:
            try:
                s = path.lstat()
                if (s.st_dev, s.st_ino) == identity:
                    path.unlink()
            except FileNotFoundError:
                pass
        raise


def main():
    global np, cKDTree

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zip_path", type=Path, metavar="ZIP")
    parser.add_argument("manifest_path", type=Path, metavar="CELL_MANIFEST_CSV")
    parser.add_argument("ccf_dir", type=Path, metavar="CCF_PAYLOAD_DIR")
    parser.add_argument("tools_dir", type=Path, metavar="TOOLS_DIR")
    parser.add_argument("out_dir", type=Path, metavar="OUT_DIR")
    args = parser.parse_args()

    require(sys.version_info >= (3, 11), "Python 3.11 or newer is required.")
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "").strip()
    require(slurm_job_id, "Refusing to run: SLURM_JOB_ID is not set.")
    started = utc_now()
    hostname = socket.gethostname()

    zip_path = args.zip_path.resolve(strict=True)
    manifest_path = args.manifest_path.resolve(strict=True)
    ccf_dir = args.ccf_dir.resolve(strict=True)
    tools_dir = args.tools_dir.resolve(strict=True)
    out_dir = args.out_dir.resolve()
    require(ccf_dir.is_dir(), f"Not a directory: {ccf_dir}")
    require(tools_dir.is_dir(), f"Not a directory: {tools_dir}")
    require(not out_dir.exists() or out_dir.is_dir(), f"Not a directory: {out_dir}")

    for name in OUTPUT_NAMES:
        require(
            not os.path.lexists(out_dir / name),
            f"Refusing to overwrite existing output: {out_dir / name}",
        )

    annotation_path = ccf_dir / "annotation_25_ccf_2017.nrrd"
    graph_path = ccf_dir / "structure_graph_1.json"
    zip_signature = file_signature(zip_path)

    ledger = InputLedger(zip_path)
    ledger.add(manifest_path, "cell_manifest")
    ledger.add(annotation_path, "atlas_annotation")
    ledger.add(graph_path, "structure_graph")
    ledger.add(Path(__file__), "precheck_script")
    cells, brains, animal_roles = read_manifest(manifest_path)
    cell_counts = Counter(cell["brain"] for cell in cells)

    thread_variables = (
        "OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    for name in thread_variables:
        os.environ[name] = "1"

    import numpy as np
    import scipy
    from scipy.spatial import cKDTree

    sys.dont_write_bytecode = True
    sys.path.insert(0, str(tools_dir))
    importlib.invalidate_caches()
    spec = importlib.util.find_spec("derive_ccf_soma_layers")
    require(spec is not None and spec.origin, "Cannot locate derive_ccf_soma_layers.")
    require(
        Path(spec.origin).absolute().is_relative_to(tools_dir),
        "derive_ccf_soma_layers did not resolve inside TOOLS_DIR.",
    )
    ledger.add(spec.origin, "atlas_loader")
    if spec.cached and Path(spec.cached).is_file():
        ledger.add(spec.cached, "tool_bytecode")
    loader = importlib.import_module("derive_ccf_soma_layers")
    collect_tool_inputs(tools_dir, ledger)

    check_self_contained_nrrd(annotation_path)
    _header, volume = loader.read_nrrd(str(annotation_path))
    require(isinstance(volume, np.ndarray), "Atlas loader did not return an ndarray.")
    require(tuple(volume.shape) == ATLAS_SHAPE, f"Unexpected atlas shape: {volume.shape}")
    require(
        volume.dtype.kind == "u" and volume.dtype.itemsize == 4,
        f"Expected uint32 atlas labels; found {volume.dtype}.",
    )
    nodes, by_acr = loader.load_graph(str(graph_path))
    graph = AtlasGraph(nodes, by_acr)
    collect_tool_inputs(tools_dir, ledger)

    manifest_areas = []
    for cell in cells:
        require(
            cell["area"] in graph.by_acr,
            f"Unknown manifest area acronym for {cell['cell']}: {cell['area']}",
        )
        manifest_areas.append(graph.by_acr[cell["area"]])

    (
        xyz, exact_points, soma_sides, coordinate_areas,
        somata_found, zip_member_count,
    ) = read_somata(zip_path, cells, volume, graph)

    ambiguous_count = int(np.count_nonzero(soma_sides < 0))
    background_soma_count = sum(area is None for area in coordinate_areas)
    disagreement_count = sum(
        manifest != coordinate
        for manifest, coordinate in zip(manifest_areas, coordinate_areas)
    )

    labels, frequencies = np.unique(volume, return_counts=True)
    targets, label_owner, vocabulary_report = build_vocabulary(
        graph, labels, frequencies
    )
    family_by_node = make_family_map(targets, graph)
    packed, offsets = pack_target_voxels(
        volume, labels, frequencies, label_owner, len(targets)
    )
    brain_to_column = {brain: i for i, brain in enumerate(brains)}
    animal_indices = np.asarray(
        [brain_to_column[cell["brain"]] for cell in cells], dtype=np.intp
    )

    counts = count_pairs(
        targets, graph, packed, offsets, xyz, exact_points, soma_sides,
        manifest_areas, coordinate_areas, animal_indices, len(brains),
    )
    minimum_counts = counts.min(axis=1)
    animals_covered = np.count_nonzero(counts >= 3, axis=1)
    v3_mask = minimum_counts >= 3
    v3_rows = np.flatnonzero(v3_mask).tolist()

    v3_gate = family_gate(v3_mask, targets, family_by_node, graph)
    n_max = min(cell_counts.values())

    survival = []
    for n in range(3, n_max + 1):
        survives = minimum_counts >= n
        survivor_count = int(np.count_nonzero(survives))
        survival.append({
            "n": n,
            "sided_target_count": survivor_count,
            "primary_count_gate_passes": survivor_count > 0,
            **family_gate(survives, targets, family_by_node, graph),
        })

    per_target_diagnostics = [
        {
            **target_record(row, targets, graph),
            "animals_with_N_ge_3": int(animals_covered[row]),
            "minimum_N_over_animals": int(minimum_counts[row]),
        }
        for row in range(counts.shape[0])
    ]
    ranked_rows = sorted(
        range(counts.shape[0]),
        key=lambda row: (
            -int(animals_covered[row]),
            -int(minimum_counts[row]),
            targets[row // 2],
            row % 2,
        ),
    )[:25]

    collect_tool_inputs(tools_dir, ledger)
    ledger.verify_unchanged()
    require(
        file_signature(zip_path) == zip_signature,
        "ZIP changed during the run.",
    )

    summary = {
        "provenance": {
            "SLURM_JOB_ID": slurm_job_id,
            "hostname": hostname,
            "utc_timestamp": started,
            "completed_utc": utc_now(),
            "python_version": sys.version,
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "tools_dir": str(tools_dir),
            "input_files": ledger.as_list(),
            "zip": {
                "path": str(zip_path),
                "size_bytes": zip_signature[2],
                "member_count": zip_member_count,
                "needed_swc_member_count": len(cells),
                "sha256": None,
                "sha256_omission_reason": "Contract: record ZIP size and member count only.",
            },
            "thread_limits": {name: os.environ[name] for name in thread_variables},
            "kdtree_workers": 1,
        },
        "contract": {
            "version": "EP09-measurement-contract-v1.2-adopted",
            "orchestrator_readings": {
                "R1": (
                    "Traverse grey, excluding fibre-tract, ventricular and "
                    "zero-volume subtrees. Pool a CTX node including its own "
                    "direct voxels when all nonempty eligible children are "
                    "laminar. Otherwise recurse, leaving parent-direct voxels "
                    "unassigned; retain nonempty leaves."
                ),
                "R2": (
                    "Laminar means a case-insensitive 'layer' name inside "
                    "CTX; additionally, only inside Isocortex, accept parent "
                    "acronym plus 1, 2, 3, 2/3, 4, 5, 6, 6a or 6b."
                ),
                "R3": (
                    "Map every anatomical target uniquely to one of the "
                    "12 fixed major divisions among its ancestors-or-self; "
                    "both relative sides share that division. No LCA cut "
                    "or V3-dependent family construction."
                ),
                "R4": (
                    "A division is supported by at least 3 sided targets. "
                    "Secondary family analyses use exactly supported divisions "
                    "and require at least 3 of them. Divisions with 1-2 targets "
                    "keep those targets in primary scoring. All 12 having "
                    "at least 3 targets is diagnostic only."
                ),
                "R5": (
                    "The precheck primary gate is nonempty V3; each n count "
                    "gate is nonempty V_n. Family support never selects or "
                    "blocks n. Empty final T blocks primary scoring; inadequate "
                    "family support blocks only secondary family analyses. "
                    "Never change vocabulary, families or n after axon access."
                ),
            },
            "atlas_shape_ap_dv_ml": list(ATLAS_SHAPE),
            "voxel_size_um": VOXEL_UM,
            "midline_ml_um": MIDLINE_UM,
            "minimum_distal_distance_um": DISTAL_UM,
            "vocabulary_root_node_id": ROOT_ID,
            "cerebral_cortex_node_id": CTX_ID,
            "major_division_node_ids": list(MAJOR_DIVISION_IDS),
            "excluded_subtree_node_ids": sorted(EXCLUDED_IDS),
            "distance_method": (
                "Exact-radius centre search at 1222 um; outward-rounded "
                "point-to-box bounds; exact rational threshold comparisons "
                "where bounds straddle 1200 um. Original decimal soma values "
                "determine sides, voxel membership, and exact comparisons."
            ),
            "family_gate_interpretation": (
                "At least three fixed major divisions must each retain at "
                "least three sided targets. This gates only secondary family "
                "analyses over exactly the supported divisions; all twelve "
                "division counts, including zeros, remain in reports."
            ),
            "n_selection_note": (
                "This precheck reports anatomical count gates only. Selecting "
                "n additionally requires every frozen development-only "
                "synthetic precision scenario to pass."
            ),
            "n_min": 3,
            "n_max": n_max,
            "animal_column_order": brains,
            "top_25_tie_break": "minimum N descending, node ID ascending, ipsi before contra",
        },
        "vocabulary": {
            "anatomical_size": len(targets),
            "sided_size": 2 * len(targets),
            "antichain_verified": True,
            "pairwise_disjoint_masks_verified": True,
            "eligible_labels_partition_verified": True,
            **vocabulary_report,
        },
        "target_to_division_map": [
            {
                "node_id": nid,
                "acronym": graph.nodes[nid]["acronym"],
                "name": graph.nodes[nid]["name"],
                "division_node_id": family_by_node[nid],
                "division_acronym": graph.nodes[family_by_node[nid]]["acronym"],
                "division_name": graph.nodes[family_by_node[nid]]["name"],
            }
            for nid in targets
        ],
        "V3": {
            "sided_target_count": len(v3_rows),
            "targets": [target_record(row, targets, graph) for row in v3_rows],
            "primary_gate_passes": bool(v3_rows),
            **v3_gate,
        },
        "V_n": survival,
        "diagnostics": {
            "per_sided_target": per_target_diagnostics,
            "top_25_by_animals_covered": [
                per_target_diagnostics[row] for row in ranked_rows
            ],
            "per_animal": [
                {
                    "brain": brain,
                    "ep09_role": animal_roles[brain],
                    "manifest_cell_count": cell_counts[brain],
                    "sided_targets_with_N_ge_3": int(
                        np.count_nonzero(counts[:, column] >= 3)
                    ),
                }
                for column, brain in enumerate(brains)
            ],
            "development_only_V3": role_v3(
                "development", brains, animal_roles, counts, targets, graph,
                family_by_node,
            ),
            "audit_only_V3": role_v3(
                "audit", brains, animal_roles, counts, targets, graph,
                family_by_node,
            ),
        },
        "cells_read": len(cells),
        "animals_read": len(brains),
        "somata_found": somata_found,
        "ambiguous_somata": ambiguous_count,
        "coordinate_versus_manifest_area_disagreements": disagreement_count,
        "somata_in_background_voxels": background_soma_count,
        "reviewer_edits": ["background soma voxel is non-fatal; coordinate area None"],
        "axon_or_dendrite_geometry_read": False,
    }

    write_outputs(out_dir, brains, targets, graph, counts, summary)

    secondary_gate_text = (
        "PASS" if v3_gate["passes_secondary_family_gate"] else "FAIL"
    )
    primary_gate_text = "PASS" if v3_rows else "FAIL"
    supported_text = ", ".join(
        f"{item['acronym']}={item['sided_target_count']}"
        for item in v3_gate["supported_divisions"]
    ) or "none"
    survival_text = ", ".join(
        f"{item['n']}:{item['sided_target_count']}" for item in survival
    ) or "none"

    print(
        f"{len(cells)} cells, {len(brains)} animals, {somata_found} somata; "
        f"{len(targets)} anatomical / {2 * len(targets)} sided targets; "
        f"unassigned voxels={vocabulary_report['total_unassigned_voxels']}."
    )
    print(
        f"V3={len(v3_rows)}; primary_gate={primary_gate_text}; "
        f"supported divisions at V3: {supported_text}; "
        f"secondary_family_gate={secondary_gate_text}; "
        f"ambiguous={ambiguous_count}; area disagreements={disagreement_count}."
    )
    print(f"V_n counts (n:count): {survival_text}.")
    print(f"Read-only outputs: {out_dir}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("ERROR: interrupted.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
