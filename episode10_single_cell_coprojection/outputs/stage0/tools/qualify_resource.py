#!/usr/bin/env python3
"""Outcome-blind Stage-0 qualifier for the EP10 SEU resource.

The program deliberately never reads SWC member contents or NRRD voxel data.
It also skips the forbidden ``Projection class`` cell before decoding its XLSX
value.  Its outputs are aggregate resource-readiness evidence, not scientific
results or a development/final role assignment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import Any, Iterable
import xml.etree.ElementTree as ET
import zipfile


SHEET_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
FORBIDDEN_FIELDS = {"Projection class"}
CELL_ID_FIELD = "Morphology Name"
GROUP_FIELD = "fMOST Brain ID"
SOURCE_FIELD = "Soma region"
ALLOWED_FIELDS = {
    CELL_ID_FIELD,
    GROUP_FIELD,
    "Soma_X(Raw brain, in voxel)",
    "Soma_Y(Raw brain, in voxel)",
    "Soma_Z(Raw brain, in voxel)",
    "Soma_X(CCFv3_1𝜇𝑚)",
    "Soma_Y(CCFv3_1𝜇𝑚)",
    "Soma_Z(CCFv3_1𝜇𝑚)",
    "isManuallyChecked",
    SOURCE_FIELD,
    "Cortical Lamination of soma",
}
REQUIRED_FIELDS = ALLOWED_FIELDS | FORBIDDEN_FIELDS
CELL_REF_RE = re.compile(r"^([A-Z]+)([0-9]+)$")
EXCEL_ESCAPE_RE = re.compile(r"_x([0-9A-Fa-f]{4})_")


class QualificationError(RuntimeError):
    """Raised when a source cannot be parsed safely."""


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _column_index(cell_ref: str) -> int:
    match = CELL_REF_RE.match(cell_ref)
    if not match:
        raise QualificationError(f"invalid XLSX cell reference: {cell_ref!r}")
    value = 0
    for char in match.group(1):
        value = value * 26 + ord(char) - ord("A") + 1
    return value - 1


def _selected_shared_strings(
    archive: zipfile.ZipFile, wanted_indexes: set[int]
) -> dict[int, str]:
    """Resolve only explicitly selected shared-string indexes.

    In particular, indexes referenced solely by forbidden data cells are never
    joined into application strings or retained by the qualifier.
    """

    name = "xl/sharedStrings.xml"
    if not wanted_indexes:
        return {}
    if name not in archive.namelist():
        raise QualificationError("workbook references absent sharedStrings.xml")
    strings: dict[int, str] = {}
    index = 0
    with archive.open(name) as handle:
        for _, item in ET.iterparse(handle, events=("end",)):
            if item.tag != f"{{{SHEET_NS}}}si":
                continue
            if index in wanted_indexes:
                strings[index] = _decode_excel_escapes(
                    "".join(
                        node.text or ""
                        for node in item.iter(f"{{{SHEET_NS}}}t")
                    )
                )
            index += 1
            item.clear()
            if len(strings) == len(wanted_indexes) and index > max(wanted_indexes):
                break
    missing = wanted_indexes - set(strings)
    if missing:
        raise QualificationError(
            f"invalid XLSX shared-string references: {sorted(missing)[:5]}"
        )
    return strings


def _first_sheet_path(archive: zipfile.ZipFile) -> str:
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    sheets = workbook.find(f"{{{SHEET_NS}}}sheets")
    if sheets is None or len(sheets) != 1:
        raise QualificationError("metadata workbook must contain exactly one sheet")
    relation_id = sheets[0].attrib[f"{{{REL_NS}}}id"]
    relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    target = None
    for relation in relationships.findall(f"{{{PKG_REL_NS}}}Relationship"):
        if relation.attrib.get("Id") == relation_id:
            target = relation.attrib.get("Target")
            break
    if not target:
        raise QualificationError("could not resolve metadata worksheet")
    if target.startswith("/"):
        resolved = target.lstrip("/")
    else:
        resolved = str(PurePosixPath("xl") / target)
    parts: list[str] = []
    for part in PurePosixPath(resolved).parts:
        if part == "..":
            if not parts:
                raise QualificationError("worksheet relationship escapes workbook")
            parts.pop()
        elif part != ".":
            parts.append(part)
    return "/".join(parts)


def _decode_excel_escapes(value: str) -> str:
    """Decode one pass of SpreadsheetML's ``_xHHHH_`` string escaping."""

    return EXCEL_ESCAPE_RE.sub(lambda match: chr(int(match.group(1), 16)), value)


def _shared_string_index(cell: ET.Element) -> int | None:
    if cell.attrib.get("t") != "s":
        return None
    value_node = cell.find(f"{{{SHEET_NS}}}v")
    if value_node is None or not (value_node.text or ""):
        return None
    try:
        return int(value_node.text or "")
    except ValueError as exc:
        raise QualificationError("invalid XLSX shared-string reference") from exc


def _decode_cell(cell: ET.Element, shared: dict[int, str]) -> str:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return _decode_excel_escapes(
            "".join(node.text or "" for node in cell.iter(f"{{{SHEET_NS}}}t"))
        )
    value_node = cell.find(f"{{{SHEET_NS}}}v")
    value = "" if value_node is None else (value_node.text or "")
    if cell_type == "s" and value:
        try:
            return shared[int(value)]
        except (ValueError, KeyError) as exc:
            raise QualificationError("unresolved XLSX shared-string reference") from exc
    if cell_type == "b":
        return "true" if value == "1" else "false"
    return value


def read_redacted_metadata(path: Path) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Read allowed XLSX columns while never decoding forbidden data cells."""

    with zipfile.ZipFile(path) as archive:
        root = ET.fromstring(archive.read(_first_sheet_path(archive)))
        rows = root.findall(f".//{{{SHEET_NS}}}sheetData/{{{SHEET_NS}}}row")
        if not rows:
            raise QualificationError("metadata workbook contains no rows")

        header_cells = rows[0].findall(f"{{{SHEET_NS}}}c")
        header_shared_indexes = {
            index
            for cell in header_cells
            if (index := _shared_string_index(cell)) is not None
        }
        header_shared = _selected_shared_strings(archive, header_shared_indexes)
        header_by_index: dict[int, str] = {}
        for cell in header_cells:
            index = _column_index(cell.attrib["r"])
            header_by_index[index] = _decode_cell(cell, header_shared).strip()
        headers = [header_by_index[i] for i in sorted(header_by_index)]
        if len(headers) != len(set(headers)):
            raise QualificationError("metadata workbook contains duplicate headers")
        missing = sorted(REQUIRED_FIELDS - set(headers))
        if missing:
            raise QualificationError(f"required metadata fields missing: {missing}")
        unknown = sorted(set(headers) - ALLOWED_FIELDS - FORBIDDEN_FIELDS)
        if unknown:
            raise QualificationError(f"metadata fields are not allowlisted: {unknown}")
        forbidden_indexes = {
            index for index, name in header_by_index.items() if name in FORBIDDEN_FIELDS
        }
        allowed_indexes = {
            index for index, name in header_by_index.items() if name in ALLOWED_FIELDS
        }
        if not forbidden_indexes:
            raise QualificationError("forbidden Projection class column was not found")

        allowed_data_shared_indexes: set[int] = set()
        for row in rows[1:]:
            for cell in row.findall(f"{{{SHEET_NS}}}c"):
                column = _column_index(cell.attrib["r"])
                if column in forbidden_indexes:
                    continue
                if column not in allowed_indexes:
                    raise QualificationError(
                        f"metadata data cell has no allowlisted header: {cell.attrib['r']}"
                    )
                shared_index = _shared_string_index(cell)
                if shared_index is not None:
                    allowed_data_shared_indexes.add(shared_index)
        shared = dict(header_shared)
        shared.update(_selected_shared_strings(archive, allowed_data_shared_indexes))

        records: list[dict[str, str]] = []
        for row in rows[1:]:
            record: dict[str, str] = {
                name: ""
                for index, name in header_by_index.items()
                if index not in forbidden_indexes and name in ALLOWED_FIELDS
            }
            for cell in row.findall(f"{{{SHEET_NS}}}c"):
                index = _column_index(cell.attrib["r"])
                # This branch is the core leakage guard: the forbidden cell's
                # shared-string or inline value is never dereferenced.
                if index in forbidden_indexes:
                    continue
                name = header_by_index.get(index)
                if index not in allowed_indexes or name is None:
                    raise QualificationError(
                        f"metadata data cell has no allowlisted header: {cell.attrib['r']}"
                    )
                record[name] = _decode_cell(cell, shared).strip()
            if any(record.values()):
                records.append(record)

    schema = {
        "sheet_count": 1,
        "row_count": len(records),
        "observed_headers": headers,
        "emitted_headers": [h for h in headers if h in ALLOWED_FIELDS],
        "forbidden_headers": sorted(FORBIDDEN_FIELDS),
        "forbidden_data_cells_dereferenced": False,
        "retained_shared_string_count": len(shared),
        "projection_class_values_decoded": False,
        "projection_class_values_emitted": False,
        "projection_class_decode_semantics": (
            "forbidden data cells are not looked up or retained by qualifier logic; "
            "the XML transport parser still processes the workbook stream"
        ),
    }
    return records, schema


def _normalize_cell_id(value: str) -> str:
    name = PurePosixPath(value.replace("\\", "/")).name.strip()
    if name.lower().endswith(".swc"):
        name = name[:-4]
    return name.casefold()


def inspect_zip_names(
    path: Path, expected_cell_ids: Iterable[str]
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Inspect only archive member names; never open an SWC member."""

    issues: list[dict[str, str]] = []
    seen: dict[str, str] = {}
    member_count = 0
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            member = PurePosixPath(info.filename)
            if info.is_dir():
                continue
            if member.is_absolute() or ".." in member.parts:
                issues.append(
                    {
                        "issue_type": "zip_path_traversal",
                        "cell_id": "",
                        "detail": info.filename,
                    }
                )
                continue
            if member.suffix.casefold() != ".swc":
                continue
            member_count += 1
            normalized = _normalize_cell_id(member.name)
            if normalized in seen:
                issues.append(
                    {
                        "issue_type": "duplicate_zip_cell_id",
                        "cell_id": normalized,
                        "detail": f"{seen[normalized]} | {info.filename}",
                    }
                )
            else:
                seen[normalized] = info.filename

    expected = {_normalize_cell_id(value) for value in expected_cell_ids if value}
    observed = set(seen)
    for cell_id in sorted(expected - observed):
        issues.append(
            {
                "issue_type": "metadata_cell_missing_from_zip",
                "cell_id": cell_id,
                "detail": path.name,
            }
        )
    for cell_id in sorted(observed - expected):
        issues.append(
            {
                "issue_type": "zip_cell_missing_from_metadata",
                "cell_id": cell_id,
                "detail": path.name,
            }
        )
    summary = {
        "path": str(path),
        "swc_member_count": member_count,
        "unique_normalized_cell_ids": len(observed),
        "member_names_inspected": True,
        "swc_contents_opened": False,
        "identity_sets_match": observed == expected and not any(
            issue["issue_type"].startswith(("zip_path", "duplicate_zip"))
            for issue in issues
        ),
    }
    return summary, issues


def inspect_nrrd_header(path: Path) -> dict[str, Any]:
    fields: dict[str, str] = {}
    returned_bytes = 0
    with path.open("rb") as handle:
        first = handle.readline()
        returned_bytes += len(first)
        if not first.startswith(b"NRRD"):
            raise QualificationError(f"not an NRRD file: {path}")
        for _ in range(4096):
            line = handle.readline()
            returned_bytes += len(line)
            if returned_bytes > 1024 * 1024:
                raise QualificationError(f"NRRD header exceeds 1 MiB: {path}")
            if line in (b"", b"\n", b"\r\n"):
                break
            text = line.decode("ascii", errors="strict").strip()
            if not text or text.startswith("#") or ":" not in text:
                continue
            key, value = text.split(":", 1)
            fields[key.strip()] = value.strip()
        else:
            raise QualificationError(f"NRRD header is unterminated: {path}")
    retained = {
        key: fields[key]
        for key in (
            "type",
            "dimension",
            "space",
            "sizes",
            "space directions",
            "space origin",
            "encoding",
            "endian",
        )
        if key in fields
    }
    return {
        "path": str(path),
        "header": retained,
        "header_bytes_returned": returned_bytes,
        "voxel_payload_decoded": False,
    }


def _count_structure_nodes(value: Any) -> int:
    if isinstance(value, dict):
        own = 1 if "id" in value and ("name" in value or "acronym" in value) else 0
        return own + sum(_count_structure_nodes(item) for item in value.values())
    if isinstance(value, list):
        return sum(_count_structure_nodes(item) for item in value)
    return 0


def inspect_structure_graph(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {
        "path": str(path),
        "success": payload.get("success") if isinstance(payload, dict) else None,
        "structure_node_count": _count_structure_nodes(payload),
        "sha256": sha256_file(path),
    }


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _csv_text(fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def qualify(
    metadata: Path,
    output_dir: Path,
    morphology_zip: Path | None = None,
    nrrds: Iterable[Path] = (),
    structure_graph: Path | None = None,
    priority_source: str = "MOp",
    minimum_development_groups: int = 12,
    minimum_final_groups: int = 8,
    expected_metadata_sha256: str | None = None,
) -> dict[str, Any]:
    metadata = metadata.resolve()
    if expected_metadata_sha256:
        actual = sha256_file(metadata)
        if actual != expected_metadata_sha256:
            raise QualificationError(
                f"metadata SHA-256 mismatch: expected {expected_metadata_sha256}, got {actual}"
            )
    else:
        actual = sha256_file(metadata)

    records, schema = read_redacted_metadata(metadata)
    issues: list[dict[str, str]] = []
    seen_cells: set[str] = set()
    group_sets: dict[str, set[str]] = {}
    cell_counts: dict[str, int] = {}
    position_complete: dict[str, int] = {}
    layer_nonempty: dict[str, int] = {}
    safe_coordinate_fields = [
        name for name in schema["emitted_headers"] if name.startswith("Soma_") and "CCF" in name
    ]

    for row_number, record in enumerate(records, start=2):
        cell_id = record.get(CELL_ID_FIELD, "").strip()
        group_id = record.get(GROUP_FIELD, "").strip()
        source = record.get(SOURCE_FIELD, "").strip()
        normalized = _normalize_cell_id(cell_id) if cell_id else ""
        if not cell_id:
            issues.append(
                {"issue_type": "missing_cell_id", "cell_id": "", "detail": str(row_number)}
            )
        elif normalized in seen_cells:
            issues.append(
                {
                    "issue_type": "duplicate_metadata_cell_id",
                    "cell_id": normalized,
                    "detail": str(row_number),
                }
            )
        else:
            seen_cells.add(normalized)
        if not group_id:
            issues.append(
                {
                    "issue_type": "missing_conservative_group",
                    "cell_id": normalized,
                    "detail": str(row_number),
                }
            )
        if not source:
            issues.append(
                {
                    "issue_type": "missing_soma_region",
                    "cell_id": normalized,
                    "detail": str(row_number),
                }
            )
            source = "<missing>"
        group_sets.setdefault(source, set())
        if group_id:
            group_sets[source].add(group_id)
        cell_counts[source] = cell_counts.get(source, 0) + 1
        if safe_coordinate_fields and all(record.get(name, "") for name in safe_coordinate_fields):
            position_complete[source] = position_complete.get(source, 0) + 1
        if record.get("Cortical Lamination of soma", ""):
            layer_nonempty[source] = layer_nonempty.get(source, 0) + 1

    zip_summary = None
    if morphology_zip is not None:
        zip_summary, zip_issues = inspect_zip_names(
            morphology_zip.resolve(),
            (record.get(CELL_ID_FIELD, "") for record in records),
        )
        issues.extend(zip_issues)

    minimum_total = minimum_development_groups + minimum_final_groups
    support_rows: list[dict[str, Any]] = []
    for source in sorted(
        cell_counts, key=lambda key: (-len(group_sets[key]), -cell_counts[key], key)
    ):
        groups = len(group_sets[source])
        support_rows.append(
            {
                "source": source,
                "cells": cell_counts[source],
                "conservative_specimen_groups": groups,
                "complete_ccf_positions": position_complete.get(source, 0),
                "nonempty_soma_layer": layer_nonempty.get(source, 0),
                "minimum_total_groups": minimum_total,
                "supports_12_development_8_final": str(groups >= minimum_total).lower(),
                "maximum_final_groups_after_minimum_development": max(
                    0, groups - minimum_development_groups
                ),
            }
        )
    priority = next((row for row in support_rows if row["source"] == priority_source), None)
    structurally_valid = not issues
    priority_group_gate = bool(
        priority and priority["conservative_specimen_groups"] >= minimum_total
    )
    decision = (
        "continue_remaining_outcome_blind_gates"
        if structurally_valid and priority_group_gate
        else "revise_resource_strategy_before_outcome_search"
    )

    nrrd_summaries = [inspect_nrrd_header(path.resolve()) for path in nrrds]
    atlas_header_match = None
    if len(nrrd_summaries) >= 2:
        keys = ("space", "sizes", "space directions", "space origin")
        first = nrrd_summaries[0]["header"]
        atlas_header_match = all(
            all(summary["header"].get(key) == first.get(key) for key in keys)
            for summary in nrrd_summaries[1:]
        )
    structure_summary = (
        inspect_structure_graph(structure_graph.resolve()) if structure_graph else None
    )

    result: dict[str, Any] = {
        "schema_version": "ep10.stage0_qualification_result.v1",
        "metadata": {
            "path": str(metadata),
            "sha256": actual,
            **schema,
            "unique_normalized_cell_ids": len(seen_cells),
            "unique_conservative_specimen_groups": len(
                {record.get(GROUP_FIELD, "") for record in records if record.get(GROUP_FIELD, "")}
            ),
            "group_interpretation": "conservative_specimen_group_not_verified_animal",
        },
        "morphology_zip": zip_summary,
        "atlas": {
            "nrrd_headers": nrrd_summaries,
            "shared_header_geometry": atlas_header_match,
            "structure_graph": structure_summary,
            "seu_transform_compatibility_validated": False,
            "voxel_payloads_opened": False,
        },
        "requirements": {
            "minimum_development_groups": minimum_development_groups,
            "minimum_final_groups": minimum_final_groups,
            "minimum_total_groups_per_source": minimum_total,
            "priority_source": priority_source,
        },
        "priority_source_support": priority,
        "eligible_sources_by_group_count_only": [
            row["source"]
            for row in support_rows
            if row["conservative_specimen_groups"] >= minimum_total
        ],
        "identity_issue_count": len(issues),
        "qualifier_structurally_valid": structurally_valid,
        "priority_source_group_gate_passed": priority_group_gate,
        "decision": decision,
        "shared_ep09_ep10_ep11_role_ledger_frozen": False,
        "separate_development_and_final_views_emitted": False,
        "morphology_outcomes_opened": False,
        "final_outcomes_opened": False,
        "scientific_validity_established": False,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    qualification_path = output_dir / "qualification.json"
    support_path = output_dir / "source_group_support.csv"
    issues_path = output_dir / "identity_issues.csv"
    _atomic_write(qualification_path, json.dumps(result, indent=2, sort_keys=True) + "\n")
    _atomic_write(
        support_path,
        _csv_text(
            [
                "source",
                "cells",
                "conservative_specimen_groups",
                "complete_ccf_positions",
                "nonempty_soma_layer",
                "minimum_total_groups",
                "supports_12_development_8_final",
                "maximum_final_groups_after_minimum_development",
            ],
            support_rows,
        ),
    )
    _atomic_write(
        issues_path,
        _csv_text(["issue_type", "cell_id", "detail"], issues),
    )
    manifest_lines = []
    for path in (qualification_path, support_path, issues_path):
        manifest_lines.append(f"{sha256_file(path)}  {path.name}")
    _atomic_write(output_dir / "result_manifest.sha256", "\n".join(manifest_lines) + "\n")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--morphology-zip", type=Path)
    parser.add_argument("--nrrd", type=Path, action="append", default=[])
    parser.add_argument("--structure-graph", type=Path)
    parser.add_argument("--priority-source", default="MOp")
    parser.add_argument("--minimum-development-groups", type=int, default=12)
    parser.add_argument("--minimum-final-groups", type=int, default=8)
    parser.add_argument("--expected-metadata-sha256")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = qualify(
        metadata=args.metadata,
        output_dir=args.output_dir,
        morphology_zip=args.morphology_zip,
        nrrds=args.nrrd,
        structure_graph=args.structure_graph,
        priority_source=args.priority_source,
        minimum_development_groups=args.minimum_development_groups,
        minimum_final_groups=args.minimum_final_groups,
        expected_metadata_sha256=args.expected_metadata_sha256,
    )
    print(json.dumps({"ok": True, "decision": result["decision"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
