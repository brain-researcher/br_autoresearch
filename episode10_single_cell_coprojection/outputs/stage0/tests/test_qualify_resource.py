from __future__ import annotations

import importlib.util
import io
from unittest import mock
import json
from pathlib import Path
import tempfile
import unittest
import xml.sax.saxutils as saxutils
import zipfile


TOOL = Path(__file__).parents[1] / "tools" / "qualify_resource.py"
SPEC = importlib.util.spec_from_file_location("qualify_resource", TOOL)
assert SPEC and SPEC.loader
qualifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qualifier)


HEADERS = [
    "Morphology Name",
    "fMOST Brain ID",
    "Soma_X(Raw brain, in voxel)",
    "Soma_Y(Raw brain, in voxel)",
    "Soma_Z(Raw brain, in voxel)",
    "Soma_X(CCFv3_1𝜇𝑚)",
    "Soma_Y(CCFv3_1𝜇𝑚)",
    "Soma_Z(CCFv3_1𝜇𝑚)",
    "isManuallyChecked",
    "Soma region",
    "Projection class",
    "Cortical Lamination of soma",
]


def column_name(index: int) -> str:
    value = index + 1
    out = ""
    while value:
        value, remainder = divmod(value - 1, 26)
        out = chr(ord("A") + remainder) + out
    return out


def make_xlsx(
    path: Path,
    groups: int,
    duplicate: bool = False,
    extra_column: bool = False,
    unheaded_data: bool = False,
) -> list[str]:
    headers = list(HEADERS)
    if extra_column:
        headers.append("Unexpected outcome")
    table = [headers]
    cell_ids = []
    for index in range(groups):
        cell_id = "cell_000" if duplicate and index == groups - 1 else f"cell_{index:03d}"
        cell_ids.append(cell_id)
        values = {
            "Morphology Name": cell_id,
            "fMOST Brain ID": f"brain_{index:03d}",
            "Soma_X(Raw brain, in voxel)": str(index + 1),
            "Soma_Y(Raw brain, in voxel)": str(index + 2),
            "Soma_Z(Raw brain, in voxel)": str(index + 3),
            "Soma_X(CCFv3_1𝜇𝑚)": str(index + 4),
            "Soma_Y(CCFv3_1𝜇𝑚)": str(index + 5),
            "Soma_Z(CCFv3_1𝜇𝑚)": str(index + 6),
            "isManuallyChecked": "true",
            "Soma region": "MOp",
            "Projection class": "SECRET_PROJECTION_PAIR",
            "Cortical Lamination of soma": "L5",
            "Unexpected outcome": "SECRET_UNALLOWLISTED_VALUE",
        }
        row = [values[header] for header in headers]
        if unheaded_data:
            row.append("SECRET_UNHEADED_VALUE")
        table.append(row)
    shared = []
    positions = {}
    for row in table:
        for value in row:
            if value not in positions:
                positions[value] = len(shared)
                shared.append(value)
    shared_xml = "".join(
        f"<si><t>{saxutils.escape(value)}</t></si>" for value in shared
    )
    row_xml = []
    for row_index, row in enumerate(table, start=1):
        cells = []
        for column_index, value in enumerate(row):
            ref = f"{column_name(column_index)}{row_index}"
            cells.append(f'<c r="{ref}" t="s"><v>{positions[value]}</v></c>')
        row_xml.append(f'<row r="{row_index}">{"".join(cells)}</row>')
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "xl/sharedStrings.xml",
            '<?xml version="1.0"?><sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            + shared_xml
            + "</sst>",
        )
        archive.writestr(
            "xl/workbook.xml",
            '<?xml version="1.0"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>'
            '<sheet name="metadata" sheetId="1" r:id="rId1"/></sheets></workbook>',
        )
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            '<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="worksheet" Target="worksheets/sheet1.xml"/></Relationships>',
        )
        archive.writestr(
            "xl/worksheets/sheet1.xml",
            '<?xml version="1.0"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>'
            + "".join(row_xml)
            + "</sheetData></worksheet>",
        )
    return cell_ids


def make_swc_zip(path: Path, names: list[str], traversal: bool = False) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name in names:
            archive.writestr(f"nested/{name}.swc", "1 1 0 0 0 1 -1\n")
        if traversal:
            archive.writestr("../escape.swc", "1 1 0 0 0 1 -1\n")


class QualifierTests(unittest.TestCase):
    def test_twenty_groups_pass_and_forbidden_value_never_emitted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = root / "metadata.xlsx"
            names = make_xlsx(metadata, 20)
            morphology = root / "morphology.zip"
            make_swc_zip(morphology, names)
            output = root / "out"
            original_decode = qualifier._decode_excel_escapes

            def guarded_decode(value: str) -> str:
                if "SECRET_PROJECTION_PAIR" in value:
                    raise AssertionError("forbidden shared string was dereferenced")
                return original_decode(value)

            with mock.patch.object(
                qualifier, "_decode_excel_escapes", side_effect=guarded_decode
            ):
                result = qualifier.qualify(metadata, output, morphology_zip=morphology)
            self.assertTrue(result["priority_source_group_gate_passed"])
            self.assertTrue(result["qualifier_structurally_valid"])
            self.assertFalse(result["metadata"]["projection_class_values_decoded"])
            self.assertFalse(result["metadata"]["forbidden_data_cells_dereferenced"])
            self.assertNotIn("forbidden_data_cell_count", result["metadata"])
            for path in output.iterdir():
                self.assertNotIn("SECRET_PROJECTION_PAIR", path.read_text())

    def test_nineteen_groups_require_resource_revision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = root / "metadata.xlsx"
            make_xlsx(metadata, 19)
            result = qualifier.qualify(metadata, root / "out")
            self.assertFalse(result["priority_source_group_gate_passed"])
            self.assertEqual(
                result["decision"], "revise_resource_strategy_before_outcome_search"
            )

    def test_duplicate_metadata_identity_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = root / "metadata.xlsx"
            make_xlsx(metadata, 20, duplicate=True)
            result = qualifier.qualify(metadata, root / "out")
            self.assertFalse(result["qualifier_structurally_valid"])
            issues = (root / "out" / "identity_issues.csv").read_text()
            self.assertIn("duplicate_metadata_cell_id", issues)

    def test_zip_traversal_and_identity_mismatch_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = root / "metadata.xlsx"
            names = make_xlsx(metadata, 20)
            morphology = root / "morphology.zip"
            make_swc_zip(morphology, names[:-1] + ["unexpected"], traversal=True)
            result = qualifier.qualify(metadata, root / "out", morphology_zip=morphology)
            self.assertFalse(result["qualifier_structurally_valid"])
            issues = (root / "out" / "identity_issues.csv").read_text()
            self.assertIn("zip_path_traversal", issues)
            self.assertIn("metadata_cell_missing_from_zip", issues)
            self.assertIn("zip_cell_missing_from_metadata", issues)

    def test_zip_member_payloads_are_not_opened(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            morphology = root / "morphology.zip"
            names = ["cell_000", "cell_001"]
            make_swc_zip(morphology, names)
            with mock.patch.object(
                zipfile.ZipFile,
                "open",
                side_effect=AssertionError("ZIP member payload opened"),
            ):
                summary, issues = qualifier.inspect_zip_names(morphology, names)
            self.assertTrue(summary["identity_sets_match"])
            self.assertEqual(issues, [])

    def test_unknown_metadata_column_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = root / "metadata.xlsx"
            make_xlsx(metadata, 20, extra_column=True)
            with self.assertRaisesRegex(
                qualifier.QualificationError, "not allowlisted"
            ):
                qualifier.qualify(metadata, root / "out")

    def test_unheaded_data_column_is_rejected_before_value_decode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = root / "metadata.xlsx"
            make_xlsx(metadata, 20, unheaded_data=True)
            original_decode = qualifier._decode_excel_escapes

            def guarded_decode(value: str) -> str:
                if "SECRET_UNHEADED_VALUE" in value:
                    raise AssertionError("unheaded shared string was dereferenced")
                return original_decode(value)

            with mock.patch.object(
                qualifier, "_decode_excel_escapes", side_effect=guarded_decode
            ):
                with self.assertRaisesRegex(
                    qualifier.QualificationError, "no allowlisted header"
                ):
                    qualifier.qualify(metadata, root / "out")

    def test_nrrd_header_does_not_decode_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "tiny.nrrd"
            payload = (
                b"NRRD0005\n"
                b"type: uint16\n"
                b"dimension: 3\n"
                b"space: left-posterior-superior\n"
                b"sizes: 1 1 1\n"
                b"encoding: raw\n\n"
                b"SECRET_VOXEL_PAYLOAD"
            )
            path.write_bytes(payload)
            header_end = payload.index(b"\n\n") + 2

            class HeaderReadGuard:
                def __init__(self, content: bytes) -> None:
                    self._stream = io.BytesIO(content)

                def __enter__(self) -> "HeaderReadGuard":
                    return self

                def __exit__(self, *args: object) -> None:
                    self._stream.close()

                def readline(self, *args: object, **kwargs: object) -> bytes:
                    if self._stream.tell() >= header_end:
                        raise AssertionError("NRRD voxel payload read attempted")
                    line = self._stream.readline(*args, **kwargs)
                    if self._stream.tell() > header_end:
                        raise AssertionError("NRRD header read crossed into voxel payload")
                    return line

                def __getattr__(self, name: str) -> object:
                    if name.startswith("read"):
                        raise AssertionError(
                            f"unexpected NRRD read method could access payload: {name}"
                        )
                    return getattr(self._stream, name)

            def guarded_open(
                open_path: Path, mode: str = "r", *args: object, **kwargs: object
            ) -> HeaderReadGuard:
                if open_path != path or mode != "rb":
                    raise AssertionError(f"unexpected file open: {open_path} {mode}")
                return HeaderReadGuard(payload)

            with mock.patch.object(Path, "open", new=guarded_open):
                summary = qualifier.inspect_nrrd_header(path)
            self.assertFalse(summary["voxel_payload_decoded"])
            self.assertEqual(summary["header_bytes_returned"], header_end)
            self.assertNotIn("SECRET", json.dumps(summary))

    def test_spreadsheetml_escape_is_normalized_for_identity_join(self) -> None:
        self.assertEqual(
            qualifier._decode_excel_escapes("17109_1701_x005F_x8048_y22277"),
            "17109_1701_x8048_y22277",
        )


if __name__ == "__main__":
    unittest.main()
