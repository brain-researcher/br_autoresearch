"""Minimal XLSX readers that decode only explicitly requested cells.

Shared strings are decoded only for indices referenced by requested cells, so
non-whitelisted columns (for example an axon-derived class) are never read
into Python strings.
"""
import re
import xml.etree.ElementTree as ET
import zipfile

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
REL_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
PKG_REL_NS = "{http://schemas.openxmlformats.org/package/2006/relationships}"


_ESC = re.compile(r"_x([0-9A-Fa-f]{4})_")


def ooxml_unescape(text):
    """Decode OOXML _xHHHH_ escapes (e.g. _x005F_ is a literal underscore)."""
    return _ESC.sub(lambda m: chr(int(m.group(1), 16)), text) if text else text


def col_of(ref):
    return re.sub(r"[0-9]", "", ref)


def sheet_paths(z):
    """Return [(sheet name, worksheet member path)] in workbook order."""
    wb = ET.fromstring(z.read("xl/workbook.xml"))
    rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
    targets = {r.attrib["Id"]: r.attrib["Target"] for r in rels.findall(PKG_REL_NS + "Relationship")}
    out = []
    for s in wb.findall(NS + "sheets/" + NS + "sheet"):
        target = targets[s.attrib[REL_NS + "id"]]
        target = target.lstrip("/")
        if not target.startswith("xl/"):
            target = "xl/" + target
        out.append((s.attrib["name"], target))
    return out


def _raw_cell(cell):
    t = cell.attrib.get("t")
    if t == "inlineStr":
        return ("str", ooxml_unescape("".join(cell.itertext())))
    v = cell.find(NS + "v")
    if v is None or v.text is None:
        return None
    if t == "s":
        return ("shared", int(v.text))
    return ("value", v.text)


def resolve_shared(z, wanted):
    strings = {}
    if not wanted or "xl/sharedStrings.xml" not in z.namelist():
        return strings
    with z.open("xl/sharedStrings.xml") as handle:
        index = 0
        for _, element in ET.iterparse(handle, events=("end",)):
            if element.tag == NS + "si":
                if index in wanted:
                    strings[index] = ooxml_unescape("".join(element.itertext()))
                element.clear()
                index += 1
                if wanted <= strings.keys():
                    break
    return strings


def header_rows(path, max_rows=5, min_cells=3):
    """Per sheet: rows up to and including the first row with >= min_cells cells."""
    report = []
    with zipfile.ZipFile(path) as z:
        for name, member in sheet_paths(z):
            rows, wanted, n_rows, dim = [], set(), 0, None
            with z.open(member) as handle:
                for _, element in ET.iterparse(handle, events=("end",)):
                    if element.tag == NS + "dimension":
                        dim = element.attrib.get("ref")
                    if element.tag == NS + "row":
                        n_rows += 1
                        cells = []
                        for cell in element.findall(NS + "c"):
                            raw = _raw_cell(cell)
                            if raw is not None:
                                cells.append((cell.attrib["r"], raw))
                                if raw[0] == "shared":
                                    wanted.add(raw[1])
                        rows.append((element.attrib.get("r"), cells))
                        element.clear()
                        if len(cells) >= min_cells or n_rows >= max_rows:
                            break
            strings = resolve_shared(z, wanted)
            out_rows = []
            for r, cells in rows:
                out_rows.append({"row": r, "cells": [
                    {"ref": ref, "value": strings[raw[1]] if raw[0] == "shared" else raw[1]}
                    for ref, raw in cells]})
            report.append({"sheet": name, "member": member, "dimension": dim, "rows": out_rows})
    return report


def whitelisted_rows(path, member, columns, header_row):
    """Decode only `columns` (letters) for every row after `header_row`."""
    columns = set(columns)
    records, wanted = [], set()
    with zipfile.ZipFile(path) as z:
        with z.open(member) as handle:
            for _, element in ET.iterparse(handle, events=("end",)):
                if element.tag != NS + "row":
                    continue
                r = int(element.attrib["r"])
                if r <= header_row:
                    element.clear()
                    continue
                rec = {"_row": r}
                for cell in element.findall(NS + "c"):
                    col = col_of(cell.attrib["r"])
                    if col not in columns:
                        continue  # never decode a non-whitelisted column
                    raw = _raw_cell(cell)
                    rec[col] = raw
                    if raw is not None and raw[0] == "shared":
                        wanted.add(raw[1])
                element.clear()
                records.append(rec)
        strings = resolve_shared(z, wanted)
    for rec in records:
        for col in list(rec):
            if col == "_row":
                continue
            raw = rec[col]
            if raw is None:
                rec[col] = None
            elif raw[0] == "shared":
                rec[col] = strings[raw[1]]
            else:
                rec[col] = raw[1]
    return records
