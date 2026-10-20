#!/usr/bin/env python3
"""Steward-only, aggregate-only prelock structural gate on the eleven EP02 primary mice.

Checks the primary-mouse part of `prelock_structural_support_gate` in
`coordinate_and_outcome_contract.primary_mouse_score`, using the eligibility,
session and code definitions frozen in policy revision 6
(`source_definitions_20261016`). Run it as the steward inside a Slurm allocation.

The provider MAT stores all 24 records in one compressed variable, in record
order. This program inflates that stream sequentially and never past the bytes
it requests. It decodes `trialID`, `seshID` and only the finiteness of
`lickState` for primary records 3, 5, 7, 8, 10, 12, 13, 14, 17, 18 and 20. The
same three fields of the nine development controls are decoded only to
cross-check the parser bit for bit against the released development pack. The
bytes of every other field, including `stimState` and `lick`, are discarded
unparsed, and inflation stops at the end of record 20, so boundary records
21-24 are never decompressed.

Per-mouse results stay in memory. The output packet holds only an overall
pass/fail and arm-level aggregate counts: no per-mouse value, no `lickState`
or `lick` value, and no audit pack.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import materialize_roles as trusted  # noqa: E402
from materialize_development_pack import verify_source  # noqa: E402

MI_INT8, MI_INT32, MI_UINT32, MI_MATRIX, MI_COMPRESSED = 1, 5, 6, 14, 15
STORAGE_DTYPES = {1: "<i1", 2: "<u1", 3: "<i2", 4: "<u2", 5: "<i4", 6: "<u4",
                  7: "<f4", 9: "<f8", 12: "<i8", 13: "<u8"}
STRUCT_CLASS = 2
NUMERIC_CLASSES = frozenset(range(6, 16))  # double, single and the integer classes
COMPLEX_FLAG = 0x0800
CHUNK = 1 << 20

VARIABLE = "seshMerge"
COLUMNS = 24
LAST_DECODED_RECORD = 20
DECODED_FIELDS = ("trialID", "seshID", "lickState")
CODES = frozenset({0, 1, 2})
EPOCHS = (1, 8)
MIN_ELIGIBLE = 30
DEVELOPMENT = [1, 2, 4, 6, 9, 11, 15, 16, 19]
ARMS = {"stimLick_minus": ("audit_calibrated_stim_lick_minus", [3, 5, 10, 14, 18, 20]),
        "stimLick_plus": ("audit_calibrated_stim_lick_plus", [7, 8, 12, 13, 17])}
BOUNDARY = [21, 22, 23, 24]
CRITERIA = ("equal_field_lengths", "trialID_codes_within_0_1_2", "seshID_contiguous_1_to_K_in_row_order",
            "sessions_1_and_8_present", "at_least_30_eligible_in_sessions_1_and_8")
COUNTS = ("eligible_trials_session_1", "eligible_trials_session_8",
          "sessions_with_nonfinite_lickState_rows", "nonfinite_lickState_rows_in_sessions_1_and_8")


class Inflater:
    """Sequential reader over one zlib payload that never inflates past a request."""

    def __init__(self, stream, compressed_bytes: int) -> None:
        self._stream = stream
        self._remaining = compressed_bytes
        self._zlib = zlib.decompressobj()
        self._pending = b""
        self._buffer = bytearray()
        self.position = 0  # decompressed bytes consumed by the caller
        self.produced = 0  # decompressed bytes produced by zlib

    def _inflate(self, limit: int) -> bytes:
        while True:
            if not self._pending:
                if self._zlib.eof or self._remaining <= 0:
                    raise EOFError("compressed variable ended before the requested bytes")
                data = self._stream.read(min(CHUNK, self._remaining))
                if not data:
                    raise EOFError("file ended inside the compressed variable")
                self._remaining -= len(data)
                self._pending = data
            out = self._zlib.decompress(self._pending, limit)
            self._pending = self._zlib.unconsumed_tail
            if out:
                self.produced += len(out)
                return out
            if self._zlib.eof:
                raise EOFError("compressed variable ended before the requested bytes")

    def read(self, count: int) -> bytes:
        while len(self._buffer) < count:
            self._buffer += self._inflate(min(CHUNK, count - len(self._buffer)))
        out = bytes(self._buffer[:count])
        del self._buffer[:count]
        self.position += count
        return out

    def skip(self, count: int) -> None:
        """Discard `count` decompressed bytes without interpreting them."""
        while count:
            take = min(count, len(self._buffer))
            if take:
                del self._buffer[:take]
            else:
                take = len(self._inflate(min(CHUNK, count)))
            count -= take
            self.position += take


def read_tag(source) -> tuple[int, int, bytes | None]:
    """Return (data type, byte count, inline payload or None) of one MAT v5 element."""
    head = source.read(8)
    first, second = struct.unpack("<II", head)
    if first >> 16:  # small data element: type and count share the first four bytes
        count = first >> 16
        if count > 4:
            raise ValueError("malformed small data element")
        return first & 0xFFFF, count, head[4:4 + count]
    return first, second, None


def read_payload(source, count: int, inline: bytes | None) -> bytes:
    if inline is not None:
        return inline
    payload = source.read(count)
    source.skip(-count % 8)
    return payload


def skip_payload(source, count: int, inline: bytes | None) -> None:
    if inline is None:
        source.skip(count + (-count % 8))


def read_element(source, expected: int) -> bytes:
    mdtype, count, inline = read_tag(source)
    if mdtype != expected:
        raise ValueError("unexpected MAT data element type")
    return read_payload(source, count, inline)


def read_array_header(source) -> tuple[int, int, tuple[int, ...], bytes]:
    flags = read_element(source, MI_UINT32)
    if len(flags) != 8:
        raise ValueError("unexpected array-flags element")
    word = struct.unpack("<I", flags[:4])[0]
    raw_dims = read_element(source, MI_INT32)
    if not raw_dims or len(raw_dims) % 4:
        raise ValueError("unexpected dimensions element")
    dims = struct.unpack(f"<{len(raw_dims) // 4}i", raw_dims)
    name = read_element(source, MI_INT8)
    return word & 0xFF, word, dims, name


def open_variable(stream, name: str = VARIABLE, fields=trusted.EXPECTED_FIELDS, columns: int = COLUMNS) -> Inflater:
    """Check the MAT header and the 1 x `columns` struct header; return a reader at record 1."""
    header = stream.read(128)
    if len(header) != 128 or header[124:126] != b"\x00\x01" or header[126:128] != b"IM":
        raise ValueError("not a little-endian MAT v5 file")
    mdtype, nbytes = struct.unpack("<II", stream.read(8))
    if mdtype != MI_COMPRESSED:
        raise ValueError("first variable is not compressed")
    source = Inflater(stream, nbytes)
    mdtype, _ = struct.unpack("<II", source.read(8))
    if mdtype != MI_MATRIX:
        raise ValueError("compressed payload is not a matrix")
    mxclass, _, dims, var_name = read_array_header(source)
    if mxclass != STRUCT_CLASS or dims != (1, columns) or var_name != name.encode("ascii"):
        raise ValueError("unexpected top-level variable")
    raw_length = read_element(source, MI_INT32)
    if len(raw_length) != 4:
        raise ValueError("unexpected field-name length element")
    length = struct.unpack("<i", raw_length)[0]
    raw_names = read_element(source, MI_INT8)
    if length <= 0 or len(raw_names) != length * len(fields):
        raise ValueError("unexpected field-name table")
    names = tuple(raw_names[i:i + length].split(b"\0", 1)[0].decode("ascii")
                  for i in range(0, len(raw_names), length))
    if names != tuple(fields):
        raise ValueError("struct fields differ from the frozen schema")
    return source


def read_field(source, decode: str | None):
    """Skip one struct-field matrix (decode None) or decode a real numeric vector.

    decode="values" returns the array with its stored dtype and MATLAB shape;
    decode="finite" returns only a flat finiteness mask. Integer storage cannot
    hold NaN or Inf, so its payload is then skipped without being decoded.
    """
    mdtype, nbytes = struct.unpack("<II", source.read(8))
    if mdtype != MI_MATRIX or nbytes % 8:
        raise ValueError("unexpected struct-field element")
    if decode is None:
        source.skip(nbytes)
        return None
    if nbytes == 0:
        raise ValueError("decoded field is empty")
    start = source.position
    mxclass, word, dims, _ = read_array_header(source)
    if mxclass not in NUMERIC_CLASSES or word & COMPLEX_FLAG or len(dims) != 2 or 1 not in dims:
        raise ValueError("decoded field is not a real numeric vector")
    mdtype, count, inline = read_tag(source)
    if mdtype not in STORAGE_DTYPES:
        raise ValueError("unsupported numeric storage type")
    dtype = np.dtype(STORAGE_DTYPES[mdtype])
    size = math.prod(dims)
    if count != size * dtype.itemsize:
        raise ValueError("numeric payload length differs from its dimensions")
    if decode == "finite" and dtype.kind in "iu":
        skip_payload(source, count, inline)
        result = np.ones(size, dtype=bool)
    else:
        values = np.frombuffer(read_payload(source, count, inline), dtype=dtype)
        result = np.isfinite(values) if decode == "finite" else values.reshape(dims, order="F")
    if source.position - start != nbytes:
        raise ValueError("struct-field element length mismatch")
    return result


def decode_records(source: Inflater, fields, plan: dict[int, dict[str, str]], last_record: int) -> dict:
    """Walk records 1..last_record in order; decode only the fields named in `plan`."""
    decoded = {}
    for record in range(1, last_record + 1):
        modes = plan.get(record, {})
        values = {}
        for field in fields:
            value = read_field(source, modes.get(field))
            if value is not None:
                values[field] = value
        if values:
            decoded[record] = values
    return decoded


def mouse_support(trial, session, finite) -> tuple[dict[str, bool], dict[str, int]]:
    """Structural checks for one mouse. The results never leave this process."""
    trial, session, finite = (np.asarray(item).reshape(-1) for item in (trial, session, finite))
    checks = dict.fromkeys(CRITERIA, False)
    counts = dict.fromkeys(COUNTS, 0)
    if not (trial.size == session.size == finite.size > 0) or finite.dtype != bool:
        return checks, counts
    checks["equal_field_lengths"] = True
    codes = trial.astype(np.float64)
    checks["trialID_codes_within_0_1_2"] = bool(np.isfinite(codes).all() and set(np.unique(codes).tolist()) <= CODES)
    days = session.astype(np.float64)
    if not (np.isfinite(days).all() and (days == np.round(days)).all()):
        return checks, counts
    days = days.astype(np.int64)
    present = np.unique(days)
    checks["seshID_contiguous_1_to_K_in_row_order"] = bool(
        np.array_equal(present, np.arange(1, present.size + 1)) and (np.diff(days) >= 0).all())
    checks["sessions_1_and_8_present"] = bool(np.isin(EPOCHS, present).all())
    eligible = (codes == 0) & finite
    counts["eligible_trials_session_1"] = int(np.count_nonzero(eligible & (days == 1)))
    counts["eligible_trials_session_8"] = int(np.count_nonzero(eligible & (days == 8)))
    checks["at_least_30_eligible_in_sessions_1_and_8"] = bool(
        checks["sessions_1_and_8_present"]
        and min(counts["eligible_trials_session_1"], counts["eligible_trials_session_8"]) >= MIN_ELIGIBLE)
    counts["sessions_with_nonfinite_lickState_rows"] = int(np.unique(days[~finite]).size)
    counts["nonfinite_lickState_rows_in_sessions_1_and_8"] = int(np.count_nonzero(~finite & np.isin(days, EPOCHS)))
    return checks, counts


def arm_summary(per_mouse) -> dict[str, int]:
    """Arm-level aggregate: mice meeting each criterion and summed structural counts."""
    summary = {"mice": len(per_mouse),
               "mice_meeting_all_criteria": sum(all(checks.values()) for checks, _ in per_mouse)}
    for criterion in CRITERIA:
        summary[f"mice_{criterion}"] = sum(checks[criterion] for checks, _ in per_mouse)
    for key in COUNTS:
        summary[f"{key}_total"] = sum(counts[key] for _, counts in per_mouse)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mat", type=Path, required=True)
    parser.add_argument("--cohort-map", type=Path, required=True)
    parser.add_argument("--development-pack", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-bytes", type=int, required=True)
    parser.add_argument("--expected-md5", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--expected-cohort-map-sha256", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite an existing gate packet")

    mat = args.mat.absolute()
    source_sha256 = verify_source(mat, args.expected_bytes, args.expected_md5, args.expected_sha256)
    cohort_bytes = trusted.read_regular_bytes(args.cohort_map.absolute())
    if trusted.sha256_bytes(cohort_bytes) != args.expected_cohort_map_sha256:
        raise ValueError("cohort-map SHA-256 differs from the frozen value")
    _, lookup = trusted.load_cohort_map(cohort_bytes, source_sha256)
    arms = {arm: [r for r in sorted(lookup) if lookup[r]["cohort_id"] == cohort] for arm, (cohort, _) in ARMS.items()}
    if (trusted.role_records(lookup, "development_full") != DEVELOPMENT
            or any(arms[arm] != expected for arm, (_, expected) in ARMS.items())
            or trusted.role_records(lookup, "post_primary_boundary_evaluator_only") != BOUNDARY):
        raise ValueError("cohort map does not yield the fixed evidence roles")
    primary = sorted(record for records in arms.values() for record in records)
    if sorted(DEVELOPMENT + primary) != list(range(1, LAST_DECODED_RECORD + 1)) or min(BOUNDARY) <= LAST_DECODED_RECORD:
        raise ValueError("decoded records must be exactly 1-20 and precede the boundary records")

    pack_root = args.development_pack.absolute()
    manifest = json.loads(trusted.read_regular_bytes(pack_root / "MANIFEST.json"))
    entries = manifest.get("record_entries", [])
    if (pack_root.name != "development_full" or manifest.get("role") != "development_full"
            or manifest.get("source_payload_sha256") != source_sha256
            or [entry["source_index_1_based"] for entry in entries] != DEVELOPMENT):
        raise ValueError("released development pack does not match the trusted source")

    plan = {record: dict.fromkeys(DECODED_FIELDS, "values") for record in DEVELOPMENT}
    plan.update({record: {"trialID": "values", "seshID": "values", "lickState": "finite"} for record in primary})
    descriptor = os.open(mat, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as stream:
        source = open_variable(stream)
        decoded = decode_records(source, trusted.EXPECTED_FIELDS, plan, LAST_DECODED_RECORD)
        if source.produced != source.position:
            raise RuntimeError("inflation ran past the end of record 20")

    mismatches = 0
    for entry in entries:
        for field in DECODED_FIELDS:
            released = np.load(pack_root / entry["fields"][field]["relative_path"], allow_pickle=False)
            ours = decoded[entry["source_index_1_based"]][field]
            mismatches += not (released.dtype == ours.dtype and released.shape == ours.shape
                               and released.tobytes() == np.ascontiguousarray(ours).tobytes())
    for record in DEVELOPMENT:
        del decoded[record]

    packet = {
        "analysis": "source_semantics_design.primary_structural_gate",
        "role": "steward_procedural_same_account",
        "policy_revision": 6,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "source_identity_verified": True,
        "cohort_map_verified": True,
        "records_inflated_1_based": "1-20",
        "boundary_records_21_24_inflated": False,
        "fields_decoded": ("trialID_and_seshID_for_records_1_20_lickState_finiteness_only_for_the_eleven_primary_"
                           "records_and_lickState_values_only_for_the_development_parser_crosscheck"),
        "fields_never_decoded": [field for field in trusted.EXPECTED_FIELDS if field not in DECODED_FIELDS],
        "development_parser_crosscheck": {"records": len(entries), "fields_compared": len(entries) * len(DECODED_FIELDS),
                                          "mismatches": mismatches, "passed": mismatches == 0},
        "per_mouse_values_emitted": False,
        "lickState_or_lick_values_emitted": False,
        "audit_pack_written": False,
    }
    if mismatches:
        packet.update(overall_pass=None, gate_valid=False, arms=None)
        trusted.atomic_write_json(args.output, packet)
        print("parser cross-check failed; no primary result emitted", file=sys.stderr)
        return 3

    per_arm = {arm: arm_summary([mouse_support(decoded[r]["trialID"], decoded[r]["seshID"], decoded[r]["lickState"])
                                 for r in records]) for arm, records in arms.items()}
    del decoded
    overall = all(summary["mice_meeting_all_criteria"] == summary["mice"] for summary in per_arm.values())
    packet.update(
        gate_valid=True,
        per_mouse_criteria=list(CRITERIA),
        eligible_trial="trialID_code_0_with_finite_lickState",
        minimum_eligible_trials_per_epoch=MIN_ELIGIBLE,
        arms=per_arm,
        overall_pass=overall,
    )
    trusted.atomic_write_json(args.output, packet)
    print(f"overall_pass={overall}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
