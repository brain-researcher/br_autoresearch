"""Read only three scalar LED fields from ws.win in a classic little-endian MAT.

This is a narrowly scoped metadata reader, not a general loadmat replacement.
Unrequested matrix payloads are skipped by their byte lengths, never converted
to numeric arrays. MAT compression may require materializing uncompressed bytes;
this does not imply that only metadata bytes were physically read.
"""
import math
import struct
import zlib


MAX_INFLATED_BYTES = 512 * 1024 * 1024
FIELDS = {"trial_start_time", "LEDon", "LEDoff"}


def tag(buffer, position, limit):
    if position + 8 > limit:
        raise ValueError("Truncated MAT element tag")
    first, second = struct.unpack_from("<II", buffer, position)
    if first >> 16:
        kind, size = first & 0xffff, first >> 16
        if size > 4:
            raise ValueError("Invalid small MAT element")
        return kind, position + 4, position + 4 + size, position + 8
    end = position + 8 + second
    if end > limit:
        raise ValueError("MAT element exceeds its containing payload")
    following = end if first == 15 else end + (-second % 8)
    if following > limit:
        raise ValueError("Truncated MAT element padding")
    return first, position + 8, end, following


def matrix_header(buffer, start, end):
    kind, a, b, position = tag(buffer, start, end)
    if kind != 6 or b - a != 8:
        raise ValueError("Expected MAT array flags")
    flags = struct.unpack_from("<I", buffer, a)[0]
    if flags & 0x800:
        raise ValueError("Complex selected MAT object unsupported")
    kind, a, b, position = tag(buffer, position, end)
    if kind != 5 or (b - a) % 4 or not 8 <= b - a <= 28:
        raise ValueError("Unsupported selected MAT dimensions")
    dims = struct.unpack_from("<" + "i" * ((b - a) // 4), buffer, a)
    if any(dimension < 0 for dimension in dims):
        raise ValueError("Negative MAT dimension")
    count = math.prod(dims)
    kind, a, b, position = tag(buffer, position, end)
    if kind not in (1, 2) or b - a > 128:
        raise ValueError("Unsupported MAT variable name")
    name = bytes(buffer[a:b]).decode("ascii")
    return flags & 0xff, count, name, position


def read_scalar(buffer, start, end):
    cls, count, name, position = matrix_header(buffer, start, end)
    if count != 1 or cls not in range(6, 16):
        raise ValueError("Selected LED metadata must be a numeric scalar")
    kind, a, b, following = tag(buffer, position, end)
    formats = {1: "b", 2: "B", 3: "h", 4: "H", 5: "i", 6: "I",
               7: "f", 9: "d", 12: "q", 13: "Q"}
    fmt = formats.get(kind)
    if fmt is None or b - a != struct.calcsize(fmt) or following != end:
        raise ValueError("Unsupported scalar LED metadata representation")
    value = float(struct.unpack_from("<" + fmt, buffer, a)[0])
    if not math.isfinite(value):
        raise ValueError("Nonfinite official LED metadata")
    return value


def read_structure(buffer, start, end, path, report):
    cls, count, name, position = matrix_header(buffer, start, end)
    if cls != 2 or not 1 <= count <= 128 or (path == "ws" and count != 1):
        raise ValueError("Only scalar ws and at most128 ws.win structure entries are supported")
    kind, a, b, position = tag(buffer, position, end)
    if kind != 5 or b - a != 4:
        raise ValueError("Expected MAT field-name width")
    width = struct.unpack_from("<i", buffer, a)[0]
    kind, a, b, position = tag(buffer, position, end)
    if kind not in (1, 2) or not 1 <= width <= 128 or (b - a) % width:
        raise ValueError("Invalid MAT field-name table")
    names = [bytes(buffer[offset:offset + width]).split(b"\0", 1)[0].decode("ascii")
             for offset in range(a, b, width)]
    if len(names) > 128 or len(set(names)) != len(names):
        raise ValueError("Invalid or duplicate MAT structure fields")
    if path == "ws" and "win" not in names:
        raise ValueError("ws.win is absent")
    if path == "ws.win" and not FIELDS <= set(names):
        raise ValueError("Required official LED scalar fields are absent")
    entries = []
    for element in range(count):
        item = {}
        for field in names:
            kind, a, b, position = tag(buffer, position, end)
            if kind != 14:
                raise ValueError("Expected nested MAT matrix")
            if path == "ws" and field == "win":
                item["win"] = read_structure(buffer, a, b, "ws.win", report)
            elif path == "ws.win" and field in FIELDS:
                item[field] = read_scalar(buffer, a, b)
                report["LED_scalar_fields_decoded"] += 1
            else:
                # Includes all EEG, peripheral signals, participant initials,
                # condition values and every provider movement-event field.
                report["unrequested_matrix_payload_bytes_skipped"] += b - a
        entries.append(item)
    if position != end:
        raise ValueError("Unexpected trailing selected MAT structure content")
    return entries


def read_cues(payload):
    if len(payload) < 128 or payload[126:128] != b"IM":
        raise ValueError("Only little-endian classic MAT v5 is supported; no fallback reader")
    report = {"LED_scalar_fields_decoded": 0, "unrequested_matrix_payload_bytes_skipped": 0,
              "compressed_matrix_bytes_materialized": False,
              "waveform_numeric_arrays_decoded": False, "provider_movement_event_values_decoded": False}
    buffer = memoryview(payload)
    position, trials = 128, None
    while position < len(buffer):
        kind, a, b, position = tag(buffer, position, len(buffer))
        if kind == 15:
            decompressor = zlib.decompressobj()
            expanded = decompressor.decompress(buffer[a:b], MAX_INFLATED_BYTES + 1)
            if len(expanded) > MAX_INFLATED_BYTES or not decompressor.eof or decompressor.unused_data:
                raise ValueError("Compressed MAT exceeds512MiB or is malformed")
            report["compressed_matrix_bytes_materialized"] = True
            inner = memoryview(expanded)
            inner_kind, ia, ib, following = tag(inner, 0, len(inner))
            if inner_kind != 14 or following != len(inner):
                raise ValueError("Unsupported compressed MAT layout")
        elif kind == 14:
            inner, ia, ib = buffer, a, b
        else:
            raise ValueError("Unexpected top-level MAT element")
        cls, count, name, unused = matrix_header(inner, ia, ib)
        if name != "ws":
            report["unrequested_matrix_payload_bytes_skipped"] += ib - ia
            continue
        if trials is not None:
            raise ValueError("Duplicate ws variable")
        trials = read_structure(inner, ia, ib, "ws", report)[0]["win"]
    if trials is None or report["LED_scalar_fields_decoded"] != 3 * len(trials):
        raise ValueError("Incomplete official cue metadata")
    return trials, report
