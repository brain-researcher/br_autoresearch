#!/usr/bin/env python3
"""Fetch only the initial byte range needed for Gao MOp soma coordinates."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


SWC_ROOT = (
    "https://mouse.digital-brain.cn/projectome/2/srv/info/mouse/cortex/"
    "neuron_download"
)
RANGE_END = 4_095
SAFE_FILE = re.compile(r"^[A-Za-z0-9.-]+_[A-Za-z0-9.-]+\.swc$")
CONTENT_RANGE = re.compile(r"bytes (\d+)-(\d+)/(\d+)")


class BoundaryError(RuntimeError):
    """A response or SWC prefix violates the outcome-blind access contract."""


class RejectRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


OPENER = urllib.request.build_opener(RejectRedirects())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def select_mop(metadata_path: Path) -> list[dict[str, str]]:
    with metadata_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    neuron_data = payload.get("neuron_data")
    if not isinstance(neuron_data, dict):
        raise RuntimeError("provider metadata lacks neuron_data")

    selected: list[dict[str, str]] = []
    for provider_id, raw in neuron_data.items():
        acronym = str(raw.get("acronym", ""))
        if not acronym.startswith("MOp"):
            continue
        if str(raw.get("id", "")) != str(provider_id):
            raise RuntimeError(f"provider ID mismatch for metadata key {provider_id}")
        filename = str(raw.get("file", ""))
        if not SAFE_FILE.fullmatch(filename):
            raise RuntimeError(f"unsafe provider filename: {filename!r}")
        selected.append(
            {
                "provider_neuron_id": str(provider_id),
                "file": filename,
                "filename_specimen_prefix": filename.split("_", 1)[0],
                "source_acronym": acronym,
                "hemisphere": str(raw.get("hemisphere", "")),
                "reconstruction_type": str(raw.get("reconstruction_type", "")),
            }
        )
    selected.sort(key=lambda row: (row["filename_specimen_prefix"], row["file"]))
    if len(selected) != 837:
        raise RuntimeError(f"expected 837 MOp records, found {len(selected)}")
    if len({row["file"] for row in selected}) != len(selected):
        raise RuntimeError("duplicate MOp filename")
    return selected


def fetch_soma(
    row: dict[str, str], stop_event: threading.Event
) -> dict[str, str | float | int]:
    filename = row["file"]
    url = f"{SWC_ROOT}/{filename}"
    last_error: Exception | None = None
    for attempt in range(1, 5):
        if stop_event.is_set():
            raise RuntimeError(f"cancelled before fetch: {filename}")
        try:
            request = urllib.request.Request(
                url,
                headers={
                    "Accept-Encoding": "identity",
                    "Range": f"bytes=0-{RANGE_END}",
                    "User-Agent": "EP10-outcome-blind-soma-qualification/1.0",
                },
            )
            with OPENER.open(request, timeout=120) as response:
                status = response.getcode()
                content_range = response.headers.get("Content-Range", "")
                range_match = CONTENT_RANGE.fullmatch(content_range.strip())
                if status != 206 or range_match is None:
                    raise BoundaryError(
                        f"range request not honored for {filename}: "
                        f"status={status} content_range={content_range!r}"
                    )
                range_start, range_end, representation_size = (
                    int(value) for value in range_match.groups()
                )
                expected_length = RANGE_END + 1
                if (
                    range_start != 0
                    or range_end != RANGE_END
                    or representation_size <= expected_length
                ):
                    raise BoundaryError(
                        f"response is not the requested strict prefix for {filename}: "
                        f"{content_range!r}"
                    )
                content_length = response.headers.get("Content-Length", "")
                if content_length != str(expected_length):
                    raise BoundaryError(
                        f"unexpected prefix length for {filename}: {content_length!r}"
                    )
                content_encoding = response.headers.get("Content-Encoding", "identity")
                if content_encoding.lower() != "identity":
                    raise BoundaryError(
                        f"encoded range response for {filename}: {content_encoding!r}"
                    )
                content_type = response.headers.get("Content-Type", "")
                if not content_type.lower().startswith("text/plain"):
                    raise BoundaryError(
                        f"unexpected content type for {filename}: {content_type!r}"
                    )
                if response.geturl() != url:
                    raise BoundaryError(
                        f"unexpected response URL for {filename}: {response.geturl()!r}"
                    )
                prefix_bytes_consumed = 0
                while True:
                    raw_line = response.readline(1_025)
                    if not raw_line:
                        break
                    prefix_bytes_consumed += len(raw_line)
                    if len(raw_line) == 1_025 and not raw_line.endswith(b"\n"):
                        raise BoundaryError(f"overlong SWC prefix line for {filename}")
                    line = raw_line.decode("utf-8")
                    stripped = line.strip()
                    if not stripped or stripped.startswith("#"):
                        continue
                    fields = stripped.split()
                    if len(fields) != 7:
                        raise BoundaryError(f"malformed first SWC row for {filename}")
                    node_id = int(fields[0])
                    node_type_value = float(fields[1])
                    if not node_type_value.is_integer():
                        raise BoundaryError(f"noninteger SWC type for {filename}")
                    node_type = int(node_type_value)
                    x, y, z, radius = (float(fields[index]) for index in (2, 3, 4, 5))
                    parent_id = int(fields[6])
                    if node_type != 1:
                        raise BoundaryError(
                            f"first SWC data row is not soma for {filename}"
                        )
                    if node_id <= 0 or parent_id != -1 or radius < 0:
                        raise BoundaryError(f"invalid soma root row for {filename}")
                    if not all(
                        math.isfinite(value) for value in (x, y, z, radius)
                    ):
                        raise BoundaryError(f"nonfinite soma row for {filename}")
                    return {
                        **row,
                        "soma_x_ccfv3_um": x,
                        "soma_y_ccfv3_um": y,
                        "soma_z_ccfv3_um": z,
                        "prefix_bytes_consumed": prefix_bytes_consumed,
                        "range_bytes_requested": expected_length,
                    }
            raise BoundaryError(
                f"no soma node in bytes 0-{RANGE_END} for {filename}"
            )
        except urllib.error.HTTPError as error:
            if error.code != 429 and not 500 <= error.code <= 599:
                raise BoundaryError(
                    f"nontransient HTTP {error.code} for {filename}"
                ) from error
            last_error = error
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as error:
            last_error = error
        except (BoundaryError, UnicodeError, ValueError) as error:
            raise BoundaryError(f"boundary failure for {filename}: {error}") from error
        if attempt < 4 and not stop_event.wait(attempt * 2):
            continue
        break
    raise RuntimeError(f"failed soma fetch for {filename}: {last_error}")


def main() -> None:
    args = parse_args()
    if args.workers < 1 or args.workers > 16:
        raise RuntimeError("workers must be between 1 and 16")
    selected = select_mop(args.metadata)
    if args.limit is not None:
        if args.limit < 1:
            raise RuntimeError("limit must be positive")
        selected = selected[: args.limit]

    by_file: dict[str, dict[str, str | float | int]] = {}
    stop_event = threading.Event()
    executor = ThreadPoolExecutor(max_workers=args.workers)
    futures = {
        executor.submit(fetch_soma, row, stop_event): row["file"] for row in selected
    }
    try:
        for future in as_completed(futures):
            result = future.result()
            by_file[str(result["file"])] = result
    except BaseException:
        stop_event.set()
        for future in futures:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)
    if len(by_file) != len(selected):
        raise RuntimeError("incomplete soma result set")

    fieldnames = [
        "provider_neuron_id",
        "file",
        "filename_specimen_prefix",
        "source_acronym",
        "hemisphere",
        "reconstruction_type",
        "soma_x_ccfv3_um",
        "soma_y_ccfv3_um",
        "soma_z_ccfv3_um",
        "prefix_bytes_consumed",
        "range_bytes_requested",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(args.output.suffix + ".part")
    with partial.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, lineterminator="\n"
        )
        writer.writeheader()
        for row in selected:
            writer.writerow(by_file[row["file"]])
    os.replace(partial, args.output)
    print(f"soma_rows={len(selected)} output={args.output}")


if __name__ == "__main__":
    main()
