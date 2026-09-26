#!/usr/bin/env python3
"""Download only the frozen EP10 Gao MOp model-eligible manifest on a DTN.

The manifest is outcome-blind.  A normal run reads SWC bodies and therefore is
permitted only after the EP10 target-observation contract has been frozen.
The supplied manifest is not trusted by itself: it is recomputed from the
frozen role ledger, safe geometry inventory, and outcome-blind common-support
ledger and compared field-for-field.
Use ``--validate-only`` to check all inputs and the destination without making
any provider request or creating the destination.  Transfer additionally
requires the explicit ``--execute-development-transfer`` acknowledgement.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import socket
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import build_mop_development_manifest as manifest_builder


SWC_ROOT = (
    "https://mouse.digital-brain.cn/projectome/2/srv/info/mouse/cortex/"
    "neuron_download"
)
EXPECTED_FILES = manifest_builder.EXPECTED_ACQUISITION_FILES
EXPECTED_GROUPS = 12
SAFE_BRAIN_ID = re.compile(r"^[0-9]+$")
SAFE_FILE = re.compile(r"^[A-Za-z0-9.-]+_[A-Za-z0-9.-]+\.swc$")
LOG_FIELDS = [
    "provider_neuron_id",
    "file",
    "fMOST_brain_id",
    "provider_sample_id",
    "download_status",
    "content_length_bytes",
    "actual_bytes",
]


class AcquisitionError(RuntimeError):
    """The role boundary, provider response, or downloaded file is invalid."""


class RejectRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


OPENER = urllib.request.build_opener(RejectRedirects())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--common-support", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "recompute and validate the manifest and Scratch path; make no "
            "network request and create no output"
        ),
    )
    parser.add_argument(
        "--execute-development-transfer",
        action="store_true",
        help=(
            "explicitly authorize transfer of the recomputed 200-cell "
            "common-support development set; mutually exclusive with "
            "--validate-only"
        ),
    )
    return parser.parse_args()


def recompute_manifest(
    roles_path: Path, inventory_path: Path, common_support_path: Path
) -> list[dict[str, str]]:
    try:
        role_rows = manifest_builder.read_csv(
            roles_path, manifest_builder.ROLE_FIELDS
        )
        inventory_rows = manifest_builder.read_csv(
            inventory_path, manifest_builder.INVENTORY_FIELDS
        )
        support_rows = manifest_builder.read_support_csv(common_support_path)
        return manifest_builder.build_rows(
            role_rows, inventory_rows, support_rows
        )
    except (manifest_builder.ManifestError, OSError) as error:
        raise AcquisitionError(
            f"could not recompute frozen development manifest: {error}"
        ) from error


def validate_manifest(
    path: Path,
    roles_path: Path,
    inventory_path: Path,
    common_support_path: Path,
) -> list[dict[str, str]]:
    expected = recompute_manifest(
        roles_path, inventory_path, common_support_path
    )
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if fields != manifest_builder.OUTPUT_FIELDS:
            raise AcquisitionError(
                "manifest columns/order differ from the deterministic builder: "
                f"observed={fields!r} expected={manifest_builder.OUTPUT_FIELDS!r}"
            )
        rows = list(reader)

    if len(rows) != EXPECTED_FILES:
        raise AcquisitionError(
            f"expected {EXPECTED_FILES} development rows, found {len(rows)}"
        )
    if len(expected) != EXPECTED_FILES:
        raise AcquisitionError(
            f"recomputed manifest has {len(expected)} rows, expected {EXPECTED_FILES}"
        )
    if rows != expected:
        for index, (observed_row, expected_row) in enumerate(
            zip(rows, expected), start=2
        ):
            for field in manifest_builder.OUTPUT_FIELDS:
                if observed_row[field] != expected_row[field]:
                    raise AcquisitionError(
                        "supplied manifest differs from deterministic builder at "
                        f"CSV row {index}, field {field!r}: "
                        f"observed={observed_row[field]!r} "
                        f"expected={expected_row[field]!r}"
                    )
        raise AcquisitionError(
            "supplied manifest row sequence differs from deterministic builder"
        )

    provider_ids: set[str] = set()
    filenames: set[str] = set()
    brain_samples: dict[str, str] = {}
    for row in rows:
        filename = row["file"]
        brain_id = row["fMOST_brain_id"]
        provider_id = row["provider_neuron_id"]
        sample_id = row["provider_sample_id"]
        if row["episode10_role"] != "development":
            raise AcquisitionError(
                f"refusing non-development manifest row: {filename}"
            )
        if row["EP11_SSp_tr_overlap"] != "no":
            raise AcquisitionError(f"refusing EP11-shared manifest row: {filename}")
        if row["role_status"] != "frozen_before_projection_outcomes":
            raise AcquisitionError(f"role is not frozen for {filename}")
        if row["projection_outcome_exposed"] != "no":
            raise AcquisitionError(f"outcome exposure is not clean for {filename}")
        if row["reconstruction_type"] != "Axon_and_dendrite":
            raise AcquisitionError(f"refusing non-primary reconstruction: {filename}")
        if row["primary_common_support"] != "yes":
            raise AcquisitionError(
                f"refusing development row outside primary common support: {filename}"
            )
        if row["primary_model_eligible"] != "yes":
            raise AcquisitionError(
                f"refusing development row outside primary model scope: {filename}"
            )
        if row["geometry_condition"] != manifest_builder.FROZEN_GEOMETRY_CONDITION:
            raise AcquisitionError(
                f"refusing unexpected geometry condition for {filename}: "
                f"{row['geometry_condition']!r}"
            )
        if not row["source_acronym"].startswith("MOp"):
            raise AcquisitionError(f"refusing non-MOp source row: {filename}")
        if not SAFE_BRAIN_ID.fullmatch(brain_id):
            raise AcquisitionError(f"unsafe fMOST brain ID: {brain_id!r}")
        if not SAFE_FILE.fullmatch(filename):
            raise AcquisitionError(f"unsafe SWC filename: {filename!r}")
        if filename.split("_", 1)[0] != brain_id:
            raise AcquisitionError(
                f"filename prefix does not match fMOST brain: {filename} vs {brain_id}"
            )
        if not provider_id.isdigit() or not sample_id:
            raise AcquisitionError(f"invalid provider identity for {filename}")
        if row["hemisphere"] not in {"Left", "Right"}:
            raise AcquisitionError(f"invalid provider hemisphere for {filename}")
        try:
            soma = tuple(
                float(row[field])
                for field in (
                    "soma_x_ccfv3_um",
                    "soma_y_ccfv3_um",
                    "soma_z_ccfv3_um",
                )
            )
        except ValueError as error:
            raise AcquisitionError(f"nonnumeric soma coordinate for {filename}") from error
        if not all(math.isfinite(value) for value in soma):
            raise AcquisitionError(f"nonfinite soma coordinate for {filename}")
        if provider_id in provider_ids:
            raise AcquisitionError(f"duplicate provider neuron ID: {provider_id}")
        if filename in filenames:
            raise AcquisitionError(f"duplicate filename: {filename}")
        provider_ids.add(provider_id)
        filenames.add(filename)
        prior_sample = brain_samples.setdefault(brain_id, sample_id)
        if prior_sample != sample_id:
            raise AcquisitionError(f"multiple sample IDs for fMOST brain {brain_id}")

    if len(brain_samples) != EXPECTED_GROUPS:
        raise AcquisitionError(
            f"expected {EXPECTED_GROUPS} development groups, "
            f"found {len(brain_samples)}"
        )
    if rows != sorted(rows, key=lambda row: (row["fMOST_brain_id"], row["file"])):
        raise AcquisitionError("development manifest is not canonically sorted")
    return rows


def validate_scratch_output(path: Path) -> Path:
    if not path.is_absolute():
        raise AcquisitionError("output root must be an absolute Scratch path")
    resolved = path.resolve(strict=False)
    allowed_roots: list[Path] = []
    for variable in ("SCRATCH", "GROUP_SCRATCH"):
        value = os.environ.get(variable)
        if value:
            allowed_roots.append(Path(value).resolve(strict=False))
    if not allowed_roots:
        raise AcquisitionError("SCRATCH and GROUP_SCRATCH are both unset")
    if not any(root != resolved and root in resolved.parents for root in allowed_roots):
        raise AcquisitionError(
            f"output root must be below SCRATCH or GROUP_SCRATCH: {resolved}"
        )
    if path.is_symlink():
        raise AcquisitionError(f"refusing symlink output root: {path}")
    if path.exists() and not path.is_dir():
        raise AcquisitionError(f"output root is not a directory: {path}")
    swc_root = resolved / "swc"
    if swc_root.is_symlink():
        raise AcquisitionError(f"refusing symlink SWC root: {swc_root}")
    return resolved


def assert_dtn_host() -> None:
    host = socket.gethostname().split(".", 1)[0].lower()
    if not host.startswith("dtn"):
        raise AcquisitionError(
            f"full-tree transfer must run on a Sherlock DTN, not {host!r}"
        )


def response_length(response, filename: str) -> int:
    if response.getcode() != 200:
        raise AcquisitionError(
            f"unexpected HTTP status for {filename}: {response.getcode()}"
        )
    expected_url = f"{SWC_ROOT}/{filename}"
    if response.geturl() != expected_url:
        raise AcquisitionError(
            f"unexpected response URL for {filename}: {response.geturl()!r}"
        )
    content_type = response.headers.get("Content-Type", "")
    if not content_type.lower().startswith("text/plain"):
        raise AcquisitionError(
            f"unexpected content type for {filename}: {content_type!r}"
        )
    content_encoding = response.headers.get("Content-Encoding", "identity")
    if content_encoding.lower() != "identity":
        raise AcquisitionError(
            f"encoded response is not permitted for {filename}: {content_encoding!r}"
        )
    raw_length = response.headers.get("Content-Length", "")
    try:
        length = int(raw_length)
    except ValueError as error:
        raise AcquisitionError(
            f"invalid Content-Length for {filename}: {raw_length!r}"
        ) from error
    if length <= 0:
        raise AcquisitionError(f"nonpositive Content-Length for {filename}: {length}")
    return length


def open_provider(filename: str, method: str):
    url = f"{SWC_ROOT}/{filename}"
    request = urllib.request.Request(
        url,
        method=method,
        headers={
            "Accept-Encoding": "identity",
            "User-Agent": "EP10-role-safe-development-acquisition/1.0",
        },
    )
    return OPENER.open(request, timeout=300)


def is_transient(error: BaseException) -> bool:
    if isinstance(error, urllib.error.HTTPError):
        return error.code == 429 or 500 <= error.code <= 599
    return isinstance(
        error,
        (urllib.error.URLError, TimeoutError, ConnectionError, OSError),
    )


def acquire_one(
    row: dict[str, str], swc_root: Path, stop_event: threading.Event
) -> dict[str, str | int]:
    filename = row["file"]
    brain_id = row["fMOST_brain_id"]
    destination = swc_root / brain_id / filename
    if destination.is_symlink():
        raise AcquisitionError(f"refusing symlink destination: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.parent.is_symlink():
        raise AcquisitionError(
            f"refusing symlink fMOST directory: {destination.parent}"
        )
    partial = destination.with_suffix(destination.suffix + ".part")
    if partial.is_symlink():
        raise AcquisitionError(f"refusing symlink partial: {partial}")

    last_error: BaseException | None = None
    for attempt in range(1, 5):
        if stop_event.is_set():
            raise AcquisitionError(f"cancelled before acquisition: {filename}")
        try:
            if destination.is_file():
                with open_provider(filename, "HEAD") as response:
                    expected_bytes = response_length(response, filename)
                actual_bytes = destination.stat().st_size
                if actual_bytes != expected_bytes:
                    raise AcquisitionError(
                        f"existing file length mismatch for {filename}: "
                        f"{actual_bytes} != {expected_bytes}"
                    )
                return {
                    **{field: row[field] for field in LOG_FIELDS[:4]},
                    "download_status": "existing_verified_by_header",
                    "content_length_bytes": expected_bytes,
                    "actual_bytes": actual_bytes,
                }
            if destination.exists():
                raise AcquisitionError(f"destination is not a regular file: {destination}")

            partial.unlink(missing_ok=True)
            with open_provider(filename, "GET") as response:
                expected_bytes = response_length(response, filename)
                actual_bytes = 0
                with partial.open("xb") as handle:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
                        actual_bytes += len(chunk)
            if actual_bytes != expected_bytes or partial.stat().st_size != expected_bytes:
                raise AcquisitionError(
                    f"downloaded length mismatch for {filename}: "
                    f"header={expected_bytes} stream={actual_bytes} "
                    f"file={partial.stat().st_size}"
                )
            os.replace(partial, destination)
            return {
                **{field: row[field] for field in LOG_FIELDS[:4]},
                "download_status": "downloaded",
                "content_length_bytes": expected_bytes,
                "actual_bytes": actual_bytes,
            }
        except BaseException as error:
            partial.unlink(missing_ok=True)
            if isinstance(error, AcquisitionError) or not is_transient(error):
                raise
            last_error = error
            if attempt < 4 and not stop_event.wait(attempt * 2):
                continue
            break
    raise AcquisitionError(f"failed to acquire {filename}: {last_error}")


def actual_swc_paths(swc_root: Path) -> set[str]:
    if not swc_root.exists():
        return set()
    observed: set[str] = set()
    for directory, directory_names, filenames in os.walk(swc_root, followlinks=False):
        base = Path(directory)
        for name in directory_names:
            if (base / name).is_symlink():
                raise AcquisitionError(f"refusing symlink directory: {base / name}")
        for name in filenames:
            path = base / name
            if path.is_symlink():
                raise AcquisitionError(f"refusing symlink file: {path}")
            if name.endswith(".swc"):
                observed.add(path.relative_to(swc_root).as_posix())
    return observed


def assert_exact_file_set(rows: list[dict[str, str]], swc_root: Path) -> None:
    expected = {
        f"{row['fMOST_brain_id']}/{row['file']}"
        for row in rows
    }
    observed = actual_swc_paths(swc_root)
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing or extra:
        raise AcquisitionError(
            "development SWC identity set mismatch: "
            f"missing={missing[:5]} extra={extra[:5]}"
        )


def write_log(path: Path, rows: list[dict[str, str | int]]) -> None:
    partial = path.with_suffix(path.suffix + ".part")
    if path.is_symlink() or partial.is_symlink():
        raise AcquisitionError(f"refusing symlink log path below {path.parent}")
    try:
        with partial.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=LOG_FIELDS, lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    if args.validate_only == args.execute_development_transfer:
        raise AcquisitionError(
            "choose exactly one of --validate-only or "
            "--execute-development-transfer"
        )
    if args.workers < 1 or args.workers > 16:
        raise AcquisitionError("workers must be between 1 and 16")
    rows = validate_manifest(
        args.manifest, args.roles, args.inventory, args.common_support
    )
    output_root = validate_scratch_output(args.output_root)

    if args.validate_only:
        print(
            json.dumps(
                {
                    "development_files": len(rows),
                    "development_groups": len(
                        {row["fMOST_brain_id"] for row in rows}
                    ),
                    "primary_common_support_yes": sum(
                        row["primary_common_support"] == "yes" for row in rows
                    ),
                    "network_requests": 0,
                    "output_created": False,
                    "output_root": str(output_root),
                    "status": "validated_only",
                },
                sort_keys=True,
            )
        )
        return

    assert_dtn_host()
    os.umask(0o077)
    output_root.mkdir(parents=True, exist_ok=True)
    swc_root = output_root / "swc"
    swc_root.mkdir(parents=True, exist_ok=True)
    if swc_root.is_symlink():
        raise AcquisitionError(f"refusing symlink SWC root: {swc_root}")

    # Refuse a mixed destination before making any provider request.
    existing = actual_swc_paths(swc_root)
    expected = {
        f"{row['fMOST_brain_id']}/{row['file']}"
        for row in rows
    }
    unexpected = sorted(existing - expected)
    if unexpected:
        raise AcquisitionError(
            f"output contains non-development SWCs: {unexpected[:5]}"
        )

    by_file: dict[str, dict[str, str | int]] = {}
    stop_event = threading.Event()
    executor = ThreadPoolExecutor(max_workers=args.workers)
    futures = {
        executor.submit(acquire_one, row, swc_root, stop_event): row["file"]
        for row in rows
    }
    try:
        for future in as_completed(futures):
            record = future.result()
            filename = str(record["file"])
            if filename in by_file:
                raise AcquisitionError(f"duplicate completed filename: {filename}")
            by_file[filename] = record
    except BaseException:
        stop_event.set()
        for future in futures:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)

    assert_exact_file_set(rows, swc_root)
    records = [by_file[row["file"]] for row in rows]
    if len(records) != EXPECTED_FILES:
        raise AcquisitionError("incomplete development acquisition log")
    write_log(output_root / "download_log.csv", records)
    print(
        json.dumps(
            {
                "development_files": len(records),
                "development_groups": EXPECTED_GROUPS,
                "downloaded": sum(
                    row["download_status"] == "downloaded" for row in records
                ),
                "existing_verified_by_header": sum(
                    row["download_status"] == "existing_verified_by_header"
                    for row in records
                ),
                "total_bytes": sum(int(row["actual_bytes"]) for row in records),
                "output_root": str(output_root),
                "status": "complete",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
