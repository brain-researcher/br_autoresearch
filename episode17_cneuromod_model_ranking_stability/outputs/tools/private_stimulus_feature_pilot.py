"""Unactivated EP17 private fixed-32 stimulus-to-feature pilot.

Callable only: no credential discovery, subprocess, CLI, reader release or
isolation mechanism. The coordinator must separately select execution and
provide an authorized inherited descriptor inside the required reader boundary.
Only fixed operational codes leave this module; exceptions from decryption,
decoding and inference are never printed or serialized. A stopped attempt is
preserved and is not a releasable cache. Run only in an attended Slurm context.
"""
from __future__ import annotations

from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import resource
import signal
import socket
import stat
import time
import warnings
import zipfile

import inspect_stimulus_archive as archive_metadata


EPISODE = Path(__file__).resolve().parents[2]
PRIVATE_OUTPUT = EPISODE / "outputs/cache/private/stimulus_features/pilot001/development"
REPORT_PATH = EPISODE / "outputs/runs/stimulus_features/pilot001/pilot_report.json"
RANDOM_CHECKPOINT = EPISODE / "outputs/cache/model_checkpoints/random_resnet50_seed17.pth"
MODELS = ("DINO_R50", "SWAV_R50_800", "SUP_R50_V1", "RANDOM_R50_S17")
LAYERS = ("layer1", "layer2", "layer3", "layer4")
WIDTHS = (256, 512, 1024, 2048)
IMAGE_COUNT = 32
MEMBER_CAP = 16 * 1024 * 1024
PIXEL_CAP = 16_000_000
SECRET_CAP = 4096
INTERNAL_SECONDS = 600
INDEX_FIELDS = tuple(field for field in archive_metadata.SELECTION_FIELDS if field != "member_match_count")
REPORT_CAP = 256 * 1024
CACHE_CAP = 4 * 1024 * 1024  # 1,966,080 feature bytes plus four HDF5/index containers.
RANDOM_CAP = 128 * 1024 * 1024


class PilotRefusal(RuntimeError):
    """Internal fixed code, never provider text or a secret-bearing exception."""


def require(condition, code):
    if not condition:
        raise PilotRefusal(code)


def pilot_paths(attempt="pilot001"):
    """Derive one explicitly selected attempt; never discover or advance it."""
    require(isinstance(attempt, str) and re.fullmatch(r"pilot[0-9]{3,6}", attempt) is not None,
            "explicit_pilot_attempt_required")
    return (EPISODE / "outputs/cache/private/stimulus_features" / attempt / "development",
            EPISODE / "outputs/runs/stimulus_features" / attempt / "pilot_report.json")


def _stamp():
    return datetime.now(timezone.utc).isoformat()


def _ordinary_private_file(path):
    info = Path(path).lstat()
    require(stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid()
            and stat.S_IMODE(info.st_mode) == 0o600, "private_file_required")
    return info


def _mkdir_private(path, *, exclusive=False):
    """Private staging is storage hygiene, explicitly not same-UID isolation."""
    path = Path(path)
    if not path.parent.exists():
        _mkdir_private(path.parent)
    created = False
    try:
        path.mkdir(mode=0o700)
        created = True
    except FileExistsError:
        require(not exclusive, "attempt_already_exists")
    if created:
        # Oak can inherit setgid even when mkdir requests0700. Normalize only
        # this call's new owned directory, never an existing path or boundary.
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            info = os.fstat(descriptor)
            require(stat.S_ISDIR(info.st_mode) and info.st_uid == os.geteuid(),
                    "new_private_directory_owner_changed")
            os.fchmod(descriptor, 0o700)
        finally:
            os.close(descriptor)
    info = path.lstat()
    require(stat.S_ISDIR(info.st_mode) and not path.is_symlink(), "ordinary_directory_required")
    # Pre-existing parents above the private attempt have the repository's
    # established permissions. The fresh final directory must be private.
    if exclusive:
        require(info.st_uid == os.getuid() and stat.S_IMODE(info.st_mode) == 0o700,
                "private_directory_required")


@contextmanager
def _exclusive_file(path, mode="w+b"):
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    options = {} if "b" in mode else {"encoding": "utf-8", "newline": ""}
    with os.fdopen(descriptor, mode, **options) as stream:
        yield stream


@contextmanager
def _deadline():
    require(signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0), "existing_alarm_required_by_caller")
    previous = signal.getsignal(signal.SIGALRM)

    def expired(_signal, _frame):
        raise PilotRefusal("deadline_exceeded")

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, INTERNAL_SECONDS)
    start = time.monotonic()
    try:
        def check():
            require(time.monotonic() - start < INTERNAL_SECONDS, "deadline_exceeded")
        yield check
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _read_password(password_fd):
    """Read caller-owned inherited fd, never a path; remove one file newline.

    The fd is duplicated and the duplicate closed. Reading advances the shared
    descriptor offset. At most4096 bytes including an optional final LF/CRLF
    are accepted; embedded NUL/newline or an empty value refuses. No secure
    erasure of Python/ZIP-library copies is claimed.
    """
    require(type(password_fd) is int and password_fd >= 3, "inherited_password_descriptor_required")
    info = os.fstat(password_fd)
    require(stat.S_ISREG(info.st_mode) or stat.S_ISFIFO(info.st_mode), "password_descriptor_type")
    with os.fdopen(os.dup(password_fd), "rb", buffering=0) as stream:
        pieces, size = [], 0
        while size <= SECRET_CAP:
            piece = stream.read(min(1024, SECRET_CAP + 1 - size))
            if not piece:
                break
            pieces.append(piece)
            size += len(piece)
        require(0 < size <= SECRET_CAP, "password_delivery_size")
        secret = b"".join(pieces)
    if secret.endswith(b"\r\n"):
        secret = secret[:-2]
    elif secret.endswith(b"\n"):
        secret = secret[:-1]
    require(secret and not any(marker in secret for marker in (b"\0", b"\r", b"\n")),
            "password_delivery_format")
    return secret


def _selection():
    selected = archive_metadata.load_selection()
    # The existing selector is authoritative for first32 numeric concepts and
    # lexical eligible images. Refuse role/order drift before any secret read.
    require(len(selected) == IMAGE_COUNT, "fixed_pilot_count_changed")
    require([row["feature_row_zero_based"] for row in selected] == list(range(IMAGE_COUNT)),
            "fixed_pilot_order_changed")
    require(all(row["role"] == "development" and row["eligible_image"] is True
                and row["outer_fold"] and row["development_block"] for row in selected),
            "development_only_selection_required")
    ids = [row["things_category_nr"] for row in selected]
    require(ids == sorted(set(ids)) and len(ids) == IMAGE_COUNT, "fixed_concept_order_changed")
    require(len({row["provider_image_path"] for row in selected}) == IMAGE_COUNT,
            "duplicate_selected_image")
    # Reuse completed metadata, not a new assignment or an inferred pool size.
    images = archive_metadata.read_csv(archive_metadata.ROLES / "image_assignments.csv")
    development_count = sum(row["eligible_image"] == "True" and row["role"] == "development" for row in images)
    require(IMAGE_COUNT <= development_count <= 480 * 6, "development_pool_count_changed")
    return selected, development_count


def _identity(stream):
    info = os.fstat(stream.fileno())
    require(stat.S_ISREG(info.st_mode), "ordinary_archive_required")
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _verify_archive(stream, selected, check):
    archive_metadata.validate_manifest()
    before = _identity(stream)
    require(before[2] == archive_metadata.EXPECTED_SIZE, "archive_size_changed")
    md5, sha256 = hashlib.md5(), hashlib.sha256()
    stream.seek(0)
    remaining = before[2]
    while remaining:
        check()
        chunk = stream.read(min(1024 * 1024, remaining))
        require(chunk, "archive_short_read")
        remaining -= len(chunk)
        md5.update(chunk)
        sha256.update(chunk)
    require(md5.hexdigest() == archive_metadata.EXPECTED_MD5
            and sha256.hexdigest() == archive_metadata.EXPECTED_SHA256, "archive_content_identity_changed")
    facts, entries = archive_metadata.inspect_stream(stream, before[2])
    complete, matches = archive_metadata.match_selection(selected, entries)
    require(complete, "selected_member_mapping_changed")
    for match in matches:
        member = match["member"]
        require(member["compression_method"] == zipfile.ZIP_DEFLATED
                and member["general_purpose_flags"] == 9 and member["aes"] is None
                and not member["strong_encryption_flag"]
                and 0 < member["compressed_bytes"] <= MEMBER_CAP
                and 0 < member["uncompressed_bytes"] <= MEMBER_CAP,
                "selected_archive_format_or_size_unsupported")
    require(_identity(stream) == before, "archive_changed_during_read")
    return before, facts


def _save_index(path, selected):
    with _exclusive_file(path, "w+") as output:
        writer = csv.DictWriter(output, fieldnames=INDEX_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: row[key] for key in INDEX_FIELDS} for row in selected)
        output.flush()
        os.fsync(output.fileno())


def _write_report(stream, report):
    payload = (json.dumps(report, sort_keys=True, allow_nan=False, indent=2) + "\n").encode("utf-8")
    require(len(payload) <= REPORT_CAP, "operational_report_cap")
    stream.seek(0)
    stream.write(payload)
    stream.truncate()
    stream.flush()
    os.fsync(stream.fileno())


def _prepare_random(features, checkpoint_path, check):
    """Create once, or load the exact existing artifact; never repair/reroll it."""
    import torch
    if checkpoint_path.exists() or checkpoint_path.is_symlink():
        info = _ordinary_private_file(checkpoint_path)
        require(0 < info.st_size <= RANDOM_CAP, "random_checkpoint_size")
        with checkpoint_path.open("rb") as stream:
            payload = torch.load(stream, map_location="cpu", weights_only=True)
        require(isinstance(payload, dict) and set(payload) == {"definition", "state_dict"}
                and payload["definition"] == _random_definition(), "random_checkpoint_definition_changed")
        # A fresh architecture consumes the same fixed CPU seed in an isolated
        # scope, then every stored backbone parameter/buffer replaces it strictly.
        model = features.build_backbone(features.RANDOM_ID)
        model.backbone.load_state_dict(payload["state_dict"], strict=True)
        require(all(torch.isfinite(value).all().item() for value in model.state_dict().values()),
                "random_checkpoint_nonfinite")
        del model, payload
        check()
        return "reused"
    _mkdir_private(checkpoint_path.parent)
    model = features.build_backbone(features.RANDOM_ID)
    with _exclusive_file(checkpoint_path) as stream:
        torch.save({"definition": _random_definition(), "state_dict": model.backbone.state_dict()}, stream)
        stream.flush()
        os.fsync(stream.fileno())
    require(_ordinary_private_file(checkpoint_path).st_size <= RANDOM_CAP, "random_checkpoint_size")
    del model
    check()
    return "created"


def _random_definition():
    return {"model_id": MODELS[-1], "seed": 17, "torch": "2.4.1", "torchvision": "0.19.1",
            "constructor": "resnet50(weights=None,zero_init_residual=False);fc=Identity_after_construction"}


def _load_backbone(features, model_id, checkpoint_path):
    import torch
    model = features.build_backbone(model_id)
    if model_id == features.RANDOM_ID:
        _ordinary_private_file(checkpoint_path)
        with checkpoint_path.open("rb") as stream:
            payload = torch.load(stream, map_location="cpu", weights_only=True)
        require(payload["definition"] == _random_definition(), "random_checkpoint_definition_changed")
        model.backbone.load_state_dict(payload["state_dict"], strict=True)
        del payload
    return model


def _decode_inputs(stream, selected, password, features, report, check):
    from PIL import Image
    inputs = []
    # The previously bounded directory has already been validated, and this is
    # the same open immutable archive object used for both content digests.
    with zipfile.ZipFile(stream, "r", allowZip64=False) as archive:
        for row in selected:
            check()
            info = archive.getinfo(row["provider_image_path"])
            require(info.filename == info.orig_filename == row["provider_image_path"]
                    and not info.is_dir() and not stat.S_ISLNK(info.external_attr >> 16),
                    "selected_member_type_changed")
            if not report["selected_member_read_attempted"]:
                report["selected_member_read_attempted"] = True
                report["first_member_attempt_utc"] = _stamp()
            with archive.open(info, "r", pwd=password) as member:
                if not report["image_pixels_read"]:
                    # Set before the first decoded member read, including a
                    # later CRC/decoder failure after data entered the process.
                    report["image_pixels_read"] = True
                    report["first_image_access_utc"] = _stamp()
                payload = member.read(MEMBER_CAP + 1)
            require(0 < len(payload) == info.file_size <= MEMBER_CAP, "selected_member_decode_size")
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                with Image.open(io.BytesIO(payload)) as image:
                    require(image.format == "JPEG" and getattr(image, "n_frames", 1) == 1
                            and 0 < image.width * image.height <= PIXEL_CAP,
                            "selected_image_format_or_pixel_cap")
                    # No exif_transpose or persistent plaintext image output.
                    with image.convert("RGB") as rgb:
                        inputs.append(features.preprocess(rgb))
            del payload
            report["images_preprocessed"] += 1
    require(len(inputs) == IMAGE_COUNT, "incomplete_preprocessed_bank")
    return inputs


def _write_features(features, inputs, output_dir, checkpoint_path, report, check):
    import h5py
    import numpy as np
    for model_id in MODELS:
        check()
        started = time.monotonic()
        model = _load_backbone(features, model_id, checkpoint_path)
        report["timing_seconds"]["model_load"] += time.monotonic() - started
        path = output_dir / (model_id + ".h5")
        with _exclusive_file(path) as file_stream:
            with h5py.File(file_stream, "w") as store:
                datasets = {name: store.create_dataset(name, (IMAGE_COUNT, width), dtype="float32")
                            for name, width in zip(LAYERS, WIDTHS)}
                for row_index, value in enumerate(inputs):
                    check()
                    if not report["model_features_computed"]:
                        report["model_features_computed"] = True
                        report["first_feature_computation_utc"] = _stamp()
                    started = time.monotonic()
                    vectors = model(value.unsqueeze(0))
                    report["timing_seconds"]["inference"] += time.monotonic() - started
                    require(tuple(vectors) == LAYERS, "feature_layers_changed")
                    started = time.monotonic()
                    for name, width in zip(LAYERS, WIDTHS):
                        vector = vectors[name].detach().cpu().numpy()
                        require(vector.shape == (1, width) and vector.dtype == np.dtype("float32")
                                and bool(np.isfinite(vector).all()), "feature_representation_or_finiteness")
                        datasets[name][row_index, :] = vector[0]
                    del vectors, vector
                    report["timing_seconds"]["feature_write"] += time.monotonic() - started
                store.attrs["model_id"] = model_id
                store.attrs["role"] = "development"
                store.attrs["rows_complete"] = IMAGE_COUNT
                store.flush()
            file_stream.flush()
            os.fsync(file_stream.fileno())
        del model
        report["completed_models"].append(model_id)
        report["feature_files"].append({"model_id": model_id, "dtype": "float32", "finite": True,
                                        "shapes": [[IMAGE_COUNT, width] for width in WIDTHS],
                                        "bytes": path.stat().st_size})
        require(sum(item["bytes"] for item in report["feature_files"])
                + (output_dir / "image_index.csv").stat().st_size <= CACHE_CAP, "private_cache_size_cap")


def run_private_development_feature_pilot(*, password_fd: int, private_output_dir: Path,
                                        operational_report_path: Path, random_checkpoint_path: Path,
                                        attempt: str = "pilot001"):
    """Execute only when separately selected; return an operational report.

    No CLI/default credential path or authority booleans. Paths must match the
    explicitly selected attempt (default pilot001) and may not exist already,
    except the single persistent seed17 checkpoint shared by all attempts.
    No automatic advance/retry. Failure never releases artifacts.
    The 600-second internal alarm also bounds blocked fd reads. Caller supplies
    the specified 660+10-second external timeout and1CPU/4GiB allocation.
    """
    start = time.monotonic()
    require(bool(os.environ.get("SLURM_JOB_ID")) and "-ln" not in socket.gethostname(),
            "compute_allocation_required")
    private_output_dir, operational_report_path, random_checkpoint_path = map(
        Path, (private_output_dir, operational_report_path, random_checkpoint_path))
    selected_private, selected_report = pilot_paths(attempt)
    require(private_output_dir == selected_private and operational_report_path == selected_report
            and random_checkpoint_path == RANDOM_CHECKPOINT, "fixed_pilot_paths_required")
    require(not private_output_dir.exists() and not private_output_dir.is_symlink()
            and not private_output_dir.parent.exists()
            and not operational_report_path.parent.exists(), "attempt_already_exists")
    # Reserve fresh output/report directories first; never overwrite a failed
    # attempt. File modes do not assert effective reader isolation.
    _mkdir_private(operational_report_path.parent, exclusive=True)
    _mkdir_private(private_output_dir, exclusive=True)
    report = {"analysis_id": "stimulus_features", "attempt": attempt, "status": "started",
              "pilot_feature_extraction_complete": False, "private_artifacts_releasable": False,
              "physical_role_isolation_verified": False, "reader_handoffs_activated": False,
              "source_roles_complete": False, "candidate_scoring_authorized": False,
              "audit_pixels_or_features_opened": False, "neural_values_read": False,
              "selected_member_read_attempted": False, "image_pixels_read": False,
              "model_features_computed": False, "images_preprocessed": 0,
              "completed_models": [], "feature_files": [], "selected_images": [],
              "model_ids": list(MODELS), "layers": list(LAYERS), "started_utc": _stamp(),
              "job_id": os.environ["SLURM_JOB_ID"], "host": socket.gethostname(),
              "caps": {"internal_seconds": INTERNAL_SECONDS, "member_bytes": MEMBER_CAP,
                       "decoded_pixels": PIXEL_CAP, "password_delivery_bytes": SECRET_CAP,
                       "feature_cache_bytes": CACHE_CAP, "random_checkpoint_bytes": RANDOM_CAP},
              "timing_seconds": {"archive_identity_and_metadata": 0.0, "random_checkpoint": 0.0,
                                 "secret_delivery": 0.0, "decode_transform": 0.0,
                                 "model_load": 0.0, "inference": 0.0, "feature_write": 0.0}}
    phase = "runtime"
    interrupted = False
    with _exclusive_file(operational_report_path) as report_stream:
        _write_report(report_stream, report)
        try:
            with _deadline() as check:
                import model_features as features
                features.check_runtime()
                report["runtime"] = features.runtime_metadata()
                phase = "selection"
                selected, development_count = _selection()
                report["selected_images"] = [{key: row[key] for key in INDEX_FIELDS} for row in selected]
                report["eligible_development_images"] = development_count
                _save_index(private_output_dir / "image_index.csv", selected)
                # Selected IDs are durable and public before any pixel opening.
                _write_report(report_stream, report)
                phase = "archive_identity"
                with archive_metadata.ARCHIVE.open("rb") as stream:
                    started = time.monotonic()
                    identity, directory = _verify_archive(stream, selected, check)
                    report["timing_seconds"]["archive_identity_and_metadata"] = time.monotonic() - started
                    report["archive_content_identity_verified"] = True
                    report["archive_directory"] = directory
                    phase = "random_checkpoint"
                    started = time.monotonic()
                    report["random_checkpoint_action"] = _prepare_random(features, random_checkpoint_path, check)
                    report["timing_seconds"]["random_checkpoint"] = time.monotonic() - started
                    phase = "secret_delivery"
                    started = time.monotonic()
                    password = _read_password(password_fd)
                    report["timing_seconds"]["secret_delivery"] = time.monotonic() - started
                    phase = "image_decode"
                    started = time.monotonic()
                    try:
                        inputs = _decode_inputs(stream, selected, password, features, report, check)
                    finally:
                        del password
                        report["timing_seconds"]["decode_transform"] = time.monotonic() - started
                    require(_identity(stream) == identity, "archive_changed_during_read")
                phase = "feature_extraction"
                _write_features(features, inputs, private_output_dir, random_checkpoint_path, report, check)
                del inputs
                require(report["completed_models"] == list(MODELS) and report["images_preprocessed"] == IMAGE_COUNT,
                        "incomplete_pilot")
                check()
                fixed = sum(report["timing_seconds"][name] for name in
                            ("archive_identity_and_metadata", "random_checkpoint", "secret_delivery", "model_load"))
                variable = sum(report["timing_seconds"][name] for name in
                               ("decode_transform", "inference", "feature_write"))
                measured = time.monotonic() - start
                residual = max(0.0, measured - fixed - variable)
                # Include selection/index/report preparation, container creation,
                # flush/close/fsync and other actual preprojection work. Assign
                # residual to per-image work conservatively rather than omit it.
                report["timing_seconds"]["other_preprojection_work"] = residual
                variable += residual
                report["cost_projection_coverage"] = {
                    "measured_preprojection_seconds": measured,
                    "fixed_seconds": fixed,
                    "variable_seconds_including_residual": variable,
                    "residual_assignment": "conservatively_scaled_with_development_image_count",
                    "includes_selection_index_container_creation_flush_close_fsync": True,
                    "final_operational_report_serialization_and_flush_included": False}
                report["buffered_full_development_seconds_estimate_not_guarantee"] = (
                    2 * (fixed + development_count / IMAGE_COUNT * variable) + 60)
                report["pilot_feature_extraction_complete"] = True
                report["status"] = "completed_private_staging"
        except (KeyboardInterrupt, SystemExit):
            report["status"], report["error_code"] = "interrupted", "interrupted"
            interrupted = True
        except Exception:
            # Never serialize str(exc), traceback, repr, image bytes, password,
            # feature tensors or exception context from a downstream library.
            report["status"], report["error_code"] = "stopped", phase + "_failed"
        finally:
            report["finished_utc"] = _stamp()
            report["elapsed_seconds"] = time.monotonic() - start
            report["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            _write_report(report_stream, report)
    if interrupted:
        raise KeyboardInterrupt("private_pilot_interrupted") from None
    return report
