#!/usr/bin/env python3
"""Audit inert public-source method semantics for EP21.

This supplemental verifier reads an exact allowlist of text members from the
authenticated Wang et al. V1.0.0 source archive.  It never imports or executes
upstream code, and it never reads bundled result tables, model/data payloads,
empirical handoffs, OOD material, EP04 state, or legacy Scratch.  The report
records tagged-source facts but deliberately adopts no scientific choice and
does not emit a protocol lock.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import csv
import datetime as dt
import hashlib
import io
import json
import os
import re
import secrets
import stat
import zipfile
from pathlib import Path
from typing import Any, Iterator, Mapping


EPISODE_DIR = Path(__file__).resolve().parents[3]
SOURCE_ARCHIVE = Path(
    "/scratch/users/zijiao/br_autoresearch/"
    "episode21_wang_neural_code_conversion_reproduction/source_qualification/"
    "attempt-001/KamitaniLab/InterSiteNeuralCodeConversion-V1.0.0.zip"
)
QUALIFICATION_REPORT = (
    EPISODE_DIR
    / "outputs/source_qualification/attempt-017/qualification_report.json"
)
NEUTRAL_CONFORMANCE_REPORT = (
    EPISODE_DIR
    / "outputs/pipeline_conformance/attempt-005/neutral_conformance_report.json"
)
PROTOCOL_TABLE = EPISODE_DIR / "PROTOCOL_TABLE.yaml"
OUTPUT_DIR = EPISODE_DIR / "outputs/source_qualification/attempt-019"
OUTPUT_NAME = "public_method_semantics_audit.json"

PINNED_ARCHIVE_MD5 = "bb80dcc1a038737d6d8fc2c95b3e5586"
PINNED_ARCHIVE_SHA256 = (
    "bf57aa748df80c8647cb3a773063f670a132aea24e92673bfae7a019de86b6c6"
)
PINNED_ARCHIVE_SIZE = 23_110_908
QUALIFICATION_SHA256 = (
    "98b9e5e30947f12484dc2df66d5ef06b666cb84f2e84cc01b8b0a1c609e07126"
)
NEUTRAL_CONFORMANCE_SHA256 = (
    "54fb25d6ffeb8197fde920f631c10258e1d2c82709518d4c2e958d8da2e4a187"
)
PROTOCOL_TABLE_SHA256 = (
    "c12b12b172c5e9916c4843d1b287b16c5831a29736c97a0f1cca2ad66bb68f82"
)

ARCHIVE_PREFIX = "KamitaniLab-InterSiteNeuralCodeConversion-edbff02/"
MEMBER_SPECS: dict[str, tuple[str, int, str]] = {
    "brain_train": (
        "NCC_brain_loss/ncc_train.py",
        7641,
        "28c74907a963dba3bd6063a86d8706a535c356129757054407065275409cefea",
    ),
    "brain_test": (
        "NCC_brain_loss/ncc_test.py",
        6904,
        "06f8ecdb5876b30bf9e6febe1c0517117818d6e717d98185de6affa4cab48b6f",
    ),
    "brain_parameters": (
        "NCC_brain_loss/params/converter_params_VC.csv",
        700,
        "5e0749536a636dcc49e426441a6114efbd0a01edf1ce26115601436112d06b59",
    ),
    "feature_decoder_train": (
        "feature-decoding/featdec_fastl2lir_train.py",
        8723,
        "16400a088e409b781d4084a4b28b013bfe4ca604dae8981e76b8ba6cdaba567e",
    ),
    "feature_decoder_predict": (
        "feature-decoding/featdec_fastl2lir_predict.py",
        7867,
        "68f435078533627ee42c7b299dbec6a6d69abed3e417e1fe3e6d3ebca1b8c27b",
    ),
    "feature_decoder_config": (
        "feature-decoding/config/deeprecon_pyfastl2lir_alpha100_vgg19_allunits.yaml",
        1537,
        "10ea56716c5a2b7f602488ad8e29bff94c65102203ba2b529c36812f988aeaae",
    ),
    "feature_evaluator": (
        "evaluation/decoding_accuracy/featdec_eval.py",
        5124,
        "9b5a4192d030c0b5ccb8ac3de6f572b03cef219f5ed49e76f01fd87851b111a7",
    ),
    "conversion_bootstrap": (
        "evaluation/conversion_accuracy/plot_figure.py",
        11647,
        "638811c8038d1c0d43e03577d7ddd2243eb2bf9c5b061412288add7e51fd8566",
    ),
    "decoding_bootstrap": (
        "evaluation/decoding_accuracy/plot_figure.py",
        7204,
        "a42b4a6d5c391910c919d86c9d800370c08523a4b7b69ab33ff9c52ad80a088c",
    ),
    "pixel_evaluator": (
        "evaluation/identification_accuracy/recon_image_eval.py",
        4049,
        "803e53da3d7b7f462255e1e2ce85b35fe338baf80fc38b08d992b98e68f5b283",
    ),
    "dnn_evaluator": (
        "evaluation/identification_accuracy/recon_image_eval_dnn.py",
        8420,
        "922cd26405c276bb5ec0a0c5edd2368011bb6457ddca300d4e1f2eb5ab958d91",
    ),
    "identification_bootstrap": (
        "evaluation/identification_accuracy/plot_figure.py",
        9349,
        "8ac408b10d8ff1b3579515dd30de1982a191e450055ddcfd0ba6a26ef842f731",
    ),
}

MAX_JSON_BYTES = 5 * 1024 * 1024
MAX_PROTOCOL_BYTES = 2 * 1024 * 1024
MAX_MEMBER_BYTES = 128 * 1024
READ_CHUNK = 1024 * 1024


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def _identity(info: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
        info.st_nlink,
    )


def open_nofollow(path: Path, flags: int) -> int:
    """Open an absolute path through held, no-follow directory descriptors."""
    if not path.is_absolute() or path.name in {"", ".", ".."}:
        raise ValueError(f"path is not a fixed absolute file path: {path}")
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    current = os.open("/", directory_flags)
    try:
        for component in path.parts[1:-1]:
            if component in {"", ".", ".."}:
                raise ValueError(f"unsafe path component in {path}")
            following = os.open(component, directory_flags, dir_fd=current)
            os.close(current)
            current = following
        return os.open(
            path.name,
            flags | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=current,
        )
    finally:
        os.close(current)


def read_stable_bytes(
    path: Path,
    *,
    label: str,
    maximum_bytes: int,
    expected_sha256: str,
) -> tuple[bytes, dict[str, Any]]:
    try:
        descriptor = open_nofollow(
            path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0)
        )
    except OSError as error:
        raise ValueError(f"cannot securely open {label}: {path}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError(f"{label} is not a single-link regular file")
        if before.st_size <= 0 or before.st_size > maximum_bytes:
            raise ValueError(f"{label} has an invalid size")
        chunks: list[bytes] = []
        count = 0
        while True:
            chunk = os.read(descriptor, min(READ_CHUNK, maximum_bytes + 1 - count))
            if not chunk:
                break
            chunks.append(chunk)
            count += len(chunk)
            if count > maximum_bytes:
                raise ValueError(f"{label} exceeded its size bound")
        after = os.fstat(descriptor)
        if _identity(after) != _identity(before) or count != before.st_size:
            raise ValueError(f"{label} changed while reading")
    finally:
        os.close(descriptor)
    payload = b"".join(chunks)
    observed_sha256 = hashlib.sha256(payload).hexdigest()
    if observed_sha256 != expected_sha256:
        raise ValueError(f"{label} SHA256 mismatch")
    return payload, {
        "path": str(path),
        "size_bytes": len(payload),
        "sha256": observed_sha256,
        "regular_file": True,
        "single_link": True,
    }


@contextlib.contextmanager
def verified_archive(path: Path) -> Iterator[tuple[zipfile.ZipFile, dict[str, Any]]]:
    try:
        descriptor = open_nofollow(
            path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0)
        )
    except OSError as error:
        raise ValueError(f"cannot securely open source archive: {path}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError("source archive is not a single-link regular file")
        if before.st_size != PINNED_ARCHIVE_SIZE:
            raise ValueError("source archive size mismatch")
        md5_digest = hashlib.md5()  # Release identity, not a security primitive.
        sha256_digest = hashlib.sha256()
        count = 0
        while True:
            chunk = os.read(descriptor, 4 * READ_CHUNK)
            if not chunk:
                break
            md5_digest.update(chunk)
            sha256_digest.update(chunk)
            count += len(chunk)
        if _identity(os.fstat(descriptor)) != _identity(before) or count != before.st_size:
            raise ValueError("source archive changed while hashing")
        observed_md5 = md5_digest.hexdigest()
        observed_sha256 = sha256_digest.hexdigest()
        if observed_md5 != PINNED_ARCHIVE_MD5:
            raise ValueError("source archive MD5 mismatch")
        if observed_sha256 != PINNED_ARCHIVE_SHA256:
            raise ValueError("source archive SHA256 mismatch")
        os.lseek(descriptor, 0, os.SEEK_SET)
        with os.fdopen(descriptor, "rb", closefd=True) as raw:
            descriptor = -1
            with zipfile.ZipFile(raw) as archive:
                yield archive, {
                    "path": str(path),
                    "size_bytes": count,
                    "md5": observed_md5,
                    "sha256": observed_sha256,
                    "regular_file": True,
                    "single_link": True,
                    "held_descriptor_verified": True,
                }
            if _identity(os.fstat(raw.fileno())) != _identity(before):
                raise ValueError("source archive changed during ZIP inspection")
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def read_allowlisted_members(
    archive: zipfile.ZipFile,
    specs: Mapping[str, tuple[str, int, str]] = MEMBER_SPECS,
) -> dict[str, dict[str, Any]]:
    by_name: dict[str, list[zipfile.ZipInfo]] = {}
    for info in archive.infolist():
        by_name.setdefault(info.filename, []).append(info)
    observed: dict[str, dict[str, Any]] = {}
    for key, (relative_path, expected_size, expected_sha256) in specs.items():
        full_path = ARCHIVE_PREFIX + relative_path
        matches = by_name.get(full_path, [])
        if len(matches) != 1:
            raise ValueError(f"expected exactly one allowlisted member {full_path}")
        info = matches[0]
        unix_mode = info.external_attr >> 16
        file_type = stat.S_IFMT(unix_mode)
        if info.is_dir() or info.flag_bits & 0x1:
            raise ValueError(f"allowlisted member is a directory or encrypted: {full_path}")
        if file_type not in {0, stat.S_IFREG}:
            raise ValueError(f"allowlisted member is not a regular file: {full_path}")
        if info.file_size != expected_size or not (0 < info.file_size <= MAX_MEMBER_BYTES):
            raise ValueError(f"allowlisted member size mismatch: {full_path}")
        raw = archive.read(info)
        if len(raw) != expected_size:
            raise ValueError(f"allowlisted member short read: {full_path}")
        observed_sha256 = hashlib.sha256(raw).hexdigest()
        if observed_sha256 != expected_sha256:
            raise ValueError(f"allowlisted member SHA256 mismatch: {full_path}")
        try:
            text = raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise ValueError(f"allowlisted member is not UTF-8 text: {full_path}") from error
        observed[key] = {
            "path": full_path,
            "size_bytes": len(raw),
            "sha256": observed_sha256,
            "text": text,
        }
    if set(observed) != set(specs):
        raise ValueError("allowlisted member set is incomplete")
    return observed


def require_once(source: str, needle: str, *, label: str) -> dict[str, Any]:
    hits = [
        {"line": number, "text": line.strip()}
        for number, line in enumerate(source.splitlines(), start=1)
        if needle in line
    ]
    if len(hits) != 1:
        raise ValueError(f"{label}: expected one occurrence of {needle!r}, got {len(hits)}")
    return hits[0]


def require_present(source: str, needle: str, *, label: str) -> dict[str, Any]:
    hits = [
        {"line": number, "text": line.strip()}
        for number, line in enumerate(source.splitlines(), start=1)
        if needle in line
    ]
    if not hits:
        raise ValueError(f"{label}: required source fragment is absent: {needle!r}")
    return hits[0]


def require_python(source: str, *, label: str) -> ast.Module:
    try:
        return ast.parse(source)
    except SyntaxError as error:
        raise ValueError(f"{label} is not valid Python source") from error


def calls_named(tree: ast.AST, name: str) -> list[ast.Call]:
    calls: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        called = None
        if isinstance(node.func, ast.Name):
            called = node.func.id
        elif isinstance(node.func, ast.Attribute):
            called = node.func.attr
        if called == name:
            calls.append(node)
    return calls


def parse_feature_config(source: str) -> dict[str, Any]:
    alpha_match = re.findall(r"^alpha:\s*(\d+)\s*$", source, re.MULTILINE)
    voxel_match = re.findall(r"^\s{2}VC:\s+(\d+)\s*$", source, re.MULTILINE)
    label_match = re.findall(r"^label key:\s*\n\s{2}(\S+)\s*$", source, re.MULTILINE)
    network_match = re.findall(r"^network:\s*(\S+)\s*$", source, re.MULTILINE)
    layers_block = re.search(
        r"^layers:\s*\n(?P<body>(?:\s{2}-\s+\S+\s*\n)+)", source, re.MULTILINE
    )
    if (
        alpha_match != ["100"]
        or voxel_match != ["500"]
        or label_match != ["image_index"]
        or network_match != ["caffe/VGG19"]
        or layers_block is None
    ):
        raise ValueError("feature-decoder config does not match the tagged method")
    layers = re.findall(r"^\s{2}-\s+(\S+)\s*$", layers_block.group("body"), re.MULTILINE)
    expected_layers = [
        "conv1_1", "conv1_2", "conv2_1", "conv2_2",
        "conv3_1", "conv3_2", "conv3_3", "conv3_4",
        "conv4_1", "conv4_2", "conv4_3", "conv4_4",
        "conv5_1", "conv5_2", "conv5_3", "conv5_4",
        "fc6", "fc7", "fc8",
    ]
    if layers != expected_layers:
        raise ValueError("feature-decoder layer list changed")
    return {
        "ridge_alpha": 100,
        "requested_voxels_per_feature_unit": 500,
        "label_key": "image_index",
        "network": "caffe/VGG19",
        "layers": layers,
        "chunk_axis": 1,
    }


def parse_brain_parameters(source: str) -> dict[str, Any]:
    rows = list(csv.DictReader(io.StringIO(source)))
    required = ["Source", "Target", "Number of samples", "Alpha", "ROI", "Method"]
    if not rows or list(rows[0]) != required:
        raise ValueError("brain-loss parameter columns changed")
    subjects = [f"sub0{index}" for index in range(1, 6)]
    expected_pairs = {(source_id, target_id) for source_id in subjects for target_id in subjects if source_id != target_id}
    observed_pairs = [(row["Source"], row["Target"]) for row in rows]
    if len(rows) != 20 or len(set(observed_pairs)) != 20 or set(observed_pairs) != expected_pairs:
        raise ValueError("brain-loss parameter table is not the exact 20 directed pairs")
    if any(
        row["Number of samples"] != "6000"
        or row["ROI"] != "VC"
        or row["Method"] != "wholeVC"
        for row in rows
    ):
        raise ValueError("brain-loss parameter-table fixed columns changed")
    alpha_by_pair = {
        f"{row['Source']}_to_{row['Target']}": int(row["Alpha"]) for row in rows
    }
    return {
        "row_count": len(rows),
        "directed_pair_count": len(set(observed_pairs)),
        "declared_sample_count": 6000,
        "roi": "VC",
        "method": "wholeVC",
        "alpha_by_source_target": alpha_by_pair,
        "unique_alpha_values": sorted(set(alpha_by_pair.values())),
    }


def derive_method_facts(members: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    texts = {key: str(value["text"]) for key, value in members.items()}
    for key, source in texts.items():
        if key not in {"brain_parameters", "feature_decoder_config"}:
            require_python(source, label=key)

    config = parse_feature_config(texts["feature_decoder_config"])
    feature_train = texts["feature_decoder_train"]
    feature_predict = texts["feature_decoder_predict"]
    brain_train = texts["brain_train"]
    brain_test = texts["brain_test"]
    feature_eval = texts["feature_evaluator"]
    pixel_eval = texts["pixel_evaluator"]
    dnn_eval = texts["dnn_evaluator"]

    feature_training_evidence = [
        require_once(feature_train, "x_mean = np.mean(x, axis=0)[np.newaxis, :]", label="feature decoder"),
        require_once(feature_train, "x_norm = np.std(x, axis=0, ddof=1)[np.newaxis, :]", label="feature decoder"),
        require_once(feature_train, "y_mean = np.mean(y, axis=0)[np.newaxis, :]", label="feature decoder"),
        require_once(feature_train, "y_norm = np.std(y, axis=0, ddof=1)[np.newaxis, :]", label="feature decoder"),
        require_once(feature_train, "y_labels = np.unique(x_labels)", label="feature decoder"),
        require_once(feature_train, "y_index = np.array([np.where(np.array(y_labels) == xl) for xl in x_labels]).flatten()", label="feature decoder"),
        require_once(feature_train, "'alpha':  alpha", label="feature decoder"),
        require_once(feature_train, "'n_feat': num_voxel[roi]", label="feature decoder"),
        require_once(feature_train, "'dtype': np.float32", label="feature decoder"),
    ]
    if len(calls_named(require_python(feature_train, label="feature decoder"), "FastL2LiR")) != 1:
        raise ValueError("feature decoder must instantiate FastL2LiR exactly once")
    if "fit_intercept" in feature_train or "intercept" in feature_train:
        raise ValueError("unexpected explicit feature-decoder intercept option")
    prediction_evidence = [
        require_once(feature_predict, "x_labels_unique = np.unique(x_labels)", label="feature prediction"),
        require_once(feature_predict, "np.mean(x[(np.array(x_labels) == lb).flatten(), :], axis=0)", label="feature prediction"),
        require_once(feature_predict, "x = (x - x_mean) / x_norm", label="feature prediction"),
        require_once(feature_predict, "y_pred = y_pred * y_norm + y_mean", label="feature prediction"),
    ]

    brain_training_evidence = [
        require_once(brain_train, "x_index = np.argsort(x_labels.flatten())", label="brain loss"),
        require_once(brain_train, "y_index = np.argsort(y_labels.flatten())", label="brain loss"),
        require_once(brain_train, "x_mean = np.mean(x, axis=0)[np.newaxis, :]", label="brain loss"),
        require_once(brain_train, "x_norm = np.std(x, axis=0, ddof=1)[np.newaxis, :]", label="brain loss"),
        require_once(brain_train, "y_mean = np.mean(y, axis=0)[np.newaxis, :]", label="brain loss"),
        require_once(brain_train, "y_norm = np.std(y, axis=0, ddof=1)[np.newaxis, :]", label="brain loss"),
        require_once(brain_train, "model.fit(x_normalized, y_normalized, alpha=alpha, n_feat=0,", label="brain loss"),
        require_once(brain_train, "chunk_size=0, dtype=np.float32", label="brain loss"),
        require_once(brain_train, "b = model.b.flatten()", label="brain loss"),
        require_once(brain_train, "M = np.vstack([W, b])", label="brain loss"),
    ]
    if "x_labels" + " == " + "y_labels" in brain_train:
        raise ValueError("unexpected direct brain-loss label equality assertion")
    brain_testing_evidence = [
        require_once(brain_test, "x_test_labels_unique = np.unique(x_labels)", label="brain-loss test"),
        require_once(brain_test, "np.mean(x[(x_labels == lb).flatten(), :], axis=0)", label="brain-loss test"),
        require_once(brain_test, "x = np.hstack((x, np.ones((x.shape[0], 1))))", label="brain-loss test"),
        require_once(brain_test, "converted_x = np.matmul(x, M)", label="brain-loss test"),
        require_once(brain_test, "converted_x = converted_x * y_norm + y_mean", label="brain-loss test"),
    ]

    feature_eval_evidence = [
        require_once(feature_eval, "r_prof = profile_correlation(pred_y, true_y_sorted)", label="feature evaluator"),
        require_once(feature_eval, "r_patt = pattern_correlation(pred_y, true_y_sorted, mean=train_y_mean, std=train_y_std)", label="feature evaluator"),
        require_once(feature_eval, "'Profile correlation': np.nanmean(r_prof.flatten())", label="feature evaluator"),
        require_once(feature_eval, "'Pattern correlation': np.nanmean(r_patt.flatten())", label="feature evaluator"),
    ]
    pixel_eval_evidence = [
        require_once(pixel_eval, "r_pixelt = pattern_correlation(recon_images, true_images)", label="pixel evaluator"),
        require_once(pixel_eval, "ident = pairwise_identification(recon_images, true_images)", label="pixel evaluator"),
        require_once(pixel_eval, "return np.nanmean(r_pixelt), np.nanmean(ident)", label="pixel evaluator"),
    ]
    dnn_eval_evidence = [
        require_once(dnn_eval, "ident_feat = pairwise_identification(recon_feat[layer], true_feat[layer])", label="DNN evaluator"),
        require_once(dnn_eval, "'Identification accuracy': np.nanmean(ident_feat)", label="DNN evaluator"),
        require_once(dnn_eval, "self.image_size = [227, 227]", label="DNN evaluator"),
        require_once(dnn_eval, "self.image_size = [224, 224]", label="DNN evaluator"),
    ]

    bootstrap_common_needles = [
        "np.random.choice(unique_sources, 1, replace=True)[0]",
        "np.random.choice(unique_targets, 1, replace=True)[0]",
        "if source_sample != target_sample:",
        "combined_sample = combined_sample.head(sample_size)",
    ]
    bootstrap_facts: dict[str, Any] = {}
    for key in ("conversion_bootstrap", "decoding_bootstrap", "identification_bootstrap"):
        source = texts[key]
        evidence = [require_once(source, needle, label=key) for needle in bootstrap_common_needles]
        evidence.append(require_present(source, "np.percentile", label=key))
        if "while combined_sample.shape[0] < sample_size:" not in source:
            raise ValueError(f"{key} no longer has the source while-fill behavior")
        if re.search(
            r"n_iterations\s*=\s*1000\s*,\s*alpha\s*=\s*0\.05\s*,\s*sample_size\s*=\s*20",
            source,
        ) is None:
            raise ValueError(f"{key} bootstrap defaults changed")
        if "max_attempt" in source or "max_draw" in source:
            raise ValueError(f"{key} unexpectedly contains a bounded empty-draw guard")
        bootstrap_facts[key] = {
            "source_and_target_sampled_separately_with_replacement": True,
            "source_equals_target_rejected": True,
            "requested_sample_size": 20,
            "resamples": 1000,
            "percentile_interval_alpha": 0.05,
            "duplicate_pair_rows_retained": True,
            "overshoot_truncated_with_head": True,
            "missing_or_empty_pair_draw_adds_zero_rows_and_retries": True,
            "empty_or_unfillable_draw_has_no_bounded_failure_rule": True,
            "evidence": evidence,
        }
    if texts["conversion_bootstrap"].count("np.random.seed(42)") != 1:
        raise ValueError("conversion bootstrap seed behavior changed")
    if "Parallel(n_jobs=n_jobs)" not in texts["conversion_bootstrap"] or "n_jobs=-1" not in texts["conversion_bootstrap"]:
        raise ValueError("conversion bootstrap worker behavior changed")
    for key in ("decoding_bootstrap", "identification_bootstrap"):
        if "with Pool() as pool:" not in texts[key] or "np.random.seed" in texts[key]:
            raise ValueError(f"{key} process/seed behavior changed")
    bootstrap_facts["conversion_bootstrap"].update(
        {"seed": 42, "seed_scope": "reset_inside_each_group_bootstrap", "parallel_workers": "joblib_n_jobs_minus_one"}
    )
    for key in ("decoding_bootstrap", "identification_bootstrap"):
        bootstrap_facts[key].update(
            {"seed": None, "seed_scope": "unseeded_multiprocessing", "parallel_workers": "multiprocessing_pool_default"}
        )

    return {
        "feature_decoder": {
            **config,
            "training_response_centering_axis": 0,
            "training_response_std_axis": 0,
            "training_response_std_ddof": 1,
            "training_feature_centering_axis": 0,
            "training_feature_std_axis": 0,
            "training_feature_std_ddof": 1,
            "feature_label_alignment": "unique_response_labels_then_exact_lookup_for_each_response_row",
            "fastl2lir_constructor_arguments": [],
            "fastl2lir_fit_parameters_from_script": ["alpha", "n_feat", "dtype"],
            "explicit_intercept_option_in_script": False,
            "repeated_test_response_rule": "average_by_unique_label_before_prediction_when_average_sample_true",
            "prediction_normalization": "stored_x_mean_x_norm_then_stored_y_norm_y_mean",
            "training_evidence": feature_training_evidence,
            "prediction_evidence": prediction_evidence,
        },
        "brain_loss": {
            "parameter_table": parse_brain_parameters(texts["brain_parameters"]),
            "source_and_target_rows_sorted_independently_by_label": True,
            "source_target_label_equality_assertion_in_tagged_fitter": False,
            "training_response_centering_axis": 0,
            "training_response_std_axis": 0,
            "training_response_std_ddof": 1,
            "fit_n_feat": 0,
            "fit_chunk_size": 0,
            "fit_dtype": "numpy.float32",
            "num_sample_argument_used_to_slice_fit_rows": False,
            "intercept_storage": "FastL2LiR model.b flattened and appended as final row of M",
            "test_intercept_application": "append_ones_column_then_matrix_multiply_M",
            "repeated_test_response_rule": "average_by_unique_image_label_before_conversion",
            "training_evidence": brain_training_evidence,
            "test_evidence": brain_testing_evidence,
        },
        "evaluation_calls": {
            "feature_decoding": {
                "profile": "bdpy.evals.metrics.profile_correlation",
                "pattern": "bdpy.evals.metrics.pattern_correlation_with_target_decoder_training_y_mean_y_std",
                "dyad_layer_aggregation": "numpy.nanmean",
                "evidence": feature_eval_evidence,
            },
            "pixel_identification": {
                "pixel_correlation": "bdpy.evals.metrics.pattern_correlation",
                "identification": "bdpy.evals.metrics.pairwise_identification",
                "dyad_aggregation": "numpy.nanmean",
                "evidence": pixel_eval_evidence,
            },
            "dnn_identification": {
                "identification": "bdpy.evals.metrics.pairwise_identification_per_layer",
                "dyad_layer_aggregation": "numpy.nanmean",
                "tagged_alexnet_input": [227, 227],
                "tagged_vgg19_input": [224, 224],
                "evidence": dnn_eval_evidence,
            },
        },
        "source_style_dyadic_bootstrap": bootstrap_facts,
    }


def validate_bindings(
    qualification: Mapping[str, Any],
    neutral: Mapping[str, Any],
    protocol_text: str,
) -> None:
    if (
        qualification.get("schema_version") != "ep21.source_qualification_report.v2"
        or qualification.get("attempt") != 17
        or qualification.get("qualification_outcome")
        != "completed_with_empirical_and_scientific_decision_blockers"
        or qualification.get("empirical_or_ood_payload_accessed") is not False
        or qualification.get("protocol_lock_present") is not False
        or qualification.get("laion_readiness") != "blocked"
    ):
        raise ValueError("authoritative qualification binding changed")
    gaps = qualification.get("source_pipeline_assets", {}).get(
        "unmanifested_or_unpinned_gaps", {}
    )
    if gaps.get("main_bdpy_dependency", {}).get("state") != "declared_but_unpinned":
        raise ValueError("qualification no longer retains the unpinned bdpy blocker")
    if (
        neutral.get("schema_version") != "ep21.neutral_conformance_report.v1"
        or neutral.get("attempt") != 5
        or neutral.get("outcome")
        != "passed_protocol_neutral_checks_with_protocol_and_access_blockers"
        or neutral.get("protocol_lock_emitted") is not False
        or neutral.get("empirical_readiness") != "blocked"
        or neutral.get("payload_accessed") is not False
        or neutral.get("empirical_or_ood_payload_accessed") is not False
    ):
        raise ValueError("authoritative neutral-conformance binding changed")
    required_protocol_lines = [
        "empirical_execution_authorized: false",
        "protocol_lock_exists: false",
        "regularization_value_selection_and_intercept: unresolved_authenticate_source_method_before_lock",
        "requested_output_size_weighting_duplicate_and_empty_draw_rules: unresolved_authenticate_from_source_code_before_lock",
    ]
    for line in required_protocol_lines:
        if protocol_text.count(line) != 1:
            raise ValueError(f"protocol binding changed for {line!r}")


def build_report(attempt: int) -> dict[str, Any]:
    if attempt != 19:
        raise ValueError("this frozen verifier is only valid for attempt 19")
    started_at = utc_now()
    qualification_raw, qualification_identity = read_stable_bytes(
        QUALIFICATION_REPORT,
        label="attempt-17 qualification report",
        maximum_bytes=MAX_JSON_BYTES,
        expected_sha256=QUALIFICATION_SHA256,
    )
    neutral_raw, neutral_identity = read_stable_bytes(
        NEUTRAL_CONFORMANCE_REPORT,
        label="attempt-5 neutral-conformance report",
        maximum_bytes=MAX_JSON_BYTES,
        expected_sha256=NEUTRAL_CONFORMANCE_SHA256,
    )
    protocol_raw, protocol_identity = read_stable_bytes(
        PROTOCOL_TABLE,
        label="protocol table",
        maximum_bytes=MAX_PROTOCOL_BYTES,
        expected_sha256=PROTOCOL_TABLE_SHA256,
    )
    qualification = json.loads(qualification_raw.decode("utf-8", errors="strict"))
    neutral = json.loads(neutral_raw.decode("utf-8", errors="strict"))
    protocol_text = protocol_raw.decode("utf-8", errors="strict")
    validate_bindings(qualification, neutral, protocol_text)

    with verified_archive(SOURCE_ARCHIVE) as (archive, archive_identity):
        members = read_allowlisted_members(archive)
        facts = derive_method_facts(members)

    completed_at = utc_now()
    return {
        "schema_version": "ep21.public_method_semantics_audit.v1",
        "astra_analysis_id": "source_qualification",
        "attempt": attempt,
        "started_at": started_at,
        "completed_at": completed_at,
        "process_id": os.getpid(),
        "scope": "inert_allowlisted_public_source_method_semantics",
        "outcome": "completed_tagged_source_method_facts_with_choices_and_dependencies_unresolved",
        "bindings": {
            "qualification_report": {
                **qualification_identity,
                "schema_version": qualification["schema_version"],
                "attempt": qualification["attempt"],
            },
            "neutral_conformance_report": {
                **neutral_identity,
                "schema_version": neutral["schema_version"],
                "attempt": neutral["attempt"],
            },
            "protocol_table": protocol_identity,
            "source_archive": archive_identity,
        },
        "allowlisted_source_members": {
            key: {field: value[field] for field in ("path", "size_bytes", "sha256")}
            for key, value in members.items()
        },
        "tagged_source_facts_recorded_not_adopted": facts,
        "retained_unresolved": {
            "scientist_protocol_choices": [
                "paper_ReLU_vs_tagged_LeakyReLU_0_2",
                "paper_vs_tagged_content_loss_weighting",
                "paper_eight_vs_tagged_nineteen_reconstruction_layers",
                "paper_Caffe_224_vs_official_deploy_227_vs_tagged_PyTorch_227_backend_and_input",
                "port_or_select_brain_loss_regularization_for_LAION_tau_train",
            ],
            "fastl2lir": [
                "authenticated_fastl2lir_0_9_source_and_runtime",
                "voxel_ranking_ties_and_fewer_than_500_behavior",
                "mathematical_intercept_and_coefficient_semantics",
            ],
            "bdpy": [
                "authenticated_main_environment_bdpy_revision",
                "profile_and_pattern_correlation_exact_semantics",
                "pairwise_identification_tie_and_nan_semantics",
            ],
            "bootstrap_adaptation": [
                "bounded_empty_or_missing_pair_failure_rule",
                "deterministic_seed_and_worker_rule",
                "duplicate_and_multirow_pair_weighting_rule",
                "adoption_as_secondary_only_without_independence_claim",
            ],
            "data_and_access": [
                "LAION_DUA_license_and_Sherlock_risk_clearance",
                "authenticated_release_participants_roles_tau_splits_overlaps_betas_QC_and_masks",
                "role_filtered_regular_only_feature_and_scoring_handoffs",
                "later_explicit_empirical_execution_authorization",
            ],
        },
        "boundary": {
            "upstream_python_imported_or_executed": False,
            "zip_members_extracted": False,
            "bundled_result_csv_read": False,
            "model_or_data_payload_read": False,
            "feature_or_scoring_handoff_accessed": False,
            "empirical_or_neural_value_accessed": False,
            "ood_payload_accessed": False,
            "ep04_or_legacy_scratch_accessed": False,
            "checkpoint_deserialized": False,
            "tensor_or_device_created": False,
            "optimizer_or_inference_executed": False,
        },
        "protocol_effect": {
            "tagged_source_facts_available_to_future_lock_resolver": True,
            "scientific_choice_adopted": False,
            "dependency_semantics_fully_authenticated": False,
            "protocol_lock_emitted": False,
            "empirical_access_authorized": False,
            "empirical_readiness": False,
        },
        "next_action": "authenticate the exact fastl2lir and bdpy dependency sources if available, wire the three attempt-5 guards into an EP21 adapter, and obtain the scientist/steward dispositions before a fail-closed protocol lock; do not access empirical payloads",
    }


def _read_all_descriptor(descriptor: int, maximum_bytes: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    count = 0
    while True:
        chunk = os.read(descriptor, min(READ_CHUNK, maximum_bytes + 1 - count))
        if not chunk:
            break
        chunks.append(chunk)
        count += len(chunk)
        if count > maximum_bytes:
            raise ValueError("published report exceeded its size bound")
    return b"".join(chunks)


def _same_published_file(left: os.stat_result, right: os.stat_result) -> bool:
    fields = ("st_dev", "st_ino", "st_mode", "st_uid", "st_gid", "st_size", "st_nlink")
    return all(getattr(left, field) == getattr(right, field) for field in fields)


def _unlink_if_identity(
    directory_descriptor: int,
    name: str,
    expected_device: int,
    expected_inode: int,
) -> bool:
    """Remove a private-directory entry only while it retains the known inode."""
    try:
        observed = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return False
    if observed.st_dev != expected_device or observed.st_ino != expected_inode:
        return False
    os.unlink(name, dir_fd=directory_descriptor)
    return True


def atomic_publish_json(directory: Path, name: str, report: Mapping[str, Any]) -> dict[str, Any]:
    if "/" in name or name in {"", ".", ".."}:
        raise ValueError("invalid report basename")
    directory_descriptor = open_nofollow(
        directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    )
    temporary_name = f".{name}.{os.getpid()}.{secrets.token_hex(8)}.partial"
    report_descriptor = -1
    temporary_exists = False
    temporary_device: int | None = None
    temporary_inode: int | None = None
    published_device: int | None = None
    published_inode: int | None = None
    try:
        directory_info = os.fstat(directory_descriptor)
        if not stat.S_ISDIR(directory_info.st_mode):
            raise ValueError("output path is not a directory")
        if directory_info.st_uid != os.getuid():
            raise ValueError("output directory is not owned by the current user")
        if directory_info.st_mode & 0o777 != 0o700:
            raise ValueError("output directory permission bits must be exactly 0700")
        if directory_info.st_mode & (stat.S_ISUID | stat.S_ISVTX):
            raise ValueError("output directory has forbidden special mode bits")
        try:
            os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(f"refusing existing report: {directory / name}")
        payload = (
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
        report_descriptor = os.open(
            temporary_name,
            os.O_RDWR
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=directory_descriptor,
        )
        temporary_exists = True
        created = os.fstat(report_descriptor)
        temporary_device = created.st_dev
        temporary_inode = created.st_ino
        if (
            not stat.S_ISREG(created.st_mode)
            or created.st_nlink != 1
            or stat.S_IMODE(created.st_mode) != 0o600
            or created.st_size != 0
        ):
            raise ValueError("new temporary report has an invalid identity")
        offset = 0
        while offset < len(payload):
            written = os.write(report_descriptor, payload[offset:])
            if written <= 0:
                raise OSError("zero-progress report write")
            offset += written
        os.fsync(report_descriptor)
        before = os.fstat(report_descriptor)
        if before.st_dev != created.st_dev or before.st_ino != created.st_ino:
            raise ValueError("temporary report descriptor identity changed")
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError("temporary report is not a single-link regular file")
        if stat.S_IMODE(before.st_mode) != 0o600 or before.st_size != len(payload):
            raise ValueError("temporary report mode or size mismatch")
        if _read_all_descriptor(report_descriptor, MAX_JSON_BYTES) != payload:
            raise ValueError("temporary report readback mismatch")
        if _identity(os.fstat(report_descriptor)) != _identity(before):
            raise ValueError("temporary report changed during readback")
        temporary_entry = os.stat(
            temporary_name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if not _same_published_file(temporary_entry, before):
            raise ValueError("temporary report pathname no longer names the held inode")
        os.link(
            temporary_name,
            name,
            src_dir_fd=directory_descriptor,
            dst_dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        held_after_link = os.fstat(report_descriptor)
        final_after_link = os.stat(
            name, dir_fd=directory_descriptor, follow_symlinks=False
        )
        published_device = final_after_link.st_dev
        published_inode = final_after_link.st_ino
        if (
            held_after_link.st_nlink != 2
            or not _same_published_file(final_after_link, held_after_link)
        ):
            raise ValueError("published report pathname does not name the held inode")
        temporary_after_link = os.stat(
            temporary_name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if not _same_published_file(temporary_after_link, held_after_link):
            raise ValueError("temporary report pathname changed during publication")
        if not _unlink_if_identity(
            directory_descriptor,
            temporary_name,
            held_after_link.st_dev,
            held_after_link.st_ino,
        ):
            raise ValueError("temporary report pathname changed before unlink")
        temporary_exists = False
        os.fsync(directory_descriptor)
        final_info = os.fstat(report_descriptor)
        if (
            final_info.st_nlink != 1
            or final_info.st_uid != os.getuid()
            or stat.S_IMODE(final_info.st_mode) != 0o600
            or final_info.st_size != len(payload)
            or _read_all_descriptor(report_descriptor, MAX_JSON_BYTES) != payload
        ):
            raise ValueError("published report identity or readback mismatch")
        held_path_descriptor = os.open(
            name,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0),
            dir_fd=directory_descriptor,
        )
        try:
            held_path_info = os.fstat(held_path_descriptor)
            if (
                not _same_published_file(held_path_info, final_info)
                or _read_all_descriptor(held_path_descriptor, MAX_JSON_BYTES) != payload
            ):
                raise ValueError("held-directory final pathname verification failed")
        finally:
            os.close(held_path_descriptor)

        rebound_directory_descriptor = open_nofollow(
            directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            rebound_directory_info = os.fstat(rebound_directory_descriptor)
            if (
                rebound_directory_info.st_dev != directory_info.st_dev
                or rebound_directory_info.st_ino != directory_info.st_ino
                or rebound_directory_info.st_uid != directory_info.st_uid
                or rebound_directory_info.st_mode != directory_info.st_mode
            ):
                raise ValueError("expected output-directory pathname was rebound")
            rebound_report_descriptor = os.open(
                name,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0),
                dir_fd=rebound_directory_descriptor,
            )
            try:
                rebound_report_info = os.fstat(rebound_report_descriptor)
                if (
                    not _same_published_file(rebound_report_info, final_info)
                    or _read_all_descriptor(rebound_report_descriptor, MAX_JSON_BYTES)
                    != payload
                ):
                    raise ValueError("expected final report pathname was rebound")
            finally:
                os.close(rebound_report_descriptor)
        finally:
            os.close(rebound_directory_descriptor)
        if _identity(os.fstat(report_descriptor)) != _identity(final_info):
            raise ValueError("published report changed during final verification")
        return {
            "path": str(directory / name),
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "mode": "0600",
            "single_link": True,
            "atomic_no_overwrite_publication": True,
        }
    except Exception:
        if published_device is not None and published_inode is not None:
            _unlink_if_identity(
                directory_descriptor, name, published_device, published_inode
            )
        raise
    finally:
        if report_descriptor >= 0:
            os.close(report_descriptor)
        if (
            temporary_exists
            and temporary_device is not None
            and temporary_inode is not None
        ):
            _unlink_if_identity(
                directory_descriptor,
                temporary_name,
                temporary_device,
                temporary_inode,
            )
        os.close(directory_descriptor)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", type=int, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            {
                "event": "ep21_public_method_semantics_audit_started",
                "attempt": args.attempt,
                "pid": os.getpid(),
                "started_at": utc_now(),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    report = build_report(args.attempt)
    output_identity = atomic_publish_json(OUTPUT_DIR, OUTPUT_NAME, report)
    print(
        json.dumps(
            {
                "event": "ep21_public_method_semantics_audit_completed",
                "attempt": args.attempt,
                "pid": os.getpid(),
                "completed_at": report["completed_at"],
                "output": output_identity,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
