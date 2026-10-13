"""Unactivated, explicitly authorized two-stage continuation of job 46424955.

No scheduler, retries, conformance reruns, neural access, or OOD access. The
original exporter/claim/lock are immutable. New authority must separately
approve the exact resources and conditional staging release below.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import signal
import socket
import sys
import time

BASE = Path(__file__).resolve().parents[3]
PROV = BASE / "outputs/steward_provisioning"
OLD_JOB = "46424955"
OLD_BUILDING = PROV / f"regular_vgg19_sealed.building.{OLD_JOB}"
ROOT = PROV / "regular_vgg19_continuation_20261005"
FINAL = PROV / "regular_vgg19_sealed"
FINAL_REPORT = PROV / "regular_vgg19_structural_report.json"
FILES = ("features.h5", "finite_packed.h5", "image_index.tsv")
REUSED = ("sub-01", "sub-03")
REMAINING = ("sub-06", "sub-07", "tau_train", "tau_test")
PARTITIONS = REUSED + ("sub-05",) + REMAINING
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")
STAGES = {
    "publish-sub05": {"jobs": 1, "cpus": 1, "memory_gib": 4, "gpus": 0, "seconds": 7200},
    "remaining-export": {"jobs": 1, "cpus": 4, "memory_gib": 16, "gpus": 1, "seconds": 36000},
}
DURABLE_LIMIT = 1_700_000_000_000
SCRATCH_LIMIT = 340_000_000_000
AUTH_FIELDS = {
    "schema_version": "ep21.regular_vgg19_continuation.v1", "episode_id": "ep21",
    "purpose": "complete_regular_vgg19_after_46424955", "approved": True,
    "original_job_id": OLD_JOB, "stages": STAGES,
    "durable_bytes": DURABLE_LIMIT, "scratch_bytes": SCRATCH_LIMIT,
    "reuse_sealed_partitions": list(REUSED), "publish_staged_partition": "sub-05",
    "new_forward_partitions": list(REMAINING), "new_forward_images": 10545,
    "release_sub05_staging_after_complete_durable_publication": True,
    "release_new_partition_staging_after_complete_durable_publication": True,
    "preserve_old_durable_partial": True, "reuse_existing_lock_and_conformance": True,
    "allow_regular_image_values": True, "allow_neural_values": False,
    "allow_ood": False, "allow_fitting": False, "allow_scoring": False,
    "allow_reconstruction": False, "allow_external_site_execution": False,
    "automatic_retry": False, "automatic_extension": False,
}


def require(condition, code):
    if not condition:
        raise RuntimeError(code)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, record, *, replace=False):
    """Only the current stage status is replaceable; artifacts remain immutable."""
    require(replace or not path.exists(), "existing_continuation_artifact")
    pending = path.with_suffix(path.suffix + ".pending")
    with pending.open("w" if replace else "x") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    pending.replace(path)


def scratch_paths():
    parent = Path(os.environ["SCRATCH"]) / "br_autoresearch" / BASE.name
    return parent / f"vgg19-export-{OLD_JOB}", parent / "vgg19-continuation-20261005"


def authorize(path):
    path = path.resolve()
    require(path.parent == BASE / "outputs" and "authorization" in path.name
            and path.suffix == ".json", "explicit_episode_authorization_required")
    record = read_json(path)
    require(record.get("approved") is True and all(record.get(k) == v for k, v in AUTH_FIELDS.items()),
            "new_resource_and_staging_release_approval_required")
    return path


def load_context(check):
    """Reuse passing conformance and the original lock, without rewriting either."""
    sys.path.insert(0, str(BASE / "outputs/pipeline_conformance/tools"))
    import feature_protocol as fp
    from public_runtime_harness import PUBLIC_MEMBERS, ARCHITECTURE_SHA256
    public = read_json(BASE / "outputs/pipeline_conformance/attempt-007/public_runtime_conformance_report.json")
    original = read_json(BASE / "outputs/regular_vgg19_runtime_retry_authorization_2026-10-03.json")
    lock = fp.read_yaml(fp.LOCK)
    require(lock.get("job_id") == OLD_JOB, "original_lock_job_required")
    rows, groups, alignment, _ = fp.image_roles()
    fp.validate_feature_lock(lock, original, public, alignment)
    complete = read_json(fp.CONFORMANCE)
    require(complete.get("job_id") == OLD_JOB and complete.get("outcome")
            == "passed_required_shared_and_Finding_A_protocol_conformance",
            "original_completed_conformance_required")
    claim = read_json(PROV / "regular_vgg19_export_job_claim.json")
    require(claim == {"job_id": OLD_JOB, "purpose": original["purpose"],
                      "protocol_lock": str(fp.LOCK)}, "original_claim_identity_mismatch")
    records = [json.loads(line) for line in
               (PROV / f"regular_vgg19_attempt003-{OLD_JOB}.out").read_text().splitlines()
               if line.startswith("{")]
    require(records[-1] == {"publication": "incomplete", "error_type": "RuntimeError",
                           "error_code": "deadline_reached"}, "original_deadline_failure_required")
    sealed = [r["partition"] for r in records
              if r.get("status") == "real_regular_features_provisioned_sealed"]
    require(sealed == list(REUSED), "original_sealed_partition_identity_mismatch")
    require(tuple(groups) == PARTITIONS and len(rows) == 24681,
            "original_partition_order_required")
    check()
    return fp, groups, lock, PUBLIC_MEMBERS, ARCHITECTURE_SHA256


def partition_metadata(path, selection, fp, check):
    """Header/index checks only: no feature or mask dataset values are read."""
    import h5py
    require(path.is_dir() and {p.name for p in path.iterdir()} == set(FILES),
            "complete_three_file_partition_required")
    with (path / "image_index.tsv").open() as stream:
        rows = csv.reader(stream, delimiter="\t")
        require(next(rows) == ["output_row_index", "regular_HDF5_row", "image_id", "participant", "role"],
                "index_header_mismatch")
        for target, (ordinal, row) in enumerate(selection):
            check()
            require(next(rows, None) == [str(target), str(ordinal), row["image_id"],
                                        row["participant"], row["role"]], "index_identity_mismatch")
        require(next(rows, None) is None, "index_extra_rows")
    with h5py.File(path / "features.h5", "r") as features, h5py.File(path / "finite_packed.h5", "r") as finite:
        require(set(features) == set(fp.LAYERS) == set(finite), "layer_set_mismatch")
        for layer, shape in zip(fp.LAYERS, fp.SHAPES):
            check()
            require(features[layer].shape == (len(selection),) + tuple(shape)
                    and features[layer].dtype.str == "<f4"
                    and finite[layer].shape == (len(selection), (math.prod(shape) + 7) // 8)
                    and finite[layer].dtype.str == "|u1", "partition_shape_dtype_mismatch")
    return {name: (path / name).stat().st_size for name in FILES}


def durable_bytes():
    # Exact owned/history roots only; hardlinks are counted once, not recopied.
    seen = set(); total = 10_000_000  # Original public-stage output reserve.
    for root in (OLD_BUILDING, ROOT, FINAL):
        for key in PARTITIONS:
            for name in FILES:
                path = root / key / name
                if path.is_file():
                    stat = path.stat(); identity = (stat.st_dev, stat.st_ino)
                    if identity not in seen:
                        seen.add(identity); total += stat.st_size
    metadata = [BASE / "outputs/protocol_lock.yaml",
                BASE / "outputs/pipeline_conformance/attempt-008/conformance_report.json",
                PROV / "regular_vgg19_export_job_claim.json", FINAL_REPORT]
    metadata += [PROV / f"regular_vgg19_continuation_20261005_{stage}_claim.json" for stage in STAGES]
    metadata += list(ROOT.glob("*.json"))
    total += sum(path.stat().st_size for path in metadata if path.is_file())
    total += 1_000_000  # Remaining bounded status/structural JSON publication headroom.
    return total


def scratch_bytes():
    old, new = scratch_paths()
    # These are the two narrowly owned attempt trees, not a cache/root search.
    return sum(p.stat().st_size for root in (old, new) if root.exists()
               for p in root.rglob("*") if p.is_file())


def publish_partition(stage, key, selection, fp, check):
    sizes = partition_metadata(stage, selection, fp, check)
    target = ROOT / key
    require(not target.exists(), "no_partition_retry_or_overwrite")
    require(durable_bytes() + sum(sizes.values()) <= DURABLE_LIMIT, "durable_cap_exceeded")
    target.mkdir(mode=0o700)
    for name in FILES:
        with (stage / name).open("rb") as src, (target / name).open("xb") as dst:
            while block := src.read(8 << 20):
                check(); dst.write(block)
            dst.flush(); os.fsync(dst.fileno())
        require((target / name).stat().st_size == sizes[name], "publication_size_mismatch")
        (target / name).chmod(0o400)
    require(partition_metadata(target, selection, fp, check) == sizes,
            "published_partition_metadata_mismatch")
    target.chmod(0o500)
    receipt = {"publication": "complete", "partition": key, "images": len(selection),
               "job_id": os.environ["SLURM_JOB_ID"], "source": str(stage),
               "destination": str(target), "file_bytes": sizes,
               "neural_values_read": False, "OOD_access": False,
               "old_durable_partial_preserved": True}
    write_json(ROOT / f"{key}_publication.json", receipt)
    check()
    # This exact release is part of new approval, never implied by old authority.
    for name in FILES:
        (stage / name).unlink()
    stage.rmdir()
    return receipt


def remaining_export(fp, groups, lock, members, architecture_sha, check, status, checkpoint):
    old, scratch = scratch_paths()
    scratch.mkdir(mode=0o700)
    for key in ("cache", "tmp"):
        (scratch / key).mkdir(mode=0o700)
    os.environ.update(XDG_CACHE_HOME=str(scratch / "cache"), TORCH_HOME=str(scratch / "cache/torch"),
                      TMPDIR=str(scratch / "tmp"))
    import numpy as np
    import h5py
    import torch
    import PIL
    from PIL import Image
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "one_allocated_GPU_required")
    require({"numpy": np.__version__, "torch": torch.__version__}
            == lock["runtime"]["loaded_module_versions"], "frozen_loaded_runtime_required")
    require(h5py.__version__ == lock["runtime"]["h5py"]
            and PIL.__version__ == lock["runtime"]["Pillow"], "frozen_HDF5_and_image_runtime_required")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True); torch.manual_seed(42)
    for name, digest in members.items():
        hashed = hashlib.sha256()
        with (old / name).open("rb") as stream:
            while block := stream.read(8 << 20):
                check(); hashed.update(block)
        require(hashed.hexdigest() == digest, "retained_public_asset_identity_mismatch")
    model_source = BASE / "outputs/source_pipeline_binding/bdpy_0_25/plaintext/bdpy/dl/torch/models.py"
    require(hashlib.sha256(model_source.read_bytes()).hexdigest() == architecture_sha,
            "architecture_identity_mismatch")
    definition = importlib.util.spec_from_file_location("ep21_pinned_vgg_architecture", model_source)
    module = importlib.util.module_from_spec(definition); definition.loader.exec_module(module)
    encoder = module.VGG19()
    state = torch.load(old / "VGG_ILSVRC_19_layers.pt", map_location="cpu", weights_only=True)
    encoder.load_state_dict(state, strict=True); del state
    encoder.eval().to("cuda")
    captured = {}
    indices = (0, 2, 5, 7, 10, 12, 14, 16, 19, 21, 23, 25, 28, 30, 32, 34)
    nodes = [encoder.features[i] for i in indices] + [encoder.classifier[i] for i in (0, 3, 6)]
    for layer, node in zip(fp.LAYERS, nodes):
        def hook(_module, _inputs, output, name=layer):
            captured[name] = output.detach().clone()
        node.register_forward_hook(hook)
    mean_image = np.load(old / "ilsvrc_2012_mean.npy", allow_pickle=False)
    require(mean_image.ndim == 3 and mean_image.shape[0] == 3 and np.isfinite(mean_image).all(),
            "mean_file_invalid")
    mean_bgr = np.float32([mean_image[i].mean() for i in range(3)])[:, None, None]
    del mean_image
    staged_input = old / "regular_stimuli.h5"
    require(staged_input.stat().st_size == (BASE / "inputs/laion_regular_feature_handoff/task-images_stimuli.h5").stat().st_size,
            "retained_input_copy_size_mismatch")
    with h5py.File(staged_input, "r") as stimuli:
        require(stimuli["images"].shape == (24681,), "regular_H5_shape_mismatch")
        for key in REMAINING:
            selection = groups[key]
            expected = len(selection) * sum(math.prod(s) * 4 + (math.prod(s) + 7) // 8 for s in fp.SHAPES)
            require(scratch_bytes() + expected + 1_000_000_000 <= SCRATCH_LIMIT, "scratch_cap_exceeded")
            require(durable_bytes() + expected + 1_000_000_000 <= DURABLE_LIMIT, "durable_cap_exceeded")
            stage = scratch / key; stage.mkdir(mode=0o700)
            status.update(partition=key, images_done=0, images_total=len(selection)); checkpoint()
            with h5py.File(stage / "features.h5", "x") as out, h5py.File(stage / "finite_packed.h5", "x") as finite, (stage / "image_index.tsv").open("x") as index:
                data = {l: out.create_dataset(l, (len(selection),) + tuple(s), dtype="<f4") for l, s in zip(fp.LAYERS, fp.SHAPES)}
                flags = {l: finite.create_dataset(l, (len(selection), (math.prod(s) + 7) // 8), dtype="u1") for l, s in zip(fp.LAYERS, fp.SHAPES)}
                index.write("output_row_index\tregular_HDF5_row\timage_id\tparticipant\trole\n")
                for target_row, (regular_row, row) in enumerate(selection):
                    check()
                    with Image.open(io.BytesIO(stimuli["images"][regular_row].tobytes())) as image:
                        image = image.convert("RGB").resize((224, 224), Image.Resampling.BICUBIC)
                        pixels = np.float32(np.asarray(image).transpose(2, 0, 1)[::-1]) - mean_bgr
                    captured.clear()
                    with torch.inference_mode():
                        encoder(torch.from_numpy(pixels[None]).to("cuda"))
                    require(tuple(captured) == fp.LAYERS, "feature_execution_order_mismatch")
                    for layer, shape in zip(fp.LAYERS, fp.SHAPES):
                        values = captured[layer][0].cpu().numpy()
                        require(values.shape == tuple(shape) and values.dtype == np.float32, "feature_shape_dtype_mismatch")
                        valid = np.isfinite(values)
                        require(valid.all(), "nonfinite_true_features_stop_no_filtering")
                        data[layer][target_row] = values
                        flags[layer][target_row] = np.packbits(valid.reshape(-1), bitorder="little")
                    fields = (str(target_row), str(regular_row), row["image_id"], row["participant"], row["role"])
                    require(all("\t" not in v and "\n" not in v and "\r" not in v for v in fields), "invalid_index_identifier")
                    index.write("\t".join(fields) + "\n")
                    if target_row % 256 == 0:
                        out.flush(); finite.flush(); index.flush()
                        status["images_done"] = target_row + 1; checkpoint()
                        require(scratch_bytes() <= SCRATCH_LIMIT, "scratch_cap_exceeded")
            receipt = publish_partition(stage, key, selection, fp, check)
            status["completed_partitions"].append(receipt); checkpoint()


def final_publication(fp, groups, lock, check, authorization, status):
    require(not FINAL.exists() and not FINAL_REPORT.exists(), "existing_full_handoff")
    building = PROV / "regular_vgg19_sealed.continuation-20261005.building"
    building.mkdir(mode=0o700)
    summaries = []
    for key in PARTITIONS:
        source = (OLD_BUILDING if key in REUSED else ROOT) / key
        sizes = partition_metadata(source, groups[key], fp, check)
        target = building / key; target.mkdir(mode=0o700)
        for name in FILES:
            check(); os.link(source / name, target / name)
        target.chmod(0o500)
        summaries.append({"partition": key, "images": len(groups[key]), "nonfinite_elements": 0,
                          "physical_source": str(source), "file_bytes": sizes})
    check(); building.chmod(0o500); building.rename(FINAL)
    report = {"publication": "complete", "job_id": os.environ["SLURM_JOB_ID"],
              "original_job_id": OLD_JOB, "authorization": str(authorization),
              "protocol_lock": str(fp.LOCK), "partitions": summaries,
              "layer_shapes": dict(zip(fp.LAYERS, fp.SHAPES)), "feature_dtype": "float32",
              "finite_mask_bitorder": "little", "feature_axes": "NCHW_or_Nvector",
              "contains_feature_values": False, "contains_neural_values": False,
              "model_fitting_scoring_run": False, "OOD_access": False,
              "HDF5_row_rule": "source_index_CSV_ordinal", "feature_recipe": lock["feature_recipe"],
              "runtime": lock["runtime"], "old_partial_preserved": True,
              "physical_durable_bytes_including_old_partial_and_metadata_reserve": durable_bytes(),
              "new_forward_images": 10545, "reused_generated_images": 14136,
              "hardlinks_add_no_second_feature_payload_copy": True}
    require(report["physical_durable_bytes_including_old_partial_and_metadata_reserve"] <= DURABLE_LIMIT,
            "durable_cap_exceeded")
    check(); write_json(FINAL_REPORT, report)
    status["final_report_created_by_this_stage"] = True
    check()


def run(args):
    started = time.monotonic(); os.umask(0o077)
    resources = STAGES[args.stage]
    require(os.environ.get("SLURM_JOB_ID") and "-ln" not in socket.gethostname(), "compute_shell_required")
    require(sys.version_info[:3] == (3, 12, 1), "original_python_required")
    require(os.environ.get("SLURM_CPUS_PER_TASK") == str(resources["cpus"])
            and all(os.environ.get(k) == str(resources["cpus"]) for k in THREADS), "cpu_thread_binding_mismatch")
    require(0 < args.max_seconds <= resources["seconds"] - 120, "explicit_stage_cap_required")
    deadline = started + args.max_seconds
    end = os.environ.get("SLURM_JOB_END_TIME", "")
    if end.isdigit():
        deadline = min(deadline, time.monotonic() + int(end) - time.time() - 120)

    def check():
        require(time.monotonic() < deadline, "deadline_reached")

    def interrupted(_signum, _frame):
        raise RuntimeError("deadline_or_scheduler_signal")

    check()
    authorization = authorize(args.authorization)
    claim = PROV / f"regular_vgg19_continuation_20261005_{args.stage}_claim.json"
    with claim.open("x") as stream:
        json.dump({"job_id": os.environ["SLURM_JOB_ID"], "stage": args.stage,
                   "authorization": str(authorization), "original_claim_preserved": True}, stream)
        stream.flush(); os.fsync(stream.fileno())
    if args.stage == "publish-sub05":
        ROOT.mkdir(mode=0o700)
    else:
        prior = read_json(ROOT / "publish-sub05_status.json")
        require(prior.get("status") == "completed" and prior.get("staging_released") is True
                and prior.get("authorization") == str(authorization), "successful_publication_stage_required")
    status = {"status": "running", "stage": args.stage, "job_id": os.environ["SLURM_JOB_ID"],
              "hostname": socket.gethostname(), "authorization": str(authorization),
              "resources": resources, "max_seconds": args.max_seconds,
              "completed_partitions": [], "neural_values_read": False, "OOD_access": False,
              "automatic_retry": False, "old_claim_and_partial_preserved": True}
    status_path = ROOT / f"{args.stage}_status.json"

    def checkpoint():
        status["elapsed_seconds"] = time.monotonic() - started
        write_json(status_path, status, replace=True)

    previous = {s: signal.getsignal(s) for s in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT)}
    try:
        for s in previous:
            signal.signal(s, interrupted)
        signal.setitimer(signal.ITIMER_REAL, max(0.001, deadline - time.monotonic()))
        checkpoint(); check()
        fp, groups, lock, members, architecture = load_context(check)
        for key in REUSED:
            partition_metadata(OLD_BUILDING / key, groups[key], fp, check)
        require(durable_bytes() <= DURABLE_LIMIT and scratch_bytes() <= SCRATCH_LIMIT, "existing_storage_cap_exceeded")
        old, _ = scratch_paths()
        if args.stage == "publish-sub05":
            status["completed_partitions"].append(publish_partition(old / "sub-05", "sub-05", groups["sub-05"], fp, check))
            status["staging_released"] = not (old / "sub-05").exists()
        else:
            require(not (old / "sub-05").exists(), "published_staging_must_be_released_before_next_partition")
            partition_metadata(ROOT / "sub-05", groups["sub-05"], fp, check)
            remaining_export(fp, groups, lock, members, architecture, check, status, checkpoint)
            final_publication(fp, groups, lock, check, authorization, status)
        check(); status["status"] = "completed"; checkpoint(); check()
    except Exception as exc:
        status.update(status="technical_incomplete", error_type=type(exc).__name__,
                      error_code=str(exc) if isinstance(exc, RuntimeError) else "see_failure_type")
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for s, handler in previous.items():
            signal.signal(s, handler)
    if time.monotonic() >= deadline:
        status.update(status="technical_incomplete", error_code="deadline_reached")
    checkpoint()
    if time.monotonic() >= deadline and status["status"] == "completed":
        status.update(status="technical_incomplete", error_code="deadline_reached_during_final_status")
        checkpoint()
    if status["status"] != "completed" and status.get("final_report_created_by_this_stage") is True:
        final = read_json(FINAL_REPORT)
        final.update(publication="incomplete", continuation_error=status.get("error_code"))
        write_json(FINAL_REPORT, final, replace=True)
    print(json.dumps(status), flush=True)
    return 0 if status["status"] == "completed" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=tuple(STAGES), required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, required=True)
    raise SystemExit(run(parser.parse_args()))
