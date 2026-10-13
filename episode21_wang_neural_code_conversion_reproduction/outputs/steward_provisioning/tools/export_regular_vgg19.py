"""Approved public/lock prelude followed by sealed regular-only VGG features."""
import csv
import argparse
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import zipfile

BASE = Path(__file__).resolve().parents[3]
PROV = BASE / "outputs/steward_provisioning"
# The public prelude never imports this image-bearing stage for constants.
sys.path.insert(0, str(BASE / "outputs/pipeline_conformance/tools"))
from public_runtime_harness import PUBLIC_MEMBERS, RESOURCES as PUBLIC_RESOURCES, run_in_allocation
from feature_protocol import (LAYERS, SHAPES, RESOURCES, bind_feature_export_context,
                              complete_protocol_lock, validate_authorization)


def require(condition, code):
    if not condition:
        raise RuntimeError(code)


def main(authorization_path=None):
    os.umask(0o077)
    job = os.environ.get("SLURM_JOB_ID")
    require(job is not None, "Slurm_required")
    require(sys.version_info[:3] == (3, 12, 1), "intended_interpreter_required")
    auth_path = (Path(authorization_path).resolve() if authorization_path is not None
                 else BASE / "outputs/regular_vgg19_export_authorization_2026-10-02.json")
    require(auth_path.parent == BASE / "outputs" and "authorization" in auth_path.name
            and auth_path.suffix == ".json", "explicit_episode_authorization_path_required")
    auth = json.loads(auth_path.read_text())
    validate_authorization(auth)
    require(int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) == 4 and int(os.environ.get("SLURM_MEM_PER_NODE", 0)) == 16384, "allocation_mismatch")
    # Never restart the allocation clock after either required prelude.
    remaining = int(os.environ["SLURM_JOB_END_TIME"]) - time.time() - 120
    require(remaining > 0, "allocation_deadline_reached")
    deadline = time.monotonic() + min(RESOURCES["minutes"] * 60 - 120, remaining)

    def check():
        require(time.monotonic() < deadline, "deadline_reached")

    public_report = run_in_allocation(auth_path)
    complete_protocol_lock(public_report, check, auth_path)
    context = bind_feature_export_context(public_report, auth_path)
    with (PROV / "regular_vgg19_export_job_claim.json").open("x") as f:
        json.dump({"job_id": job, "purpose": auth["purpose"],
                   "protocol_lock": str(context.lock_path)}, f)
    # Establish owned cache/tmp before heavy imports can create them.
    scratch = Path(os.environ["SCRATCH"]) / "br_autoresearch" / BASE.name / f"vgg19-export-{job}"
    scratch.mkdir(mode=0o700, parents=True, exist_ok=False)
    (scratch / "cache").mkdir(mode=0o700)
    (scratch / "tmp").mkdir(mode=0o700)
    os.environ.update(XDG_CACHE_HOME=str(scratch / "cache"),
                      TORCH_HOME=str(scratch / "cache/torch"), TMPDIR=str(scratch / "tmp"))
    def used_temp():
        return sum(p.stat().st_size for p in scratch.rglob("*") if p.is_file())

    import numpy as np
    import h5py
    import torch
    from PIL import Image
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "one_allocated_GPU_required")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(42)
    source = context.source
    rows = context.rows
    require(len(rows) == 24681 and len({r["image_id"] for r in rows}) == 24681, "image_support_mismatch")
    participants = ("sub-01", "sub-03", "sub-05", "sub-06", "sub-07")
    groups = {p: [] for p in participants}
    groups.update({"tau_train": [], "tau_test": []})
    for ordinal, row in enumerate(rows):
        role = row["role"]
        require(role in ("subject_unique", "tau_train", "tau_test"), "forbidden_image_role")
        key = row["participant"] if role == "subject_unique" else role
        require(key in groups, "participant_mismatch")
        groups[key].append((ordinal, row))  # CSV ordinal indexes the regular H5.
    require(all(len(groups[p]) == 4712 for p in participants) and len(groups["tau_train"]) == 897 and len(groups["tau_test"]) == 224, "role_count_mismatch")
    destination = PROV / "regular_vgg19_sealed"
    report_path = PROV / "regular_vgg19_structural_report.json"
    require(not destination.exists() and not report_path.exists(), "do_not_overwrite")
    building = PROV / f"regular_vgg19_sealed.building.{job}"
    building.mkdir(mode=0o700)
    archive = Path(os.environ["SCRATCH"]) / "br_autoresearch" / BASE.name / "public_model_assets/VGG_ILSVRC_19_layers.zip"
    with zipfile.ZipFile(archive) as z:
        reserve = sum(z.getinfo("VGG_ILSVRC_19_layers/" + n).file_size for n in PUBLIC_MEMBERS)
        reserve += (source / "task-images_stimuli.h5").stat().st_size
        require(reserve + used_temp() <= RESOURCES["scratch_bytes"], "scratch_cap_exceeded")
        for name, digest in PUBLIC_MEMBERS.items():
            target = scratch / name
            h = hashlib.sha256()
            with z.open("VGG_ILSVRC_19_layers/" + name) as src, target.open("xb") as dst:
                while block := src.read(1 << 20):
                    check(); h.update(block); dst.write(block)
            require(h.hexdigest() == digest, "public_member_identity_mismatch")
    model_source = BASE / "outputs/source_pipeline_binding/bdpy_0_25/plaintext/bdpy/dl/torch/models.py"
    require(hashlib.sha256(model_source.read_bytes()).hexdigest() == "d1f10981e3dd95c9196db3bb97b9431158192605a65cf0644c19eb8a43ab8d5d", "architecture_identity_mismatch")
    spec = importlib.util.spec_from_file_location("ep21_pinned_vgg_architecture", model_source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    encoder = module.VGG19()
    state = torch.load(scratch / "VGG_ILSVRC_19_layers.pt", map_location="cpu", weights_only=True)
    encoder.load_state_dict(state, strict=True)
    del state
    encoder.eval().to("cuda")
    captured = {}
    indices = (0, 2, 5, 7, 10, 12, 14, 16, 19, 21, 23, 25, 28, 30, 32, 34)
    nodes = [encoder.features[i] for i in indices] + [encoder.classifier[i] for i in (0, 3, 6)]
    for layer, node in zip(LAYERS, nodes):
        def hook(_module, _inputs, output, name=layer):
            captured[name] = output.detach().clone()
        node.register_forward_hook(hook)
    mean_image = np.load(scratch / "ilsvrc_2012_mean.npy", allow_pickle=False)
    require(mean_image.ndim == 3 and mean_image.shape[0] == 3 and np.isfinite(mean_image).all(), "mean_file_invalid")
    mean_bgr = np.float32([mean_image[i].mean() for i in range(3)])[:, None, None]
    del mean_image
    staged_input = scratch / "regular_stimuli.h5"
    with (source / "task-images_stimuli.h5").open("rb") as src, staged_input.open("xb") as dst:
        while block := src.read(8 << 20):
            check(); dst.write(block)
    require(used_temp() <= RESOURCES["scratch_bytes"], "scratch_cap_exceeded")
    require(staged_input.stat().st_size == (source / "task-images_stimuli.h5").stat().st_size, "input_copy_incomplete")
    # Reserve the public stage's entire small output allowance within the
    # parent cap, rather than adding it to the approved 1.6-TB envelope.
    public_output_reserve = PUBLIC_RESOURCES["durable_bytes"]
    protocol_output_bytes = sum(p.stat().st_size for p in (
        context.lock_path, BASE / "outputs/pipeline_conformance/attempt-008/conformance_report.json",
        PROV / "regular_vgg19_export_job_claim.json"))
    durable = public_output_reserve + protocol_output_bytes
    summaries = []
    with h5py.File(staged_input, "r") as stimuli:
        require(stimuli["images"].shape == (24681,), "regular_H5_shape_mismatch")
        for key, selection in groups.items():
            check()
            stage = scratch / key
            stage.mkdir(mode=0o700)
            expected = len(selection) * sum(math.prod(s) * 4 + (math.prod(s) + 7) // 8 for s in SHAPES)
            fixed_inputs = used_temp()
            require(expected + fixed_inputs + 1_000_000_000 <= RESOURCES["scratch_bytes"], "scratch_cap_exceeded")
            require(durable + expected + 1_000_000_000 <= RESOURCES["durable_bytes"], "durable_cap_exceeded")
            with h5py.File(stage / "features.h5", "x") as out, h5py.File(stage / "finite_packed.h5", "x") as finite, (stage / "image_index.tsv").open("w") as index:
                data = {l: out.create_dataset(l, (len(selection),) + s, dtype="<f4") for l, s in zip(LAYERS, SHAPES)}
                flags = {l: finite.create_dataset(l, (len(selection), (math.prod(s) + 7) // 8), dtype="u1") for l, s in zip(LAYERS, SHAPES)}
                index.write("output_row_index\tregular_HDF5_row\timage_id\tparticipant\trole\n")
                for target_row, (regular_row, row) in enumerate(selection):
                    check()
                    with Image.open(io.BytesIO(stimuli["images"][regular_row].tobytes())) as image:
                        image = image.convert("RGB").resize((224, 224), Image.Resampling.BICUBIC)
                        pixels = np.float32(np.asarray(image).transpose(2, 0, 1)[::-1]) - mean_bgr
                    captured.clear()
                    with torch.inference_mode():
                        encoder(torch.from_numpy(pixels[None]).to("cuda"))
                    require(tuple(captured) == LAYERS, "feature_execution_order_mismatch")
                    for layer, shape in zip(LAYERS, SHAPES):
                        values = captured[layer][0].cpu().numpy()
                        require(values.shape == shape and values.dtype == np.float32, "feature_shape_dtype_mismatch")
                        valid = np.isfinite(values)
                        require(valid.all(), "nonfinite_true_features_stop_no_filtering")
                        data[layer][target_row] = values
                        flags[layer][target_row] = np.packbits(valid.reshape(-1), bitorder="little")
                    fields = (str(target_row), str(regular_row), row["image_id"], row["participant"], row["role"])
                    require(all("\t" not in x and "\n" not in x and "\r" not in x for x in fields), "invalid_index_identifier")
                    index.write("\t".join(fields) + "\n")
                    if target_row % 256 == 0:
                        out.flush(); finite.flush()
                        require(used_temp() <= RESOURCES["scratch_bytes"], "scratch_cap_exceeded")
                        print(json.dumps({"stage": "laion_feature_preparation", "partition": key,
                                          "images_done": target_row + 1, "images_total": len(selection)}), flush=True)
            target = building / key
            target.mkdir(mode=0o700)
            for file in stage.iterdir():
                check()
                durable += file.stat().st_size
                require(durable <= RESOURCES["durable_bytes"], "durable_cap_exceeded")
                with file.open("rb") as src, (target / file.name).open("xb") as dst:
                    while block := src.read(8 << 20):
                        check(); dst.write(block)
                    dst.flush(); os.fsync(dst.fileno())
                require((target / file.name).stat().st_size == file.stat().st_size, "publication_size_mismatch")
                (target / file.name).chmod(0o400)
            target.chmod(0o500)
            shutil.rmtree(stage)
            summaries.append({"partition": key, "images": len(selection), "nonfinite_elements": 0})
            print(json.dumps({"partition": key, "status": "real_regular_features_provisioned_sealed"}), flush=True)
    report = {"publication": "complete", "job_id": job, "contains_feature_values": False,
              "contains_neural_values": False, "model_fitting_scoring_run": False,
              "partitions": summaries, "layer_shapes": dict(zip(LAYERS, SHAPES)),
              "feature_dtype": "float32", "finite_mask_bitorder": "little", "feature_axes": "NCHW_or_Nvector",
              "durable_bytes": durable - public_output_reserve - protocol_output_bytes,
              "public_stage_output_reserve_bytes": public_output_reserve,
              "protocol_and_claim_metadata_bytes": protocol_output_bytes,
              "allocation_durable_accounted_bytes": durable,
              "resources": auth["resources"], "checkpoint": PUBLIC_MEMBERS,
              "implementation_binding": "outputs/protocol_implementation_bindings_2026-09-30.yaml",
              "runtime": {"torch": torch.__version__, "numpy": np.__version__, "h5py": h5py.__version__},
              "TF32": False, "AMP": False, "OOD_access": False, "historical_Caffe_parity": "unverified",
              "protocol_lock": str(context.lock_path), "authorization": str(auth_path),
              "HDF5_row_rule": "source_index_CSV_ordinal"}
    encoded = (json.dumps(report, indent=2) + "\n").encode()
    require(durable + len(encoded) <= RESOURCES["durable_bytes"], "durable_cap_exceeded")
    building.chmod(0o500); building.rename(destination)
    shutil.rmtree(scratch)
    pending = report_path.with_suffix(".json.pending")
    with pending.open("xb") as f:
        f.write(encoded); f.flush(); os.fsync(f.fileno())
    pending.chmod(0o400); pending.rename(report_path)
    print(json.dumps({"stage": "laion_feature_preparation", "status": "complete_regular_only_features_sealed",
                      "images": 24681, "durable_bytes": report["durable_bytes"]}), flush=True)
    print(json.dumps({"publication": "complete", "structural_report": str(report_path)}))


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--authorization", type=Path)
        main(parser.parse_args().authorization)
    except Exception as exc:
        print(json.dumps({"publication": "incomplete", "error_type": type(exc).__name__,
                          "error_code": str(exc) if type(exc) is RuntimeError else "see_failure_type_no_values_logged"}), flush=True)
        raise SystemExit(1)
