#!/usr/bin/env python3
"""Convert public BVLC AlexNet into the pinned BdPy architecture, on CPU.

This creates an explicitly adapted evaluator, not the unavailable author-lab
checkpoint. Only synthetic RGB images are used. Both exact parameter mapping
and independent OpenCV/Caffe versus PyTorch pre-ReLU execution must pass before
the converted state dictionary is published. Run inside a Slurm allocation on
Sherlock; this command never downloads assets or starts empirical evaluation.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import socket
import tempfile


LAYERS = {
    "conv1": "features.0",
    "conv2": "features.4",
    "conv3": "features.8",
    "conv4": "features.10",
    "conv5": "features.12",
    "fc6": "classifier.0",
    "fc7": "classifier.2",
    "fc8": "classifier.4",
}
# Fixed before inspecting outputs. Float32 implementations accumulate sums in
# different orders, especially the 9,216-input fc6 products. Every element must
# satisfy abs(torch - caffe) <= ATOL + RTOL * abs(caffe); not a global-peak test.
ATOL = 1e-3
RTOL = 1e-4


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--caffemodel", type=Path, required=True)
    parser.add_argument("--deploy", type=Path, required=True)
    parser.add_argument("--bdpy-models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def require_compute():
    hostname = socket.gethostname()
    if "-ln" in hostname or not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run this model conversion inside a Slurm compute allocation.")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[key] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    return hostname


def load_bdpy_model(path):
    spec = importlib.util.spec_from_file_location("ep21_pinned_bdpy_models", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import pinned model definitions: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.AlexNet().cpu().eval()


def extract_parameters(net, model, np, torch):
    expected = model.state_dict()
    mapped = {}
    for caffe_name, torch_name in LAYERS.items():
        layer_id = net.getLayerId(caffe_name)
        if layer_id <= 0:
            raise ValueError(f"Missing Caffe layer: {caffe_name}")
        blobs = net.getLayer(layer_id).blobs
        if len(blobs) != 2:
            raise ValueError(f"Expected weight and bias for {caffe_name}, got {len(blobs)}")
        for suffix, blob in zip(("weight", "bias"), blobs):
            key = f"{torch_name}.{suffix}"
            array = np.asarray(blob)
            if array.dtype != np.float32 or not np.isfinite(array).all():
                raise ValueError(f"Not a finite float32 parameter: {key}")
            # OpenCV exposes Caffe biases as (1, n), but torch expects (n,).
            # Do not reshape weights: an incompatible FC order must fail.
            if suffix == "bias":
                array = array.reshape(-1)
            if tuple(array.shape) != tuple(expected[key].shape):
                raise ValueError(f"Shape mismatch for {key}: {array.shape} != {expected[key].shape}")
            tensor = torch.from_numpy(array.copy())
            if not np.array_equal(tensor.numpy(), array):
                raise ValueError(f"Parameter mapping changed values for {key}")
            mapped[key] = tensor
    if set(mapped) != set(expected) or len(mapped) != 16:
        raise ValueError("Converted keys do not exactly cover the 16 BdPy AlexNet tensors.")
    model.load_state_dict(mapped, strict=True)
    for key, tensor in model.state_dict().items():
        if not torch.equal(tensor, mapped[key]):
            raise ValueError(f"Strict loading changed values for {key}")
    return mapped


def synthetic_fixtures(np):
    # Deliberately non-square RGB fixtures exercise direct resize, channel order
    # and pixel scale, not center crop or ImageNet [0, 1] normalization.
    yield "black_nonsquare", np.zeros((91, 137, 3), dtype=np.uint8)
    yield "color_nonsquare", np.broadcast_to(
        np.array([17, 101, 233], dtype=np.uint8), (109, 71, 3)
    ).copy()
    y, x = np.indices((103, 151))
    yield "structured_nonsquare", np.stack(
        (x * 7 % 256, y * 11 % 256, ((x // 9 + y // 7) % 2) * 255), axis=-1
    ).astype(np.uint8)
    yield "random_nonsquare_seed_21", np.random.default_rng(21).integers(
        0, 256, size=(149, 97, 3), dtype=np.uint8
    )


def preprocess(rgb, np, Image):
    resized = np.asarray(Image.fromarray(rgb).convert("RGB").resize(
        (227, 227), resample=Image.Resampling.LANCZOS
    ), dtype=np.float32)
    bgr = resized[:, :, ::-1] - np.array([104, 117, 123], dtype=np.float32)
    return np.ascontiguousarray(bgr.transpose(2, 0, 1)[None, ...])


def verify_execution(net, model, np, torch, Image, report):
    captured = {}
    handles = []

    def hook_for(name):
        def capture(_module, _args, value):
            # Clone now so the comparison is pre-ReLU even if upstream code
            # later changes a following activation to operate in place.
            captured[name] = value.detach().cpu().numpy().copy()
        return capture

    for name, module_name in LAYERS.items():
        handles.append(model.get_submodule(module_name).register_forward_hook(hook_for(name)))
    try:
        for fixture_name, rgb in synthetic_fixtures(np):
            blob = preprocess(rgb, np, Image)
            if fixture_name == "color_nonsquare":
                expected = np.array([129, -16, -106], dtype=np.float32)
                if not np.array_equal(blob[0], np.broadcast_to(expected[:, None, None], (3, 227, 227))):
                    raise AssertionError("RGB-to-BGR, pixel scale or mean subtraction is wrong.")
            captured.clear()
            with torch.inference_mode():
                model(torch.from_numpy(blob))
            for layer in LAYERS:
                net.setInput(blob)
                # End each independent forward at this layer. Subsequent
                # in-place Caffe ReLUs cannot overwrite the requested output.
                reference = net.forward(layer).copy()
                actual = captured[layer]
                if actual.shape != reference.shape:
                    raise AssertionError(f"{fixture_name}/{layer}: {actual.shape} != {reference.shape}")
                if not np.isfinite(actual).all() or not np.isfinite(reference).all():
                    raise AssertionError(f"{fixture_name}/{layer}: nonfinite activation")
                difference = np.abs(actual.astype(np.float64) - reference.astype(np.float64))
                limit = ATOL + RTOL * np.abs(reference.astype(np.float64))
                passed = bool(np.all(difference <= limit))
                result = {
                    "fixture": fixture_name,
                    "layer": layer,
                    "shape": list(actual.shape),
                    "max_abs_error": float(difference.max()),
                    "max_tolerance_fraction": float(np.max(difference / limit)),
                    "violating_elements": int(np.count_nonzero(difference > limit)),
                    "passed": passed,
                }
                report["activation_checks"].append(result)
                print(json.dumps(result, sort_keys=True), flush=True)
                if not passed:
                    raise AssertionError(f"Predeclared float32 parity failed: {fixture_name}/{layer}")
    finally:
        for handle in handles:
            handle.remove()


def publish_checkpoint(path, state, model, torch):
    # Same-filesystem temporary file, then exclusive hard-link publication:
    # incomplete or unverifiable state is never exposed at the final path.
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.",
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            torch.save(state, stream)
        loaded = torch.load(temporary, map_location="cpu", weights_only=True)
        model.load_state_dict(loaded, strict=True)
        if set(loaded) != set(state) or not all(torch.equal(loaded[k], state[k]) for k in state):
            raise AssertionError("Safe checkpoint reload changed parameter values.")
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    args = parse_args()
    hostname = require_compute()
    for path in (args.output, args.report):
        if path is not None and (path.exists() or path.is_symlink()):
            raise FileExistsError(f"Refusing to overwrite: {path}")
    for path in (args.caffemodel, args.deploy, args.bdpy_models):
        if not path.is_file():
            raise FileNotFoundError(path)
    import cv2
    import numpy as np
    from PIL import Image, __version__ as pillow_version
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    report = {
        "status": "running",
        "evaluator": "public_BVLC_AlexNet_converted_to_pinned_BdPy",
        "author_lab_checkpoint_identity_verified": False,
        "hostname": hostname,
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "sources": {"caffemodel": str(args.caffemodel), "deploy": str(args.deploy),
                    "bdpy_models": str(args.bdpy_models)},
        "runtime": {"torch": torch.__version__, "opencv": cv2.__version__,
                    "numpy": np.__version__, "pillow": pillow_version},
        "preprocessing": "RGB_direct_Lanczos_227_BCHW_BGR_float32_0_255_minus_104_117_123",
        "reference_backend": "OpenCV_Caffe_importer_CPU_fusion_disabled",
        "tolerance": {"atol": ATOL, "rtol": RTOL, "rule": "every_element"},
        "empirical_images_or_neural_values_read": False,
        "activation_checks": [],
    }
    try:
        net = cv2.dnn.readNetFromCaffe(str(args.deploy), str(args.caffemodel))
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        net.enableFusion(False)
        model = load_bdpy_model(args.bdpy_models)
        state = extract_parameters(net, model, np, torch)
        report["exact_parameter_mapping"] = {"tensor_count": len(state), "passed": True}
        verify_execution(net, model, np, torch, Image, report)
        if torch.cuda.is_initialized():
            raise AssertionError("Unexpected CUDA initialization")
        publish_checkpoint(args.output, state, model, torch)
        report.update(status="passed", output=str(args.output), output_bytes=args.output.stat().st_size,
                      safe_weights_only_reload=True)
        print(f"PASS: published adapted evaluator to {args.output}", flush=True)
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            with args.report.open("x", encoding="utf-8") as stream:
                json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
                stream.write("\n")


if __name__ == "__main__":
    main()
