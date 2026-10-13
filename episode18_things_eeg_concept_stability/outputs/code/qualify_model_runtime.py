#!/usr/bin/env python3
"""Load EP18's selected local models sequentially on CPU, without inference.

Reads only the explicit asset map, frozen registry, and listed public model
files. No images, EEG, concept metadata, downloads, or runtime boundary claim.
Run only in a coordinator-owned allocation after separate asset acquisition.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import resource
import signal
import socket
import time


EPISODE = Path(__file__).resolve().parents[2]
REGISTRY = EPISODE / "outputs/config/feature_registry.json"
REPORT_NAME = "model_runtime_qualification.json"
MAX_SECONDS = 1200
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")
ROUTES = ("supervised_resnet50_v1", "dino_resnet50", "clip_vit_b32_image",
          "concept_name_clip_text", "blip_caption_clip_text", "dinov2_vit_b14")
ASSETS = {"supervised_resnet50_v1": (None, "resnet50-0676ba61.pth"),
          "dino_resnet50": (None, "dino_resnet50_pretrain.pth"),
          "clip": ("openai/clip-vit-base-patch32", "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"),
          "blip": ("Salesforce/blip-image-captioning-base", "82a37760796d32b1411fe092ab5d4e227313294b"),
          "dinov2": ("facebook/dinov2-base", "f9e44c814b77203eaa57a6bdbbd535f21ede1415")}


def require_compute():
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise RuntimeError("Use the coordinator's actual compute-node allocation")
    if os.environ.get("SLURM_CPUS_PER_TASK") != "1":
        raise RuntimeError("This sequential loader requires exactly one CPU per task")
    if any(os.environ.get(name) != "1" for name in THREADS):
        raise RuntimeError("All six numerical thread limits must explicitly equal one")


def require(condition, description):
    if not condition:
        raise ValueError(description)


def read_json(path):
    with Path(path).open(encoding="utf-8") as source:
        return json.load(source)


def qualify_files(mapping, registry, check):
    """Hash only the declared selected files; no cache search or network access."""
    require(mapping["schema"] == "ep18_model_runtime_asset_map_v1", "Wrong map schema")
    require(set(mapping["assets"]) == set(ASSETS), "Selected asset set changed")
    root = Path(mapping["asset_root"])
    root = (EPISODE / root).resolve() if not root.is_absolute() else root.resolve()
    permitted = [(EPISODE / "outputs/cache/model_assets").resolve()]
    if os.environ.get("SCRATCH"):
        permitted.append((Path(os.environ["SCRATCH"]) /
                          "br_autoresearch/episode18_things_eeg_concept_stability/model_assets").resolve())
    require(root in permitted, "Asset root must be the explicit episode cache or task Scratch root")
    require(registry["text_encoder"]["revision"] == ASSETS["clip"][1], "CLIP revision changed")
    require(registry["features"]["blip_caption_clip_text"]["revision"] == ASSETS["blip"][1], "BLIP revision changed")
    require(registry["features"]["dinov2_vit_b14"]["revision"] == ASSETS["dinov2"][1], "DINOv2 revision changed")
    paths, evidence = {}, {}
    for key, (model_id, basename) in ASSETS.items():
        check()
        entry = mapping["assets"][key]
        local = root / entry["local_path"]
        require(not local.is_symlink() and local.resolve().is_relative_to(root), "Unselected asset routing")
        require(local.name == basename, "Checkpoint/revision basename changed")
        if model_id:
            require(entry["model_id"] == model_id and entry["revision"] == basename, "Selected model changed")
            require(local.is_dir(), "Missing exact local snapshot")
            require({p.name for p in local.iterdir()} == {f["name"] for f in entry["files"]},
                    "Snapshot must contain exactly the acquired declared files (no alternate weights/code)")
        else:
            require(len(entry["files"]) == 1 and entry["files"][0]["name"] == basename,
                    "Wrong direct checkpoint entry")
        paths[key] = str(local)
        evidence[key] = []
        for item in entry["files"]:
            check()
            require(Path(item["name"]).name == item["name"], "Asset file must be a basename")
            path = local / item["name"] if model_id else local
            require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root),
                    "Missing or externally routed declared file")
            size = path.stat().st_size
            require(size == item["bytes"], f"Wrong byte count: {path}")
            if "sha256" in item or "sha256_prefix" in item:
                kind, digest = "sha256", hashlib.sha256()
            elif "git_blob_sha1" in item:
                kind, digest = "git_blob_sha1", hashlib.sha1(f"blob {size}\0".encode())
            elif "source_metadata_md5" in item:
                kind, digest = "source_metadata_md5", hashlib.md5()
            else:
                raise ValueError("Declared asset has no source integrity evidence")
            with path.open("rb") as source:
                while chunk := source.read(4 * 1024 * 1024):
                    check()
                    digest.update(chunk)
            observed = digest.hexdigest()
            if "sha256_prefix" in item:
                require(observed.startswith(item["sha256_prefix"]), f"Official filename hash mismatch: {path}")
            else:
                require(observed == item[kind], f"Source integrity mismatch ({kind}): {path}")
            evidence[key].append({"path": str(path), "bytes": size, "integrity_algorithm": kind, "digest": observed,
                                  "source_url": item["url"], "source_integrity_checked": True})
    return paths, evidence


def qualify_versions(registry):
    modules = {"torch": "torch", "torchvision": "torchvision", "transformers": "transformers",
               "Pillow": "PIL", "numpy": "numpy", "scipy": "scipy", "scikit-image": "skimage"}
    observed = {"python": {"version": platform.python_version(), "path": os.sys.executable}}
    for name, module_name in modules.items():
        module = importlib.import_module(module_name)
        observed[name] = {"version": str(module.__version__), "path": str(module.__file__)}
    for name, expected in registry["runtime_recipe"]["target_versions"].items():
        require(observed[name]["version"].split("+")[0] == expected, f"Unselected {name} runtime")
    require("pillow-simd" not in observed["Pillow"]["path"].lower(), "Plain Pillow must precede SIMD")
    # Ancillary versions are acquisition/runtime evidence, not new feature choices.
    for name in ("tokenizers", "huggingface-hub", "safetensors", "ftfy", "wcwidth",
                 "filelock", "fsspec", "packaging", "PyYAML", "regex", "requests", "tqdm", "typing-extensions"):
        try:
            observed[name] = {"version": importlib.metadata.version(name)}
        except importlib.metadata.PackageNotFoundError:
            observed[name] = {"version": None}
    # Coordinator selected this cleaning branch prospectively, before text
    # inference. Optional dependency availability may not silently change it.
    require(observed["ftfy"]["version"] == "6.3.1" and observed["wcwidth"]["version"] == "0.2.13",
            "Selected CLIP cleaning requires ftfy6.3.1 and wcwidth0.2.13")
    return observed


def qualify_layers(route, adapter):
    """Check the selected actual loaded layers; deliberately no forward pass."""
    model, torch = adapter.model, adapter.torch
    result = {"class": type(model).__name__, "selected_dimension": 512,
              "loading_info": getattr(model, "ep18_loading_info", {"strict_state_dict": True})}
    require(not model.training, "Model must be in eval mode")
    require(all(p.device.type == "cpu" and p.dtype == torch.float32 for p in model.parameters()),
            "Only CPU float32 models qualify")
    if route in {"supervised_resnet50_v1", "dino_resnet50"}:
        require(isinstance(model.fc, torch.nn.Identity), "ResNet classifier was not removed")
        require(model.layer4[-1].conv3.out_channels == 2048, "ResNet selected width changed")
        require(tuple(model.avgpool.output_size) == (1, 1), "ResNet selected pool changed")
        result["selected_dimension"] = 2048
    elif route == "clip_vit_b32_image":
        require(tuple(model.visual_projection.weight.shape) == (512, 768), "CLIP image projection changed")
        require(model.config.image_size == 224 and model.config.patch_size == 32, "CLIP image geometry changed")
    elif route == "concept_name_clip_text":
        require(tuple(model.text_projection.weight.shape) == (512, 512), "CLIP text projection changed")
        require(model.config.max_position_embeddings == 77, "CLIP text context changed")
        tokenizer = adapter.tokenizer
        require((tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id) == (49406, 49407, 49407),
                "CLIP tokenizer special tokens changed")
        require(tokenizer.model_max_length == 77 and tokenizer.truncation_side == "right", "CLIP token rule changed")
        require(tokenizer.fix_text is not None and tokenizer.fix_text.__module__ == "ftfy",
                "Selected CLIP text cleaning must not fall back to BasicTokenizer")
        result["text_cleaner"] = "ftfy.fix_text"
    elif route == "dinov2_vit_b14":
        require(model.config.hidden_size == 768 and model.config.patch_size == 14, "DINOv2 encoder changed")
        require(tuple(model.embeddings.position_embeddings.shape) == (1, 1370, 768), "DINOv2 native positions changed")
        require(adapter.processor.crop_size == {"height": 224, "width": 224}, "DINOv2 selected crop changed")
        require(adapter.processor.size == {"shortest_edge": 256}, "DINOv2 selected resize changed")
        result.update(selected_dimension=1536, native_positions=1370,
                      selected_input_patch_slots=256, selected_input_CLS_slots=1,
                      output_geometry_inference_verified=False)
    else:
        require(model.config.vision_config.image_size == 384 and model.config.vision_config.hidden_size == 768,
                "BLIP selected vision configuration changed")
        require((model.config.text_config.bos_token_id, model.config.text_config.sep_token_id,
                 model.config.text_config.pad_token_id) == (30522, 102, 0), "BLIP generation tokens changed")
        require(callable(model.generate), "BLIP generation interface missing")
        result["caption_text_encoder"] = qualify_layers("concept_name_clip_text", adapter.caption_encoder)
    result["parameter_count"] = sum(p.numel() for p in model.parameters())
    return result


def run(args):
    require_compute()
    require(math.isfinite(args.max_seconds) and 0 < args.max_seconds <= MAX_SECONDS, "Invalid bounded loader cap")
    started = time.monotonic()
    output = args.output_dir.resolve()
    require(output.is_relative_to((EPISODE / "outputs/runs").resolve()),
            "Use a fresh episode-local run directory, separate from assets")
    output.mkdir(parents=True, exist_ok=False)
    report = {"phase": "EP18_local_model_asset_and_runtime_qualification", "status": "running",
              "job_id": os.environ["SLURM_JOB_ID"], "hostname": socket.gethostname(),
              "started_utc": datetime.now(timezone.utc).isoformat(), "internal_seconds": args.max_seconds,
              "asset_map": str(args.asset_map.resolve()), "registry": str(REGISTRY),
              "files": {}, "load_results": {}, "asset_runtime_qualified": False,
              "image_or_EEG_or_concept_data_read": False, "forward_passes": 0,
              "models_downloaded": False, "audit_seal_verified": False,
              "image_worker_boundary_verified": False, "frozen_feature_record_emitted": False,
              "text_inference_selected_by_this_process": False, "low_level_checks_repeated": False}

    def check():
        if time.monotonic() - started >= args.max_seconds:
            raise TimeoutError("Whole-process model qualification deadline reached")

    def interrupted(signum, frame):
        raise TimeoutError(f"Model qualification interrupted by signal {signum}")

    def write():
        report["elapsed_seconds"] = time.monotonic() - started
        report["max_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        temporary = output / (REPORT_NAME + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(output / REPORT_NAME)

    old_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT)}
    try:
        for sig in old_handlers:
            signal.signal(sig, interrupted)
        signal.setitimer(signal.ITIMER_REAL, args.max_seconds)
        write()
        # These flags prevent Hugging Face's optional network fallbacks. They
        # are not an OS isolation boundary and do not release image/EEG work.
        require(os.environ.get("L_SCRATCH_JOB"), "Offline model caches require Slurm node-local job scratch")
        offline_cache = Path(os.environ["L_SCRATCH_JOB"]) / f"ep18-offline-models-{os.environ['SLURM_JOB_ID']}"
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                          HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false",
                          HF_HOME=str(offline_cache))
        mapping, registry = read_json(args.asset_map), read_json(REGISTRY)
        check()
        paths, report["files"] = qualify_files(mapping, registry, check)
        write()
        check()
        report["runtime"] = qualify_versions(registry)
        import feature_extractors as adapters
        torch = adapters.configure_torch("cpu")
        torch.set_num_interop_threads(1)
        write()
        for route in ROUTES:
            check()
            report["active_route"] = route
            write()
            check()
            stage_started = time.monotonic()
            if route == "concept_name_clip_text":
                adapter = adapters.TextEncoder(paths["clip"], device="cpu")
            else:
                adapter = adapters.ImageExtractor(route, paths, device="cpu")
            detail = qualify_layers(route, adapter)
            detail["load_and_layer_check_seconds"] = time.monotonic() - stage_started
            report["load_results"][route] = detail
            del adapter
            gc.collect()
            write()
            check()
        report["active_route"] = None
        report["asset_runtime_qualified"] = True
        report["status"] = "asset_runtime_qualification_completed"
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        check()
        write()
        check()
    except Exception as error:
        signal.setitimer(signal.ITIMER_REAL, 0)
        report.update(status="technical_runtime_qualification_failure", asset_runtime_qualified=False,
                      error=f"{type(error).__name__}: {error}", finished_utc=datetime.now(timezone.utc).isoformat())
        write()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
    if report["asset_runtime_qualified"] and time.monotonic() - started >= args.max_seconds:
        report.update(status="technical_runtime_qualification_failure", asset_runtime_qualified=False,
                      error="TimeoutError: Whole-process deadline reached during handler restoration",
                      finished_utc=datetime.now(timezone.utc).isoformat())
        write()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-map", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=MAX_SECONDS)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({"status": report["status"], "elapsed_seconds": report["elapsed_seconds"]}))
    return 0 if report["asset_runtime_qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
