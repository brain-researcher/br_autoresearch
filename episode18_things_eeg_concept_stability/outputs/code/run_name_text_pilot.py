#!/usr/bin/env python3
"""Fixed first-64 concept-name cost pilot, after offline model qualification.

Compute only. Reads the prepared label-name table and pinned local CLIP assets;
never reads private image routing, pixels, event tables or EEG. This separate
label process does not establish an image-worker boundary or an EEG audit seal.
There is deliberately no all-concept mode or automatic successor.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import resource
import signal
import socket
import sys
import time

import qualify_model_runtime as qualification

EPISODE = Path(__file__).resolve().parents[2]
MAPPING = EPISODE / "outputs/config/model_runtime_asset_map_proposal.json"
PROMPTS = EPISODE / "outputs/runs/feature_preparation/20261003-mapping01/label_name_prompts.tsv"
QUALIFICATION = EPISODE / "outputs/runs/feature_preparation/exact_cpu_provision_46842156/model_runtime/model_runtime_qualification.json"
PILOT_COUNT = 64
BATCH_SIZE = 32
CONSISTENCY_COUNT = 4
MAX_SECONDS = 600


def select_first64(rows):
    """Use the already prepared prompts verbatim; identifiers only order rows."""
    rows = list(rows)
    qualification.require(len(rows) == 1854, "The fixed prepared name inventory must contain all1854 concepts")
    keys = [row["uniqueID"] for row in rows]
    qualification.require(all(isinstance(key, str) and key for key in keys)
                          and len(set(keys)) == len(keys), "Nonempty unique source concept IDs required")
    qualification.require(all(isinstance(row["name_prompt"], str) and row["name_prompt"].strip()
                              and row["provider_Word_fallback"] in {"0", "1"} for row in rows),
                          "Use the complete prepared names and explicit source fallback flags")
    return sorted(rows, key=lambda row: row["uniqueID"])[:PILOT_COUNT]


def qualified_clip(mapping, registry, prior, executable):
    qualification.require(prior.get("status") == "asset_runtime_qualification_completed"
                          and prior.get("asset_runtime_qualified") is True,
                          "Completed offline asset/runtime qualification is required before text inference")
    qualification.require(set(prior.get("load_results", {})) == set(qualification.ROUTES)
                          and set(prior.get("files", {})) == set(qualification.ASSETS),
                          "Reuse the completed whole selected asset/runtime qualification")
    qualification.require(prior.get("forward_passes") == 0
                          and prior.get("image_or_EEG_or_concept_data_read") is False,
                          "Use the selected data-free loading prerequisite")
    qualification.require(prior["runtime"]["python"]["path"] == executable,
                          "Use the same explicitly qualified private interpreter")
    for name, expected in registry["runtime_recipe"]["target_versions"].items():
        qualification.require(prior["runtime"][name]["version"].split("+")[0] == expected,
                              f"Qualified {name} version differs from the unchanged registry")
    qualification.require(mapping["schema"] == "ep18_model_runtime_asset_map_v1"
                          and mapping["asset_root"] == "outputs/cache/model_assets",
                          "Use the selected episode-local asset root")
    entry = mapping["assets"]["clip"]
    revision = qualification.ASSETS["clip"][1]
    qualification.require(entry["model_id"] == qualification.ASSETS["clip"][0]
                          and entry["revision"] == registry["text_encoder"]["revision"] == revision,
                          "Pinned CLIP text revision must remain unchanged")
    root = EPISODE / mapping["asset_root"] / entry["local_path"]
    qualification.require(root.name == revision and not root.is_symlink()
                          and root.resolve().is_relative_to((EPISODE / mapping["asset_root"]).resolve()),
                          "Use the exact qualified local CLIP snapshot")
    observed = {item["path"]: item for item in prior["files"]["clip"]}
    qualification.require(set(observed) == {str(root / item["name"]) for item in entry["files"]},
                          "Qualified CLIP file identities changed")
    for item in entry["files"]:
        evidence = observed[str(root / item["name"])]
        qualification.require(evidence["source_integrity_checked"] is True
                              and evidence["bytes"] == item["bytes"] and evidence["source_url"] == item["url"],
                              "Reuse source-verified CLIP files, not a different snapshot")
    return root


def run(args):
    qualification.require_compute()
    qualification.require(os.environ.get("SLURM_MEM_PER_NODE") == "7168"
                          and not os.environ.get("SLURM_JOB_GPUS"), "Selected pilot envelope is1CPU7GiB0GPU")
    qualification.require(math.isfinite(args.max_seconds) and 0 < args.max_seconds <= MAX_SECONDS,
                          "Pilot deadline must be positive and no more than600seconds")
    qualification.require(os.environ.get("L_SCRATCH_JOB"), "Pilot caches require Slurm node-local job scratch")
    output = args.output_dir.absolute()
    qualification.require(not output.exists() and not output.is_symlink()
                          and output.parent.resolve() == (EPISODE / "outputs/runs/feature_preparation").resolve()
                          and output.name.startswith("name_text_pilot_"), "Use a fresh episode-local name_text_pilot attempt")
    output.mkdir()
    started = time.monotonic()
    report = {"analysis_id": "feature_preparation", "phase": "first64_prepared_name_text_cost_pilot",
              "status": "running", "job_id": os.environ["SLURM_JOB_ID"], "hostname": socket.gethostname(),
              "started_utc": datetime.now(timezone.utc).isoformat(), "maximum_seconds": args.max_seconds,
              "qualification": str(QUALIFICATION), "prompt_source": str(PROMPTS),
              "selection": "first64_uniqueID_lexicographic", "batch_size": BATCH_SIZE,
              "selected_concepts": PILOT_COUNT, "completed_concepts": 0,
              "unique_concepts_inferred": 0, "singleton_consistency_concepts": CONSISTENCY_COUNT,
              "image_pixels_or_private_routing_or_events_or_EEG_read": False,
              "audit_seal_verified": False, "image_worker_boundary_verified": False,
              "frozen_feature_record_emitted": False, "full1854_released": False, "automatic_successor": False}

    def check():
        if time.monotonic() - started >= args.max_seconds:
            raise TimeoutError("Fixed name-text pilot deadline reached; no automatic continuation")

    def interrupted(signum, _frame):
        raise TimeoutError(f"Name-text pilot interrupted by signal{signum}")

    def write():
        report["elapsed_seconds"] = time.monotonic() - started
        report["max_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        temporary = output / "name_text_pilot_report.json.tmp"
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(output / "name_text_pilot_report.json")

    old_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT)}
    try:
        for sig in old_handlers:
            signal.signal(sig, interrupted)
        signal.setitimer(signal.ITIMER_REAL, args.max_seconds)
        write()
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                          HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false",
                          HF_HOME=str(Path(os.environ["L_SCRATCH_JOB"]) / "ep18-name-text-pilot-hf"))
        registry = qualification.read_json(qualification.REGISTRY)
        mapping = qualification.read_json(MAPPING)
        prior = qualification.read_json(QUALIFICATION)
        clip = qualified_clip(mapping, registry, prior, sys.executable)
        check()
        with PROMPTS.open(newline="", encoding="utf-8") as source:
            selected = select_first64(csv.DictReader(source, delimiter="\t"))
        names = [row["name_prompt"] for row in selected]
        report["provider_Word_fallbacks"] = sum(int(row["provider_Word_fallback"]) for row in selected)
        import numpy as np
        import feature_extractors as adapters
        torch = adapters.configure_torch("cpu")
        torch.set_num_interop_threads(1)
        loading_started = time.monotonic()
        encoder = adapters.TextEncoder(str(clip), device="cpu")
        report["text_model"] = qualification.qualify_layers("concept_name_clip_text", encoder)
        report["load_seconds"] = time.monotonic() - loading_started
        chunks, lengths, truncated, batch_seconds = [], [], [], []
        for first in range(0, PILOT_COUNT, BATCH_SIZE):
            check()
            batch_started = time.monotonic()
            values, tokens = encoder.encode(names[first:first + BATCH_SIZE])
            batch_seconds.append(time.monotonic() - batch_started)
            chunks.append(values)
            lengths.extend(tokens["original_token_lengths"])
            truncated.extend(tokens["truncated"])
            report["completed_concepts"] += len(values)
            report["unique_concepts_inferred"] = report["completed_concepts"]
            write()
        features = np.concatenate(chunks)
        qualification.require(features.shape == (PILOT_COUNT, 512) and features.dtype == np.float32
                              and np.isfinite(features).all(), "All64 finite float32 projected text rows required")
        qualification.require(np.allclose(np.linalg.norm(features, axis=1), 1., atol=1e-5, rtol=0.),
                              "Every selected text feature must retain unit L2 norm")
        check()
        singleton = []
        for name in names[:CONSISTENCY_COUNT]:
            check()
            values, _ = encoder.encode([name])
            singleton.append(values[0])
        singleton = np.asarray(singleton)
        report["batch_consistency_max_abs_error"] = float(np.max(np.abs(features[:CONSISTENCY_COUNT] - singleton)))
        report["batch_consistency_tolerance"] = {"atol": 1e-6, "rtol": 1e-5}
        qualification.require(np.allclose(features[:CONSISTENCY_COUNT], singleton, atol=1e-6, rtol=1e-5),
                              "Batched and singleton text inference disagree; no bulk release")
        check()
        with (output / "text_features_first64.npy").open("xb") as destination:
            np.save(destination, features, allow_pickle=False)
        with (output / "selected_ids.json").open("x") as destination:
            json.dump([row["uniqueID"] for row in selected], destination)
            destination.write("\n")
        report.update(status="first64_name_text_cost_pilot_completed", batch_seconds=batch_seconds,
                      encoding_seconds=sum(batch_seconds), original_token_lengths=lengths,
                      token_truncations=truncated, truncated_concepts=sum(truncated),
                      feature_shape=[PILOT_COUNT, 512], feature_dtype="float32",
                      projected_full1854_encoding_seconds=sum(batch_seconds) * 1854 / PILOT_COUNT,
                      projection_limitation="Extrapolates64 sorted names; excludes model load, diagnostics and I/O; not measured full1854 runtime.",
                      finished_utc=datetime.now(timezone.utc).isoformat())
        check()
        write()
        check()
    except Exception as error:
        signal.setitimer(signal.ITIMER_REAL, 0)
        report.update(status="technical_name_text_pilot_failure", error=f"{type(error).__name__}: {error}",
                      finished_utc=datetime.now(timezone.utc).isoformat())
        write()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=MAX_SECONDS)
    report = run(parser.parse_args())
    print(json.dumps({key: report[key] for key in ("status", "completed_concepts", "elapsed_seconds", "max_rss_kib")}), flush=True)
    return 0 if report["status"] == "first64_name_text_cost_pilot_completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
