#!/usr/bin/env python3
"""Conditional all1854 label-only extraction after the measured first64 pilot.

Prepare only until the coordinator records release and explicitly invokes
--run-full-labels. Same fixed prompts, local model, batch32 and resource bounds;
no image/EEG inputs, automatic successor or overall frozen feature record.
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

import run_name_text_pilot as pilot

TOTAL = 1854
require = pilot.qualification.require


def require_successful_pilot(record, max_seconds):
    require(record.get("status") == "first64_name_text_cost_pilot_completed"
            and record.get("completed_concepts") == record.get("selected_concepts") == 64
            and record.get("unique_concepts_inferred") == 64
            and record.get("batch_size") == pilot.BATCH_SIZE
            and record.get("qualification") == str(pilot.QUALIFICATION)
            and record.get("prompt_source") == str(pilot.PROMPTS),
            "Completed fixed64 pilot of these same prompts and model is required")
    estimate, loading, memory = (record.get(name) for name in
                                ("projected_full1854_encoding_seconds", "load_seconds", "max_rss_kib"))
    require(all(type(value) in (int, float) and math.isfinite(value) for value in (estimate, loading, memory))
            and estimate > 0 and loading >= 0 and 0 < memory < 7 * 1024**2
            and estimate + loading < max_seconds,
            "Measured pilot projection must fit the unchanged time and memory envelope")


def run(args):
    pilot.qualification.require_compute()
    require(args.run_full_labels, "Explicit coordinator invocation after logged release is required")
    require(os.environ.get("SLURM_MEM_PER_NODE") == "7168" and not os.environ.get("SLURM_JOB_GPUS"),
            "Selected label-only envelope is1CPU7GiB0GPU")
    require(math.isfinite(args.max_seconds) and 0 < args.max_seconds <= pilot.MAX_SECONDS,
            "Full-label deadline must be positive and no more than600seconds")
    require(os.environ.get("L_SCRATCH_JOB"), "Label caches require node-local job scratch")
    parent = pilot.EPISODE / "outputs/runs/feature_preparation"
    pilot_path = args.pilot_report.absolute()
    require(pilot_path.name == "name_text_pilot_report.json" and not pilot_path.is_symlink()
            and pilot_path.parent.parent.resolve() == parent.resolve()
            and pilot_path.parent.name.startswith("name_text_pilot_"), "Use the episode-local completed pilot report")
    measured = pilot.qualification.read_json(pilot_path)
    require_successful_pilot(measured, args.max_seconds)
    output = args.output_dir.absolute()
    require(not output.exists() and not output.is_symlink() and output.parent.resolve() == parent.resolve()
            and output.name.startswith("all_name_text_"), "Use a fresh all_name_text attempt; preserve the pilot")
    output.mkdir()
    started = time.monotonic()
    report = {"analysis_id": "feature_preparation", "phase": "all1854_prepared_name_text_features",
              "status": "running", "job_id": os.environ["SLURM_JOB_ID"], "hostname": socket.gethostname(),
              "started_utc": datetime.now(timezone.utc).isoformat(), "maximum_seconds": args.max_seconds,
              "pilot_report": str(pilot_path), "explicit_full_label_invocation": True,
              "prompt_source": str(pilot.PROMPTS), "qualification": str(pilot.QUALIFICATION),
              "batch_size": pilot.BATCH_SIZE, "selected_concepts": TOTAL, "completed_concepts": 0,
              "image_pixels_or_private_routing_or_events_or_EEG_read": False,
              "audit_seal_verified": False, "image_worker_boundary_verified": False,
              "frozen_feature_record_emitted": False, "automatic_successor": False}

    def check():
        if time.monotonic() - started >= args.max_seconds:
            raise TimeoutError("Full-label deadline reached; no automatic continuation")

    def interrupted(signum, _frame):
        raise TimeoutError(f"Full-label extraction interrupted by signal{signum}")

    def write():
        report["elapsed_seconds"] = time.monotonic() - started
        report["max_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        temporary = output / "label_feature_report.json.tmp"
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(output / "label_feature_report.json")

    old_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT)}
    try:
        for sig in old_handlers:
            signal.signal(sig, interrupted)
        signal.setitimer(signal.ITIMER_REAL, args.max_seconds)
        write()
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                          HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false",
                          HF_HOME=str(Path(os.environ["L_SCRATCH_JOB"]) / "ep18-all-name-text-hf"))
        registry = pilot.qualification.read_json(pilot.qualification.REGISTRY)
        clip = pilot.qualified_clip(pilot.qualification.read_json(pilot.MAPPING), registry,
                                   pilot.qualification.read_json(pilot.QUALIFICATION), sys.executable)
        with pilot.PROMPTS.open(newline="", encoding="utf-8") as source:
            rows = list(csv.DictReader(source, delimiter="\t"))
        pilot.select_first64(rows)  # Reuse full source-inventory validation; no inference here.
        selected = sorted(rows, key=lambda row: row["uniqueID"])
        check()
        import numpy as np
        import feature_extractors as adapters
        torch = adapters.configure_torch("cpu")
        torch.set_num_interop_threads(1)
        loading_started = time.monotonic()
        encoder = adapters.TextEncoder(str(clip), device="cpu")
        report["text_model"] = pilot.qualification.qualify_layers("concept_name_clip_text", encoder)
        report["load_seconds"] = time.monotonic() - loading_started
        chunks, tokens_by_id, batch_seconds = [], [], []
        for first in range(0, TOTAL, pilot.BATCH_SIZE):
            check()
            batch = selected[first:first + pilot.BATCH_SIZE]
            batch_started = time.monotonic()
            values, tokens = encoder.encode([row["name_prompt"] for row in batch])
            batch_seconds.append(time.monotonic() - batch_started)
            require(values.shape == (len(batch), 512) and values.dtype == np.float32
                    and np.isfinite(values).all()
                    and np.allclose(np.linalg.norm(values, axis=1), 1., atol=1e-5, rtol=0.),
                    "Every prepared label must yield its finite normalized float32 feature")
            chunks.append(values)
            tokens_by_id.extend({"uniqueID": row["uniqueID"], "original_token_length": length,
                                 "truncated": truncated, "provider_Word_fallback": int(row["provider_Word_fallback"])}
                                for row, length, truncated in zip(batch, tokens["original_token_lengths"], tokens["truncated"]))
            report["completed_concepts"] += len(values)
            write()
        check()
        features = np.concatenate(chunks)
        require(features.shape == (TOTAL, 512) and len(tokens_by_id) == TOTAL, "All1854 aligned label rows are required")
        with (output / "concept_name_clip_text.npy").open("xb") as destination:
            np.save(destination, features, allow_pickle=False)
        with (output / "concept_ids_and_tokens.json").open("x") as destination:
            json.dump(tokens_by_id, destination, indent=2)
            destination.write("\n")
        report.update(status="all1854_label_text_features_completed", feature_shape=[TOTAL, 512], feature_dtype="float32",
                      batch_seconds=batch_seconds, encoding_seconds=sum(batch_seconds),
                      truncated_concepts=sum(row["truncated"] for row in tokens_by_id),
                      provider_Word_fallbacks=sum(row["provider_Word_fallback"] for row in tokens_by_id),
                      finished_utc=datetime.now(timezone.utc).isoformat())
        check()
        write()
        check()
    except Exception as error:
        signal.setitimer(signal.ITIMER_REAL, 0)
        report.update(status="technical_label_text_feature_failure", error=f"{type(error).__name__}: {error}",
                      finished_utc=datetime.now(timezone.utc).isoformat())
        write()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-full-labels", action="store_true", required=True,
                        help="Coordinator explicitly selects this invocation after logging the measured pilot release")
    parser.add_argument("--pilot-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=pilot.MAX_SECONDS)
    report = run(parser.parse_args())
    print(json.dumps({key: report[key] for key in ("status", "completed_concepts", "elapsed_seconds", "max_rss_kib")}), flush=True)
    return 0 if report["status"] == "all1854_label_text_features_completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
