#!/usr/bin/env python3
"""One bounded public HTTPS transfer in coordinator-owned Slurm compute.

Sherlock's DTN has no interactive shell. This selected compute route uses only
the already listed public URLs; it never retries the failed DTN authentication.
No credential handling,
installation, model loading, retries, resumed files, or implicit HF downloads.
Byte counts are checked during transfer; the compute loader subsequently checks
the source-declared integrity hashes before loading anything.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import socket
import time
from urllib.parse import urlparse
from urllib.request import urlopen


EPISODE = Path(__file__).resolve().parents[2]
EXPECTED_ASSETS = {"supervised_resnet50_v1", "dino_resnet50", "clip", "blip", "dinov2"}
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")


def require_compute_transfer():
    hostname = socket.gethostname()
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in hostname or "dtn" in hostname.lower():
        raise RuntimeError("Public model acquisition requires the coordinator's actual Slurm compute node")
    if os.environ.get("SLURM_CPUS_PER_TASK") != "1" or any(os.environ.get(name) != "1" for name in THREADS):
        raise RuntimeError("The selected transfer requires one allocated CPU and six explicit one-thread limits")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-map", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, default=1800)
    args = parser.parse_args()
    require_compute_transfer()
    if not 1 <= args.max_seconds <= 1800:
        raise ValueError("Whole transfer limit must be within 1..1800 seconds")
    started = time.monotonic()

    def check():
        if time.monotonic() - started >= args.max_seconds:
            raise TimeoutError("Bounded public HTTPS transfer deadline reached; no retry")

    def interrupted(signum, frame):
        raise TimeoutError(f"Transfer interrupted by signal {signum}; no retry")

    old_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT)}
    try:
        for sig in old_handlers:
            signal.signal(sig, interrupted)
        signal.setitimer(signal.ITIMER_REAL, args.max_seconds)
        with args.asset_map.open(encoding="utf-8") as source:
            mapping = json.load(source)
        if mapping["schema"] != "ep18_model_runtime_asset_map_v1" or set(mapping["assets"]) != EXPECTED_ASSETS:
            raise ValueError("Wrong selected asset map")
        root = Path(mapping["asset_root"])
        root = (EPISODE / root).resolve() if not root.is_absolute() else root.resolve()
        allowed = [(EPISODE / "outputs/cache/model_assets").resolve()]
        if os.environ.get("SCRATCH"):
            allowed.append((Path(os.environ["SCRATCH"]) /
                            "br_autoresearch/episode18_things_eeg_concept_stability/model_assets").resolve())
        if root not in allowed:
            raise ValueError("Use only the selected episode cache or task Scratch root")
        targets = []
        for asset in mapping["assets"].values():
            local = root / asset["local_path"]
            for item in asset["files"]:
                if Path(item["name"]).name != item["name"]:
                    raise ValueError("Asset names must be basenames")
                target = local / item["name"] if asset["kind"] == "huggingface_snapshot" else local
                temporary = target.with_name(target.name + ".partial")
                if not target.resolve().is_relative_to(root) or target.exists() or temporary.exists():
                    raise ValueError(f"Only fresh selected targets are permitted: {target}")
                parsed = urlparse(item["url"])
                if parsed.scheme != "https" or parsed.hostname not in {
                        "huggingface.co", "download.pytorch.org", "dl.fbaipublicfiles.com"}:
                    raise ValueError("Only the explicit official model URLs are permitted")
                targets.append((item, target, temporary))
        if len({target for _, target, _ in targets}) != len(targets):
            raise ValueError("Duplicate transfer targets")
        print(json.dumps({"phase": "public_model_asset_acquisition", "job_id": os.environ["SLURM_JOB_ID"],
                          "hostname": socket.gethostname(), "allocated_cpus": 1,
                          "maximum_seconds": args.max_seconds, "files": len(targets),
                          "declared_bytes": sum(item["bytes"] for item, _, _ in targets),
                          "credentials_used": False, "automatic_retry": False}), flush=True)
        for item, target, temporary in targets:
            check()
            target.parent.mkdir(parents=True, exist_ok=True)
            count = 0
            # No retry loop. A failed request retains its .partial file and
            # stops; it cannot be mistaken for a qualified local checkpoint.
            with urlopen(item["url"], timeout=60) as response, temporary.open("xb") as destination:
                while chunk := response.read(1024 * 1024):
                    check()
                    count += len(chunk)
                    if count > item["bytes"]:
                        raise ValueError(f"Transfer exceeds declared file size: {target}")
                    destination.write(chunk)
            if count != item["bytes"]:
                raise ValueError(f"Incomplete transfer {target}: {count} != {item['bytes']}")
            check()
            temporary.replace(target)
            print(json.dumps({"transferred": str(target), "bytes": count,
                              "elapsed_seconds": time.monotonic() - started}), flush=True)
        check()
        print("All declared model files transferred; compute integrity/runtime qualification is still required.", flush=True)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    main()
