"""Render only a successfully completed future amended_full002 evaluation.

Reuse the current figure/summary and all completed-report checks, supplying only
the fresh input/output paths. No scientific arithmetic or current-run files are
changed, and the original renderer's default paths remain amended_full001.
"""
import os
from pathlib import Path
import socket

from render_finding_a_result import load_completed_report, render


def main():
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise RuntimeError("Scientific rendering requires an actual compute-node Slurm shell")
    episode = Path(__file__).resolve().parents[3]
    run = episode / "outputs/runs/finding_a_feature_decoding/amended_full002"
    figures = episode / "outputs/figures/finding_a_amended_full002"
    report, phase, scientific = load_completed_report(run)
    render(report, phase, scientific, figures=figures)


if __name__ == "__main__":
    main()
