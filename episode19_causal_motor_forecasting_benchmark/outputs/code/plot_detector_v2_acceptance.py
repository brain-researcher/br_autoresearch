#!/usr/bin/env python3
"""One source-backed EP19 acceptance figure; artifact inputs only."""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import csv
import hashlib
import json
import math
import socket
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

COLORS = {"v1": "#737373", "D2": "#0072B2", "D1": "#D55E00"}
PEOPLE = ("sub-02", "sub-03", "sub-04", "sub-06", "sub-08",
          "sub-09", "sub-11", "sub-12", "sub-14", "sub-15",
          "sub-17", "sub-18", "sub-20", "sub-21", "sub-22")


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(folder, stage):
    record = read_json(folder / "closed.json")
    if record["stage"] != stage or record["complete"] is not True:
        raise RuntimeError("Incomplete plotting input")
    files = {
        str(p.relative_to(folder)): digest(p)
        for p in sorted(folder.rglob("*"))
        if p.is_file() and p != folder / "closed.json"
    }
    if files != record["files"]:
        raise RuntimeError("Closed plotting artifacts changed")
    return record["metadata"]


def rank(values, q=.95):
    values = np.sort(np.asarray(values, float))
    return float(values[max(0, math.ceil(q * len(values)) - 1)]) if len(values) else None


def metric_from_rows(rows):
    determined = [float(r["lag_ms"]) for r in rows if r["status"] == "determinate"]
    return {
        "dot": rank(determined),
        "lower": rank([float(r["lag_lower_ms"]) for r in rows]),
        "upper": rank([float(r["lag_upper_ms"]) for r in rows]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector-dir", required=True, type=Path)
    parser.add_argument("--test-dir", required=True, type=Path)
    parser.add_argument("--v4-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--png-name", type=Path, default=Path("detector_v2_acceptance.png"),
        help="PNG filename relative to --output-dir (no directory components).",
    )
    args = parser.parse_args()
    if (
        args.png_name.is_absolute()
        or len(args.png_name.parts) != 1
        or args.png_name.suffix.lower() != ".png"
    ):
        parser.error("--png-name must be a PNG filename within --output-dir")
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname().lower():
        raise RuntimeError("Plotting requires an actual Slurm compute-node shell")
    detector, test = args.detector_dir.resolve(), args.test_dir.resolve()
    verify(detector / "settings", "settings")
    verify(test / "controls", "controls")
    event_meta = verify(test / "events", "events")
    verify(test / "provider", "provider")
    if event_meta["health_only"]:
        raise RuntimeError("Real-event plots are disabled in health mode")
    output = args.output_dir.resolve()
    if output.is_symlink() or (
        output.exists() and (not output.is_dir() or any(output.iterdir()))
    ):
        raise FileExistsError("Use a fresh/empty figure directory")
    output.mkdir(parents=True, exist_ok=True)

    settings = read_json(detector / "settings" / "detector_settings.json")
    controls = read_json(test / "controls" / "positive_control_summary.json")
    selection = read_json(test / "events" / "selection.json")
    raw = {
        "v1": read_csv(args.v4_dir / "raw_onsets.csv"),
        "D2": read_csv(test / "events" / "D2" / "raw_onsets.csv"),
        "D1": read_csv(test / "events" / "D1" / "raw_onsets.csv"),
    }
    raw = {model: [r for r in rows if int(r["k"]) == 5]
           for model, rows in raw.items()}
    people = event_meta["participants"]
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "legend.frameon": False, "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax in axes.flat:
        ax.grid(alpha=.2)

    ax = axes[0, 0]
    with np.load(detector / "settings" / "noise_samples.npz",
                 allow_pickle=False) as saved:
        full, corrected = np.sort(saved["fit_full"]), np.sort(saved["fit_corrected"])
    x = np.geomspace(1e-5, max(float(full[-1]), .18), 1200)
    for label, values, color in (
        ("Full 1-s fit windows", full, "#737373"),
        ("Corrected final 480 ms", corrected, "#0072B2"),
    ):
        survival = (len(values) - np.searchsorted(values, x, side="right")) / len(values)
        ax.plot(x, np.maximum(survival, .5 / len(values)), color=color, label=label)
    levels = [
        ("v1 LOW", .08509930134406786, "#737373", "--"),
        ("v1 HIGH", .16721586763828217, "#737373", ":"),
        ("v2 LOW", settings["shared_threshold"]["low_g"], "#009E73", "--"),
        ("v2 HIGH", settings["shared_threshold"]["high_g"], "#009E73", ":"),
    ]
    for label, value, color, style in levels:
        ax.axvline(value, color=color, linestyle=style, label=f"{label}={value:.4f}")
    ax.set(yscale="log", xlabel="X (g)", ylabel="Empirical fraction > X",
           title="(A) Stationary fit-noise survival")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    all_finite = [288.0]
    for rows in raw.values():
        for row in rows:
            for field in ("lag_lower_ms", "lag_upper_ms"):
                value = float(row[field])
                if np.isfinite(value):
                    all_finite.append(value)
    left, right = min(all_finite), max(all_finite)
    grid = np.unique(np.r_[np.linspace(left - 20, right + 20, 1500), 288])
    tail_notes = []
    for model, rows in raw.items():
        if not rows:
            tail_notes.append(f"{model}: 0; 0 unresolved")
            continue
        lower = np.sort([float(r["lag_lower_ms"]) for r in rows])
        upper = np.sort([float(r["lag_upper_ms"]) for r in rows])
        det = np.sort([float(r["lag_ms"]) for r in rows
                       if r["status"] == "determinate"])
        tail_notes.append(
            f"{model}: {int(np.count_nonzero(det > 1000))}; "
            f"{len(rows) - len(det)} unresolved"
        )
        lo_cdf = np.searchsorted(upper, grid, side="right") / len(rows)
        hi_cdf = np.searchsorted(lower, grid, side="right") / len(rows)
        ax.fill_between(grid, lo_cdf, hi_cdf, step="post",
                        color=COLORS[model], alpha=.10)
        if len(det):
            ax.step(det, np.arange(1, len(det) + 1) / len(det), where="post",
                    color=COLORS[model],
                    label=f"{model}: determinate {len(det)}/{len(rows)}")
        U95 = rank(upper)
        if U95 is not None and np.isfinite(U95):
            ax.axvline(U95, color=COLORS[model], linestyle="--", alpha=.75,
                       label=f"{model} U95={U95:g} ms")
        else:
            ax.plot([], [], color=COLORS[model], linestyle="--",
                    label=f"{model} U95=∞")
    ax.axvline(288, color="black", linestyle=":", label="288-ms margin")
    ax.set(title="(B) k=5 lag ECDF; shading brackets unresolved events",
           xlabel="L − R (ms)", ylabel="Cumulative fraction",
           xlim=(-50, 1000), ylim=(0, 1.02))
    ax.text(
        .58, .66,
        "Events >1,000 ms (determinate only)\n" + "\n".join(tail_notes),
        transform=ax.transAxes, fontsize=7.5, ha="left", va="top",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": .85},
    )
    ax.legend(fontsize=7.5, loc="lower right")

    ax = axes[1, 0]
    person_metrics = {
        (model, p): metric_from_rows([r for r in rows if r["participant"] == p])
        for model, rows in raw.items() for p in people
    }
    finite = [288.0, 0.0] + [
        value for item in person_metrics.values() for value in item.values()
        if value is not None and np.isfinite(value)]
    bottom, top = min(finite) - 60, max(finite) + 120
    for model, offset in (("v1", -.22), ("D2", 0), ("D1", .22)):
        for i, p in enumerate(people):
            item = person_metrics[(model, p)]
            if item["lower"] is None:
                continue
            lo, hi = item["lower"], item["upper"]
            x_pos = i + offset
            lower_plot = lo if np.isfinite(lo) else bottom + 10
            upper_plot = hi if np.isfinite(hi) else top - 10
            ax.vlines(x_pos, lower_plot, upper_plot, color=COLORS[model], lw=1.4)
            if not np.isfinite(hi):
                ax.plot(x_pos, upper_plot, "^", color=COLORS[model], ms=4)
            if not np.isfinite(lo):
                ax.plot(x_pos, lower_plot, "v", color=COLORS[model], ms=4)
            if item["dot"] is not None:
                ax.plot(x_pos, item["dot"], "o", color=COLORS[model], ms=4)
    ax.axhline(288, color="black", linestyle=":")
    ax.set_xticks(range(len(people)), people, rotation=55, ha="right")
    ax.set(title="(C) k=5 person p95: dots determinate; bars censoring bounds",
           ylabel="L − R (ms)", ylim=(bottom, top))
    ax.legend(handles=[
        Line2D([], [], color=COLORS[m], marker="o", label=m)
        for m in ("v1", "D2", "D1")], fontsize=8)

    ax = axes[1, 1]
    gate_colors = {"near": "#0072B2", "long": "#D55E00",
                   "fast": "#009E73", "sham": "#CC79A7"}
    offsets = {"near": -.24, "long": -.08, "fast": .08, "sham": .24}
    markers = {"linear": "o", "quadratic": "s", "pulse_growth": "^"}
    passed = 0
    for i, p in enumerate(people):
        person = controls["by_k"]["5"]["participants"][p]
        passed += person["pass"]
        for family, marker in markers.items():
            item = person["families"][family]
            for name in ("near", "long", "fast"):
                if name == "fast" and family == "pulse_growth":
                    continue
                value = item[name]["fraction"]
                if value is not None:
                    ax.plot(i + offsets[name], value, marker,
                            color=gate_colors[name], ms=4, alpha=.8)
        if person["sham_alarm_fraction"] is not None:
            ax.plot(i + offsets["sham"], person["sham_alarm_fraction"], "D",
                    color=gate_colors["sham"], ms=4)
        if not person["pass"]:
            ax.plot(i, -.035, "x", color="black", ms=5)
    for name, threshold in (("near", .90), ("long", .95),
                            ("fast", .95), ("sham", .05)):
        ax.axhline(threshold, color=gate_colors[name], linestyle=":", alpha=.65)
    ax.set_xticks(range(len(people)), people, rotation=55, ha="right")
    ax.set(title=f"(D) k=5 controls: {passed}/{len(people)} people pass all gates",
           ylabel="Classification fraction / sham alarm fraction", ylim=(-.07, 1.04))
    handles = [
        Line2D([], [], color=gate_colors[name], marker="o",
               label=label, linestyle="none")
        for name, label in (
            ("near", "Near ≥.90"), ("long", "Long ≥.95"),
            ("fast", "Fast ≥.95"), ("sham", "Sham ≤.05"))
    ]
    handles += [
        Line2D([], [], color="black", marker=marker, linestyle="none", label=family)
        for family, marker in markers.items()
    ]
    ax.legend(handles=handles, fontsize=7, ncol=2, loc="center right")

    selection_text = (
        f"selected={selection['selected']}"
        if selection["selected"]
        else "no candidate frozen (v5 UNDERIDENTIFIED)"
    )
    fig.suptitle(
        "development calibration/diagnostic evidence; not an EEG result\n"
        f"EP19 detector v2 / raw-onset v5; {selection_text}",
        fontsize=14,
    )
    fig.text(
        .02, .015,
        "Post hoc follow-up to job 48230171. Censoring bounds are not sampling confidence intervals.\n"
        "Controls reuse backgrounds and sham positions; usable coverage ≥.95 is also required.\n"
        "No worst-case timing certification or EEG scoring clearance.",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, .09, 1, .94))
    path = output / args.png_name
    with path.open("xb") as handle:
        fig.savefig(handle, format="png", dpi=160)
    plt.close(fig)
    with (output / "provenance.json").open("x", encoding="utf-8") as handle:
        json.dump({
            "detector_dir": str(detector), "test_dir": str(test),
            "v4_dir": str(args.v4_dir.resolve()),
            "settings_closure_sha256": digest(detector / "settings" / "closed.json"),
            "events_closure_sha256": digest(test / "events" / "closed.json"),
            "v4_raw_onsets_sha256": digest(args.v4_dir / "raw_onsets.csv"),
            "figure_sha256": digest(path),
        }, handle, indent=2, allow_nan=False)
    print(path, flush=True)


if __name__ == "__main__":
    main()
