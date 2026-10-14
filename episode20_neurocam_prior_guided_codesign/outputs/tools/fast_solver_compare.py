"""Stage 1 acceptance comparison: fast solver versus the unmodified reference.

Windows are prefixes of cycle 0 starting from the zero state: the 5-us case
(2 actions), the first dwell (25 actions) and the first frame (1600 actions).
Criteria (proposed_fast_forward_formulation.md section 7, item 3):
* all 69,848 final states and every hold mean within 1e-8 V + 1e-5 |v_ref|;
* each per-window physical work total within 1e-18 J + 1e-5 |W_ref|;
* ADC codes identical, except +/-1 where the reference mean lies within that
  tolerance of a threshold; pixels within 1/3 LSB.
Conversion means replicate the reference readout arithmetic (interval mean x
duration accumulated in time order, divided by the aperture length) and the
reference quantizer (bisect_right on its own threshold table).
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

STATE_ATOL, STATE_RTOL = 1e-8, 1e-5
WORK_ATOL, WORK_RTOL = 1e-18, 1e-5
WINDOWS = (("5us", 2), ("dwell", 25), ("frame", 1600))


def _quantizer():
    from conditional_acquisition_recipe import _quantizer as build
    q = build(lambda: None)
    return np.asarray(q.thresholds_volts), np.asarray(q.code_volts)


def conversion_means(intervals, conv_start, conv_end, conv_adc):
    """Reference readout arithmetic on per-interval hold means."""
    t0, t1 = intervals["t_start"], intervals["t_end"]
    mean = intervals["hold_mean"]
    out = np.full(len(conv_start), np.nan)
    order = np.argsort(t0)
    t0s, t1s = t0[order], t1[order]
    for c in range(len(conv_start)):
        lo = np.searchsorted(t0s, conv_start[c], side="left")
        hi = np.searchsorted(t1s, conv_end[c], side="right")
        if hi <= lo or not np.isfinite(t1s[hi - 1]) or t0s[lo] != conv_start[c] or t1s[hi - 1] != conv_end[c]:
            continue
        integral = 0.0
        for i in order[lo:hi]:
            integral += float(mean[i, conv_adc[c]]) * (float(t1[i]) - float(t0[i]))
        out[c] = integral / (conv_end[c] - conv_start[c])
    return out


def scaled(new, ref):
    return np.abs(new - ref) / (STATE_ATOL + STATE_RTOL * np.abs(ref))


def compare(reference_dir, solver_dir, *, state_ids=None):
    ref = dict(np.load(reference_dir / "reference_intervals.npz", allow_pickle=False))
    new = dict(np.load(solver_dir / "solver_intervals.npz", allow_pickle=False))
    ref_ck = dict(np.load(reference_dir / "reference_checkpoints.npz", allow_pickle=False))
    new_ck = dict(np.load(solver_dir / "solver_checkpoints.npz", allow_pickle=False))
    thresholds, levels = _quantizer()
    lsb = float(levels[1] - levels[0])
    results = {}
    for name, slots in WINDOWS:
        window = {}
        in_ref = np.flatnonzero(ref_ck["slots"] == slots)
        in_new = np.flatnonzero(new_ck["slots"] == slots)
        if not len(in_ref) or not len(in_new):
            results[name] = dict(available=False)
            continue
        t_end = float(ref_ck["times"][in_ref[0]])
        if float(new_ck["times"][in_new[0]]) != t_end:
            raise ValueError("checkpoint times differ")
        xr, xn = ref_ck["states"][in_ref[0]], new_ck["states"][in_new[0]]
        s = scaled(xn, xr)
        worst = int(np.argmax(s))
        window["state"] = dict(count=int(len(xr)), max_scaled=float(s.max()), max_abs=float(np.abs(xn - xr).max()),
                               worst_index=worst, worst_id=None if state_ids is None else state_ids[worst],
                               worst_ref=float(xr[worst]), worst_new=float(xn[worst]),
                               violations=int(np.sum(s > 1)), passed=bool(s.max() <= 1))
        # per-interval hold means inside the window
        mask_r = np.isfinite(ref["t_end"]) & (ref["t_end"] <= t_end)
        mask_n = np.isfinite(new["t_end"]) & (new["t_end"] <= t_end)
        if not np.array_equal(ref["t_start"][mask_r], new["t_start"][mask_n]):
            raise ValueError("interval grids differ")
        hr, hn = ref["hold_mean"][mask_r], new["hold_mean"][mask_n]
        sh = scaled(hn, hr)
        window["hold_mean"] = dict(count=int(hr.size), max_scaled=float(sh.max()),
                                   max_abs=float(np.abs(hn - hr).max()), violations=int(np.sum(sh > 1)),
                                   passed=bool(sh.max() <= 1))
        se = scaled(new["hold_end"][mask_n], ref["hold_end"][mask_r])
        window["hold_end"] = dict(max_scaled=float(se.max()), max_abs=float(np.abs(new["hold_end"][mask_n]
                                                                                 - ref["hold_end"][mask_r]).max()))
        ss = scaled(new["source_mean"][mask_n], ref["source_mean"][mask_r])
        window["source_mean"] = dict(max_scaled=float(ss.max()),
                                     max_abs=float(np.abs(new["source_mean"][mask_n] - ref["source_mean"][mask_r]).max()))
        # works per window
        fields = [str(v) for v in ref["work_fields"]]
        wr = ref["work"][mask_r].sum(axis=0)
        wn = new["work"][mask_n].sum(axis=0)
        work = {}
        for i, field in enumerate(fields):
            tol = WORK_ATOL + WORK_RTOL * abs(wr[i])
            work[field] = dict(reference=float(wr[i]), solver=float(wn[i]), abs_diff=float(abs(wn[i] - wr[i])),
                               scaled=float(abs(wn[i] - wr[i]) / tol), passed=bool(abs(wn[i] - wr[i]) <= tol))
        window["work"] = work
        window["work_passed"] = all(v["passed"] for v in work.values())
        # conversions and codes
        conv_start, conv_end, conv_adc = ref["conv_start"], ref["conv_end"], ref["conv_adc"]
        inside = conv_end <= t_end
        sub_r = {k: v[mask_r] for k, v in ref.items() if k in ("t_start", "t_end", "hold_mean")}
        sub_n = {k: v[mask_n] for k, v in new.items() if k in ("t_start", "t_end", "hold_mean")}
        mr = conversion_means(sub_r, conv_start[inside], conv_end[inside], conv_adc[inside])
        mn = conversion_means(sub_n, conv_start[inside], conv_end[inside], conv_adc[inside])
        valid = np.isfinite(mr) & np.isfinite(mn)
        cr = np.searchsorted(thresholds, mr[valid], side="right")
        cn = np.searchsorted(thresholds, mn[valid], side="right")
        tol = STATE_ATOL + STATE_RTOL * np.abs(mr[valid])
        nearest = np.minimum(np.abs(mr[valid] - thresholds[np.clip(cr - 1, 0, len(thresholds) - 1)]),
                             np.abs(mr[valid] - thresholds[np.clip(cr, 0, len(thresholds) - 1)]))
        near_threshold = nearest <= tol
        differ = cn != cr
        bad = differ & ~(near_threshold & (np.abs(cn - cr) == 1))
        sm = scaled(mn[valid], mr[valid])
        window["codes"] = dict(conversions=int(valid.sum()), identical=int(np.sum(~differ)),
                               differ=int(differ.sum()), differ_allowed=int(np.sum(differ & ~bad)),
                               violations=int(bad.sum()), reference_near_threshold=int(near_threshold.sum()),
                               conversion_mean_max_scaled=float(sm.max()) if sm.size else 0.0,
                               conversion_mean_max_abs=float(np.abs(mn[valid] - mr[valid]).max()) if sm.size else 0.0,
                               passed=bool(bad.sum() == 0))
        # pixels whose three words are inside the window
        pix = ref["pixel_conv"]
        index_inside = -np.ones(len(conv_start), dtype=np.int64)
        index_inside[np.flatnonzero(inside)[valid]] = np.arange(int(valid.sum()))
        usable = np.all(index_inside[pix] >= 0, axis=1)
        if usable.any():
            rows = index_inside[pix[usable]]
            pr = levels[cr[rows]].sum(axis=1) / 3.0
            pn = levels[cn[rows]].sum(axis=1) / 3.0
            dp = np.abs(pn - pr)
            window["pixels"] = dict(count=int(usable.sum()), max_abs=float(dp.max()), max_lsb=float(dp.max() / lsb),
                                    violations=int(np.sum(dp > lsb / 3.0 * (1 + 1e-9))),
                                    passed=bool(np.all(dp <= lsb / 3.0 * (1 + 1e-9))))
        else:
            window["pixels"] = dict(count=0, passed=True)
        # solver diagnostics within the window
        window["solver"] = dict(max_sweeps=int(new["sweeps"][mask_n].max()),
                                mean_sweeps=float(new["sweeps"][mask_n].mean()),
                                max_wr_change=float(np.nanmax(new["wr_change"][mask_n])),
                                max_rep_pad=float(np.nanmax(new["rep_pad"][mask_n])),
                                max_rep_source=float(np.nanmax(new["rep_source"][mask_n])),
                                clip_events=int(new["clip_events"][mask_n].sum()),
                                solver_seconds=float(np.nansum(new["elapsed"][mask_n])))
        window["reference"] = dict(max_subintervals=int(ref["subintervals"][mask_r].max()),
                                   max_partition_error=float(np.nanmax(ref["partition_error"][mask_r])),
                                   solver_seconds=float(np.nansum(ref["elapsed"][mask_r])))
        # energy audit (Norton identity) over the window, from the solver run
        e0 = float(new_ck["energy"][np.flatnonzero(new_ck["slots"] == 0)[0], 1])
        e1 = float(new_ck["energy"][in_new[0], 1])
        audit = new["audit"][mask_n].sum(axis=0)
        supplied = float(new["work"][mask_n][:, fields.index("supplied_charge_boundary_work_joule")].sum())
        residual = e1 - e0 - supplied - audit[0] + audit[1]
        scale = max(abs(e1), abs(e0), abs(audit[0]), abs(audit[1]), abs(supplied))
        window["energy_audit"] = dict(stored_start=e0, stored_end=e1, input_work=float(audit[0]),
                                      heat=float(audit[1]), supplied_charge_work=supplied, residual=float(residual),
                                      relative_residual=float(residual / scale) if scale else 0.0)
        window["passed"] = bool(window["state"]["passed"] and window["hold_mean"]["passed"] and window["work_passed"]
                                and window["codes"]["passed"] and window["pixels"]["passed"])
        results[name] = window
    return results


def difference(run_a, run_b, reference_dir):
    """Report-only differences between two solver runs (e.g. ref_fed vs exact inputs)."""
    a = dict(np.load(run_a / "solver_intervals.npz", allow_pickle=False))
    b = dict(np.load(run_b / "solver_intervals.npz", allow_pickle=False))
    ca = dict(np.load(run_a / "solver_checkpoints.npz", allow_pickle=False))
    cb = dict(np.load(run_b / "solver_checkpoints.npz", allow_pickle=False))
    ref = dict(np.load(reference_dir / "reference_intervals.npz", allow_pickle=False))
    thresholds, levels = _quantizer()
    lsb = float(levels[1] - levels[0])
    common = np.intersect1d(ca["slots"], cb["slots"])
    states = []
    for slot in common:
        xa = ca["states"][np.flatnonzero(ca["slots"] == slot)[0]]
        xb = cb["states"][np.flatnonzero(cb["slots"] == slot)[0]]
        states.append(dict(slot=int(slot), max_abs=float(np.abs(xa - xb).max()),
                           max_abs_pads=float(np.abs(xa[:4096] - xb[:4096]).max()),
                           max_abs_sources=float(np.abs(xa[4096:4160] - xb[4096:4160]).max()),
                           max_abs_holds=float(np.abs(xa[-24:] - xb[-24:]).max())))
    mask = np.isfinite(a["t_end"]) & np.isfinite(b["t_end"])
    out = dict(states=states, hold_mean_max_abs=float(np.abs(a["hold_mean"][mask] - b["hold_mean"][mask]).max()),
               source_mean_max_abs=float(np.abs(a["source_mean"][mask] - b["source_mean"][mask]).max()))
    t_end = float(np.max(a["t_end"][mask]))
    inside = ref["conv_end"] <= t_end
    ma = conversion_means({k: a[k][mask] for k in ("t_start", "t_end", "hold_mean")}, ref["conv_start"][inside],
                          ref["conv_end"][inside], ref["conv_adc"][inside])
    mb = conversion_means({k: b[k][mask] for k in ("t_start", "t_end", "hold_mean")}, ref["conv_start"][inside],
                          ref["conv_end"][inside], ref["conv_adc"][inside])
    valid = np.isfinite(ma) & np.isfinite(mb)
    code_a = np.searchsorted(thresholds, ma[valid], side="right")
    code_b = np.searchsorted(thresholds, mb[valid], side="right")
    out["conversions"] = dict(count=int(valid.sum()), mean_max_abs=float(np.abs(ma[valid] - mb[valid]).max()),
                              mean_max_lsb=float(np.abs(ma[valid] - mb[valid]).max() / lsb),
                              codes_differ=int(np.sum(code_a != code_b)),
                              max_code_difference=int(np.abs(code_a - code_b).max()))
    fields = [str(v) for v in a["work_fields"]]
    wa, wb = a["work"][mask].sum(axis=0), b["work"][mask].sum(axis=0)
    out["work"] = {f: dict(a=float(wa[i]), b=float(wb[i]), abs_diff=float(abs(wa[i] - wb[i])),
                           rel_diff=float(abs(wa[i] - wb[i]) / max(abs(wa[i]), 1e-300))) for i, f in enumerate(fields)}
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--solver", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--solver-b", type=Path, default=None,
                        help="report differences between --solver and this second solver run")
    args = parser.parse_args(argv)
    from conditional_forward_recipe import expected_state_ids
    if args.solver_b is not None:
        result = difference(args.solver, args.solver_b, args.reference)
        encoded = json.dumps(dict(solver_a=str(args.solver), solver_b=str(args.solver_b), difference=result),
                             indent=2, sort_keys=True, default=float)
        with open(args.output, "x") as stream:
            stream.write(encoded + "\n")
        print(encoded[:6000])
        return 0
    results = compare(args.reference, args.solver, state_ids=expected_state_ids())
    encoded = json.dumps(dict(reference=str(args.reference), solver=str(args.solver), windows=results),
                         indent=2, sort_keys=True, default=float)
    with open(args.output, "x") as stream:
        stream.write(encoded + "\n")
    print(encoded[:6000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
