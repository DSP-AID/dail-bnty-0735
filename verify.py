"""Reproducible verification run for the Riemann Hypothesis deliverable.

Run from this directory:

    python3 verify.py                     # full run (~5 min, no network, no keys)
    python3 verify.py --height 100 --xi-height 60 --quick   # smoke run

Needs only numpy + mpmath and the public datasets in ./data.  What it does:

  A. blind scan of Z(t) on [0.5, height] with the fast numpy implementation,
     sign certification of every bracket endpoint at two independent mpmath
     precisions, and bisection refinement of every certified zero;
  B. comparison of the refined zeros with two published tables shipped in the
     repository (Plouffe/Gourdon ~1000-digit first 100 zeros, Odlyzko's
     100000-zero table);
  C. spot-check of the whole Odlyzko table up to t = 74920: for a stride
     sample of its entries, a certified sign change of Z on (g-d, g+d), which
     proves a zero in each such window and that it is the tabulated one;
  D. count of *all* nontrivial zeros with |Im s| <= xi_height inside the strip
     0.4 <= Re s <= 6 by the argument principle applied to xi, compared with
     2 x (number of critical-line zeros found in A).  Equality proves that up
     to that height there is no off-line zero and no multiple zero.

Numerical verification is not a proof of RH; see README.md for exactly which
claims are certified and which rest on the correctness of the code.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import zfast          # noqa: E402
import zrigor         # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def load_plouffe(path=None):
    """First 100 zero ordinates, one per line, ~1000 exact decimal digits."""
    with open(path or os.path.join(DATA, "plouffe_first100.txt")) as f:
        return [ln.strip() for ln in f if ln.strip()]


def load_odlyzko(path=None):
    """Odlyzko's table: 100000 ordinates, 9 decimals, one per line, ordered;
    line i (1-based) is the i-th zero of zeta above the real axis."""
    return np.loadtxt(path or os.path.join(DATA, "odlyzko_zeros1.txt"))


def _log(res, lines, msg):
    print(msg, flush=True)
    lines.append(msg)
    return res


def run_scan(height, step=1e-4, dps_refine=45, verbose=True):
    """Steps A: blind scan -> certified brackets -> refined ordinates."""
    t0 = time.time()
    brackets = zfast.scan_brackets(0.5, height, h=step, chunk_points=4096)
    t_scan = time.time() - t0

    t0 = time.time()
    certified, rejected = [], []
    for a, b in brackets:
        sa, ok_a = zrigor.sign_certified(a)
        if a != b:
            sb, ok_b = zrigor.sign_certified(b)
        else:
            sb, ok_b = 0, True
        if ok_a and ok_b and sa != 0 and sb != 0 and sa != sb:
            certified.append((a, b))
        else:
            rejected.append((a, b, sa, ok_a, sb, ok_b))
    t_cert = time.time() - t0

    t0 = time.time()
    roots, bad = [], []
    for a, b in certified:
        r, ok = zrigor.refine_root_bisect(a, b, dps=dps_refine)
        if not ok or r is None:
            bad.append((a, b))
        else:
            roots.append(r)
    t_refine = time.time() - t0

    if verbose:
        print(f"  scan {t_scan:.1f}s  certify {t_cert:.1f}s  refine {t_refine:.1f}s")
    return dict(step=step, n_brackets=len(brackets), n_certified=len(certified),
                n_rejected=len(rejected), n_roots=len(roots), bad=bad,
                rejected=rejected[:5], roots=roots)


def compare_tables(roots, odlyzko):
    """Step B: agreement with the two published tables.

    The blind scan finds the zeros in increasing order, so our i-th root is
    the i-th tabulated zero; a missed or spurious zero breaks the elementwise
    match rather than shifting it silently.
    """
    import mpmath as mp
    out = {}
    with zrigor.workdps(60):
        pl = load_plouffe()
        n = min(len(pl), len(roots))
        worst = mp.mpf(0)
        for i in range(n):
            worst = max(worst, abs(mp.mpf(pl[i]) - roots[i]))
        out["plouffe_zeros_compared"] = n
        out["plouffe_min_matching_digits"] = int(-mp.log10(worst))

    ours = np.array([float(r) for r in roots])
    ref = odlyzko[:len(ours)]
    if len(ref) == len(ours):
        out["odlyzko_low_zeros_compared"] = int(len(ref))
        out["odlyzko_low_max_abs_diff"] = float(np.max(np.abs(ours - ref)))
        out["odlyzko_count_matches_index"] = bool(
            np.allclose(ours, ref, atol=1e-8))
    else:
        out["odlyzko_count_matches_index"] = False
        out["odlyzko_low_count_mismatch"] = [int(len(ref)), int(len(ours))]
    return out


def probe_odlyzko(odlyzko, stride=50, delta=1e-4, mpmath_samples=40):
    """Step C: every `stride`-th tabulated zero gets a sign-change check.

    delta = 1e-4 keeps the fast float64 tier's ~1e-10 rounding noise far below
    the signal |Z'(g)|*delta, while being far below the smallest known gap
    between consecutive zeros (~0.1), so a sign change pins the tabulated zero.
    """
    sample = odlyzko[::stride]
    ts = np.empty(2 * sample.size, dtype=np.float64)
    ts[0::2], ts[1::2] = sample - delta, sample + delta
    t0 = time.time()
    z = zfast.Z(ts, dtype=np.float64)
    zr = z.reshape(-1, 2)
    ok_fast = int(np.sum((zr[:, 0] > 0) != (zr[:, 1] > 0)))
    t_fast = time.time() - t0

    idx = np.linspace(0, sample.size - 1, mpmath_samples).astype(int)
    certified = 0
    worst = 0.0
    for i in idx:
        g = float(odlyzko[int(i * stride)])
        s_lo, ok_a = zrigor.sign_certified(g - delta)
        s_hi, ok_b = zrigor.sign_certified(g + delta)
        if ok_a and ok_b and s_lo != 0 and s_hi != 0 and s_lo != s_hi:
            certified += 1
            r, _ = zrigor.refine_root_bisect(g - delta, g + delta, dps=30)
            if r is not None:
                worst = max(worst, abs(float(r) - g))
    print(f"  fast probes {t_fast:.1f}s")
    return dict(probed=int(sample.size), sign_changes_fast=ok_fast,
                mpmath_certified=int(certified), mpmath_probes=len(idx),
                max_offset_from_table=float(worst),
                t_range=[float(odlyzko[0]), float(odlyzko[-1])],
                stride=stride, delta=delta)


def count_offline(height, scan_roots, **kw):
    """Step D: zeros of xi in the strip by the argument principle."""
    t0 = time.time()
    n_rect, residual, worst_step = zrigor.xi_rect_zero_count(height, **kw)
    on_line = sum(1 for r in scan_roots if 0 < float(r) < height)
    print(f"  contour {time.time() - t0:.1f}s")
    return dict(height=float(height), n_rectangle=int(n_rect),
                residual=float(residual), max_arg_step_rad=float(worst_step),
                on_line_zeros=on_line, expected_if_all_on_line=2 * on_line,
                off_line_orbits=int(round((n_rect - 2 * on_line) / 4)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--height", type=float, default=300.0)
    ap.add_argument("--step", type=float, default=1e-4)
    ap.add_argument("--xi-height", type=float, default=200.0)
    ap.add_argument("--odlyzko-stride", type=int, default=50)
    ap.add_argument("--mpmath-samples", type=int, default=24)
    ap.add_argument("--quick", action="store_true",
                    help="small parameters, for a fast smoke run")
    ap.add_argument("--json", dest="json_path", default=None)
    args = ap.parse_args()
    if args.quick:
        args.height, args.step = min(args.height, 60.0), 1e-3
        args.xi_height = min(args.xi_height, 40.0)
        args.odlyzko_stride, args.mpmath_samples = 20000, 5

    odlyzko = load_odlyzko()
    res, lines = {}, []
    _log(res, lines, "A. blind scan + sign certification + refinement")
    scan = run_scan(args.height, step=args.step,
                    dps_refine=30 if args.quick else 45)
    _log(res, lines, f"   brackets={scan['n_brackets']} "
                     f"certified={scan['n_certified']} rejected={scan['n_rejected']} "
                     f"refined={scan['n_roots']}")
    assert scan["n_rejected"] == 0 and not scan["bad"], scan["rejected"]

    res["A"] = dict(height=args.height, step=args.step,
                    brackets=scan["n_brackets"], certified=scan["n_certified"],
                    rejected=scan["n_rejected"], refined=scan["n_roots"],
                    first_zero=mp_nstr(scan["roots"][0], 30),
                    last_zero=mp_nstr(scan["roots"][-1], 30))

    res["B"] = compare_tables(scan["roots"], odlyzko)
    _log(res, lines, f"B. published tables: {json.dumps(res['B'])}")

    res["C"] = probe_odlyzko(odlyzko, stride=args.odlyzko_stride,
                             mpmath_samples=args.mpmath_samples)
    _log(res, lines, f"C. Odlyzko table probes: {json.dumps(res['C'])}")

    res["D"] = count_offline(args.xi_height, scan["roots"])
    _log(res, lines, f"D. strip zero count: {json.dumps(res['D'])}")

    res["all_checks_pass"] = bool(
        scan["n_rejected"] == 0 and not scan["bad"]
        and res["B"].get("odlyzko_count_matches_index")
        and res["B"]["plouffe_min_matching_digits"] >= 15
        and res["C"]["sign_changes_fast"] == res["C"]["probed"]
        and res["C"]["mpmath_certified"] == res["C"]["mpmath_probes"]
        and res["C"]["max_offset_from_table"] < 1e-8
        and res["D"]["n_rectangle"] == res["D"]["expected_if_all_on_line"]
        and res["D"]["residual"] < 1e-6)
    _log(res, lines, f"\nALL CHECKS PASS: {res['all_checks_pass']}")

    if args.json_path:
        with open(args.json_path, "w") as f:
            json.dump(res, f, indent=2, default=str)
    if not res["all_checks_pass"]:
        sys.exit(1)


def mp_nstr(x, digits):
    import mpmath as mp
    with zrigor.workdps(digits + 5):
        return mp.nstr(mp.mpf(x), digits, strip_zeros=False)


if __name__ == "__main__":
    main()
