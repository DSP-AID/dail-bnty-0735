"""Self-checks for the two independent Z(t) implementations and for the
verification machinery (contour count, sign certification, shipped datasets).

Run with:  python3 -m pytest tests/test_rh.py   (from the src/ directory)
"""
import os
import sys

import mpmath as mp
import numpy as np
import pytest

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC)

import verify          # noqa: E402
import zfast           # noqa: E402
import zrigor          # noqa: E402


def test_datasets_are_the_public_ones():
    pl = verify.load_plouffe()
    assert len(pl) == 100
    assert pl[0].startswith("14.134725141734693790457251983562470270784257115699")
    assert pl[1].startswith("21.022039638771554992628479593896902777334")
    od = verify.load_odlyzko()
    assert od.size == 100000 and od[0] == pytest.approx(14.134725142, abs=1e-9)
    assert np.all(np.diff(od) > 0)


@pytest.mark.parametrize("t", [0.5, 10.0, 100.0, 1000.37, 30425.9])
def test_fast_tier_matches_mpmath(t):
    with zrigor.workdps(40):
        ref = float(mp.siegelz(mp.mpf(t)))
    # float64 trig in the main sum bounds the longdouble tier at ~1e-13 for
    # t <= 2000 and ~3e-10 for t <= 75000 (see zfast docstring).
    ld_tol = 1e-13 if t <= 2000 else 1e-9
    for dtype, tol in ((np.float64, 1e-8), (np.longdouble, ld_tol)):
        got = float(zfast.Z([t], dtype=dtype)[0])
        assert abs(got - ref) < tol * max(1.0, abs(ref)), (t, dtype, got, ref)


def test_theta_matches_mpmath():
    with zrigor.workdps(40):
        for t in (50.0, 1000.0, 30000.0):
            ref = float(mp.siegeltheta(mp.mpf(t)))
            got = float(zfast.theta(np.longdouble(t))[0])
            assert abs(got - ref) < 1e-9 * max(1.0, abs(ref))


def test_sign_certification_agrees_with_reference():
    ts = [14.5, 20.0, 100.0, 1000.0, 50000.0]
    with zrigor.workdps(50):
        for t in ts:
            ref = mp.siegelz(mp.mpf(t))
            sign, ok = zrigor.sign_certified(t)
            assert ok and sign == int((ref > 0) - (ref < 0))


def test_first_zero_reproduced_from_scratch():
    scan = verify.run_scan(20.0, step=1e-3, dps_refine=35, verbose=False)
    assert scan["n_brackets"] == 1 and scan["n_rejected"] == 0
    pl = verify.load_plouffe()
    with zrigor.workdps(60):
        diff = abs(scan["roots"][0] - mp.mpf(pl[0]))
    assert float(-mp.log10(diff)) > 14


def test_contour_count_catches_every_zero_below_height():
    # zeros below t=20: 14.13 (one) -> rectangle count must be 2*1;
    # below t=35: 14.13, 21.02, 25.01, 30.42, 32.93 -> 2*5.
    for T, expect in ((20.0, 2), (35.0, 10)):
        n, residual, step = zrigor.xi_rect_zero_count(T)
        assert n == expect and residual < 1e-8 and step < 2.8
