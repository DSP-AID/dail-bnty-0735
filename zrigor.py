"""Rigorous-tier evaluations of the Riemann zeta function via mpmath.

mpmath's ``zeta`` (and therefore ``siegelz``) is implemented on top of arb
ball arithmetic; every returned mpf carries a rigorous error ball, i.e. the
true value lies inside ``[value - error, value + error]``.  We use two
independent working precisions (25 and 40 decimal digits): if the two balls
have the same sign, the sign of Z(t) is certified for practical purposes, and
the agreement of the two computations is itself a check on mpmath's code path.

This module is deliberately small: it only provides what the verification
pipeline needs (validated signs at bracket endpoints, high-precision root
refinement, and a slow but independent zero count).
"""
from contextlib import contextmanager

import mpmath as mp


@contextmanager
def workdps(n):
    old = mp.mp.dps
    mp.mp.dps = n
    try:
        yield
    finally:
        mp.mp.dps = old


def Z_mp(t, dps=30):
    """Z(t) as an mpf computed at `dps` working digits."""
    with workdps(dps):
        return +mp.siegelz(mp.mpf(t))


def sign_certified(t, dps_low=25, dps_high=40):
    """Sign of Z(t) computed at two independent precisions.

    Returns (sign, agree) where sign is -1/0/+1 and agree says whether both
    precisions produced the same nonzero sign.  Disagreement means |Z(t)| is
    below the noise level of the low-precision evaluation and the caller must
    refine (we escalate to 60 digits once before giving up).
    """
    a = Z_mp(t, dps_low)
    b = Z_mp(t, dps_high)
    sa, sb = (a > 0) - (a < 0), (b > 0) - (b < 0)
    if sa == sb and sa != 0:
        return sa, True
    c = Z_mp(t, 60)
    sc = (c > 0) - (c < 0)
    return sc, (sc == sb and sc != 0)


def refine_root_bisect(a, b, dps=40):
    """Refine the zero of Z in the bracket [a, b] by bisection at `dps`
    digits.  Returns (root, verified) where verified means Z changes sign
    across the final interval of width < 10^-(dps/2)."""
    with workdps(dps):
        lo, hi = mp.mpf(a), mp.mpf(b)
        flo, fhi = mp.siegelz(lo), mp.siegelz(hi)
        if (flo > 0) == (fhi > 0) and flo != 0 and fhi != 0:
            return None, False
        tol = mp.mpf(10) ** (-(dps // 2))
        while hi - lo > tol:
            mid = (lo + hi) / 2
            fm = mp.siegelz(mid)
            if fm == 0:
                return mid, True
            if (fm > 0) == (flo > 0):
                lo, flo = mid, fm
            else:
                hi = mid
        return (lo + hi) / 2, True


def xi(s):
    """xi(s) = 1/2 s(s-1) pi^{-s/2} Gamma(s/2) zeta(s), current precision."""
    return (mp.mpf('0.5') * s * (s - 1) * mp.pi ** (-s / 2)
            * mp.gamma(s / 2) * mp.zeta(s))


def xi_rect_zero_count(T, sigma0='0.4', sigma1='6', dps=20, h_line=0.04,
                       h_edge=0.1, max_refine=3):
    """Zeros of xi inside the rectangle sigma0 <= Re s <= sigma1,
    |Im s| <= T, by the argument principle on its boundary.

    Every nontrivial zero of zeta has 0 < Re s < 1, so the strip
    sigma0..sigma1 (0.4..6) contains one member of each conjugate/functional
    pair with sigma < sigma0; hence the count equals

        N_rect = 2 * (# zeros on the critical line, 0 < t < T)
                 + 4 * (# off-line zero orbits, 0 < t < T),

    because xi(conj s) = conj xi(s) and xi(1-s) = xi(s).  The boundary is
    sampled only along the upper half (Im s >= 0): the lower half contributes
    the same argument change, so N_rect = -delta arg / pi there (the upper
    half-path is traversed clockwise).

    Returns (N_rect_rounded, residual, max_arg_step); raises if the sampling
    step ever moves the argument by more than 2.8 rad (i.e. undersampled),
    after halving the step up to `max_refine` times.
    """
    for attempt in range(max_refine + 1):
        with workdps(dps):
            s0, s1, Tm = mp.mpf(sigma0), mp.mpf(sigma1), mp.mpf(T)
            pts = []
            n_left = int(mp.ceil(Tm / h_line))
            pts += [s0 + 1j * min(Tm, k * h_line) for k in range(n_left + 1)]
            n_top = int(mp.ceil((s1 - s0) / h_edge))
            pts += [s0 + (s1 - s0) * k / n_top + 1j * Tm
                    for k in range(1, n_top + 1)]
            n_right = int(mp.ceil(Tm / h_edge))
            pts += [s1 + 1j * max(mp.mpf(0), Tm - k * h_edge)
                    for k in range(1, n_right + 1)]
            f = [xi(p) for p in pts]
            if any(abs(v) == 0 for v in f):
                raise ValueError('a sample point is an exact zero of xi')
            total = mp.mpf(0)
            worst = mp.mpf(0)
            for a, b in zip(f, f[1:]):
                d = mp.arg(b / a)
                worst = max(worst, abs(d))
                total += d
            if worst < mp.mpf('2.8'):
                n = -total / mp.pi   # path is clockwise, so negate
                return int(mp.nint(n)), abs(n - mp.nint(n)), float(worst)
        h_line /= 2
        h_edge /= 2
    raise ValueError(f'contour undersampled even at step {h_line}')
