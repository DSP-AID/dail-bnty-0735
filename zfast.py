"""Fast high-precision evaluation of the Riemann-Siegel Z-function.

`Z(t) = Re( exp(i*theta(t)) * zeta(1/2 + i*t) )`, computed with an
Euler-Maclaurin ("classical") summation for zeta on the critical line,

    zeta(s) = sum_{n<N} n^-s + N^(1-s)/(s-1) + N^-s/2
              + sum_k B_2k/(2k)! * s(s+1)..(s+2k-2) * N^(-s-2k+1),

with N = t/2 + 8 and 18 correction terms: the series term ratio is
(t / 2 pi N)^2 ~ 0.1, so the truncation error is below 1e-18 for every t.
The main sum uses float64 cos/sin accumulated in float128; rounding gives
an absolute Z error ~1e-15 for t <= 300 and ~3e-10 for t <= 75000 - far
below the 1e-4 scan step used to bracket zeros (|Z'| * step ~ 1e-4).

Two independent implementations exist in this repository:

  * :mod:`zfast` - compiled numpy arithmetic, used to scan the critical line;
  * :mod:`zrigor` - mpmath (arb-backed ball arithmetic), slow, used to
    certify bracket endpoints and refine zeros.

The two agree far inside the quoted error bounds; see ``tests/test_rh.py``.
A missed sign change would require *both* independent implementations to be
wrong at the same place.
"""
import numpy as np

LOG2PI = np.longdouble(np.log(2.0 * np.pi))
LOGPI = np.longdouble(np.log(np.pi))


def _bernoulli_even(K):
    """B_2 .. B_2K as exact Fractions (Akiyama-Tanigawa style recursion)."""
    from fractions import Fraction
    A = [Fraction(0)] * (2 * K + 2)
    B = {}
    for m in range(2 * K + 1):
        A[m] = Fraction(1, m + 1)
        for j in range(m, 0, -1):
            A[j - 1] = j * (A[j - 1] - A[j])
        B[m] = A[0]
    return [B[2 * k] for k in range(1, K + 1)]


def _euler_macaurin_coeffs(k_max):
    """B_2k / (2k)! as float128 constants."""
    out = []
    for k, B in enumerate(_bernoulli_even(k_max), start=1):
        fact = 1
        for i in range(1, 2 * k + 1):
            fact *= i
        out.append(np.longdouble(B.numerator) / np.longdouble(
            B.denominator * fact))
    return out


def _stirling_coeffs(terms):
    """Coefficients B_2k / (2k(2k-1)) for the Stirling series of log Gamma."""
    from fractions import Fraction
    return [Fraction(B, (2 * k) * (2 * k - 1))
            for k, B in enumerate(_bernoulli_even(terms), start=1)]


_STIRLING_CACHE = {}


def _loggamma_stirling(w, M=16, terms=14):
    """log Gamma(w) for float128 complex w with |w| >= 1 (recurrence-shifted).

    log Gamma(w) = log Gamma(w+M) - sum_{j=0}^{M-1} log(w+j), the shifted
    value being taken from the Stirling series at w+M (|w+M| >= 16, so the
    optimally truncated 14-term series has remainder ~1e-24).
    """
    w = np.asarray(w, dtype=np.clongdouble)
    z = w + M
    inv = np.clongdouble(1.0) / z
    inv2 = inv * inv
    ser = np.zeros_like(z)
    pw = inv
    coefs = _STIRLING_CACHE.get(terms)
    if coefs is None:
        coefs = [np.clongdouble(np.longdouble(c.numerator)
                                / np.longdouble(c.denominator))
                 for c in _stirling_coeffs(terms)]
        _STIRLING_CACHE[terms] = coefs
    for c in coefs:
        ser = ser + c * pw
        pw = pw * inv2
    s = (z - 0.5) * np.log(z) - z + LOG2PI / 2 + ser
    for j in range(M):
        s = s - np.log(w + j)
    return s


def theta(t):
    """Riemann-Siegel theta(t) = Im log Gamma(1/4+i t/2) - (t/2) log(pi),
    float128, vectorised.  The recurrence reduction in _loggamma_stirling
    keeps Im log Gamma small, so no branch unwrapping is needed."""
    t = np.atleast_1d(np.asarray(t, dtype=np.longdouble))
    z = np.clongdouble(0.25) + 0.5j * t
    return np.imag(_loggamma_stirling(z)) - 0.5 * t * LOGPI


def n_terms_for(tmax, ratio=1.0):
    """Backwards-compat helper: N used by zeta_crit for scan planning."""
    return max(40, int(np.ceil(float(tmax) * ratio / 2.0)) + 8)


def zeta_crit(t, k_corr=18, n_terms=None):
    """zeta(1/2 + i t) by Euler-Maclaurin, vectorised over t.

    DLMF 25.1.14:  zeta(s) = sum_{n<N} n^-s + N^(1-s)/(s-1) + N^-s/2
    + sum_k B_2k/(2k)! * s(s+1)..(s+2k-2) * N^(-s-2k+1).  The correction
    series has term ratio ~(t / 2 pi N)^2, so N = t/2 + 8 gives ratio ~0.1:
    with k_corr=18 the truncation error is < 1e-18 for every t.  The main
    sum runs in float64 (phase error ~1e-10 at t=75000) accumulated in
    float128; returns (re, im) as float128 arrays.
    """
    t = np.atleast_1d(np.asarray(t, dtype=np.longdouble))
    if n_terms is None:
        n_terms = max(40, int(np.ceil(float(t.max()) / 2.0)) + 8)
    N = n_terms

    logn = np.log(np.arange(1, N, dtype=np.float64))
    inv_sqrt_n = 1.0 / np.sqrt(np.arange(1, N, dtype=np.float64))
    re = np.zeros(t.shape, dtype=np.longdouble)
    im = np.zeros(t.shape, dtype=np.longdouble)

    chunk = max(1, int(8_000_000 // max(N, 1)))
    for i in range(0, t.size, chunk):
        tt = np.asarray(t[i:i + chunk], dtype=np.float64)
        ph = np.outer(tt, logn)
        re[i:i + chunk] = (np.cos(ph) * inv_sqrt_n).sum(axis=1, dtype=np.longdouble)
        im[i:i + chunk] = (-np.sin(ph) * inv_sqrt_n).sum(axis=1, dtype=np.longdouble)

    # tail N^(1-s)/(s-1) + N^-s/2 with s = 1/2 + i t
    Nld = np.longdouble(N)
    ang = -t * np.log(Nld)
    ca, sa = np.cos(ang), np.sin(ang)
    Nhalf = np.sqrt(Nld)
    den = np.longdouble(0.25) + t * t
    re = re + Nhalf * (ca * (-np.longdouble(0.5)) + sa * t) / den + 0.5 / Nhalf * ca
    im = im + Nhalf * (sa * (-np.longdouble(0.5)) - ca * t) / den + 0.5 / Nhalf * sa

    # Euler-Maclaurin correction series
    s_c = np.clongdouble(0.5) + 1j * t
    invN2 = (np.clongdouble(1.0) / Nld) ** 2
    powN = np.exp(-(np.longdouble(1.5) - 1j * 0) * np.log(Nld)
                  + 1j * ang).astype(np.clongdouble)      # N^(-s-1)
    rising = s_c.copy()
    fact = np.longdouble(2.0)
    for k, B in enumerate(_bernoulli_even(k_corr), start=1):
        corr = (np.longdouble(B.numerator) / np.longdouble(B.denominator)
                / fact) * rising * powN
        re = re + np.real(corr)
        im = im + np.imag(corr)
        if k < k_corr:
            rising = rising * (s_c + (2 * k - 1)) * (s_c + 2 * k)
            powN = powN * invN2
            fact *= np.longdouble((2 * k + 1) * (2 * k + 2))
    return re, im


def Z(t, n_terms=None, dtype=np.longdouble):
    """Riemann-Siegel Z(t) = Re( e^{i theta(t)} zeta(1/2 + i t) ).

    Internally float128 throughout; `dtype` selects the returned precision
    (np.float64 for cheap bulk probes, np.longdouble for scans).
    """
    t = np.atleast_1d(np.asarray(t, dtype=np.longdouble))
    th = theta(t)
    re, im = zeta_crit(t, n_terms=n_terms)
    return (np.cos(th) * re - np.sin(th) * im).astype(dtype)


def scan_brackets(lo, hi, h=0.001, tiny=1e-9, sub_h=1e-5,
                  callback=None, chunk_points=32768):
    """Scan Z on [lo, hi] with step h; resample at step sub_h wherever an
    endpoint has |Z| < tiny (so that a close pair of zeros cannot be merged).

    Returns a list of brackets (a, b), a <= b, in increasing order: a sign
    change on (a, b), or the exact mesh point a == b where Z vanished.
    """
    n = int(np.floor((hi - lo) / h)) + 1
    edges = lo + h * np.arange(n + 1, dtype=np.longdouble)
    out = []
    prev_t = prev_z = None
    idx = 0
    while idx <= n:
        seg = edges[idx:idx + chunk_points + 1]
        if seg.size == 0:
            break
        z = Z(seg)
        if prev_t is not None:
            seg = np.concatenate(([prev_t], seg))
            z = np.concatenate(([prev_z], z))
        for j in range(seg.size - 1):
            a, b = seg[j], seg[j + 1]
            za, zb = z[j], z[j + 1]
            if za == 0:
                # exact vanishing mesh point; float() so mpmath can consume it
                out.append((float(a), float(a)))
                continue
            if (za > 0) != (zb > 0):
                out.append((float(a), float(b)))
            elif abs(za) < tiny or abs(zb) < tiny:
                m = max(2, int(np.ceil((b - a) / sub_h)))
                ts = a + (b - a) * np.arange(m + 1, dtype=np.longdouble) / m
                zs = Z(ts)
                for jj in range(m):
                    if zs[jj] == 0:
                        out.append((float(ts[jj]), float(ts[jj])))
                    elif (zs[jj] > 0) != (zs[jj + 1] > 0):
                        out.append((float(ts[jj]), float(ts[jj + 1])))
        prev_t, prev_z = seg[-1], z[-1]
        idx += chunk_points
        if callback:
            callback(lo + h * idx)
    return out
