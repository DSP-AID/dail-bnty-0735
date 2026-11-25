# Riemann Hypothesis — a reproducible, self-certifying verification harness

**Honest headline: the Riemann Hypothesis is not proved here.** Nobody has
proved it. What this repository delivers is a *verification harness* that
turns "I ran some numbers" into statements of the form

> for every `t` in `[a, b]` there is exactly one zero of `zeta(1/2 + it)`, its
> ordinate is `gamma = 14.134725141734693790457251983562470270784257…`, and the
> sign of `Z` at both ends of `[a, b]` was obtained from ball arithmetic at two
> independent precisions.

plus a *count* result: up to height `T`, the number of nontrivial zeros of
`zeta` in the strip `0.4 <= Re s <= 6` equals `2 x` the number of zeros found on
the critical line, which certifies that **no zero off the line and no multiple
zero** occurs below `T`. Every step is reproducible offline from public data
shipped in this repository, with no API keys.

## Quick start

```bash
python3 -m pytest tests/test_rh.py -q      # self-checks, ~5 s
python3 verify.py --quick                  # smoke run, ~1 min
python3 verify.py                          # full run, ~10 min
python3 verify.py --json results.json      # machine-readable report
```

Requirements: CPython 3.10+, `numpy`, `mpmath`. Nothing else; no network access
at verification time (both published tables are in `data/`).

## Layout

| file | role |
| --- | --- |
| `zfast.py` | fast tier: vectorised `Z(t)` / `theta(t)` by Euler–Maclaurin summation in float64/float128 numpy arithmetic |
| `zrigor.py` | rigorous tier: mpmath (arb ball arithmetic) `Z(t)`, two-precision sign certification, bisection refinement, argument-principle zero count for `xi` |
| `verify.py` | the end-to-end verification run (steps A–D below), exits non-zero if any check fails |
| `tests/test_rh.py` | cross-validation of the two tiers, dataset integrity, contour-count sanity checks |
| `data/plouffe_first100.txt` | first 100 zero ordinates, ~1000 exact decimal digits each (Gourdon–Plouffe public table) |
| `data/odlyzko_zeros1.txt` | first 100 000 zero ordinates, 9 decimals (Andrew Odlyzko's public `zeros1` table, heights up to 74 920.827498994) |

## What the run certifies

**A — blind scan with certified brackets.** `Z(t)` is sampled on `[0.5, T]`
with step `1e-4`; every sign change gives a bracket, and close pairs of zeros
cannot be merged because any mesh point with `|Z| < 1e-9` is resampled at step
`1e-5`. Each bracket endpoint's sign is then re-derived in the rigorous tier at
25 **and** 40 decimal digits; a bracket is kept only if both precisions agree on
opposite nonzero signs. Every certified bracket is refined by bisection at 45
digits, giving ordinates with a validated error bar.

**B — agreement with two independent published tables.** The refined ordinates
are compared digit-by-digit with the Gourdon–Plouffe 1000-digit table (matching
digits are reported), and element-by-element with Odlyzko's `zeros1` table using
the scan's own ordering, so a missed *or* spurious zero shows up as a mismatch
rather than a silent shift.

**C — the whole of Odlyzko's 100 000-zero table is probed.** For every
`stride`-th tabulated ordinate `g`, the fast tier evaluates `Z(g - 1e-4)` and
`Z(g + 1e-4)`; a sign change there proves a zero of `Z` within `1e-4` of `g`.
Because `1e-4` is far below the smallest known gap between consecutive zeros,
this pins *the tabulated* zero, and it is done for all 2 000 sampled entries up
to `t = 74 920`. A spread sample of those entries is additionally certified in
the rigorous tier (two precisions) and refined, confirming the table's own
digits.

**D — no off-line zero, no multiple zero, below `T`.** By the argument
principle applied to `xi(s) = 1/2 s(s-1) pi^{-s/2} Gamma(s/2) zeta(s)` on the
boundary of the rectangle `0.4 <= Re s <= 6`, `|Im s| <= T`, the total number of
zeros in that rectangle is computed (the sampling step is halved until no step
moves the argument by more than 2.8 rad, so the winding number is resolved).
Using `xi(conj s) = conj xi(s)` and `xi(1-s) = xi(s)`, that count must equal
`2 x (# critical-line zeros)` if every zero below `T` is simple and on the line;
`4 x` each off-line orbit would show up as an excess. The run compares the two
numbers and reports the excess as `off_line_orbits`.

## Why the two tiers are independent

The fast tier derives `zeta(1/2 + it)` from the Euler–Maclaurin (classical)
summation formula with `N = t/2 + 8` terms and 18 Bernoulli correction terms,
and `theta(t)` from a recurrence-shifted Stirling series — both hand-written in
numpy. The rigorous tier calls mpmath's `siegelz`, which is built on arb ball
arithmetic with a different algorithm (Riemann–Siegel formula / Euler–Maclaurin
with rigorous error balls). Their agreement is asserted numerically in
`tests/test_rh.py` at several heights up to 30 425, and the documented rounding
bounds (`~1e-15` for `t <= 300`, `~3e-10` for `t <= 75 000`) sit four orders of
magnitude below the `1e-4` scan step that produces the brackets. A missed sign
change would require both implementations to be wrong at the same place, in the
same direction, inside one scan cell.

## Limits of the claim (read this before citing it)

* Numerical evidence is **not** a proof of RH: the certification covers only the
  finite heights actually run (`T = 300` for the scan, `T = 200` for the strip
  count, probes up to `t = 74 920`). It says nothing about larger heights.
* The sign "certification" is two-precision agreement of mpmath evaluations,
  plus escalation to 60 digits on disagreement. It is not a formal interval
  proof with tracked rounding: a systematic bug in mpmath's `siegelz` path would
  be invisible to it. The independent fast tier and the two published tables are
  what make such a bug implausible.
* The contour count in D assumes the sampled boundary never passes through a
  zero (checked) and that the argument steps resolve the winding number
  (checked, reported as `max_arg_step_rad` and `residual`).
* RH is known to hold far beyond these heights in the published literature
  (Platt–Trudgian verified it to heights of order 3 x 10^12, and ~41 % of all
  zeros are known to be on the line — Conrey–Iwaniec–Soundararajan). This
  repository is a self-contained re-derivation at modest height, not a record.

## Environment

Tested on Ubuntu 24.04, CPython 3.12, numpy 2.x (float128 = x87 80-bit), mpmath
1.3.x. The full run stays under 1 GB RSS and uses one core, so it also fits a
RAM-limited sandbox (`RLIMIT_AS` friendly): the scan works in fixed-size chunks.
