"""Half-life statistics of the experiment scripts (write-up, "Observables" and "Memory half-life"), NumPy only.

The theory's m½ is the root of the MEAN curve: Σ(m) = E[s_k(m)]/E[s_k(0)], the seed mean of the signal at
age m divided by the seed mean at age 0 (eq. Sigma), crosses ½ at m½ (first crossing, linear interpolation).
That is the statistic ``"mean"`` here and the default of every script.  The other statistic, ``"seed"``, is
the mean over seeds of each seed's own half-life; it is a different number — a noisy curve crosses ½ earlier
on average — and is kept as an option and in the ``.npz`` files, since the write-up's remarks on it refer to it.

A signal array ``S`` is (n_ages, n_seeds): row m the signal after m further tasks, one column per seed; a
column with a non-finite entry (an infeasible seed) is left out of every statistic.

So that the figures, tables and files of the scripts need not know which statistic they hold, both come as an
(n_seeds,) array ``v`` to be summarised by its mean and its standard error over seeds (``stat``):

``"seed"``   v_j = the half-life of seed j (NaN when its curve never reaches half: censored).
``"mean"``   v_j = m½ + (n − 1)(m̄_(·) − m_(−j)), with m½ the half-life of the mean curve, m_(−j) that of the
             mean curve without seed j and m̄_(·) their mean: the jackknife values centred on m½.  Their mean
             is m½ exactly and their standard error √((n − 1)/n · Σ_j (m_(−j) − m̄_(·))²) is the jackknife
             standard error of m½.  All NaN when the mean curve never reaches half.
"""
from __future__ import annotations

import math
from typing import Callable, Iterable

import numpy as np

from sscl_core.theory import first_half_crossing

#: The statistics, in the order the scripts offer them (``add_arg``).
STATS = ("mean", "seed")
#: What a point of a figure is, for captions.
STAT_TEXT = {
    "mean": "m½: first crossing of ½ by the mean curve, seed mean of the signal over its value at age 0 (the "
            "write-up's Σ(m)); bars: jackknife s.e. over seeds",
    "seed": "m½: each seed's own first crossing of half its initial signal, mean ± 1 s.e. over the seeds that reach it",
}


def half_life(curve: np.ndarray) -> float:
    """First index at which ``curve`` falls to ``curve[0]/2``, linearly interpolated; NaN if never."""
    return first_half_crossing(curve)


def _finite(S: np.ndarray) -> np.ndarray:
    """Mask of the columns of ``S`` without a non-finite entry."""
    return np.isfinite(S).all(0)


def per_seed(S: np.ndarray) -> np.ndarray:
    """(n_seeds,) half-life of every column of ``S``; NaN when censored or the column is not finite."""
    S = np.asarray(S, float)
    ok = _finite(S)
    return np.array([half_life(S[:, j]) if ok[j] else float("nan") for j in range(S.shape[1])])


def mean_curve(S: np.ndarray) -> np.ndarray:
    """Σ(m) = mean_j S[m, j] / mean_j S[0, j] over the finite columns (eq. Sigma); all NaN when there is none."""
    S = np.asarray(S, float)
    ok = _finite(S)
    if not ok.any():
        return np.full(S.shape[0], np.nan)
    m = S[:, ok].mean(1)
    return m / m[0]


def mean_curve_se(S: np.ndarray) -> np.ndarray:
    """Jackknife standard error over seeds of ``mean_curve`` at every age (0 with fewer than two seeds)."""
    S = np.asarray(S, float)
    S = S[:, _finite(S)]
    n = S.shape[1]
    if n < 2:
        return np.zeros(S.shape[0])
    tot = S.sum(1, keepdims=True)
    loo = (tot - S) / (n - 1)                                   # (n_ages, n): mean without seed j
    with np.errstate(invalid="ignore", divide="ignore"):
        loo = loo / loo[:1]
    return np.sqrt((n - 1) / n * ((loo - loo.mean(1, keepdims=True)) ** 2).sum(1))


def of_mean(S: np.ndarray) -> np.ndarray:
    """(n_seeds,) jackknife values of the half-life of the mean curve, centred on it (module docstring).

    NaN for a seed whose column is not finite and for one whose leave-one-out curve never reaches half; all
    NaN when the mean curve itself never does.
    """
    S = np.asarray(S, float)
    ok = _finite(S)
    out = np.full(S.shape[1], np.nan)
    n = int(ok.sum())
    if n == 0:
        return out
    Sk = S[:, ok]
    mh = half_life(Sk.mean(1))
    if not np.isfinite(mh):
        return out
    if n == 1:
        out[ok] = mh
        return out
    tot = Sk.sum(1)
    loo = np.array([half_life((tot - Sk[:, j]) / (n - 1)) for j in range(n)])
    out[ok] = mh + (n - 1) * (np.nanmean(loo) - loo)
    return out


def values(S: np.ndarray, stat: str = "mean") -> np.ndarray:
    """(n_seeds,) values of statistic ``stat`` for one condition's signal ``S`` (n_ages, n_seeds)."""
    if stat not in STATS:
        raise ValueError(f"unknown half-life statistic {stat!r}; known: {STATS}")
    return of_mean(S) if stat == "mean" else per_seed(S)


def stat(v: np.ndarray) -> tuple[float, float]:
    """(mean, s.e.) of an (n_seeds,) ``values`` array over its finite entries; (NaN, NaN) when none, s.e. 0 for one."""
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return float("nan"), float("nan")
    return float(v.mean()), float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else 0.0


def by_condition(columns: Iterable[tuple], key: Callable, stat_: str = "mean") -> dict:
    """Group per-run signal columns into conditions and reduce each to its half-life values.

    Parameters
    ----------
    columns : iterable of (run, column)
        One (n_ages,) signal column per ``config.Run``; a condition's seeds may come from several jobs.
    key : callable
        ``key(run)`` → the condition (hashable); its runs differ in the seed only.
    stat_ : {"mean", "seed"}

    Returns
    -------
    dict
        ``{condition: {"mhalf": (n_seeds,) ``values`` of ``stat_``, "seed": (n_seeds,) per-seed half-lives,
        "S": (n_ages, n_seeds) the columns, "seeds": the seed numbers}}``, columns in seed order.
    """
    groups: dict = {}
    for r, col in columns:
        groups.setdefault(key(r), []).append((r.seed, np.asarray(col, float)))
    out = {}
    for k, rows in groups.items():
        rows.sort(key=lambda t: t[0])
        S = np.stack([c for _, c in rows], 1)
        out[k] = {"mhalf": values(S, stat_), "seed": per_seed(S), "S": S, "seeds": [s for s, _ in rows]}
    return out


def add_arg(ap) -> None:
    """Add ``--stat`` to a script's argument parser (default ``mean``, the write-up's m½)."""
    ap.add_argument("--stat", choices=STATS, default="mean",
                    help="half-life statistic: 'mean' = first crossing of ½ by the seed-mean curve, the write-up's "
                         "m½ (root of Σ(m) = ½), with a jackknife s.e.; 'seed' = mean of the per-seed half-lives")
