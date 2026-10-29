"""Host-side access to a sweep's results for the figure scripts (PLAN.md §12).

``analyze.py``, ``plot_norms.py`` and ``plot_check2.py`` all read the same
``batch_*.npz`` files; this module turns them into one record per run with
its per-task scalars and running moments, and groups records by condition.
Nothing here touches raw state.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np

from mup import io
from mup.config import Run, is_gated
from mup.observe import Moments, curves, half_life

#: Condition key fields, in order.
COND_KEYS = ("N", "kappa_label", "readout", "delta0", "lam", "solver", "eta")
#: Schema-1 → schema-2 scalar names (``load_runs`` fills the new names from the old ones).
LEGACY = {"Q_mean": "Q_all_post", "a2_mean": "a2_all_post", "aQ_mean": "aQ_all_post", "aQ_max": "aQ_max_post"}


@dataclass
class RunRecord:
    """One run's results.

    Attributes
    ----------
    run : Run
    scalars : dict of (T,) arrays
        Every per-task scalar of the batch file (schema-2 names, legacy names mapped).
    moments : dict of (n_ages, 2) arrays
        ``sum_s``, ``sum_s2``, ``sum_e``, ``sum_e2``, ``count`` per (age, half).
    ages : (n_ages,) int array
    T_burn, T_meas : int
    alpha_task : float
    """

    run: Run
    scalars: dict
    moments: dict
    ages: np.ndarray
    T_burn: int
    T_meas: int
    alpha_task: float

    @property
    def T(self) -> int:
        """Number of tasks, T_burn + T_meas."""
        return self.T_burn + self.T_meas

    def key(self) -> tuple:
        """Condition key ``(N, kappa_label, readout, delta0, lam, solver, eta)``."""
        return tuple(getattr(self.run, k) for k in COND_KEYS)

    def lifetimes(self) -> np.ndarray:
        """Task index in lifetimes, ``t κ²/N`` (κ = √N for an ungated run, so simply ``t``)."""
        return np.arange(self.T) * self.run.kappa ** 2 / self.run.N

    def curve(self, halves=(0, 1)) -> dict:
        """``S``, ``E``, ``cv_s``, ``cv_e``, ``n`` of this run as (n_ages,) arrays."""
        c = curves({k: v[None] for k, v in self.moments.items()}, halves)
        return {k: v[0] for k, v in c.items()}


def load_runs(out_dir: str) -> tuple[list[RunRecord], list[str]]:
    """Load every finished batch of a sweep into per-run records.

    Parameters
    ----------
    out_dir : str
        ``results/<sweep>``.

    Returns
    -------
    (records, skipped)
        ``records`` one ``RunRecord`` per run; ``skipped`` the labels of runs
        the sweep refused as infeasible.
    """
    records, skipped = [], []
    for res, runs, meta in io.load_sweep(out_dir):
        skipped += meta.get("skipped", [])
        if not runs:
            continue
        T_burn, T_meas = int(res["T_burn"]), int(res["T_meas"])
        T = T_burn + T_meas
        scalar_names = [k for k, v in res.items() if getattr(v, "ndim", 0) == 2 and v.shape[0] == T and k not in Moments.FIELDS]
        for i, r in enumerate(runs):
            sc = {k: np.asarray(res[k][:, i]) for k in scalar_names}
            for old, new in LEGACY.items():
                if new not in sc and old in sc:
                    sc[new] = sc[old]
            mom = {f: np.asarray(res[f][i]) for f in Moments.FIELDS}
            records.append(RunRecord(r, sc, mom, np.asarray(res["ages"]), T_burn, T_meas, float(res["alpha_task"][i])))
    return records, skipped


def group_by(records: Iterable[RunRecord], fields: tuple) -> dict:
    """Group records by the given ``COND_KEYS`` fields.

    Parameters
    ----------
    records : iterable of RunRecord
    fields : tuple of str
        Subset of ``COND_KEYS``.

    Returns
    -------
    dict
        ``{tuple of field values: [records]}``.
    """
    g = defaultdict(list)
    for rec in records:
        g[tuple(getattr(rec.run, f) for f in fields)].append(rec)
    return dict(g)


def by_kappa(records: Iterable[RunRecord]) -> dict:
    """Split records by κ label for the gated readouts, repeating the ungated ones in every group.

    Returns
    -------
    dict
        ``{kappa_label: [records]}``; the ungated records (no κ) appear under
        every label, so each κ figure can show all five readouts.
    """
    records = list(records)
    ungated = [r for r in records if not is_gated(r.run.readout)]
    out = defaultdict(list)
    for r in records:
        if is_gated(r.run.readout):
            out[r.run.kappa_label].append(r)
    if not out:
        out["sqrtN"] = []
    for members in out.values():
        members.extend(ungated)
    return dict(out)


def seed_stack(records: Iterable[RunRecord], key: str) -> np.ndarray:
    """Stack one per-task scalar across seeds.

    Parameters
    ----------
    records : iterable of RunRecord sharing a condition
    key : str
        Scalar name.

    Returns
    -------
    (n_seeds, T) array
    """
    return np.stack([r.scalars[key] for r in records])


def sem(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """Standard error of the mean along ``axis``; NaN with fewer than two samples.

    Parameters
    ----------
    x : array
    axis : int
    """
    x = np.asarray(x, float)
    n = x.shape[axis]
    if n < 2:
        return np.full(np.delete(x.shape, axis), np.nan)
    return np.std(x, axis=axis, ddof=1) / np.sqrt(n)


def half_stats(x: np.ndarray, T_burn: int, T_meas: int) -> dict:
    """Half-window means of a per-task quantity (check 2).

    Parameters
    ----------
    x : (T,) array
    T_burn, T_meas : int

    Returns
    -------
    dict
        ``half0``, ``half1`` (means), ``se`` (√(se₀² + se₁²), tasks treated as
        independent, so conservative), ``z``, ``ratio`` (half1/half0 − 1).
    """
    mid = T_burn + T_meas // 2
    segs = (x[T_burn:mid], x[mid:T_burn + T_meas])
    m = [float(np.nanmean(s)) if len(s) else np.nan for s in segs]
    se = [float(np.nanstd(s, ddof=1) / np.sqrt(len(s))) if len(s) > 1 else np.nan for s in segs]
    se_tot = np.float64(np.sqrt(se[0] ** 2 + se[1] ** 2))
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.abs(np.float64(m[1] - m[0])) / se_tot              # inf when the halves are constant and differ
        ratio = np.float64(m[1]) / np.float64(m[0]) - 1.0
    return {"half0": m[0], "half1": m[1], "se": float(se_tot), "z": float(z), "ratio": float(ratio)}


def drift_rate(x: np.ndarray, lifetimes: np.ndarray, T_burn: int, T_meas: int) -> tuple[float, float]:
    """Least-squares slope β of ``log x`` against task index in lifetimes over the measurement window (C2-C).

    Parameters
    ----------
    x : (T,) array, positive
    lifetimes : (T,) array
        ``t κ²/N``.
    T_burn, T_meas : int

    Returns
    -------
    (beta, se_beta)
        Fractional change per lifetime and its standard error (tasks treated
        as independent); NaN when fewer than three finite points.
    """
    sl = slice(T_burn, T_burn + T_meas)
    with np.errstate(divide="ignore", invalid="ignore"):
        y = np.log(np.asarray(x[sl], float))            # −inf / NaN for non-positive values, dropped below
    u = np.asarray(lifetimes[sl], float)
    ok = np.isfinite(y) & np.isfinite(u)
    if ok.sum() < 3:
        return np.nan, np.nan
    y, u = y[ok], u[ok]
    A = np.stack([u, np.ones_like(u)], 1)
    coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
    dof = len(y) - 2
    s2 = (res[0] / dof) if len(res) and dof > 0 else np.nan
    cov = s2 * np.linalg.inv(A.T @ A)
    return float(coef[0]), float(np.sqrt(cov[0, 0]))


def condition_mhalf(records: Iterable[RunRecord], halves=(0, 1)) -> tuple[float, bool]:
    """Half-life of the seed-pooled S(m) of a condition.

    Parameters
    ----------
    records : iterable of RunRecord sharing a condition
    halves : tuple
        Measurement halves to pool.

    Returns
    -------
    (m_half, censored)
    """
    records = list(records)
    hs = list(halves)
    tot = sum(r.moments["sum_s"][:, hs].sum(-1) for r in records)
    n = sum(r.moments["count"][:, hs].sum(-1) for r in records)
    with np.errstate(invalid="ignore", divide="ignore"):
        S = tot / n
    return half_life(records[0].ages, S)
