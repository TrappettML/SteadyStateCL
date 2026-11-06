"""Host-side access to a sweep's results for the figure scripts (PLAN.md §12).

``analyze.py``, ``plot_norms.py`` and ``plot_check2.py`` all read the same
``batches/batch_*.npz`` files; this module turns them into one record per run with
its per-task scalars and running moments, and groups records by condition.
Nothing here touches raw state.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np

import warnings

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
    schema : int
        Result schema of the batch file (``io.SCHEMA_VERSION`` when current).
    """

    run: Run
    scalars: dict
    moments: dict
    ages: np.ndarray
    T_burn: int
    T_meas: int
    alpha_task: float
    schema: int = io.SCHEMA_VERSION

    @property
    def T(self) -> int:
        """Number of tasks, T_burn + T_meas."""
        return self.T_burn + self.T_meas

    def key(self) -> tuple:
        """Condition key ``(N, kappa_label, readout, delta0, lam, solver, eta)``."""
        return tuple(getattr(self.run, k) for k in COND_KEYS)

    def kappa_sched(self, kappa_min: float = 1.0) -> float:
        """Schedule κ of PLAN.md §5: ``run.kappa_sched`` when the batch file recorded it; else the run's κ when
        gated and ``kappa_min`` (the sweep's smallest gated κ) when ungated (older batch files)."""
        if self.run.kappa_sched:
            return float(self.run.kappa_sched)
        return float(self.run.kappa) if is_gated(self.run.readout) else float(kappa_min)

    def lifetimes(self, kappa_min: float = 1.0) -> np.ndarray:
        """Task index in lifetimes, ``t / (N ln 2)``: the half-life of every gated readout is ≈ N ln 2 at every κ
        (write-up eq. mhalf_heads), so this unit is comparable across κ (until 2026-11-06 it was ``t κ_s²/N``,
        the N/κ² axis the write-up drops; ``kappa_min`` is unused since)."""
        return np.arange(self.T) / (self.run.N * np.log(2.0))

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

    Warns (``UserWarning``) when some batch files were written under an older
    schema than ``io.SCHEMA_VERSION``: their records lack the newer scalars,
    which the figures then mark as not recorded.  Rerunning the sweep
    (``run_sweep.py``) regenerates exactly those batches.
    """
    records, skipped, stale, n_batches = [], [], [], 0
    seen: dict = {}
    for res, runs, meta in io.load_sweep(out_dir):
        n_batches += 1
        skipped += meta.get("skipped", [])
        schema = int(meta.get("schema", 1))
        if schema < io.SCHEMA_VERSION:
            stale.append(int(meta.get("batch_id", len(stale))))
        if not runs:
            continue
        T_burn, T_meas = int(res["T_burn"]), int(res["T_meas"])
        T = T_burn + T_meas
        scalar_names = [k for k, v in res.items() if getattr(v, "ndim", 0) == 2 and v.shape[0] == T and k not in Moments.FIELDS]
        for i, r in enumerate(runs):
            seen.setdefault(r.label(), []).append(int(meta.get("batch_id", -1)))
            sc = {k: np.asarray(res[k][:, i]) for k in scalar_names}
            for old, new in LEGACY.items():
                if new not in sc and old in sc:
                    sc[new] = sc[old]
            mom = {f: np.asarray(res[f][i]) for f in Moments.FIELDS}
            records.append(RunRecord(r, sc, mom, np.asarray(res["ages"]), T_burn, T_meas, float(res["alpha_task"][i]), schema))
    dup = {lab: ids for lab, ids in seen.items() if len(ids) > 1}
    if dup:
        lab, ids = next(iter(dup.items()))
        warnings.warn(f"{len(dup)} runs appear in more than one batch file of {out_dir} (e.g. {lab} in batches {ids}): "
                      f"the sweep's config changed after some batches were written, so they count twice as extra seeds. "
                      f"Rerun run_sweep.py on this sweep: batches holding other runs than planned are pending again.",
                      stacklevel=2)
    if stale:
        Ns = sorted({r.run.N for r in records if r.schema < io.SCHEMA_VERSION})
        warnings.warn(f"{len(stale)} of {n_batches} batch files in {out_dir} "
                      f"were written under result schema < {io.SCHEMA_VERSION} (batches {stale[:6]}{', …' if len(stale) > 6 else ''}; "
                      f"N = {Ns}); their §12.1 norm scalars are missing and show as 'not recorded'. "
                      f"Rerun run_sweep.py on this sweep: stale batches are pending again.", stacklevel=2)
    return records, skipped


def kappa_min(records: Iterable[RunRecord]) -> float:
    """Smallest κ among the gated records (the schedule κ of ungated runs, PLAN.md §5); 1.0 when there are none."""
    ks = [r.run.kappa for r in records if is_gated(r.run.readout)]
    return float(min(ks)) if ks else 1.0


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


def slice_groups(records: Iterable[RunRecord]) -> dict:
    """Group records for the slice figures S1/S2 (PLAN.md §12.3): one group per readout, κ, λ, solver, η.

    Returns
    -------
    dict
        ``{(readout, kappa_label or None, lam, solver, eta): [records]}``; the
        κ field is ``None`` for ungated readouts, which have no κ and get one
        group (and one file, with no ``k`` in the stem).
    """
    g = defaultdict(list)
    for r in records:
        k = r.run.kappa_label if is_gated(r.run.readout) else None
        g[(r.run.readout, k, r.run.lam, r.run.solver, r.run.eta)].append(r)
    return dict(g)


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
    # finite values only: an active-unit mean is NaN for a task whose gate drew no unit (small N, κ), and a
    # half-window may hold none, which must give NaN quietly rather than numpy's empty-slice warnings
    segs = [np.asarray(s, float) for s in (x[T_burn:mid], x[mid:T_burn + T_meas])]
    segs = [s[np.isfinite(s)] for s in segs]
    m = [float(s.mean()) if len(s) else np.nan for s in segs]
    se = [float(s.std(ddof=1) / np.sqrt(len(s))) if len(s) > 1 else np.nan for s in segs]
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
        ``t / (N ln 2)`` (``RunRecord.lifetimes``).
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


def seed_mean(X, axis: int = 0) -> np.ndarray:
    """Mean over seeds (``axis``) of per-seed values, ignoring seeds without a value (NaN); NaN where none has one."""
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(np.asarray(X, float), axis=axis)


def seed_sem(X, axis: int = 0) -> np.ndarray:
    """s.e.m. over seeds (``axis``) of per-seed values, ignoring seeds without a value; NaN with fewer than two."""
    X = np.asarray(X, float)
    n = np.isfinite(X).sum(axis)
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sd = np.nanstd(X, axis=axis, ddof=1)
    return np.where(n > 1, sd / np.sqrt(np.maximum(n, 1)), np.nan)


def seed_curves(records: Iterable[RunRecord], halves=(0, 1)) -> dict:
    """Every seed's own ``S``, ``E``, ``cv_s``, ``cv_e``, ``n`` (``RunRecord.curve``) as (n_seeds, n_ages) arrays."""
    cs = [r.curve(halves) for r in records]
    return {k: np.stack([c[k] for c in cs]) for k in cs[0]}


def seed_mhalf(records: Iterable[RunRecord], halves=(0, 1)) -> np.ndarray:
    """Half-life of each seed's own S(m) over the given measurement halves.

    Returns
    -------
    (n_seeds,) array
        ``inf`` where the seed's curve never reaches S(0)/2 (censored), NaN
        where the seed has no curve (S(0) not finite, e.g. an infeasible run).
    """
    out = []
    for r in records:
        S = r.curve(halves)["S"]
        if not np.isfinite(S[0]):
            out.append(np.nan)
            continue
        m, cens = half_life(r.ages, S)
        out.append(np.inf if cens else m)
    return np.asarray(out, float)


def mhalf_stats(per: np.ndarray) -> dict:
    """Seed mean of per-seed half-lives (``seed_mhalf``).

    Returns
    -------
    dict
        ``m_half`` the mean over the seeds with a curve, NaN (and ``censored``
        True) when any of them never reaches S(0)/2, since the mean is then
        not defined, or when no seed has a curve; ``m_half_se`` the s.e.m. over
        the uncensored seeds (NaN with fewer than two); ``m_half_lower`` the
        mean over the uncensored seeds (a lower bound when some are censored);
        ``n_censored``; ``n_used`` the seeds with a curve.
    """
    per = np.asarray(per, float)
    per = per[~np.isnan(per)]
    fin = per[np.isfinite(per)]
    n_cens = int(len(per) - len(fin))
    censored = n_cens > 0 or len(per) == 0
    return {"m_half": float(fin.mean()) if not censored else np.nan,
            "m_half_se": float(sem(fin)) if len(fin) > 1 else np.nan,
            "m_half_lower": float(fin.mean()) if len(fin) else np.nan,
            "censored": bool(censored), "n_censored": n_cens, "n_used": int(len(per))}


def condition_mhalf(records: Iterable[RunRecord], halves=(0, 1)) -> tuple[float, bool]:
    """Half-life of a condition: each seed's own half-life over the given halves, averaged over seeds (``mhalf_stats``).

    Parameters
    ----------
    records : iterable of RunRecord sharing a condition
    halves : tuple
        Measurement halves each seed's S(m) is taken from.

    Returns
    -------
    (m_half, censored)
        NaN and True when any seed's curve never reaches S(0)/2.
    """
    st = mhalf_stats(seed_mhalf(list(records), halves))
    return st["m_half"], st["censored"]


def mhalf_by_condition(records: Iterable[RunRecord]) -> dict:
    """Half-life of every condition, the seed mean of the per-seed half-lives (M4, M5, M8 and check 4′ of PLAN.md §12.3, §12.6).

    Parameters
    ----------
    records : iterable of RunRecord

    Returns
    -------
    dict
        ``{condition key: {"m_half", "m_half_se", "censored", "n_censored", "m_half_lower", "n_seeds"}}``
        (``mhalf_stats``): each seed's half-life from its own S(m), then the
        mean over seeds; ``censored`` (and ``m_half`` NaN) when any seed's
        curve never reaches S(0)/2.
    """
    out = {}
    for key, recs in group_by(records, COND_KEYS).items():
        st = mhalf_stats(seed_mhalf(recs))
        out[key] = {"m_half": st["m_half"], "m_half_se": st["m_half_se"], "censored": st["censored"],
                    "n_censored": st["n_censored"], "m_half_lower": st["m_half_lower"], "n_seeds": len(recs)}
    return out


def level_stats(records: Iterable[RunRecord], key: str) -> dict:
    """Steady-state level of a per-task scalar: its mean over the second half of the measurement window (M7, PLAN.md §12.3).

    Parameters
    ----------
    records : iterable of RunRecord
    key : str
        Scalar name (``aQ_act_post``, ``aQ_q95``, ``aQ_max_post``, ``Q_act_pre``, …).

    Returns
    -------
    dict
        ``{condition key: (mean, se, n_seeds)}``: the per-seed means over the
        finite values of the second half, averaged over seeds, with the
        s.e.m. over seeds (NaN with one seed).  Conditions whose batch files
        lack the scalar map to ``(nan, nan, 0)``.
    """
    out = {}
    for ck, recs in group_by(records, COND_KEYS).items():
        vals = []
        for r in recs:
            if key not in r.scalars:
                continue
            mid = r.T_burn + r.T_meas // 2
            x = np.asarray(r.scalars[key][mid:r.T_burn + r.T_meas], float)
            x = x[np.isfinite(x)]
            vals.append(x.mean() if len(x) else np.nan)
        v = np.asarray(vals, float)
        v = v[np.isfinite(v)]
        out[ck] = (float(v.mean()) if len(v) else np.nan, float(sem(v)) if len(v) > 1 else np.nan, len(vals))
    return out
