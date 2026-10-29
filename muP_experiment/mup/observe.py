"""Probes, observables, running moments and the checks of PLAN.md §5–§6."""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import mup  # noqa: F401
from mup.model import Params, State


# ----------------------------------------------------------------------------
# age schedule (host side, static)
# ----------------------------------------------------------------------------
def age_schedule(N: int, kappa: float, T: int, n_geom: int = 24, m_max_factor: float = 4.0) -> np.ndarray:
    """ages = [0..20] ∪ geomspace(24, m_max, n_geom), unique ints, m_max = 4·10N/κ² capped at T−1."""
    m_max = int(min(round(m_max_factor * 10.0 * N / kappa**2), T - 1))
    ages = list(range(0, min(20, m_max) + 1))
    if m_max > 24:
        ages += [int(round(x)) for x in np.geomspace(24, m_max, n_geom)]
    elif m_max > 20:
        ages += list(range(21, m_max + 1))
    return np.array(sorted(set(ages)), dtype=np.int32)


# ----------------------------------------------------------------------------
# probe evaluation (device side, inside run_task)
# ----------------------------------------------------------------------------
def evaluate_probes(state: State, p: Params, ages: jnp.ndarray):
    """s_k, ε_k for probes k = t − ages (one (n_ages, N) @ (N, D) matmul).

    Returns (s, eps, valid); invalid ages (k < 0) read probe 0 and are masked.
    """
    k = state.t - ages
    valid = k >= 0
    idx = jnp.where(valid, k, 0)
    wk = state.probes_w[idx]                       # (n_ages, D)
    gk = state.probes_g[idx]                       # (n_ages, N)
    ak = jnp.where(p.use_heads == 1.0, state.heads[idx], state.a[None, :])
    U = (gk * ak) @ state.W / p.pN                 # (n_ages, D)
    wn = (wk * wk).sum(-1)
    s = (U * wk).sum(-1) / wn
    eps = ((U - wk) ** 2).sum(-1) / wn
    return s, eps, valid


# ----------------------------------------------------------------------------
# running moments on the host
# ----------------------------------------------------------------------------
class Moments:
    """Σs, Σs², Σε, Σε², count per (run, age, half); nothing per-sample is kept."""

    FIELDS = ("sum_s", "sum_s2", "sum_e", "sum_e2", "count")

    def __init__(self, B: int, n_ages: int):
        self.arr = np.zeros((5, B, n_ages, 2))

    def add(self, s: np.ndarray, eps: np.ndarray, valid: np.ndarray, half: int) -> None:
        v = valid.astype(np.float64)
        self.arr[0, :, :, half] += np.where(valid, s, 0.0)
        self.arr[1, :, :, half] += np.where(valid, s * s, 0.0)
        self.arr[2, :, :, half] += np.where(valid, eps, 0.0)
        self.arr[3, :, :, half] += np.where(valid, eps * eps, 0.0)
        self.arr[4, :, :, half] += v

    def as_dict(self) -> dict:
        return {f: self.arr[i] for i, f in enumerate(self.FIELDS)}


def curves(mom: dict, halves=(0, 1)):
    """Per-run means and CVs from running moments.

    Returns dict with S, E (B, n_ages) means, cv_s, cv_e (B, n_ages), n (B, n_ages).
    """
    hs = list(halves)
    n = mom["count"][..., hs].sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        S = mom["sum_s"][..., hs].sum(-1) / n
        E = mom["sum_e"][..., hs].sum(-1) / n
        vs = mom["sum_s2"][..., hs].sum(-1) / n - S**2
        ve = mom["sum_e2"][..., hs].sum(-1) / n - E**2
        cv_s = np.sqrt(np.maximum(vs, 0.0)) / np.abs(S)
        cv_e = np.sqrt(np.maximum(ve, 0.0)) / np.abs(E)
    return {"S": S, "E": E, "cv_s": cv_s, "cv_e": cv_e, "n": n}


def half_life(ages: np.ndarray, S: np.ndarray):
    """First age with S(m) ≤ S(0)/2, linearly interpolated; (NaN, True) if censored."""
    S = np.asarray(S, dtype=float)
    target = 0.5 * S[0]
    if not np.isfinite(S[0]):
        return np.nan, True
    below = np.where(S <= target)[0]
    if len(below) == 0:
        return np.nan, True
    j = int(below[0])
    if j == 0:
        return 0.0, False
    m0, m1, s0, s1 = ages[j - 1], ages[j], S[j - 1], S[j]
    if s1 == s0:
        return float(m1), False
    return float(m0 + (target - s0) * (m1 - m0) / (s1 - s0)), False


# ----------------------------------------------------------------------------
# checks
# ----------------------------------------------------------------------------
def check_gauge(results: dict, active_only: bool = True) -> dict:
    """Check 1: max over tasks of the gauge residuals returned by run_task."""
    gd = results["gauge_delta"]   # (T, B)
    gu = results["gauge_uc"]
    return {"max_delta_resid": np.nanmax(gd, axis=0), "max_uc_resid": np.nanmax(gu, axis=0)}


def _half_means(x: np.ndarray, T_burn: int, T_meas: int):
    """Mean and s.e. of a (T, B) per-task scalar over the two measurement halves."""
    mid = T_burn + T_meas // 2
    out = []
    for lo, hi in ((T_burn, mid), (mid, T_burn + T_meas)):
        seg = x[lo:hi]
        m = seg.mean(0)
        se = seg.std(0, ddof=1) / np.sqrt(max(seg.shape[0], 1)) if seg.shape[0] > 1 else np.full_like(m, np.nan)
        out.append((m, se))
    return out


def check_bounded(results: dict, T_burn: int, T_meas: int, n_se: float = 3.0) -> dict:
    """Check 2: compare first and second halves of the measurement window.

    Returns per-run flags (True = halves differ by > n_se standard errors) for
    the per-task scalars and for S(0) and S(m) at the last age.
    """
    flags = {}
    for name in ("Q_mean", "a2_mean", "aQ_mean", "aQ_max", "E0"):
        (m0, se0), (m1, se1) = _half_means(results[name], T_burn, T_meas)
        with np.errstate(invalid="ignore", divide="ignore"):
            z = np.abs(m1 - m0) / np.sqrt(se0**2 + se1**2)
        flags[name] = {"z": z, "flag": z > n_se, "half0": m0, "half1": m1}
    mom = {f: results[f] for f in Moments.FIELDS}
    c0, c1 = curves(mom, (0,)), curves(mom, (1,))
    for label, idx in (("S0", 0), ("Sm", -1)):
        with np.errstate(invalid="ignore", divide="ignore"):
            se0 = c0["cv_s"][:, idx] * np.abs(c0["S"][:, idx]) / np.sqrt(c0["n"][:, idx])
            se1 = c1["cv_s"][:, idx] * np.abs(c1["S"][:, idx]) / np.sqrt(c1["n"][:, idx])
            z = np.abs(c1["S"][:, idx] - c0["S"][:, idx]) / np.sqrt(se0**2 + se1**2)
        flags[label] = {"z": z, "flag": z > n_se, "half0": c0["S"][:, idx], "half1": c1["S"][:, idx]}
    return flags


def cv_exponent(Ns, cvs):
    """Check 3: fit CV ∝ N^γ; returns (γ, intercept).  Expect γ ≈ −1/2 (frozen readout)."""
    x, y = np.log(np.asarray(Ns, float)), np.log(np.asarray(cvs, float))
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2:
        return np.nan, np.nan
    gamma, c = np.polyfit(x[ok], y[ok], 1)
    return float(gamma), float(c)
