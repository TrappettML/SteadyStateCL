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
    """
    Static probe-age schedule: ages = [0..20] ∪ geomspace(24, m_max, n_geom) as unique ints.

    Parameters
    ----------
    N : int
    kappa : float
    T : int
        Number of tasks; m_max = m_max_factor·10N/κ² is capped at T − 1.
    n_geom : int
        Number of geometrically spaced ages beyond 24.
    m_max_factor : float

    Returns
    -------
    (n_ages,) int32 array, increasing, starting at 0.
    """
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
    """
    s_k and ε_k of the probes k = t − ages (one (n_ages, N) @ (N, D) matmul).

    Parameters
    ----------
    state : State
        After the current task's probe has been stored.
    p : Params
    ages : (n_ages,) int32 array

    Returns
    -------
    (s, eps, valid)
        ``s = u_k·w_k/|w_k|²``, ``eps = |u_k − w_k|²/|w_k|²`` and a bool mask;
        invalid ages (k < 0) read probe 0 and must be masked out by the caller.
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
    """Running sums Σs, Σs², Σε, Σε², count per (run, age, half); nothing per-sample is kept."""

    FIELDS = ("sum_s", "sum_s2", "sum_e", "sum_e2", "count")

    def __init__(self, B: int, n_ages: int):
        """
        Allocate zero sums.

        Parameters
        ----------
        B : int
            Batch size.
        n_ages : int
        """
        self.arr = np.zeros((5, B, n_ages, 2))

    def add(self, s: np.ndarray, eps: np.ndarray, valid: np.ndarray, half: int) -> None:
        """
        Accumulate one task's probe values into one measurement half.

        Parameters
        ----------
        s, eps : (B, n_ages) arrays
        valid : (B, n_ages) bool array
        half : {0, 1}
        """
        v = valid.astype(np.float64)
        self.arr[0, :, :, half] += np.where(valid, s, 0.0)
        self.arr[1, :, :, half] += np.where(valid, s * s, 0.0)
        self.arr[2, :, :, half] += np.where(valid, eps, 0.0)
        self.arr[3, :, :, half] += np.where(valid, eps * eps, 0.0)
        self.arr[4, :, :, half] += v

    def as_dict(self) -> dict:
        """The five sums as a dict keyed by ``FIELDS``, each (B, n_ages, 2)."""
        return {f: self.arr[i] for i, f in enumerate(self.FIELDS)}


def curves(mom: dict, halves=(0, 1)):
    """
    Per-run means and coefficients of variation from running moments.

    Parameters
    ----------
    mom : dict
        ``sum_s``, ``sum_s2``, ``sum_e``, ``sum_e2``, ``count``, each (B, n_ages, 2).
    halves : tuple
        Measurement halves to pool.

    Returns
    -------
    dict
        ``S``, ``E`` (means), ``cv_s``, ``cv_e`` and ``n`` (counts), each (B, n_ages);
        NaN where the count is zero.
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
    """
    First age with S(m) ≤ S(0)/2, linearly interpolated between bracketing ages.

    Parameters
    ----------
    ages : (n_ages,) array
    S : (n_ages,) array

    Returns
    -------
    (m_half, censored)
        ``(NaN, True)`` when S never reaches half of S(0) or S(0) is not finite.
    """
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
def check_gauge(results: dict) -> dict:
    """
    Check 1: the largest gauge residuals over tasks, per run.

    Parameters
    ----------
    results : dict
        ``run_stream`` output with ``gauge_delta`` and ``gauge_uc`` as (T, B).

    Returns
    -------
    dict
        ``max_delta_resid``, ``max_uc_resid`` as (B,) arrays.
    """
    gd = results["gauge_delta"]   # (T, B)
    gu = results["gauge_uc"]
    return {"max_delta_resid": np.nanmax(gd, axis=0), "max_uc_resid": np.nanmax(gu, axis=0)}


def _half_means(x: np.ndarray, T_burn: int, T_meas: int):
    """
    Mean and standard error of a (T, B) per-task scalar over the two measurement halves.

    Parameters
    ----------
    x : (T, B) array
    T_burn, T_meas : int

    Returns
    -------
    list of two (mean (B,), se (B,)) pairs
    """
    mid = T_burn + T_meas // 2
    out = []
    for lo, hi in ((T_burn, mid), (mid, T_burn + T_meas)):
        seg = x[lo:hi]
        m = seg.mean(0)
        se = seg.std(0, ddof=1) / np.sqrt(max(seg.shape[0], 1)) if seg.shape[0] > 1 else np.full_like(m, np.nan)
        out.append((m, se))
    return out


def check_bounded(results: dict, T_burn: int, T_meas: int, n_se: float = 3.0, min_count: int = 10) -> dict:
    """
    Check 2: compare the first and second halves of the measurement window.

    Parameters
    ----------
    results : dict
        ``run_stream`` output.
    T_burn, T_meas : int
    n_se : float
        Flag threshold in standard errors (tasks are treated as independent,
        so the flags are conservative for slowly drifting norms).
    min_count : int
        S(m) is compared at the largest age both halves sample at least this
        often (the last ages are sampled by very few tasks).

    Returns
    -------
    dict
        Per key (``Q_mean``, ``a2_mean``, ``aQ_mean``, ``aQ_max``, ``E0``, ``S0``,
        ``Sm``) a dict with ``z``, ``flag``, ``half0``, ``half1`` as (B,) arrays
        (``Sm`` also ``age_index``).
    """
    flags = {}
    for name in ("Q_mean", "a2_mean", "aQ_mean", "aQ_max", "E0"):
        (m0, se0), (m1, se1) = _half_means(results[name], T_burn, T_meas)
        with np.errstate(invalid="ignore", divide="ignore"):
            z = np.abs(m1 - m0) / np.sqrt(se0**2 + se1**2)
        flags[name] = {"z": z, "flag": z > n_se, "half0": m0, "half1": m1}
    mom = {f: results[f] for f in Moments.FIELDS}
    c0, c1 = curves(mom, (0,)), curves(mom, (1,))
    # the last ages are sampled by only a few tasks (m_max = T−1 by exactly one), so compare S(m)
    # at the largest age that both halves sample at least min_count times
    both = np.minimum(c0["n"], c1["n"]).min(0)
    ok = np.where(both >= min_count)[0]
    idx_m = int(ok[-1]) if len(ok) else 0
    for label, idx in (("S0", 0), ("Sm", idx_m)):
        with np.errstate(invalid="ignore", divide="ignore"):
            se0 = c0["cv_s"][:, idx] * np.abs(c0["S"][:, idx]) / np.sqrt(c0["n"][:, idx])
            se1 = c1["cv_s"][:, idx] * np.abs(c1["S"][:, idx]) / np.sqrt(c1["n"][:, idx])
            z = np.abs(c1["S"][:, idx] - c0["S"][:, idx]) / np.sqrt(se0**2 + se1**2)
        flags[label] = {"z": z, "flag": z > n_se, "half0": c0["S"][:, idx], "half1": c1["S"][:, idx], "age_index": idx}
    return flags


def cv_exponent(Ns, cvs):
    """
    Check 3: fit CV ∝ N^γ by least squares in log–log.

    Parameters
    ----------
    Ns : sequence of int
    cvs : sequence of float

    Returns
    -------
    (gamma, intercept)
        NaN when fewer than two finite points; expect γ ≈ −1/2 for the gated frozen readout.
    """
    x, y = np.log(np.asarray(Ns, float)), np.log(np.asarray(cvs, float))
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2:
        return np.nan, np.nan
    gamma, c = np.polyfit(x[ok], y[ok], 1)
    return float(gamma), float(c)
