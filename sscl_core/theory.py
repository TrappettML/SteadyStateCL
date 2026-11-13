"""Closed forms of the write-up (``muP_experiment.tex``), NumPy only: nothing here reads a run.

Every function is one equation of the write-up, named by its label there, so that the code that
checks or draws a prediction and the text that derives it can be compared line by line:

=====================  ==========================================================================
``theta``              θ = 1 − e^{−λ}, the fraction of the residual one task removes through the hidden layer
``nu``                 ν = 1 − (1 − p)^N, the probability that a task has an active unit
``abar_ss``            eq. (abarss): steady-state key norm ā_act of a projected readout
``steady_norms``       eq. (Qss)/(abarss): (ā, Q̄) of a frozen or a projected readout
``key_drift``          eq. (delta_est): δ of the learned shared readout, from ``abar_ss``
``sigma_heads``        eq. (S_heads)
``sigma_frozen``       eq. (S_frozen)
``sigma_learned``      eq. (S_learned) with δ of ``key_drift`` (no free parameter, no measured input)
``sigma`` / ``mhalf``  the curve and its half-life for a readout name of ``config.READOUTS``
``lam_realized``       eq. (lamreal): λ_c under clock (i)
``lam_a_bounds``       eq. (lama): bounds on the readout exponent λᵃ_c = Λ‖M_c‖
``lam_a_init``         eq. (lama0): its value at initialisation
``chan_diff``          eq. (chandiff): ā_c − tr M_c of a projected readout
``popstab``            eq. (popstab): left side of the sufficient contraction condition (< 2)
``lam_eff``            eq. (lameff): exponent of n forward steps
``burn_in_halflives``  the burn-in in units of the longest half-life (Protocol: 7θν/ln 2 at T_burn = 7N)
``syn_theta_bar``      eq. (syn_theta): mean removed fraction under synaptic gating, clock (ii)
``syn_rho``            eq. (syn_theta): mean residual left by a task, frozen readout
``sigma_syn_heads``    eq. (syn_heads)
=====================  ==========================================================================

Conventions: ``p`` is the gate density κ/√N (1 when ungated), ``pN`` the nominal active count,
``lam`` the hidden-layer learning per task λ, ``delta0`` the dimensionless imbalance δ₀ = a_i² − Q_ii,
``sigma0`` the initial scale (Q_ii⁽⁰⁾ ≈ σ₀²).  ``m`` may be an array of ages.
"""
from __future__ import annotations

import math

import numpy as np

LN2 = math.log(2.0)

#: Readout name → (kind of curve, gated).  Kinds: "learned" (S_learned), "frozen" (S_frozen), "heads" (S_heads).
READOUT_KIND = {
    "gated_shared": ("learned", True),
    "learned_shared": ("learned", False),
    "gated_frozen": ("frozen", True),
    "frozen": ("frozen", False),
    "task_heads": ("heads", True),
}


# ----------------------------------------------------------------------------
# elementary quantities
# ----------------------------------------------------------------------------
def theta(lam: float) -> float:
    """θ = 1 − e^{−λ}: the fraction of its residual a task removes through the hidden layer (eq. lam)."""
    return -math.expm1(-float(lam))


def density(N: int, kappa: float, gated: bool = True) -> float:
    """Gate density p = κ/√N (at most 1), or 1 when ungated."""
    return min(1.0, float(kappa) / math.sqrt(N)) if gated else 1.0


def nu(N: int, p: float) -> float:
    """ν = 1 − (1 − p)^N: the probability that a task has at least one active unit."""
    return 1.0 - (1.0 - p) ** N


def abar_nom(delta0: float) -> float:
    """ā_nom = 1 + δ₀: a_i² = δ₀ + Q_ii at Q_ii = 1 (eq. lam)."""
    return 1.0 + float(delta0)


# ----------------------------------------------------------------------------
# steady-state norms, eqs. (Qss) and (abarss)
# ----------------------------------------------------------------------------
def abar_ss(delta0: float, pN: float, lam: float, sigma0: float = 1.0) -> float:
    """Steady-state key norm of the active units of a projected readout, eq. (abarss):

        ā_act = ½ [δ₀ + σ₀² + √((δ₀ + σ₀²)² + 4θ pN/(2 − θ))],

    the positive root of the projected form of eq. (Qss), ā(ā − δ₀ − σ₀²) = θ pN/(2 − θ).  A leading-order
    estimate (it inherits the approximations of eq. Qss); Q̄_act = ā_act − δ₀.
    """
    th = theta(lam)
    b = float(delta0) + float(sigma0) ** 2
    return 0.5 * (b + math.sqrt(b * b + 4.0 * th / (2.0 - th) * float(pN)))


def steady_norms(projected: bool, delta0: float, pN: float, lam: float, sigma0: float = 1.0) -> tuple[float, float]:
    """(ā, Q̄) at steady state, eq. (Qss).

    Projected readouts (a_i² = δ₀ + Q_ii pinned): ā = ``abar_ss``, Q̄ = ā − δ₀, growing like √pN.
    Frozen readouts (a = a⁽⁰⁾): ā = δ₀ + σ₀², Q̄ = σ₀² + θ/(2 − θ) · pN/ā, growing like pN.
    """
    if projected:
        a = abar_ss(delta0, pN, lam, sigma0)
        return a, a - float(delta0)
    th = theta(lam)
    a = float(delta0) + float(sigma0) ** 2
    return a, float(sigma0) ** 2 + th / (2.0 - th) * float(pN) / a


def key_drift(N: int, p: float, lam: float, delta0: float, sigma0: float = 1.0) -> float:
    """Key drift δ of the learned shared readout, eq. (delta_est) with the norms of eq. (abarss):

        δ ≈ θ (1 − p)/(pN) · Q̄_act/ā_act .

    The relative loss of a unit's stored content per task it is active in; 0 at p = 1.
    """
    pN = p * N
    a, Q = steady_norms(True, delta0, pN, lam, sigma0)
    return theta(lam) * (1.0 - p) / pN * Q / a


# ----------------------------------------------------------------------------
# mean normalized signal Σ(m), eqs. (S_heads), (S_frozen), (S_learned)
# ----------------------------------------------------------------------------
def _perp_rate(N: int, p: float) -> float:
    """(ν − p)/(N − 1): the eigenvalue of K on content orthogonal to the sign pattern (eq. Kfrozen); 0 at N = 1."""
    return (nu(N, p) - p) / (N - 1) if N > 1 else 0.0


def sigma_heads(m, N: int, p: float, lam: float, theta_: float | None = None):
    """Σ_heads(m) = (1 − θν/N)^m, eq. (S_heads).  ``theta_`` overrides θ (synaptic gating, eq. syn_heads)."""
    th = theta(lam) if theta_ is None else theta_
    return (1.0 - th * nu(N, p) / N) ** np.asarray(m, float)


def _two_rates(m, N: int, p: float, rho_par: float, rho_perp: float):
    """(1 − p/ν) ρ_⊥^m + (p/ν) ρ_∥^m, the common form of eqs. (S_frozen) and (S_learned)."""
    m = np.asarray(m, float)
    w = p / nu(N, p)
    return (1.0 - w) * rho_perp ** m + w * rho_par ** m


def sigma_frozen(m, N: int, p: float, lam: float):
    """Σ_frozen(m), eq. (S_frozen): ρ_∥ = 1 − θp, ρ_⊥ = 1 − θ(ν − p)/(N − 1).  Ungated (p = ν = 1): (1 − θ)^m."""
    th = theta(lam)
    return _two_rates(m, N, p, 1.0 - th * p, 1.0 - th * _perp_rate(N, p))


def sigma_learned(m, N: int, p: float, lam: float, delta0: float, sigma0: float = 1.0, delta: float | None = None):
    """Σ_learned(m), eq. (S_learned): ρ′_∥ = 1 − δp − θ(1 − δ)p, ρ′_⊥ = 1 − δp − θ(1 − δ)(ν − p)/(N − 1).

    ``delta`` defaults to ``key_drift`` (eq. delta_est from eq. abarss), so the curve has no free parameter
    and no measured input; pass a value to draw the curve for another δ.
    """
    th = theta(lam)
    d = key_drift(N, p, lam, delta0, sigma0) if delta is None else float(delta)
    return _two_rates(m, N, p, 1.0 - d * p - th * (1.0 - d) * p, 1.0 - d * p - th * (1.0 - d) * _perp_rate(N, p))


def sigma(readout: str, m, N: int, kappa: float, lam: float, delta0: float = 0.0, sigma0: float = 1.0):
    """Σ(m) of a readout of ``config.READOUTS`` under unit gating: the equation the Summary of the half-life
    section says to plot, with p = κ/√N (1 for the ungated readouts, whose κ is ignored)."""
    kind, gated = READOUT_KIND[readout]
    p = density(N, kappa, gated)
    if kind == "heads":
        return sigma_heads(m, N, p, lam)
    if kind == "frozen":
        return sigma_frozen(m, N, p, lam)
    return sigma_learned(m, N, p, lam, delta0, sigma0)


def first_half_crossing(curve) -> float:
    """First index at which ``curve`` falls to ``curve[0]/2``, linearly interpolated between the bracketing
    indices (the m½ of the Observables section); NaN when it never does."""
    c = np.asarray(curve, float)
    half = c[0] / 2.0
    below = np.nonzero(c <= half)[0]
    if len(below) == 0:
        return float("nan")
    j = int(below[0])
    if j == 0:
        return 0.0
    return j - 1 + (c[j - 1] - half) / (c[j - 1] - c[j])


def mhalf_of(curve_fn, m0: int = 64, m_cap: int = 1 << 26) -> float:
    """Half-life of a decreasing Σ(m) given as a function of an age array: the first integer-age crossing of ½,
    linearly interpolated exactly as the measured one (``first_half_crossing``).  The age range doubles until
    the curve has crossed; NaN beyond ``m_cap`` ages."""
    M = m0
    while M <= m_cap:
        h = first_half_crossing(curve_fn(np.arange(M + 1)))
        if np.isfinite(h):
            return float(h)
        M *= 2
    return float("nan")


def mhalf(readout: str, N: int, kappa: float, lam: float, delta0: float = 0.0, sigma0: float = 1.0) -> float:
    """m½ of ``sigma``: the root of Σ(m) = ½ on integer ages with linear interpolation.

    Heads: ≈ N ln 2/(θν) + ln 2/2 (eq. S_heads).  Gated frozen: eq. (mhalf_frozen), e.g. 65 and 59 tasks at
    N = 100, κ = 1, 2, λ = 4.  Ungated: Σ = (1 − θ)^m, interpolated between m = 0 and 1 to 1/(2θ).
    """
    return mhalf_of(lambda m: sigma(readout, m, N, kappa, lam, delta0, sigma0))


def mhalf_heads_exact(N: int, p: float, lam: float) -> float:
    """ln 2 / (−ln(1 − θν/N)), the continuous root of eq. (S_heads)."""
    return LN2 / -math.log1p(-theta(lam) * nu(N, p) / N)


# ----------------------------------------------------------------------------
# learning per task: hidden channel (λ_c) and readout channel (λᵃ_c)
# ----------------------------------------------------------------------------
def lam_realized(lam: float, delta0: float, n_c: float, pN: float, Qbar_c: float) -> float:
    """λ_c = λ (n_c/pN)(δ₀ + Q̄_c)/(1 + δ₀), eq. (lamreal): the hidden-layer exponent of a task under clock (i)
    (``Qbar_c`` the mean Q_ii over the active units; its initial value for a frozen readout)."""
    return float(lam) * (n_c / pN) * (delta0 + Qbar_c) / (1.0 + delta0)


def lam_a_bounds(lam: float, delta0: float, n_c: float, pN: float, Qbar_c: float) -> tuple[float, float]:
    """(lower, upper) of the readout exponent λᵃ_c = Λ‖M_c‖ at Λ = λ/(1 + δ₀), eq. (lama):

        λ/(1 + δ₀) · Q̄_c/pN  ≤  λᵃ_c  ≤  λ/(1 + δ₀) · (n_c/pN) Q̄_c ,

    from Q̄_c/pN ≤ ‖M_c‖ ≤ tr M_c (eq. Mbound).  For another task duration scale both by Λ_c (1 + δ₀)/λ.
    """
    lo = float(lam) / (1.0 + delta0) * Qbar_c / pN
    return lo, lo * n_c


def lam_a_init(lam: float, delta0: float, n_c: float, pN: float, D: int) -> float:
    """λᵃ_c(0) = λ (1 + √(n_c/D))² / ((1 + δ₀) pN), eq. (lama0): the readout exponent at initialisation, from
    the largest eigenvalue of the covariance of n_c standard normal rows in D dimensions (N → ∞)."""
    return float(lam) * (1.0 + math.sqrt(n_c / D)) ** 2 / ((1.0 + delta0) * pN)


def chan_diff(delta0: float, n_c: float, pN: float) -> float:
    """ā_c − tr M_c = (n_c/pN) δ₀ for the projected readouts, eq. (chandiff)."""
    return n_c / pN * delta0


def popstab(h_eta: float, delta0: float, n_c: float, pN: float, Qbar_c: float) -> float:
    """Left side of eq. (popstab), hη (n_c/pN)(δ₀ + 2Q̄_c): the population step contracts the residual in every
    direction whenever this is < 2 (sufficient, projected readouts; it bounds hη(ā_c + ‖M_c‖))."""
    return h_eta * (n_c / pN) * (delta0 + 2.0 * Qbar_c)


def popstab_steady(h: float, eta: float, delta0: float, pN: float, lam: float, sigma0: float = 1.0,
                   projected: bool = True, realized: bool = False, n_steps: int = 1) -> float:
    """``popstab`` of a run before it is run: n_c at its mean pN and Q̄_c the larger of its initial value σ₀² and
    the steady-state estimate of eq. (abarss).

    ``realized`` evaluates it for clock (ii), where hη = λ/(n ā_c) with ā_c = δ₀ + Q̄_c (eq. clock_real), so
    the bound is (λ/n)(δ₀ + 2Q̄)/(δ₀ + Q̄).  A frozen readout (``projected`` False) has the hidden channel
    alone: hη ā with ā = δ₀ + σ₀² (λ/n under clock (ii)).
    """
    s2 = float(sigma0) ** 2
    if not projected:
        return float(lam) / n_steps if realized else h * eta * (delta0 + s2)
    Q = max(s2, abar_ss(delta0, pN, lam, sigma0) - delta0)
    h_eta = float(lam) / (n_steps * (delta0 + Q)) if realized else h * eta
    return popstab(h_eta, delta0, pN, pN, Q)


def lam_eff(lam_c: float, n: int) -> float:
    """λ_eff = −n ln(1 − λ_c/n), eq. (lameff): the exponent n forward steps of size λ_c/n reach (frozen readout);
    inf when λ_c ≥ n."""
    x = float(lam_c) / n
    return -n * math.log1p(-x) if x < 1.0 else float("inf")


def burn_in_halflives(T_burn: int, N: int, p: float, lam: float) -> float:
    """T_burn in units of the longest half-life of any readout, that of the task heads, N ln 2/(θν)
    (Protocol): T_burn θν/(N ln 2), i.e. 7θν/ln 2 ≈ 10 at T_burn = 7N."""
    return T_burn * theta(lam) * nu(N, p) / (N * LN2)


# ----------------------------------------------------------------------------
# synaptic gating, eqs. (syn_theta), (syn_heads)
# ----------------------------------------------------------------------------
def syn_theta_bar(lam: float, N: int, p: float) -> float:
    """θ̄ = E[1 − e^{−λ n/pN}] = 1 − (1 − p(1 − e^{−λ/pN}))^N, n ~ Binomial(N, p), eq. (syn_theta): the mean
    fraction a task removes under synaptic gating with clock (ii) at the mean rate."""
    return 1.0 - (1.0 - p * -math.expm1(-float(lam) / (p * N))) ** N


def syn_rho(lam: float, N: int, p: float) -> float:
    """E e^{−2λ n/pN} = (1 − p(1 − e^{−2λ/pN}))^N ≥ e^{−2λ}, eq. (syn_theta): the mean relative residual a task of
    the frozen readout leaves under synaptic gating (initial residual spread evenly over the coordinates)."""
    return (1.0 - p * -math.expm1(-2.0 * float(lam) / (p * N))) ** N


def sigma_syn_heads(m, N: int, p: float, lam: float):
    """Σ_heads under synaptic gating with clock (ii), eq. (syn_heads): (1 − θ̄/N)^m."""
    return (1.0 - syn_theta_bar(lam, N, p) / N) ** np.asarray(m, float)
