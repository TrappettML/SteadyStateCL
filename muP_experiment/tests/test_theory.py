"""``mup.theory`` against the numbers the write-up quotes, ``mup.halflife``, and the task clock / readout exponent
of the kernels against the write-up's identities ("Step sizes and learning per task")."""
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from conftest import one_batch, params
from mup import halflife as H
from mup import model
from mup import theory as T
from mup.config import Config, make_runs, short_burn_in, unstable_runs
from mup.protocol import run_stream


# ----------------------------------------------------------------------------
# closed forms: the numbers in the text
# ----------------------------------------------------------------------------
def test_half_lives_quoted_in_the_write_up():
    # eq. (mhalf_frozen): "65 and 59 tasks at N = 100, κ = 1, 2"
    assert round(T.mhalf("gated_frozen", 100, 1, 4)) == 66 and abs(T.mhalf("gated_frozen", 100, 1, 4) - 65.6) < 0.1
    assert round(T.mhalf("gated_frozen", 100, 2, 4)) == 59
    # the first-check table of the synaptic section: columns (S_frozen) and (S_heads)
    for (N, k), fro, heads in (((50, 1), 31.3, 35.0), ((50, 2), 24.9, 35.0), ((100, 1), 65.6, 70.3)):
        assert abs(T.mhalf("gated_frozen", N, k, 4) - fro) < 0.06
        assert abs(T.mhalf("task_heads", N, k, 4) - heads) < 0.06
    # eq. (S_heads): m½ = N ln 2/(θν) + ln 2/2 + O(1/N), the continuous root
    p = 0.1
    assert abs(T.mhalf_heads_exact(100, p, 4) - (100 * T.LN2 / (T.theta(4) * T.nu(100, p)) - T.LN2 / 2)) < 0.01
    # ungated: Σ = (1 − θ)^m, interpolated between m = 0 and 1 to 1/(2θ)
    for ro in ("frozen", "learned_shared"):
        assert abs(T.mhalf(ro, 100, 1, 4) - 1 / (2 * T.theta(4))) < 1e-12


def test_learned_curve_limits():
    N, k, lam = 400, 1.0, 4.0
    p = T.density(N, k)
    # δ₀ = 0: Q̄/ā = 1, δp = θ(1 − p)/N, about half the frozen half-life
    assert abs(T.key_drift(N, p, lam, 0.0) * p - T.theta(lam) * (1 - p) / N) < 1e-15
    ratio = T.mhalf("gated_shared", N, k, lam, 0.0) / T.mhalf("gated_frozen", N, k, lam)
    assert 0.45 < ratio < 0.6
    # large δ₀: the learned readout approaches the frozen one
    assert abs(T.mhalf("gated_shared", N, k, lam, 1e4) / T.mhalf("gated_frozen", N, k, lam) - 1) < 0.01
    # δ = 0 is the frozen curve exactly
    m = np.arange(50)
    np.testing.assert_allclose(T.sigma_learned(m, N, p, lam, 0.0, delta=0.0), T.sigma_frozen(m, N, p, lam), rtol=1e-14)


def test_steady_norms_solve_the_norm_equation():
    for d0, pN, lam, s0 in ((0.0, 10.0, 4.0, 1.0), (-0.75, 20.0, 1.0, 1.0), (16.0, 10.0, 4.0, 1.0), (1.0, 31.6, 0.5, 2.0)):
        th = T.theta(lam)
        a, Q = T.steady_norms(True, d0, pN, lam, s0)
        assert a > 0 and abs(Q - (a - d0)) < 1e-14
        assert abs(a * (a - d0 - s0 ** 2) - th / (2 - th) * pN) < 1e-10          # eq. (Qss), projected form
        af, Qf = T.steady_norms(False, max(d0, 0.0), pN, lam, s0)
        assert abs(Qf - (s0 ** 2 + th / (2 - th) * pN / af)) < 1e-14             # eq. (Qss), frozen
    # "Lazy and rich regimes": ā/(ā − δ₀) ≈ 11 at δ₀ = 16 and ≈ 0.8 at δ₀ = −0.75 (N = 100, κ = 1)
    a = T.abar_ss(16.0, 10.0, 4.0)
    assert abs(a / (a - 16.0) - 11.3) < 0.1
    a = T.abar_ss(-0.75, 10.0, 4.0)
    assert abs(a / (a + 0.75) - 0.81) < 0.01


def test_step_and_burn_in_numbers():
    assert abs(T.lam_eff(4.0, 40) / 4.0 - 1.054) < 1e-3                          # "at x = 0.1, λ_eff/λ_c = 1.054"
    assert abs(7 * T.theta(4.0) / T.LN2 - 9.91) < 0.01                           # Protocol: 7θν/ln 2 ≈ 10
    assert abs(T.burn_in_halflives(700, 100, 0.1, 4.0) - 9.91) < 0.01
    assert abs(math.exp(-4.0) / T.theta(4.0) - 0.019) < 5e-4                     # "1.9 % at λ = 4"
    # eq. (lama0) lies between the bounds of eq. (lama) at Q̄ = 1 for n_c ≥ 1 … well inside D
    lo, hi = T.lam_a_bounds(4.0, 0.0, 10, 10.0, 1.0)
    assert lo <= T.lam_a_init(4.0, 0.0, 10, 10.0, 400) <= hi and abs(hi - 4.0) < 1e-12 and abs(lo - 0.4) < 1e-12


def test_synaptic_numbers():
    assert abs(T.syn_theta_bar(4.0, 100, 0.1) - 0.965) < 5e-4                    # "θ̄ = 0.965 against θ = 0.982"
    assert abs(T.syn_rho(4.0, 100, 0.1) - 3.5e-3) < 5e-5                         # "ρ_c = 3.5 × 10⁻³"
    assert abs(T.syn_theta_bar(4.0, 100, 0.2) - 0.975) < 5e-4 and abs(T.syn_rho(4.0, 100, 0.2) - 1.1e-3) < 5e-5
    assert T.syn_rho(4.0, 100, 0.1) >= math.exp(-8.0)
    assert abs(T.theta(4.0) * T.nu(100, 0.1) / T.syn_theta_bar(4.0, 100, 0.1) - 1.017) < 5e-4


# ----------------------------------------------------------------------------
# half-life statistics
# ----------------------------------------------------------------------------
def test_mean_curve_half_life_and_jackknife():
    rng = np.random.default_rng(0)
    m = np.arange(200)[:, None]
    S = 0.9 * np.exp(-m / 40.0) * (1 + 0.1 * rng.standard_normal((1, 6))) + 0.05 * rng.standard_normal((200, 6))
    v = H.values(S, "mean")
    mh = H.half_life(S.mean(1))
    assert abs(np.mean(v) - mh) < 1e-9                                          # centred on the mean curve's m½
    n = S.shape[1]
    loo = np.array([H.half_life(np.delete(S, j, 1).mean(1)) for j in range(n)])
    jk = math.sqrt((n - 1) / n * ((loo - loo.mean()) ** 2).sum())
    assert abs(H.stat(v)[1] - jk) < 1e-9                                        # its s.e. is the jackknife s.e.
    np.testing.assert_allclose(H.mean_curve(S), S.mean(1) / S.mean(1)[0])
    # noisy per-seed curves cross ½ earlier on average: the two statistics differ, the write-up's is the larger
    assert H.stat(H.values(S, "seed"))[0] < mh
    # an infeasible seed (NaN column) is left out of both
    S2 = np.concatenate([S, np.full((200, 1), np.nan)], 1)
    assert np.isnan(H.values(S2, "mean")[-1]) and abs(np.nanmean(H.values(S2, "mean")) - mh) < 1e-9
    # exact curve: the statistic reproduces the equation's half-life
    th = T.sigma("task_heads", np.arange(400), 100, 1.0, 4.0)
    assert abs(H.stat(H.values(np.stack([th, th], 1), "mean"))[0] - T.mhalf("task_heads", 100, 1.0, 4.0)) < 1e-9


# ----------------------------------------------------------------------------
# task clock, eqs. (clock_nom), (clock_real)
# ----------------------------------------------------------------------------
def _task(readout, clock, N=12, D=48, pN=4.0, lam=2.0, n_steps=20, delta0=0.5, seed=3):
    p = params(readout=readout, delta0=delta0, eta=1.0, h=lam / ((1.0 + delta0) * n_steps), pN=pN, clock=clock, lam=lam)
    key = jax.random.PRNGKey(seed)
    st = model.init_state(key, p, N, D, 4)
    w, g = model.probe_task(key, 0, N, D, p.pN, p.gated)
    return st, p, w, g, n_steps


@pytest.mark.parametrize("readout", ["gated_frozen", "frozen"])
def test_realized_clock_removes_exactly_lambda(readout):
    """Clock (ii), frozen readout: every task multiplies its residual by (1 − λ/n)^n = e^{−λ_eff}, whatever the gate."""
    st, p, w, g, n = _task(readout, 1.0)
    assert g.sum() > 0
    r0 = (g * st.a) @ st.W / p.pN - w
    out = model.train_block(st.W, st.a, g, w, p, jax.random.PRNGKey(0), n_steps=n, solver="pop", K=st.W.shape[0])
    r1 = (g * out["a"]) @ out["W"] / p.pN - w
    lam = float(p.lam)
    np.testing.assert_allclose(np.asarray(r1), np.asarray(r0) * (1 - lam / n) ** n, rtol=1e-10, atol=1e-12)
    assert abs(math.exp(-T.lam_eff(lam, n)) - (1 - lam / n) ** n) < 1e-14
    assert abs(float(out["h"]) * float(model.key_norm(st.a, g, p)) - lam / n) < 1e-14       # hη ā²_c = λ/n
    # clock (i) uses the nominal step and misses λ by the factor ā²_c/ā²_nom of eq. (lamreal)
    st, p, w, g, n = _task(readout, 0.0)
    out = model.train_block(st.W, st.a, g, w, p, jax.random.PRNGKey(0), n_steps=n, solver="pop", K=st.W.shape[0])
    assert float(out["h"]) == float(p.h)
    r1 = (g * out["a"]) @ out["W"] / p.pN - w
    x = float(p.h) * float(model.key_norm(st.a, g, p))
    np.testing.assert_allclose(np.asarray(r1), np.asarray(r0) * (1 - x) ** n, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("readout", ["gated_shared", "task_heads", "learned_shared"])
def test_realized_clock_block_matches_step_loop(readout):
    """``train_block`` under clock (ii) is the loop of ``pop_step`` + ``project_rows`` at h = ``task_h``."""
    st, p, w, g, n = _task(readout, 1.0)
    a = model.draw_readout(jax.random.PRNGKey(9), model.Q_diag(st.W), p) if readout == "task_heads" else st.a
    out = model.train_block(st.W, a, g, w, p, jax.random.PRNGKey(0), n_steps=n, solver="pop", K=st.W.shape[0])
    ph = p._replace(h=model.task_h(a, g, p, n))
    s = st._replace(a=a, g=g, w=w)
    for _ in range(n):
        s, _ = model.project_rows(model.pop_step(s, ph), ph)
    np.testing.assert_allclose(np.asarray(out["W"]), np.asarray(s.W), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(np.asarray(out["a"]), np.asarray(s.a), rtol=1e-10, atol=1e-12)
    # the residual shrinks (eq. rdecay); how far below e^{−2λ} it ends depends on how ā²_c moves during the task
    r0 = (g * a) @ st.W / p.pN - w
    r1 = (g * out["a"]) @ out["W"] / p.pN - w
    assert float(r1 @ r1) < float(r0 @ r0)


def test_empty_gate_takes_no_step():
    st, p, w, g, n = _task("gated_shared", 1.0)
    z = jnp.zeros_like(g)
    assert float(model.task_h(st.a, z, p, n)) == 0.0
    out = model.train_block(st.W, st.a, z, w, p, jax.random.PRNGKey(0), n_steps=n, solver="pop", K=st.W.shape[0])
    assert np.array_equal(np.asarray(out["W"]), np.asarray(st.W)) and np.array_equal(np.asarray(out["a"]), np.asarray(st.a))


# ----------------------------------------------------------------------------
# readout exponent and stability, eqs. (Mspec), (Mbound), (lama), (chandiff), (popstab)
# ----------------------------------------------------------------------------
def test_active_qmax_is_the_norm_of_M():
    st, p, w, g, _ = _task("gated_shared", 1.0, N=14, D=40, pN=5.0)
    W, gn = np.asarray(st.W), np.asarray(g)
    D = W.shape[1]
    Wc = W[gn == 1.0]
    M = Wc.T @ Wc / (float(p.pN) * D)                                           # M_c of eq. (rflow), D × D
    for K in (W.shape[0], int(gn.sum()) + 2):
        q = float(model.active_qmax(st.W, g, K))
        assert abs(q / float(p.pN) - np.linalg.eigvalsh(M)[-1]) < 1e-12         # eq. (Mspec): μ_max = q_max/pN
    Qbar, n_c = float((gn * np.asarray(model.Q_diag(st.W))).sum() / gn.sum()), gn.sum()
    assert Qbar * (1 - 1e-12) <= q <= n_c * Qbar * (1 + 1e-12)                  # eq. (Mbound)
    assert np.isnan(float(model.active_qmax(st.W, g, int(gn.sum()) - 1)))       # more active units than the block


@pytest.mark.parametrize("clock", ["realized", "nominal"])
def test_recorded_exponents_obey_the_write_up(clock):
    cfg = Config(N=[16], kappa=[2.0], readout=["gated_shared", "task_heads", "learned_shared"], delta0=[0.5], lam=[2.0],
                 seeds=1, T_factor=0.25, T_min=4, clock=clock)
    runs = one_batch([r for r in make_runs(cfg) if r.gated == 1.0])
    res = run_stream(runs)
    r0 = runs[0]
    n_c, pN = res["n_active"], np.array([r.pN for r in runs])
    ok = n_c > 0
    assert ok.any()
    Lam = res["lam_c"] / np.where(ok, res["abar_c"], 1.0)                       # Λ_c = λ_c/ā²_c
    lo, hi = Lam * res["Qbar_c"] / pN, Lam * n_c * res["Qbar_c"] / pN
    assert np.all(res["lam_a"][ok] >= lo[ok] * (1 - 1e-9)) and np.all(res["lam_a"][ok] <= hi[ok] * (1 + 1e-9))   # eq. (lama)
    # eq. (chandiff): ā²_c − tr M_c = (n_c/pN) δ₀ for the projected readouts, tr M_c = (n_c/pN) Q̄_c
    np.testing.assert_allclose((res["abar_c"] - n_c / pN * res["Qbar_c"])[ok], (n_c / pN * r0.delta0)[ok], atol=1e-9)
    assert np.allclose(res["abar_c"][ok], (n_c / pN * res["abar2"])[ok])        # ā²_c = (n_c/pN) × the active-unit mean
    # eq. (popstab) bounds the largest step eigenvalue hη(ā²_c + ‖M_c‖) = (λ_c + λᵃ_c)/n
    assert np.all(((res["lam_c"] + res["lam_a"]) / r0.n_steps)[ok] <= res["popstab"][ok] * (1 + 1e-9))
    if clock == "realized":
        np.testing.assert_allclose(res["lam_c"][ok], r0.lam, rtol=1e-12)        # λ means what eq. (lam) says
        assert np.nanmax(res["popstab"]) < 2.0
    else:
        np.testing.assert_allclose(res["lam_c"][ok], (r0.alpha_task * r0.eta * res["abar_c"])[ok], rtol=1e-12)
        np.testing.assert_allclose(res["lam_c"][0][ok[0]],                      # eq. (lamreal), first task
                                   [T.lam_realized(r0.lam, r0.delta0, n, p_, Q) for n, p_, Q in
                                    zip(n_c[0][ok[0]], pN[ok[0]], res["Qbar_c"][0][ok[0]])], rtol=1e-9)


# ----------------------------------------------------------------------------
# config: clock, contraction condition, burn-in, initialization
# ----------------------------------------------------------------------------
def test_clock_defaults_and_checks(capsys):
    from mup.config import Run
    assert Config().clock == "realized"
    runs = make_runs(Config(N=[10], readout=["gated_shared"], seeds=1))
    assert runs[0].clock == "realized" and runs[0].clock_id == 1.0 and "clock" not in runs[0].label()
    d = runs[0].to_dict()
    assert Run.from_dict(d).clock == "realized"
    d.pop("clock")
    assert Run.from_dict(d).clock == "nominal"                                  # batch files written before the clock ran (i)
    assert make_runs(Config(N=[10], readout=["gated_frozen"], seeds=1, solver="sgd", eta=[0.05]))[0].clock == "nominal"
    with pytest.raises(ValueError):
        make_runs(Config(clock="stop"))
    capsys.readouterr()
    stage1 = Config.from_yaml("configs/stage1.yaml")
    real = make_runs(stage1)
    assert not unstable_runs(real)                                              # eq. (popstab) holds under clock (ii)
    assert "popstab" not in capsys.readouterr().out
    nom = make_runs(Config(**{**stage1.__dict__, "clock": "nominal"}))
    bad = unstable_runs(nom)
    assert bad and all(r.do_proj == 1.0 for r in bad) and min(r.delta0 for r in bad) == -0.75
    assert "popstab" in capsys.readouterr().out
    assert not short_burn_in(real, 9.0) and short_burn_in(real, 9.95)           # 7θν/ln 2 = 9.9ν at λ = 4
    with pytest.raises(ValueError):
        make_runs(Config(delta0=[-3.0], sigma0=[1.5]))                          # δ₀ ≤ −σ₀²
    assert make_runs(Config(N=[10], readout=["frozen"], delta0=[-1.5], sigma0=[1.5], seeds=1))


def test_readout_guard():
    p = params(delta0=-0.5, eta=1.0)
    Q = jnp.asarray([1.0, 0.5 - 1e-15, 0.5 + 1e-15, 0.4])
    a = np.asarray(model.draw_readout(jax.random.PRNGKey(0), Q, p))
    assert abs(abs(a[0]) - math.sqrt(0.5)) < 1e-15
    assert a[1] == 0.0 and abs(a[2]) < 1e-7                                     # negative within rounding: a = 0, not NaN
    assert np.isnan(a[3])                                                       # δ₀ + Q_ii < 0: marked infeasible
    assert list(np.asarray(model.readout_infeasible(Q, p))) == [False, False, False, True]
