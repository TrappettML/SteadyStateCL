"""The teacher family (write-up, "Teachers"; ``model.draw_teacher``) and its switch ``shared`` / ``--rho_zero``.

Every task draws a latent angle θ; two teachers of a run have similarity ρ = shared·κ(Δθ), random for every pair
of tasks.  ``shared = 0`` (``--rho_zero``) is the independent teacher of every run before 2026-11-15.
"""
import argparse

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sscl_core import model as M
from sscl_core import nested as X
from sscl_core import npmodel as NP
from sscl_core import synapse as S
from sscl_core.config import (FOURIER_W, Config, Run, add_clock_arg, add_teacher_args, clock_override, make_runs,
                              teacher_override)

f = lambda v: jnp.asarray(v, dtype=jnp.float64)


def test_kernel_has_unit_peak_and_no_constant_mode():
    assert abs(sum(FOURIER_W) - 1.0) < 1e-15
    assert abs(float(M.kernel(0.0)) - 1.0) < 1e-15 and abs(float(M.kernel(np.pi)) + 0.4) < 1e-12
    d = np.linspace(0.0, 2.0 * np.pi, 3600, endpoint=False)
    k = np.asarray(M.kernel(d))
    assert abs(k.mean()) < 1e-12                                              # zero mean over a uniform angle
    assert abs((k ** 2).mean() - sum(w * w for w in FOURIER_W) / 2.0) < 1e-12  # ⟨κ²⟩ = Σ w_k²/2 = 0.19
    np.testing.assert_allclose(k, NP.kernel(d), atol=1e-14)
    z = NP.fourier_coef(d)                                                    # the latent vector: |z| = 1, z·z′ = κ
    np.testing.assert_allclose((z * z).sum(1), 1.0, atol=1e-14)
    np.testing.assert_allclose(z @ z[0], NP.kernel(d - d[0]), atol=1e-14)


@pytest.mark.parametrize("D", [6, 20, 400])
def test_basis_is_orthogonal_and_fixed_for_the_run(D):
    key = jax.random.PRNGKey(1)
    B = np.asarray(M.teacher_basis(key, D))
    assert B.shape == (2 * len(FOURIER_W), D)
    np.testing.assert_allclose(B @ B.T / D, np.eye(len(B)), atol=1e-12)
    assert np.array_equal(B, np.asarray(M.teacher_basis(key, D)))
    assert not np.array_equal(B, np.asarray(M.teacher_basis(jax.random.PRNGKey(2), D)))
    Bn = NP.fourier_basis(np.random.default_rng(0), D)
    np.testing.assert_allclose(Bn @ Bn.T / D, np.eye(len(Bn)), atol=1e-12)


def test_rho_zero_is_the_independent_teacher():
    key, N, D = jax.random.PRNGKey(7), 9, 36
    for t in range(4):
        kt = M.task_keys(key, t)[0]
        w_old, g_old = M.draw_task(kt, N, D, f(3.0))                          # the call of the code before 2026-11-15
        w0, g0 = M.draw_task(kt, N, D, f(3.0), 1.0, key, f(0.0))
        w1, g1 = M.draw_task(kt, N, D, f(3.0), 1.0, key, f(0.8))
        assert np.array_equal(w0, w_old) and np.array_equal(g0, g_old)        # shared = 0: the same numbers
        assert np.array_equal(g1, g_old) and not np.array_equal(w1, w_old)    # the gate does not depend on shared
        assert np.array_equal(w0, jax.random.normal(jax.random.split(kt)[0], (D,), dtype=jnp.float64))
        for sh in (0.0, 0.8):                                                 # probes and the synaptic twin read the same teacher
            w, _ = M.draw_task(kt, N, D, f(3.0), 1.0, key, f(sh))
            assert np.array_equal(w, M.probe_task(key, t, N, D, f(3.0), 1.0, f(sh))[0])
            assert np.array_equal(w, S.draw_mask(kt, N, D, f(3.0), key, f(sh))[0])
            assert np.array_equal(w, S.probe_mask(key, t, N, D, f(3.0), f(sh))[0])


@pytest.mark.parametrize("shared", [0.2, 0.8, 1.0])
def test_similarity_is_the_kernel_of_the_angle_difference(shared):
    key, N, D, T = jax.random.PRNGKey(3), 4, 400, 300
    kts = jax.vmap(lambda t: M.task_keys(key, t)[0])(jnp.arange(T))
    W = np.asarray(jax.vmap(lambda k: M.draw_task(k, N, D, f(2.0), 1.0, key, f(shared))[0])(kts))
    th = np.asarray(jax.vmap(lambda k: M.task_angle(jax.random.split(k)[0]))(kts))
    assert 0.0 <= th.min() and th.max() < 2.0 * np.pi and abs(np.exp(1j * th).mean()) < 0.15
    Tm = W @ W.T / D
    off = ~np.eye(T, dtype=bool)
    rho = shared * NP.kernel(th[:, None] - th[None, :])
    assert abs(np.diag(Tm).mean() - 1.0) < 0.02                               # E|w|² = D at every shared fraction
    assert np.abs(Tm - rho)[off].max() < (0.0 if shared == 1.0 else 6.0 / np.sqrt(D)) + 1e-9   # private parts: O(D^-1/2)
    assert abs(Tm[off].mean()) < 0.02                                         # ρ is zero on average over the angles
    assert rho[off].min() < -0.3 * shared and rho[off].max() > 0.9 * shared   # and ranges over [−0.4 s, s]
    assert abs((Tm[off] ** 2).mean() - (shared ** 2 * 0.19 + (1 - shared ** 2) / D)) < 0.02


def test_config_default_legacy_files_and_label():
    assert Config().shared == 0.8
    cfg = dict(solver="pop", N=[10], readout=["gated_shared"], seeds=1)
    r = make_runs(Config(**cfg))[0]
    r0 = make_runs(Config(**cfg, shared=0.0))[0]
    assert r.shared == 0.8 and r0.shared == 0.0
    assert r.rng_seed() == r0.rng_seed()                                      # the twins share gates, W⁽⁰⁾ and ξ
    assert "rho_zero" not in r.label() and r0.label() == r.label() + " rho_zero"
    d = r.to_dict()
    assert Run.from_dict(d).shared == 0.8
    d.pop("shared")
    assert Run.from_dict(d).shared == 0.0                                     # batch files written before 2026-11-15
    from sscl_core.protocol import make_params
    np.testing.assert_array_equal(np.asarray(make_params([r, r0]).shared), [0.8, 0.0])
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError):
            make_runs(Config(**cfg, shared=bad))
    with pytest.raises(ValueError):                                           # D = 4 < 6 fixed directions
        make_runs(Config(solver="pop", N=[4], D_factor=1, readout=["frozen"], seeds=1))
    assert make_runs(Config(solver="pop", N=[4], D_factor=1, readout=["frozen"], seeds=1, shared=0.0))


def test_rho_zero_flag():
    ap = argparse.ArgumentParser()
    add_clock_arg(ap)                                                         # every μP script's parser
    assert clock_override(ap.parse_args([])) == {}                            # no flag: the config's shared fraction
    assert clock_override(ap.parse_args(["--rho_zero"])) == {"shared": 0.0}
    assert clock_override(ap.parse_args(["--shared", "0.5"])) == {"shared": 0.5}
    assert clock_override(ap.parse_args(["--rho_zero", "--clock", "nominal"])) == {"clock": "nominal", "shared": 0.0}
    with pytest.raises(SystemExit):
        ap.parse_args(["--rho_zero", "--shared", "0.5"])
    ap2 = argparse.ArgumentParser()
    add_teacher_args(ap2)
    cfg = Config(**{**Config().__dict__, **teacher_override(ap2.parse_args(["--rho_zero"]))})
    assert cfg.shared == 0.0


def test_stream_at_rho_zero_is_unchanged_and_twins_share_gates():
    from sscl_core.protocol import run_stream
    kw = dict(solver="pop", N=[10], readout=["gated_frozen"], delta0=[1.0], lam=[1.0], seeds=2, T_factor=0.25, T_min=4)
    a = run_stream(make_runs(Config(**kw, shared=0.0)))
    b = run_stream(make_runs(Config(**kw, shared=0.8)))
    assert all(np.isfinite(np.asarray(b[k], dtype=np.float64)).all() for k in ("sum_s", "sum_e", "L_end"))
    assert not np.allclose(np.asarray(a["sum_s"]), np.asarray(b["sum_s"]))    # the teachers differ …
    np.testing.assert_allclose(np.asarray(a["abar_c"]), np.asarray(b["abar_c"]), rtol=1e-12)   # … the gates and frozen keys do not


def test_numpy_reference_and_nested_stream():
    rng = np.random.default_rng(0)
    B = NP.fourier_basis(rng, 50)
    xi = rng.standard_normal(50)
    assert NP.teacher(xi, 1.0, B, 0.0) is xi
    w = NP.teacher(xi, 1.0, B, 1.0)
    assert abs(w @ w / 50 - 1.0) < 1e-12 and abs(w @ NP.teacher(xi, 1.5, B, 1.0) / 50 - NP.kernel(0.5)) < 1e-12
    c = X.Cond(N=8, D=400, p=0.3, rho=0.7, delta0=1.0, lam=1.0)
    assert c.latent == "fourier"
    s = X.Stream(c, 0)
    Wt = np.stack([s.task(t)[1] for t in range(200)])
    th = np.array([s.angle(t) for t in range(200)])
    Tm, off = Wt @ Wt.T / c.D, ~np.eye(200, dtype=bool)
    assert np.abs(Tm - 0.7 * NP.kernel(th[:, None] - th[None, :]))[off].max() < 6.0 / np.sqrt(c.D)
    s0 = X.Stream(X.Cond(N=8, D=400, p=0.3, rho=0.0, delta0=1.0, lam=1.0), 0)  # ρ = 0: the independent teacher
    assert abs((np.stack([s0.task(t)[1] for t in range(200)]) @ s0.task(0)[1] / 400)[1:]).max() < 6.0 / 20.0
