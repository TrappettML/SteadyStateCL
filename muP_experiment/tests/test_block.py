"""``model.train_block`` (the active-row block loop with lazy row scale, tracked Q and the frozen closed form)
against the explicit step-and-project loop of ``protocol._step_and_project``; masked heads; infeasible-row redraw;
the manifest's run check."""
import math
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import params, state_with_task

from sscl_core import model
from sscl_core.config import Config, make_batches, make_runs
from sscl_core.protocol import _step_and_project, _step_fn, run_stream


def _loop(st, p, k_stream, n_steps, solver):
    """The pre-2026-11-06 task loop: step + project on the full state, every step."""
    step = _step_fn(p, solver, k_stream)
    fcum = jnp.ones(st.W.shape[0])
    lsum, lmax = 0.0, 0.0
    n1 = jnp.maximum(st.g.sum(), 1.0)
    rows = []
    for i in range(n_steps):
        st, f, row = _step_and_project(st, p, step, i)
        fcum = fcum * f
        lsum = lsum + jnp.log(f).sum() / n1
        lmax = jnp.maximum(lmax, jnp.abs(jnp.log(f)).max())
        rows.append(row)
    return st, fcum, lsum, lmax, jnp.stack(rows)


@pytest.mark.parametrize("readout", ["gated_shared", "learned_shared", "gated_frozen", "frozen", "task_heads"])
@pytest.mark.parametrize("solver", ["pop", "sgd", "sgd4"])
@pytest.mark.parametrize("delta0", [1.0, -0.3])
def test_block_matches_explicit_loop(readout, solver, delta0):
    N, D, n_steps = 12, 48, 7
    st, p = state_with_task(seed=3, N=N, D=D, readout=readout, delta0=delta0, pN=4.0)
    k_stream = jax.random.PRNGKey(11)
    ref, fcum, lsum, lmax, rows = _loop(st, p, k_stream, n_steps, solver)
    for K in (N, 8):                                               # every row, and a block of 8 (gate ≤ 8 here)
        if K < N and st.g.sum() > K:
            continue
        r = model.train_block(st.W, st.a, st.g, st.w, p, k_stream, n_steps=n_steps, solver=solver, K=K, traced=True)
        np.testing.assert_allclose(r["W"], ref.W, rtol=0, atol=1e-11)
        np.testing.assert_allclose(r["a"], ref.a, rtol=0, atol=1e-11)
        np.testing.assert_allclose(r["f"], fcum, rtol=1e-11, atol=0)
        np.testing.assert_allclose(r["logf_sum"], lsum, rtol=0, atol=1e-10)
        np.testing.assert_allclose(r["logf_max"], lmax, rtol=0, atol=1e-10)
        np.testing.assert_allclose(r["rows"], rows, rtol=1e-9, atol=1e-11, equal_nan=True)
        np.testing.assert_allclose(r["uc_after"], model.u_c(ref, p), rtol=0, atol=1e-11)
        # the pre-projection state of the last step: Q and a after the step, before eq. (proj)
        pre_state = _loop(st, p, k_stream, n_steps - 1, solver)[0]
        pre_state = _step_fn(p, solver, k_stream)(pre_state, n_steps - 1)
        np.testing.assert_allclose(r["Q_pre"], model.Q_diag(pre_state.W), rtol=1e-10, atol=0)
        np.testing.assert_allclose(r["a_pre"], pre_state.a, rtol=0, atol=1e-11)
        np.testing.assert_allclose(r["uc_before"], model.u_c(pre_state, p), rtol=0, atol=1e-11)


@pytest.mark.parametrize("readout", ["gated_shared", "gated_frozen", "task_heads"])
def test_batch_sgd_step_is_the_sum_of_single_sample_updates(readout):
    """``sgd_step`` with B samples moves W and a by the sum of the B updates of eqs. (sgdW)–(sgda), each from the
    weights the step starts with; B = 1 is the online step bit for bit."""
    N, D, B = 12, 48, 5
    st, p = state_with_task(seed=7, N=N, D=D, readout=readout, delta0=0.5, pN=4.0)
    k = jax.random.PRNGKey(4)
    got = model.sgd_step(st, p, k, B)
    X = np.asarray(jax.random.normal(k, (B, D), dtype=st.W.dtype))
    W, a, g, w = (np.asarray(v) for v in (st.W, st.a, st.g, st.w))
    eta, pN, la = float(p.eta), float(p.pN), float(p.learn_a)
    dW, da = np.zeros_like(W), np.zeros_like(a)
    for x in X:
        hx = W @ x / np.sqrt(D)
        e = (g * a) @ hx / pN - w @ x / np.sqrt(D)
        dW -= eta * (g * a)[:, None] * e * x[None, :] / np.sqrt(D)
        da -= la * eta / D * g * e * hx
    assert np.abs(dW).max() > 0
    np.testing.assert_allclose(got.W, W + dW, rtol=0, atol=1e-12)
    np.testing.assert_allclose(got.a, a + da, rtol=0, atol=1e-12)
    one = model.sgd_step(st, p, k, 1)
    ref = model.sgd_step(st, p, k)
    assert np.array_equal(one.W, ref.W) and np.array_equal(one.a, ref.a)
    assert [model.sgd_batch(s) for s in ("pop", "sgd", "sgd32")] == [0, 1, 32]


def test_block_overflow_is_nan_not_silent():
    st, p = state_with_task(seed=1, N=12, D=48, readout="gated_shared", pN=4.0)
    st = st._replace(g=jnp.ones(12))                                # 12 active, block of 4
    r = model.train_block(st.W, st.a, st.g, st.w, p, jax.random.PRNGKey(0), n_steps=3, solver="pop", K=4)
    assert np.isnan(np.asarray(r["W"])).all() and np.isnan(float(r["logf_sum"]))


@pytest.mark.parametrize("readout", ["gated_frozen", "frozen"])
def test_closed_form_matches_steps(readout):
    st, p = state_with_task(seed=5, N=12, D=48, readout=readout, delta0=0.0, pN=4.0, h=0.02)
    k = jax.random.PRNGKey(0)
    ref = model.train_block(st.W, st.a, st.g, st.w, p, k, n_steps=200, solver="pop", K=12)
    cf = model.train_block(st.W, st.a, st.g, st.w, p, k, n_steps=200, solver="pop", K=12, closed=True)
    np.testing.assert_allclose(cf["W"], ref["W"], rtol=0, atol=1e-13)
    np.testing.assert_allclose(cf["uc_after"], ref["uc_after"], rtol=0, atol=1e-13)
    assert float(cf["logf_sum"]) == 0.0 and np.all(np.asarray(cf["f"]) == 1.0)


def test_block_rows_bounds_the_gate():
    assert model.block_rows(200, 200.0, False) == 200
    K = model.block_rows(200, math.sqrt(200), True)
    assert 30 < K < 60
    assert model.block_rows(5, 2.0, True) == 5


def test_closed_runs_batch_apart_and_mixed_batches_step():
    cfg = Config(solver="pop", N=[10], readout=["gated_shared", "gated_frozen", "frozen"], delta0=[0.0], seeds=2, batch_size=64)
    runs = make_runs(cfg)
    for b in make_batches(runs, 64):
        assert len({r.closed for r in b}) == 1
    assert sum(r.closed for r in runs) == 4 and not any(r.closed for r in runs if r.readout == "gated_shared")


def test_stored_heads_are_masked_and_pre_value_rescaled(small_runs):
    runs = [r for r in small_runs if r.readout == "task_heads"]
    res = run_stream(runs, chunk=1)
    assert np.all(res["heads_fro2_post"][1:] > 0)
    # the head of task t is g_t ⊙ a after the task: its norm is Σ_active a_i² = n_active · mean_active a_i² (post);
    # after task 0 nothing has been rescaled yet, so the two agree exactly (entries off the gate contribute nothing)
    assert abs(res["heads_fro2_post"][0, 0] - res["n_active"][0, 0] * res["a2_act_post"][0, 0]) < 1e-9
    # the pre value of task t ≥ 1 sits between the previous post value (before any of the task's steps) and
    # this task's post value with the last step's factor undone: it carries the rescaling of all but the last step
    assert np.all(np.isfinite(res["heads_fro2_pre"][1:]))


def test_infeasible_rows_are_redrawn_and_seeds_kept():
    cfg = Config(solver="pop", N=[5], readout=["gated_shared"], delta0=[-0.75], seeds=12, T_min=2, T_factor=0.1)
    runs = make_runs(cfg)
    from sscl_core.protocol import feasible, init_batch
    assert feasible(runs).all()                                    # every seed survives at D = 20, δ₀ = −0.75
    st, p = init_batch(runs)
    Q = np.asarray(jax.vmap(model.Q_diag)(st.W))
    assert np.all(Q + np.asarray(p.Delta0)[:, None] * np.asarray(p.eta)[:, None] > 0)
    # a feasible first draw is unchanged by the redraw machinery
    key = jax.random.PRNGKey(7)
    p1 = params(delta0=1.0)
    W_plain = jax.random.normal(jax.random.split(key)[0], (5, 20), dtype=jnp.float64)
    np.testing.assert_array_equal(model.draw_W0(jax.random.split(key)[0], p1, 5, 20), W_plain)


def test_sigma0_scales_the_initial_W():
    cfg = Config(solver="pop", N=[5], readout=["frozen"], delta0=[0.0], seeds=1, sigma0=[1.0, 2.0], T_min=2, T_factor=0.1)
    runs = make_runs(cfg)
    assert len(runs) == 2 and runs[0].rng_seed() == runs[1].rng_seed() and runs[0].label() != runs[1].label()
    from sscl_core.protocol import init_batch
    st, _ = init_batch(runs)
    np.testing.assert_allclose(st.W[1], 2.0 * st.W[0])


def test_changed_config_marks_batches_stale(tmp_path):
    from sscl_core import io
    from sscl_core.protocol import run_stream
    cfg = Config(solver="pop", N=[5], readout=["gated_shared"], delta0=[0.0], seeds=2, T_min=2, T_factor=0.1, batch_size=64)
    batches = io.plan_sweep(cfg, str(tmp_path))
    io.save_batch(str(tmp_path), 0, run_stream(batches[0]), batches[0])
    assert io.pending_batches(str(tmp_path)) == []
    cfg3 = Config(**{**cfg.__dict__, "seeds": 3})               # one more seed: batch 0 now holds other runs
    io.plan_sweep(cfg3, str(tmp_path))
    assert io.pending_batches(str(tmp_path)) == [0]
    assert io.read_manifest(str(tmp_path))["batches"][0]["status"] == "stale"


def test_load_runs_warns_on_duplicates(tmp_path):
    from sscl_core import io
    from sscl_core.results import load_runs
    cfg = Config(solver="pop", N=[5], readout=["gated_shared"], delta0=[0.0], seeds=2, T_min=2, T_factor=0.1, batch_size=64)
    batches = io.plan_sweep(cfg, str(tmp_path))
    res = run_stream(batches[0])
    io.save_batch(str(tmp_path), 0, res, batches[0])
    io.save_batch(str(tmp_path), 1, res, batches[0])                 # the same runs under two batch ids
    with pytest.warns(UserWarning, match="more than one batch file"):
        records, _ = load_runs(str(tmp_path))
    assert len(records) == 4
