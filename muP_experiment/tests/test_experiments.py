"""``mup.experiments`` kernels against ``protocol.run_task``, and ``mup.pool`` placements."""
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import one_batch, small_runs  # noqa: F401

from mup import experiments as X
from mup import pool
from mup.config import Config, make_runs
from mup.observe import age_schedule
from mup.protocol import batched_run_task, init_batch, run_stream


def _ref_states(runs, T):
    """States after each of ``T`` tasks through the sweep kernel."""
    r0 = runs[0]
    ages = jnp.asarray(age_schedule(r0.N, r0.kappa_s, T, r0.n_ages_geom, r0.m_max_factor))
    st, p = init_batch(runs)
    f = batched_run_task(r0.n_steps, r0.solver)
    out = []
    for _ in range(T):
        st, _ = f(st, p, ages)
        out.append(jax.device_get(st))
    return out, p


def test_train_task_matches_run_task(small_runs):
    runs = small_runs
    T = runs[0].T_burn + runs[0].T_meas
    ref, p = _ref_states(runs, T)
    st, p2 = init_batch(runs)
    g = X.train_kernel(runs[0].n_steps, runs[0].solver, 1)
    for t in range(T):
        st = g(st, p2)
        got = jax.device_get(st)
        for name in ("g", "w", "key", "t"):
            assert np.array_equal(getattr(got, name), getattr(ref[t], name)), (t, name)
        for name in ("W", "a", "heads"):
            np.testing.assert_allclose(getattr(got, name), getattr(ref[t], name), atol=1e-12, rtol=0, err_msg=(t, name))


def test_chunked_train_matches(small_runs):
    runs = small_runs
    T = 6
    ref, _ = _ref_states(runs, T)
    st, p = init_batch(runs)
    st = X.train_kernel(runs[0].n_steps, runs[0].solver, T)(st, p)
    np.testing.assert_allclose(np.asarray(st.W), ref[-1].W, atol=1e-12, rtol=0)
    assert int(st.t[0]) == T


def test_signal_after_tasks_matches_loop(small_runs):
    runs = small_runs
    T = 5
    ref, p = _ref_states(runs, T)
    vs = jax.jit(jax.vmap(X.probe_signal, in_axes=(0, 0, None)))
    want = np.stack([np.asarray(vs(s, p, jnp.int32(0))) for s in ref])
    got = X.signal_after_tasks(runs, 0, T, 0, chunk=2)
    np.testing.assert_allclose(got, want, atol=1e-11, rtol=0)
    # probe T after T tasks, followed for 3 more, equals the loop too
    ref2, _ = _ref_states(runs, T + 3)
    want2 = np.stack([np.asarray(vs(s, p, jnp.int32(T))) for s in ref2[T:]])
    got2 = X.signal_after_tasks(runs, T, 3, T, chunk=4)
    np.testing.assert_allclose(got2, want2, atol=1e-11, rtol=0)


def test_norms_over_tasks_matches_loop(small_runs):
    runs = small_runs
    T = 4
    ref, p = _ref_states(runs, T)
    vn = jax.jit(jax.vmap(X.task_norms))
    want = np.stack([np.asarray(vn(s, p)) for s in ref])            # (T, B, 5)
    got = X.norms_over_tasks(runs, T, chunk=3)
    for i, q in enumerate(X.NORM_KEYS):
        np.testing.assert_allclose(got[q], want[:, :, i].T, atol=1e-11, rtol=0, err_msg=q)


def test_imbalance_over_steps_pins_with_projection(small_runs):
    runs = [r for r in small_runs if r.readout == "gated_shared"] * 2
    R = X.imbalance_over_steps(runs, [0.0, 1.0], 3, chunk=2)
    assert R.shape == (2, 3 * runs[0].n_steps, 4)
    assert R[1, :, 3].max() < 1e-9                                   # projection on: Δ_i pinned at Δ₀
    assert R[0, :, 3].max() > 1e-6                                   # off: it drifts


def test_infeasible_runs_come_back_nan():
    cfg = Config(N=[10], readout=["gated_shared"], delta0=[-0.75], lam=[1.0], seeds=6, T_factor=0.25, T_min=4)
    runs = make_runs(cfg)
    from mup.protocol import feasible
    ok = feasible(runs)
    S = X.signal_after_tasks(runs, 0, 3, 0)
    assert S.shape == (3, len(runs))
    assert np.array_equal(np.isfinite(S).all(0), ok)


def test_split_and_group():
    parts = X.split(list(range(10)), 4)
    assert sum(parts, []) == list(range(10)) and len(parts) == 4 and max(map(len, parts)) - min(map(len, parts)) <= 1
    assert X.split([1, 2], 5) == [[1], [2]]


def _jobs():
    cfg = Config(N=[10], readout=["gated_shared", "gated_frozen"], delta0=[0.0], lam=[1.0], seeds=2, T_factor=0.25, T_min=4)
    runs = one_batch(make_runs(cfg))
    return [("a", {"runs": runs[:2]}, 2.0), ("b", {"runs": runs[2:]}, 1.0)], runs


def test_pool_in_process_and_subprocess_agree(tmp_path):
    jobs, runs = _jobs()
    here = pool.run_jobs("mup.experiments:stream_batch", jobs, workers=1)
    sub = pool.run_jobs("mup.experiments:stream_batch", jobs, workers=2, log=lambda m: None)
    for a, b in zip(here, sub):
        np.testing.assert_allclose(a["sum_s"], b["sum_s"], atol=1e-10, rtol=0)
        np.testing.assert_allclose(a["E0"], b["E0"], atol=1e-10, rtol=0)
    want = run_stream(runs[:2])
    np.testing.assert_allclose(here[0]["E0"], want["E0"], atol=0, rtol=0)


def test_pool_reports_failures():
    jobs = [("bad", {"runs": []}, 1.0)]
    with pytest.raises((RuntimeError, IndexError)):
        pool.run_jobs("mup.experiments:stream_batch", jobs, workers=2, log=lambda m: None)
    with pytest.raises(IndexError):
        pool.run_jobs("mup.experiments:stream_batch", jobs, workers=1)
