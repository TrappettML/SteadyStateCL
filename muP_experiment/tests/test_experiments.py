"""``mup.experiments`` kernels against ``protocol.run_task``, and ``mup.pool`` placements."""
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import one_batch, small_runs  # noqa: F401

from mup import experiments as X
from mup import model as M
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


def test_first_and_steady_signal_matches_signal_after_tasks(small_runs):
    """One stream of T + n tasks gives the first-memory curve and the steady-state one of the two separate calls."""
    runs = small_runs
    T, n = 5, 4
    got = X.first_and_steady_signal(runs, T, n, chunk=2)
    assert got["S0"].shape == (T, len(runs)) and got["ST"].shape == (n, len(runs))
    np.testing.assert_allclose(got["S0"], X.signal_after_tasks(runs, 0, T, 0, chunk=3), atol=1e-11, rtol=0)
    np.testing.assert_allclose(got["ST"], X.signal_after_tasks(runs, T, n, T, chunk=3), atol=1e-11, rtol=0)


def test_norms_over_tasks_matches_loop(small_runs):
    runs = small_runs
    T = 4
    ref, p = _ref_states(runs, T)
    vn = jax.jit(jax.vmap(X.task_norms))
    want = np.stack([np.asarray(vn(s, p)) for s in ref])            # (T, B, 5)
    got = X.norms_over_tasks(runs, T, chunk=3)
    for i, q in enumerate(X.NORM_KEYS):
        np.testing.assert_allclose(got[q], want[:, :, i].T, atol=1e-11, rtol=0, err_msg=q)


def test_leak_norms_match_loop(small_runs):
    """``LEAK_KEYS`` of task t come from the state before it: u_b of the next task's own gate, teacher and readout."""
    runs = small_runs
    T = 4
    ref, p = _ref_states(runs, T)
    st0, _ = init_batch(runs)
    before = [jax.device_get(st0)] + ref[:-1]
    vl = jax.jit(jax.vmap(X.leak_norms))
    want = np.stack([np.asarray(vl(s, p)) for s in before])         # (T, B, 3)
    got = X.norms_over_tasks(runs, T, chunk=3)
    for i, q in enumerate(X.LEAK_KEYS):
        np.testing.assert_allclose(got[q], want[:, :, i].T, atol=1e-11, rtol=0, err_msg=q)
    # the pieces against a direct evaluation for one run: ub2 = diagonal + pairs, with the task's own draws
    s = jax.tree_util.tree_map(lambda x: x[0], before[1])
    p0 = jax.tree_util.tree_map(lambda x: x[0], p)
    N, D = s.W.shape
    k_teach, k_head, _ = M.task_keys(s.key, s.t)
    w, g = M.draw_task(k_teach, N, D, p0.pN, p0.gated)
    a = np.where(float(p0.use_heads) == 1.0, np.asarray(M.draw_readout(k_head, M.Q_diag(s.W), p0)), np.asarray(s.a))
    V = (np.asarray(g) * a)[:, None] * np.asarray(s.W) / float(p0.pN)
    G = V @ V.T / D
    np.testing.assert_allclose(want[1, 0], [G.sum(), V.sum(0) @ np.asarray(w) / D, np.trace(G)], atol=1e-11, rtol=0)


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


# ----------------------------------------------------------------------------
# pool.plan: placement from --gpus alone
# ----------------------------------------------------------------------------
def _runs(N=10, seeds=2):
    """``seeds`` frozen-readout runs of width ``N`` (one run per seed)."""
    return make_runs(Config(N=[N], readout=["frozen"], delta0=[0.0], lam=[1.0], seeds=seeds, T_factor=0.5, T_min=6))


def _args(**kw):
    import argparse
    base = {"gpus": None, "per_gpu": None, "workers": 0, "chunk": None, "split": None}
    return argparse.Namespace(**{**base, **kw})


def test_plan_cpu_in_process_keeps_groups_whole():
    from mup import pool
    groups = [(_runs(N=10, seeds=3), 20), (_runs(N=20, seeds=3), 40)]
    pl = pool.plan(_args(), groups)
    assert pl.gpus is None and pl.slots == 1 and pl.pieces == [1, 1] and pl.chunk == 1
    assert pl.describe() == "in-process"


def test_plan_gpu_fills_cards_without_overcutting(monkeypatch):
    from mup import driver, pool
    monkeypatch.setattr(driver, "gpu_memory_gb", lambda: 32.0)
    monkeypatch.setattr(driver, "visible_gpus", lambda: [])
    one = [(_runs(N=10, seeds=12), 20)]                       # 12 runs, 1 group
    pl = pool.plan(_args(gpus=[]), one)
    assert pl.gpus == [0, 1, 2, 3] and pl.per_gpu == 3 and pl.slots == 12
    assert pl.pieces == [4]                                   # one group: one piece per card
    assert pl.chunk == 16                                     # 64 and 32 exceed T = 20
    many = [(_runs(N=10, seeds=1), 2000) for _ in range(20)]  # 20 groups ≥ 12 slots
    pl = pool.plan(_args(gpus=["0", "1"]), many)
    assert pl.gpus == [0, 1] and pl.per_gpu == 3 and pl.pieces == [1] * 20 and pl.chunk == 64
    pl = pool.plan(_args(gpus=["auto"], per_gpu=1, chunk=8, split=2), many)
    assert pl.per_gpu == 1 and pl.chunk == 8 and pl.pieces == [1] * 20   # split capped by the group size
    pl = pool.plan(_args(gpus=[], split=2), [(_runs(N=10, seeds=2), 20)] * 20)
    assert pl.pieces == [2] * 20
    chains = [(_runs(N=10, seeds=8), 80), (_runs(N=10, seeds=8), 40), (_runs(N=10, seeds=8), 10)]
    assert pool.plan(_args(gpus=[]), chains).pieces == [3, 2, 1]   # ceil(cards × share of the total cost)
    equal = [(_runs(N=10, seeds=8), 80)] * 6                       # six equal groups: none is cut
    assert pool.plan(_args(gpus=[]), equal).pieces == [1] * 6


def test_plan_memory_split_is_a_multiple_of_the_cards(monkeypatch):
    from mup import driver, pool
    monkeypatch.setattr(driver, "visible_gpus", lambda: [0, 1, 2, 3])
    runs = _runs(N=10, seeds=12)
    one = pool.job_bytes(runs[:1], 20, 16, 8.0)
    # a process's share holds 2.5 runs: pieces of 2 runs, so 6 are needed and 8 (two per card) are made
    monkeypatch.setattr(driver, "gpu_memory_gb", lambda: one * 2.5 * 3 / driver.GPU_MEM_SHARE / 2 ** 30)
    pl = pool.plan(_args(gpus=[]), [(runs, 20)])
    assert pl.per_gpu == 3 and pl.pieces == [8]


def test_assign_balances_largest_first():
    from mup import pool
    unit_of = pool.assign([8, 7, 6, 5, 4, 3, 2, 1], 4)
    load = [0] * 4
    for u, c in zip(unit_of, [8, 7, 6, 5, 4, 3, 2, 1]):
        load[u] += c
    assert unit_of[:4] == [0, 1, 2, 3] and max(load) - min(load) <= 1
    assert pool.assign([], 4) == [] and pool.assign([1.0, 2.0], 1) == [0, 0]


def test_group_key_separates_block_sizes_and_job_cost_has_overhead():
    cfg = Config(N=[100], readout=["gated_shared", "gated_frozen", "learned_shared"], kappa=["1", "4"],
                 delta0=[0.0], lam=[1.0], seeds=1, T_factor=0.25, T_min=4)
    runs = make_runs(cfg)
    keys = {X.group_key(r) for r in runs}
    assert all(k[-1] == r.block_rows() for r in runs for k in [X.group_key(r)])
    assert len({X.group_key(r)[-1] for r in runs if r.readout == "gated_shared"}) == 2   # one block per κ
    ungated = [r for r in runs if r.readout == "learned_shared"]
    assert X.group_key(ungated[0])[-1] == 100 and len(keys) >= 4
    r = runs[0]
    assert X.job_cost([r], 10) > X.run_cost(r, 10) > 0
    assert X.job_cost([r, r], 10) - X.job_cost([r], 10) == pytest.approx(X.run_cost(r, 10))


def test_plan_gpu_respects_memory(monkeypatch):
    from mup import driver, pool
    monkeypatch.setattr(driver, "gpu_memory_gb", lambda: 0.001)   # ~1 MB cards
    monkeypatch.setattr(driver, "visible_gpus", lambda: [0, 1, 2, 3])
    runs = _runs(N=10, seeds=12)
    r = runs[0]
    need = pool.job_bytes(runs[:1], 20, 16, 8.0)
    big = _args(gpus=[])
    monkeypatch.setattr(driver, "gpu_memory_gb", lambda: need * 4 / driver.GPU_MEM_SHARE / 2 ** 30)
    pl = pool.plan(big, [(runs, 20)])                          # 4 runs per card, so 3 processes hold ≤ 1 run each
    assert pl.per_gpu == 3 and pl.pieces == [12]
    monkeypatch.setattr(driver, "gpu_memory_gb", lambda: need * 1.5 / driver.GPU_MEM_SHARE / 2 ** 30)
    pl = pool.plan(big, [(runs, 20)])                          # only one run per card fits: one process each
    assert pl.per_gpu == 1 and pl.pieces == [12]
    pl = pool.plan(big, [(runs[i:i + 1], 20) for i in range(12)], split=False)
    assert pl.pieces == [1] * 12 and pl.per_gpu == 1
    with pytest.raises(SystemExit):
        pool.plan(big, [(runs[:4], 20)] * 3, split=False)        # 4 runs per batch, 1 fits
    monkeypatch.setattr(driver, "gpu_memory_gb", lambda: need * 0.5 / driver.GPU_MEM_SHARE / 2 ** 30)
    with pytest.raises(SystemExit):
        pool.plan(big, [(runs, 20)])
    assert r.N == 10


def test_progress_lines_reach_the_log_in_process_and_from_workers(tmp_path):
    from mup import pool
    runs = _runs(N=10, seeds=2)
    jobs = [((i,), {"runs": runs, "T_pre": 0, "n_follow": 20, "k": 0}, 1.0) for i in range(2)]
    for kw in ({}, {"workers": 2}):
        lines = []
        pool.run_jobs("mup.experiments:signal_after_tasks", jobs, log=lines.append, chunk=5, **kw)
        prog = [ln for ln in lines if "task " in ln and "%" in ln]
        assert any("task 20/20 (100%)" in ln for ln in prog), lines
        if not kw:                                              # workers relay only the latest line per 0.5 s poll
            assert any("task 10/20 (50%)" in ln for ln in prog), lines
        assert sum("done" in ln for ln in lines) == 2
    assert pool._REPORT is None


def test_loss_curves_match_stream_trace(small_runs):
    """The learning curve's end point is the stream's L_end of that task, and its steps are the trace's loss column."""
    import dataclasses
    from mup.protocol import TRACE_QUANT, run_stream
    runs = [dataclasses.replace(r, trace=True, trace_points=1000) for r in small_runs]
    T = runs[0].T_burn + runs[0].T_meas
    ref = run_stream(runs, chunk=1)
    tasks = [0, 1, T // 2 - 1, T // 2, T - 2, T - 1]
    L = X.loss_curves(small_runs, tasks, T, chunk=3)
    assert L.shape == (len(runs), len(tasks), runs[0].n_steps + 1)
    for j, t in enumerate(tasks):
        np.testing.assert_allclose(L[:, j, -1], ref["L_end"][t], rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(L[:, j, 1:], ref["trace"][:, t, :, TRACE_QUANT.index("loss")], rtol=1e-5, atol=1e-7)
    assert np.all(L[:, :, 0] > L[:, :, -1])                     # every task is learned
