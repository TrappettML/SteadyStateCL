"""The replay kernel of ``sscl_core/replay.py`` and the scripts of this folder.

    python -m pytest -q -p no:cacheprovider          (from replay_experiment/; about a minute on the laptop)

The last test runs only where JAX sees a GPU (cyclops): the same jobs through ``pool.run_jobs`` on the cards.
"""
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sscl_core import experiments as X
from sscl_core import replay as R
from sscl_core.config import Config, make_runs

GPUS = [d for d in jax.devices() if d.platform in ("gpu", "cuda", "rocm")]


def base(N=10, seeds=2, B=4, lam=1.0, gating="none", kappa=2.0):
    cfg = Config(name="t", N=[N], readout=[R.GATING_READOUT[gating]], kappa=[kappa], delta0=[0.0], lam=[lam],
                 seeds=seeds, solver="sgd", sgd_batch=B, T_per_N=1.0)
    return make_runs(cfg)


def runs_of(methods, cap="1D", write="0.25D", mode="sleep", ratio=1.0, gating="none", **kw):
    return R.expand(base(gating=gating, **kw), methods, [cap], write, ratio, mode, gating)


def advance(runs, T):
    """The batched state after ``T`` tasks, on the host."""
    rs, rp, ok = R.init_batch(runs)
    assert ok.all()
    return jax.device_get(R.train_kernel(runs[0].spec, T)(rs, rp)), rp


@pytest.mark.parametrize("B", [1, 4])
def test_none_is_the_mup_run(B):
    """No buffer, no replay: the μP kernel's signal of ``learned_shared``, to rounding."""
    b = base(B=B)
    got = R.signal_after_tasks(R.expand(b, ["none"], [], "0.25D", 1.0, "sleep"), 2, 5, 2)
    np.testing.assert_allclose(got, X.signal_after_tasks(b, 2, 5, 2), rtol=0, atol=1e-10)


def test_spec_and_replayed_samples():
    """Both modes replay ρ·n·B samples per task; ``none`` has no buffer."""
    s = runs_of(["fifo"], mode="sleep", ratio=0.5)[0]
    i = runs_of(["fifo"], mode="interleaved", ratio=0.5)[0]
    n, B = s.n_steps, 4
    assert s.spec == (n, "sgd4", 40, 10, 0, round(0.5 * n), "none") and i.spec == (n, "sgd4", 40, 10, 2, 0, "none")
    assert s.spec[5] * B == i.spec[4] * n
    assert runs_of(["none"])[0].spec == (n, "sgd4", 0, 0, 0, 0, "none")
    assert R.parse_size("0.5D", 80) == (40, "0.5D") and R.parse_size(300, 80) == (300, "300")


def test_same_stream_for_every_method():
    """Methods and capacities at one seed share the run key: same weights, teachers and fresh inputs."""
    runs = R.expand(base(), list(R.METHODS), ["1D", "4D"], "0.25D", 1.0, "sleep")
    assert len({r.rng_seed() for r in runs if r.seed == 0}) == 1
    assert len({r.label() for r in runs}) == len(runs)


def test_fifo_holds_the_last_tasks():
    """C = 4m: after T tasks the buffer holds m samples of each of the last four tasks, targets y = x·w/√D."""
    runs = runs_of(["fifo"], seeds=1)
    (rs, _) = advance(runs, 7)
    buf, s = rs.buf, rs.s
    assert int(buf.n[0]) == 40 and float(buf.count[0]) == 70
    assert sorted(np.asarray(buf.task[0]).tolist()) == sorted([3, 4, 5, 6] * 10)
    last = np.asarray(buf.task[0]) == 6                              # the task just trained: its teacher is state.w
    D = s.W.shape[-1]
    np.testing.assert_allclose(np.asarray(buf.y[0])[last], np.asarray(buf.X[0])[last] @ np.asarray(s.w[0]) / np.sqrt(D), atol=1e-12)


def test_buffer_fills_before_it_overwrites():
    """Empty slots fill in slot order under every method, so the filled slots are 0 … n − 1."""
    runs = runs_of(["fifo", "random", "priority"], seeds=1)
    (rs, _) = advance(runs, 3)
    for b in range(3):
        assert int(rs.buf.n[b]) == 30
        assert np.asarray(rs.buf.task[b])[:30].tolist() == [0] * 10 + [1] * 10 + [2] * 10


def test_priority_overwrites_the_smallest_gradients_random_any():
    """One write into a full buffer: priority replaces exactly the m slots with the smallest gradient norm,
    random m distinct slots, fifo the m oldest."""
    runs = runs_of(["fifo", "random", "priority"], seeds=1)
    (rs, rp) = advance(runs, 6)                                      # full since task 4
    one = lambda tree, b: jax.tree_util.tree_map(lambda x: jnp.asarray(x[b]), tree)
    for b, r in enumerate(runs):
        s, buf, p = one(rs.s, b), one(rs.buf, b), one(rp.p, b)
        new = R.write_buffer(buf, s, p, rp.method[b], jax.random.PRNGKey(3), 10)
        changed = np.flatnonzero(np.asarray(new.stamp) != np.asarray(buf.stamp))
        assert len(changed) == 10 and int(new.n) == 40
        assert (np.asarray(new.task)[changed] == int(s.t) - 1).all()
        if r.method == "priority":
            g = np.asarray(R.grad_norm(s.W, s.a, p, buf.X, buf.y))
            assert set(changed) == set(np.argsort(g)[:10])
        if r.method == "fifo":
            assert set(changed) == set(np.argsort(np.asarray(buf.stamp))[:10])


def test_grad_norm_is_the_gradient_norm():
    """``grad_norm`` against autodiff of ½e² w.r.t. (W, a)."""
    runs = runs_of(["priority"], seeds=1)
    (rs, rp) = advance(runs, 3)
    W, a, pN = rs.s.W[0], rs.s.a[0], rp.p.pN[0]
    Xs, y = rs.buf.X[0, :5], rs.buf.y[0, :5]
    D = W.shape[1]
    loss = lambda W, a, x, t: 0.5 * (a @ (W @ x) / (jnp.sqrt(D) * pN) - t) ** 2
    p = jax.tree_util.tree_map(lambda x: x[0], rp.p)
    for j in range(5):
        gW, ga = jax.grad(loss, (0, 1))(W, a, Xs[j], y[j])
        want = jnp.sqrt((gW ** 2).sum() + (ga ** 2).sum())
        np.testing.assert_allclose(R.grad_norm(W, a, p, Xs[j:j + 1], y[j:j + 1])[0], want, rtol=1e-10)


@pytest.mark.parametrize("mode", R.MODES)
def test_first_task_replays_nothing(mode):
    """The buffer is read before it is written: with an empty buffer task 0 is the μP task under both modes."""
    runs = runs_of(["fifo"], mode=mode, seeds=1)
    none = runs_of(["none"], seeds=1)
    np.testing.assert_allclose(R.signal_after_tasks(runs, 0, 1, 0), R.signal_after_tasks(none, 0, 1, 0), atol=1e-9)
    assert abs(R.signal_after_tasks(runs, 0, 3, 0)[2, 0] - R.signal_after_tasks(none, 0, 3, 0)[2, 0]) > 1e-6


def test_chunks_and_curves_agree():
    """Scanning several tasks per call equals one per call; the traced curve ends where the untraced task does."""
    runs = runs_of(["fifo", "priority"], seeds=1)
    a = R.signal_after_tasks(runs, 2, 4, 2, chunk=1)
    b = R.signal_after_tasks(runs, 2, 4, 2, chunk=4)
    np.testing.assert_allclose(a, b, atol=1e-10)
    L = R.loss_curves(runs, [0, 3], 4)
    n, n_sleep = runs[0].spec[0], runs[0].spec[5]
    assert L.shape == (2, 2, n + n_sleep + 1)
    sl = R.loss_slopes(runs, 4)
    np.testing.assert_allclose(sl["loss_start"][[0, 3]].T, L[:, :, 0], atol=1e-10)
    np.testing.assert_allclose(sl["loss_sleep"][[0, 3]].T, L[:, :, -1], atol=1e-10)
    np.testing.assert_allclose(R.norms_over_tasks(runs, 4)["loss_end"][:, [0, 3]], L[:, :, -1], atol=1e-10)


def test_scripts_run(tmp_path):
    """Every script end to end on a tiny grid: figures and arrays are written."""
    import run_inter_vs_sleep, run_loss_slope, run_Nscaling, run_stask, run_szero, run_training_verify, run_weight_norms
    small = ["--N", "10", "--seeds", "2", "--sgd_batch", "8", "--lam", "1"]
    out = lambda name: ["--out", str(tmp_path / name)]
    assert run_szero.main(small + ["--tasks", "8"] + out("szero")) == 0
    assert run_stask.main(small + ["--T", "6", "--tau", "4", "--mode", "interleaved"] + out("stask")) == 0
    assert run_Nscaling.main(["--N", "10", "12"] + small[2:] + ["--T", "0.6", "--tau", "0.4"] + out("Nscaling")) == 0
    assert run_weight_norms.main(["--N", "10", "12"] + small[2:] + ["--tasks", "0.8"] + out("wn")) == 0
    assert run_training_verify.main(small[2:] + ["--cell", "10,1D,0", "--tasks", "6"] + out("tv")) == 0
    assert run_loss_slope.main(small + ["--tasks", "6"] + out("ls")) == 0
    assert run_inter_vs_sleep.main(small + ["--T", "6", "--tau", "4"] + out("ivs")) == 0
    pngs = sorted(p.name for p in tmp_path.rglob("*.png"))
    assert len(pngs) == 4 + 4 + 2 + 2 + 1 + 1 + 1, pngs               # szero and stask: signal and area per capacity
    assert "stask_area_N10_C4D_d00_lam1_interleaved_ungated.png" in pngs
    assert "stask_N10_C4D_d00_lam1_interleaved_ungated.png" in pngs and "inter_vs_sleep_N10_d00_lam1_ungated.png" in pngs


@pytest.mark.parametrize("gating", ["unit", "synapse"])
def test_scripts_run_gated(tmp_path, gating):
    """The scripts under ``--gating``: the figures are named with the gating and κ."""
    import run_stask, run_training_verify
    small = ["--N", "10", "--seeds", "2", "--sgd_batch", "4", "--lam", "1", "--gating", gating, "--kappa", "2"]
    assert run_stask.main(small + ["--T", "4", "--tau", "3", "--capacity", "1D", "--mode", "interleaved",
                                   "--out", str(tmp_path / "stask")]) == 0
    assert run_training_verify.main(small[2:] + ["--cell", "10,1D,0", "--tasks", "6", "--out", str(tmp_path / "tv")]) == 0
    pngs = sorted(p.name for p in tmp_path.rglob("*.png"))
    assert f"stask_N10_C1D_d00_lam1_interleaved_{gating}_k2.png" in pngs, pngs
    assert f"loss_N10_C1D_d00_sleep_{gating}_k2.png" in pngs, pngs


# ----------------------------------------------------------------------------
# gating
# ----------------------------------------------------------------------------
def test_gating_flag_sets_the_readout():
    """``none`` runs the ungated readout, ``unit`` and ``synapse`` the κ-sparse one; a mismatch is refused."""
    assert {r.readout for r in runs_of(["fifo"])} == {"learned_shared"}
    for g in ("unit", "synapse"):
        rr = runs_of(["none", "fifo"], gating=g)
        assert {r.readout for r in rr} == {"gated_shared"} and {r.gating for r in rr} == {g}
        assert all(r.spec[-1] == g and r.gate_label() == f"{g}_k2" for r in rr)
    with pytest.raises(ValueError):
        R.expand(base(gating="unit"), ["fifo"], ["1D"], "0.25D", 1.0, "sleep", "none")
    u, sy = runs_of(["fifo"], gating="unit", seeds=1)[0], runs_of(["fifo"], gating="synapse", seeds=1)[0]
    assert u.rng_seed() == sy.rng_seed() and u.label() != sy.label() and R.group_key(u) != R.group_key(sy)


@pytest.mark.parametrize("B", [1, 4])
def test_unit_gated_none_is_the_mup_gated_run(B):
    """Unit gating, no buffer: the μP kernel's signal of ``gated_shared`` (block of active rows), to rounding."""
    b = base(B=B, gating="unit")
    got = R.signal_after_tasks(R.expand(b, ["none"], [], "0.25D", 1.0, "sleep", "unit"), 2, 5, 2)
    np.testing.assert_allclose(got, X.signal_after_tasks(b, 2, 5, 2), rtol=0, atol=1e-9)


def test_synapse_mask_is_that_of_the_synaptic_run():
    """The regenerated mask is bit for bit ``synapse.draw_mask`` / ``probe_mask`` of the task."""
    from sscl_core import model, synapse
    key, N, D, pN = jax.random.PRNGKey(5), 10, 40, jnp.float64(6.0)
    for k in (0, 3):
        _, M = synapse.probe_mask(key, k, N, D, pN, 0.8)
        np.testing.assert_array_equal(synapse.task_mask(key, k, N, D, pN), M)
        assert 0.3 < float(M.mean()) < 0.9


def _gated_setup(gating, b=6):
    """One run's weights after 3 tasks, and b samples of three different tasks with their own gates."""
    from sscl_core import model, synapse
    runs = runs_of(["fifo"], gating=gating, seeds=1)
    (rs, rp) = advance(runs, 3)
    one = lambda tree: jax.tree_util.tree_map(lambda x: jnp.asarray(x[0]), tree)
    s, buf, p = one(rs.s), one(rs.buf), one(rp.p)
    N, D = s.W.shape
    tasks = jnp.asarray([0, 0, 1, 1, 2, 2][:b], jnp.int32)
    Xs = jax.random.normal(jax.random.PRNGKey(9), (b, D), dtype=jnp.float64)
    y = jax.random.normal(jax.random.PRNGKey(8), (b,), dtype=jnp.float64)
    if gating == "unit":
        G = jnp.stack([model.probe_task(s.key, k, N, D, p.pN, p.gated, p.shared)[1] for k in tasks])
        Gfull = G[:, :, None] * jnp.ones((1, 1, D))
    else:
        G = jnp.stack([synapse.task_mask(s.key, k, N, D, p.pN) for k in tasks])
        Gfull = G
    return s, buf, p, Xs, y, tasks, G, Gfull


@pytest.mark.parametrize("gating", ["unit", "synapse"])
def test_gated_step_is_the_gradient_under_each_samples_gate(gating):
    """One step of ``gated_phase`` (projection off) against autodiff of Σ_j ½e_j², every sample through its own
    gate: ΔW = −η pN ∂/∂W, Δa = −η pN/D ∂/∂a (the scaling of ``model.sgd_step``)."""
    s, _, p, Xs, y, tasks, G, Gfull = _gated_setup(gating)
    p = p._replace(do_proj=jnp.float64(0.0))
    D = s.W.shape[1]
    wgt = jnp.asarray([1.0, 1.0, 1.0, 0.0, 1.0, 1.0])               # one sample weighted out
    W1, a1, _ = R.gated_phase(s.W, s.a, p, lambda i: (Xs, y, wgt, G), 1, gating)

    def loss(W, a):
        h = jnp.einsum("jid,id,jd->ji", Gfull, W, Xs) / jnp.sqrt(D)
        e = h @ a / p.pN - y
        return 0.5 * (wgt * e * e).sum()
    gW, ga = jax.grad(loss, (0, 1))(s.W, s.a)
    np.testing.assert_allclose(s.W - W1, p.eta * p.pN * gW, rtol=1e-9, atol=1e-13)
    np.testing.assert_allclose(s.a - a1, p.eta * p.pN / D * ga, rtol=1e-9, atol=1e-13)
    if gating == "unit":                                            # a unit no weighted sample opened is untouched,
        p1 = p._replace(do_proj=jnp.float64(1.0))                  # projection included
        w1 = jnp.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])            # only sample 0 (task 0) counts
        W2, a2, _ = R.gated_phase(s.W, s.a, p1, lambda i: (Xs, y, w1, G), 1, gating)
        shut = np.asarray((G * w1[:, None]).max(0) == 0.0)
        assert shut.any() and (~shut).any()
        np.testing.assert_array_equal(np.asarray(W2)[shut], np.asarray(s.W)[shut])
        np.testing.assert_array_equal(np.asarray(a2)[shut], np.asarray(s.a)[shut])
    else:                                                           # a closed synapse of every sample is untouched
        closed = np.asarray(Gfull.max(0) == 0.0)
        assert closed.any()
        np.testing.assert_array_equal(np.asarray(W1)[closed], np.asarray(s.W)[closed])


@pytest.mark.parametrize("gating", ["unit", "synapse"])
def test_gated_grad_norm_is_the_gradient_norm(gating):
    """The priority under the sample's own gate against autodiff of ½e² w.r.t. (W, a)."""
    s, _, p, Xs, y, tasks, G, Gfull = _gated_setup(gating)
    D = s.W.shape[1]
    got = R.grad_norm(s.W, s.a, p, Xs, y, G if gating == "unit" else None, tasks, s.key, gating)
    for j in range(len(y)):
        loss = lambda W, a: 0.5 * (a @ ((Gfull[j] * W) @ Xs[j]) / (jnp.sqrt(D) * p.pN) - y[j]) ** 2
        gW, ga = jax.grad(loss, (0, 1))(s.W, s.a)
        np.testing.assert_allclose(got[j], jnp.sqrt((gW ** 2).sum() + (ga ** 2).sum()), rtol=1e-10)


def test_buffer_stores_each_tasks_unit_gate():
    """Unit gating: every stored sample carries the gate of the task it came from, and the tasks' gates differ."""
    from sscl_core import model
    runs = runs_of(["fifo"], gating="unit", seeds=1)
    (rs, rp) = advance(runs, 5)
    N, D = rs.s.W.shape[1:]
    p = jax.tree_util.tree_map(lambda x: x[0], rp.p)
    gates = {k: np.asarray(model.probe_task(rs.s.key[0], k, N, D, p.pN, p.gated, p.shared)[1]) for k in range(5)}
    for slot in range(int(rs.buf.n[0])):
        np.testing.assert_array_equal(np.asarray(rs.buf.G[0, slot]), gates[int(rs.buf.task[0, slot])])
    assert len({tuple(g) for g in gates.values()}) > 1


@pytest.mark.parametrize("gating", ["unit", "synapse"])
@pytest.mark.parametrize("mode", R.MODES)
def test_gated_replay_runs(gating, mode):
    """Gated replay end to end: task 0 (empty buffer) is the gated no-replay task, later tasks differ, every
    observable is finite, and the chunked and traced kernels agree."""
    runs = runs_of(["fifo", "priority"], mode=mode, gating=gating, seeds=1)
    none = runs_of(["none"], gating=gating, seeds=1)
    np.testing.assert_allclose(R.signal_after_tasks(runs, 0, 1, 0)[:, 0], R.signal_after_tasks(none, 0, 1, 0)[:, 0],
                               atol=1e-9)
    assert abs(R.signal_after_tasks(runs, 0, 3, 0)[2, 0] - R.signal_after_tasks(none, 0, 3, 0)[2, 0]) > 1e-6
    np.testing.assert_allclose(R.signal_after_tasks(runs, 2, 3, 2, chunk=1), R.signal_after_tasks(runs, 2, 3, 2, chunk=3),
                               atol=1e-10)
    L = R.loss_curves(runs, [0, 3], 4)
    assert np.isfinite(L).all() and (L[:, :, -1] < L[:, :, 0]).all()
    np.testing.assert_allclose(R.norms_over_tasks(runs, 4)["loss_end"][:, [0, 3]], L[:, :, -1], atol=1e-10)


@pytest.mark.skipif(not GPUS, reason="no GPU visible to JAX")
def test_pool_on_gpus_matches_in_process():
    """``pool.run_jobs`` with ``gpus``: the replay jobs on the cards give this process's result."""
    from sscl_core import driver, pool
    gpus = driver.visible_gpus() or list(range(len(GPUS)))
    jobs = []
    for i, mode in enumerate(R.MODES):
        runs = runs_of(["fifo", "random", "priority"], mode=mode)
        jobs.append(((i,), {"runs": runs, "T_pre": 3, "n_follow": 4, "k": 3}, 1.0))
    res = pool.run_jobs("sscl_core.replay:signal_after_tasks", jobs, gpus=gpus, per_gpu=1)
    for (_, kw, _), r in zip(jobs, res):
        np.testing.assert_allclose(r, R.signal_after_tasks(**kw), atol=1e-8, rtol=0)


def test_area_curve():
    """Area under the normalized curve: n for a constant signal, the trapezoid sum for a decay, both statistics."""
    import run_szero
    Y = np.ones((6, 3)) * np.array([1.0, 2.0, 4.0])
    for stat in ("mean", "seed"):
        A, se = run_szero.area_curve(Y, stat)
        np.testing.assert_allclose(A, np.arange(6)); np.testing.assert_allclose(se, 0, atol=1e-12)
    d = 0.5 ** np.arange(5)
    A, _ = run_szero.area_curve(np.stack([d, 3 * d], 1), "mean")
    np.testing.assert_allclose(A, [0, 0.75, 1.125, 1.3125, 1.40625])
