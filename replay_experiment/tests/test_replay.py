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


def base(N=10, seeds=2, B=4, lam=1.0):
    cfg = Config(name="t", N=[N], readout=["learned_shared"], delta0=[0.0], lam=[lam], seeds=seeds, solver="sgd",
                 sgd_batch=B, T_per_N=1.0)
    return make_runs(cfg)


def runs_of(methods, cap="1D", write="0.25D", mode="sleep", ratio=1.0, **kw):
    return R.expand(base(**kw), methods, [cap], write, ratio, mode)


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
    assert s.spec == (n, "sgd4", 40, 10, 0, round(0.5 * n)) and i.spec == (n, "sgd4", 40, 10, 2, 0)
    assert s.spec[5] * B == i.spec[4] * n
    assert runs_of(["none"])[0].spec == (n, "sgd4", 0, 0, 0, 0)
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
    assert len(pngs) == 2 + 2 + 2 + 2 + 1 + 1 + 1, pngs
    assert "stask_N10_C4D_d00_lam1_interleaved.png" in pngs and "inter_vs_sleep_N10_d00_lam1.png" in pngs


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
