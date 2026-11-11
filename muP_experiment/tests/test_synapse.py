"""``mup.synapse`` (synaptic gating) against the unit-gated kernels at p = 1, a NumPy loop, and its own closed form."""
import jax
import jax.numpy as jnp
import numpy as np

from mup import experiments as X
from mup import model as M
from mup import synapse as S
from mup.config import Config, make_runs
from mup.protocol import init_batch

GATED = ["gated_shared", "gated_frozen", "task_heads"]


def _runs(kappa, N=9, seeds=2, lam=1.0, delta0=0.5, readout=GATED):
    cfg = Config(name="t", N=[N], kappa=[kappa], readout=readout, delta0=[delta0], lam=[lam], seeds=seeds,
                 T_per_N=1.0, T_min=4)
    return make_runs(cfg)


def test_all_synapses_open_is_unit_gating():
    """At p = 1 (κ = √N) every mask and every gate is all ones, so the two gatings are the same network."""
    runs = _runs("sqrtN")
    for kw in ({"T_pre": 0, "n_follow": 6, "k": 0}, {"T_pre": 4, "n_follow": 5, "k": 4}):
        unit = X.signal_after_tasks(runs, chunk=2, **kw)
        syn = S.signal_after_tasks(runs, chunk=3, **kw)
        np.testing.assert_allclose(syn, unit, atol=1e-9, rtol=0)


def test_teacher_is_the_unit_run_teacher():
    key = jax.random.PRNGKey(3)
    k_teach = M.task_keys(key, 5)[0]
    w, mask = S.draw_mask(k_teach, 9, 36, jnp.float64(3.0))
    assert np.array_equal(w, M.draw_task(k_teach, 9, 36, jnp.float64(3.0))[0])
    assert mask.shape == (9, 36) and set(np.unique(mask)) <= {0.0, 1.0}
    assert abs(float(mask.mean()) - 1 / 3) < 0.1


def _numpy_stream(run, T, k):
    """Eq. (syn_pop) + projection on every unit, step by step in NumPy; the signal of probe ``k`` after each task."""
    st, p = init_batch([run])
    key = st.key[0]
    W, a = np.array(st.W[0]), np.array(st.a[0])
    N, D = W.shape
    pN, c, d0 = run.pN, run.h * run.eta, run.Delta0
    heads, out = {}, []
    for t in range(T):
        k_teach, k_head, _ = M.task_keys(key, t)
        w, mask = (np.asarray(v) for v in S.draw_mask(k_teach, N, D, jnp.float64(pN)))
        if run.use_heads:
            a = np.asarray(M.draw_readout(k_head, jnp.asarray((W ** 2).sum(1) / D), jax.tree_util.tree_map(lambda v: v[0], p)))
        for _ in range(run.n_steps):
            MW = mask * W
            r = a @ MW / pN - w
            W = W - c * a[:, None] * mask * r[None, :]
            if run.learn_a:
                a = a - c * (MW @ r) / D
            if run.do_proj:
                Q = (W ** 2).sum(1) / D
                A2 = 0.5 * (d0 * run.eta + np.sqrt((d0 * run.eta) ** 2 + 4 * a * a * Q))
                f = np.sqrt((A2 - d0 * run.eta) / Q)
                a, W = np.sign(a) * np.sqrt(A2), W * f[:, None]
                heads = {j: h / f for j, h in heads.items()}
        if run.use_heads:
            heads[t] = a.copy()
        if t >= k:
            wk, mk = (np.asarray(v) for v in S.probe_mask(key, k, N, D, jnp.float64(pN)))
            ak = heads[k] if run.use_heads else a
            out.append(float(((ak @ (mk * W)) / pN) @ wk / (wk @ wk)))
    return np.array(out)


def test_matches_numpy_loop():
    runs = _runs(1.5, seeds=1)
    T, k = 6, 2
    got = S.signal_after_tasks(runs, k, T - k, k, chunk=2)
    for j, r in enumerate(runs):
        np.testing.assert_allclose(got[:, j], _numpy_stream(r, T, k), atol=1e-10, rtol=0, err_msg=r.readout)
    assert np.abs(got).max() > 0.1                                         # the tasks were learned


def test_closed_form_matches_steps():
    runs = _runs(1.5, readout=["gated_frozen"])
    assert all(r.closed for r in runs)
    st, p = init_batch(runs)
    a = S.train_kernel(runs[0].n_steps, 3, True)(st, p)
    st, p = init_batch(runs)
    b = S.train_kernel(runs[0].n_steps, 3, False)(st, p)
    np.testing.assert_allclose(a.W, b.W, atol=1e-11, rtol=0)


def test_first_and_steady_signal_matches_signal_after_tasks():
    runs = _runs(1.5, seeds=1)
    T, n = 4, 3
    got = S.first_and_steady_signal(runs, T, n, chunk=2)
    np.testing.assert_allclose(got["S0"], S.signal_after_tasks(runs, 0, T, 0, chunk=3), atol=1e-11, rtol=0)
    np.testing.assert_allclose(got["ST"], S.signal_after_tasks(runs, T, n, T, chunk=3), atol=1e-11, rtol=0)
