"""A batch of 5 runs (one per readout) equals five single runs with the same keys.

The plan asks for bit-for-bit equality.  XLA's CPU backend reduces the batched
``(B, N, D)`` matmul in a different order from the single ``(N, D)`` matvec, so
the two paths differ by one ulp per step (measured 4e-16 absolute after one
task at N = 10).  Draws (teacher, gate, inputs) and all integer outputs are
compared exactly; floats to 1e-10 absolute, far below any physical effect.
"""
import functools

import jax
import jax.numpy as jnp
import numpy as np
from conftest import small_runs  # noqa: F401

from mup import model
from mup.observe import age_schedule
from mup.protocol import init_batch, run_task, batched_run_task, run_stream


def _single(run, T, ages):
    p = jax.tree_util.tree_map(lambda x: x[0], init_batch([run])[1])
    st = model.init_state(jax.random.PRNGKey(run.rng_seed()), p, run.N, run.D, T)
    f = jax.jit(functools.partial(run_task, n_steps=run.n_steps, solver=run.solver))
    outs = []
    for _ in range(T):
        st, out = f(st, p, ages)
        outs.append(out)
    return st, outs


def test_batch_equals_singles(small_runs):
    runs = small_runs
    assert [r.readout for r in runs] == ["gated_shared", "learned_shared", "gated_frozen", "frozen", "task_heads"]
    T = runs[0].T_burn + runs[0].T_meas
    ages = jnp.asarray(age_schedule(runs[0].N, runs[0].kappa, T, runs[0].n_ages_geom, runs[0].m_max_factor))
    st, p = init_batch(runs)
    f = batched_run_task(runs[0].n_steps, runs[0].solver)
    bouts = []
    for _ in range(T):
        st, out = f(st, p, ages)
        bouts.append(out)
    for i, r in enumerate(runs):
        st1, outs = _single(r, T, ages)
        sb = jax.tree_util.tree_map(lambda x: x[i], st)
        for name, a, b in zip(st1._fields, st1, sb):
            a, b = np.asarray(a), np.asarray(b)
            if name in ("g", "w", "probes_w", "probes_g", "key", "t"):
                assert np.array_equal(a, b), name          # draws are exact
            else:
                np.testing.assert_allclose(a, b, atol=1e-10, rtol=0, err_msg=name)
        for o1, ob in zip(outs, bouts):
            for k in o1:
                a, b = np.asarray(o1[k]), np.asarray(ob[k][i])
                if k in ("valid", "n_active"):
                    assert np.array_equal(a, b), k
                else:
                    np.testing.assert_allclose(a, b, atol=1e-10, rtol=0, err_msg=k)


def test_run_stream_runs(small_runs):
    res = run_stream(small_runs)
    assert res["count"].shape[0] == 5
    assert np.isfinite(res["E0"]).all()
    ros = [r.readout for r in small_runs]
    # frozen readouts: a never moves, so a² mean is constant across tasks
    for ro in ("gated_frozen", "frozen"):
        assert np.ptp(res["a2_mean"][:, ros.index(ro)]) == 0.0
    # ungated runs: every unit active
    for ro in ("learned_shared", "frozen"):
        assert np.all(res["n_active"][:, ros.index(ro)] == 10)
