"""Test 9 (PLAN.md §12.2): a traced run reproduces the untraced one bit-for-bit."""
import jax.numpy as jnp
import numpy as np
from conftest import small_runs  # noqa: F401

from mup.observe import age_schedule
from mup.protocol import SCALARS, TRACE_QUANT, batched_run_task, init_batch, trace_indices


def test_trace_matches_untraced(small_runs):
    runs = small_runs
    T = runs[0].T_burn + runs[0].T_meas
    ages = jnp.asarray(age_schedule(runs[0].N, runs[0].kappa, T, runs[0].n_ages_geom, runs[0].m_max_factor))
    tidx = jnp.asarray(trace_indices(runs[0].n_steps, runs[0].solver, 1000))
    assert len(tidx) == runs[0].n_steps
    st_a, p = init_batch(runs)
    st_b, _ = init_batch(runs)
    fa = batched_run_task(runs[0].n_steps, runs[0].solver, False)
    fb = batched_run_task(runs[0].n_steps, runs[0].solver, True)
    for _ in range(4):
        st_a, oa = fa(st_a, p, ages)
        st_b, ob = fb(st_b, p, ages, tidx)
        for name, x, y in zip(st_a._fields, st_a, st_b):
            assert np.array_equal(np.asarray(x), np.asarray(y)), name
        for k in SCALARS:
            assert np.array_equal(np.asarray(oa[k]), np.asarray(ob[k])), k
        tr = np.asarray(ob["trace"])
        assert tr.shape == (len(runs), runs[0].n_steps, len(TRACE_QUANT))
        # last traced step equals the per-task scalars of the same state
        assert np.allclose(tr[:, -1, TRACE_QUANT.index("Q_act_post")], np.asarray(ob["Q_act_post"]))
        assert np.allclose(tr[:, -1, TRACE_QUANT.index("Q_act_pre")], np.asarray(ob["Q_act_pre"]))
        assert np.allclose(tr[:, -1, TRACE_QUANT.index("loss")], np.asarray(ob["L_end"]))
        assert np.allclose(tr[:, :, TRACE_QUANT.index("logf")].sum(1), np.asarray(ob["logf_sum"]))


def test_trace_indices():
    assert list(trace_indices(10, "pop", 200)) == list(range(10))
    i = trace_indices(30000, "sgd", 200)
    assert i[0] == 0 and i[-1] == 29999 and np.all(np.diff(i) > 0) and len(i) <= 200
    j = trace_indices(1000, "pop", 200)
    assert j[0] == 0 and j[-1] == 999 and len(j) == 200
