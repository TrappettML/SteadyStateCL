"""Without projection, the population flow drifts Δ_i by O(h²) per step, SGD by O(η²)."""
import jax
import numpy as np
from conftest import state_with_task

from mup import model


def _drift_pop(h, n_alpha=0.5):
    st, p = state_with_task(seed=11, N=10, D=40, readout="gated_shared", delta0=1.0, eta=1.0, h=h)
    p = p._replace(do_proj=p.do_proj * 0)
    d0 = np.asarray(model.imbalance(st, p))
    for _ in range(int(round(n_alpha / h))):
        st = model.pop_step(st, p)
    act = np.asarray(st.g) == 1
    return np.abs(np.asarray(model.imbalance(st, p)) - d0)[act].max()


def test_pop_drift_scales_as_h2_per_step():
    # over a fixed α the total drift is O(h) (= steps × h²)
    d1, d2, d4 = _drift_pop(0.04), _drift_pop(0.02), _drift_pop(0.01)
    assert d1 > d2 > d4
    assert 1.6 < d1 / d2 < 2.4
    assert 1.6 < d2 / d4 < 2.4


def _drift_sgd(eta, n_steps=400):
    st, p = state_with_task(seed=12, N=10, D=40, readout="gated_shared", delta0=1.0, eta=eta)
    p = p._replace(do_proj=p.do_proj * 0)
    d0 = np.asarray(st.a) ** 2 - np.asarray(model.Q_diag(st.W))      # η Δ_i = a² − Q
    for i in range(n_steps):
        st = model.sgd_step(st, p, jax.random.fold_in(jax.random.PRNGKey(1), i))
    act = np.asarray(st.g) == 1
    return np.abs(np.asarray(st.a) ** 2 - np.asarray(model.Q_diag(st.W)) - d0)[act].max()


def test_sgd_drift_scales_as_eta2():
    # same input stream, same number of steps: drift of a² − Q is O(η²)
    d1, d2, d4 = _drift_sgd(0.04), _drift_sgd(0.02), _drift_sgd(0.01)
    assert d1 > d2 > d4
    assert 3.0 < d1 / d2 < 5.0
    assert 3.0 < d2 / d4 < 5.0
