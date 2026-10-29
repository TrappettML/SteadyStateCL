"""(sgdW), (sgda) equal pN·η·∇_W ℓ and (pN·η/D)·∇_a ℓ of the literal loss."""
import jax
import jax.numpy as jnp
import numpy as np
from conftest import state_with_task

from mup import model


def test_sgd_step_matches_autograd():
    st, p = state_with_task(seed=1, N=12, D=48, readout="gated_shared", eta=0.3)
    k = jax.random.PRNGKey(99)
    x = jax.random.normal(k, (48,), dtype=jnp.float64)
    gW, ga = jax.grad(lambda W, a: model.sample_loss(st._replace(W=W, a=a), p, x), argnums=(0, 1))(st.W, st.a)
    new = model.sgd_step(st, p, k)
    D = 48
    np.testing.assert_allclose(st.W - new.W, p.pN * p.eta * gW, rtol=0, atol=1e-12)
    np.testing.assert_allclose(st.a - new.a, p.pN * p.eta / D * ga, rtol=0, atol=1e-12)


def test_frozen_readout_does_not_move():
    st, p = state_with_task(seed=2, readout="gated_frozen")
    new = model.sgd_step(st, p, jax.random.PRNGKey(3))
    assert np.array_equal(np.asarray(new.a), np.asarray(st.a))
    new = model.pop_step(st, p)
    assert np.array_equal(np.asarray(new.a), np.asarray(st.a))


def test_inactive_units_receive_zero():
    st, p = state_with_task(seed=4, N=20, D=80)
    new = model.sgd_step(st, p, jax.random.PRNGKey(5))
    off = np.asarray(st.g) == 0
    assert off.any() and (~off).any()
    assert np.array_equal(np.asarray(new.W)[off], np.asarray(st.W)[off])
    assert np.array_equal(np.asarray(new.a)[off], np.asarray(st.a)[off])
    new = model.pop_step(st, p)
    assert np.array_equal(np.asarray(new.W)[off], np.asarray(st.W)[off])


def test_ungated_gate_is_all_ones():
    for ro in ("learned_shared", "frozen"):
        st, p = state_with_task(seed=4, N=20, D=80, readout=ro)
        assert np.all(np.asarray(st.g) == 1.0)
    st, p = state_with_task(seed=4, N=20, D=80, readout="gated_shared")
    assert 0 < np.asarray(st.g).sum() < 20


def test_pop_step_is_expected_sgd_step():
    """E_x[SGD update] = population update at h = 1/D (Monte-Carlo, loose)."""
    st, p = state_with_task(seed=6, N=8, D=32, eta=0.2, h=1.0 / 32)
    keys = jax.random.split(jax.random.PRNGKey(0), 200_000)
    dW = jax.vmap(lambda k: model.sgd_step(st, p, k).W - st.W)(keys).mean(0)
    pop = model.pop_step(st, p).W - st.W
    np.testing.assert_allclose(np.asarray(dW), np.asarray(pop), atol=3e-3, rtol=0)
