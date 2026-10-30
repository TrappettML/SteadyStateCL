"""u_c unchanged to 1e-13; Δ_i = Δ₀ to 1e-13; stored heads keep every stored u_k."""
import jax
import numpy as np
from conftest import state_with_task

from mup import model
from mup.observe import evaluate_probes
import jax.numpy as jnp


def _perturb(st, p):
    # a few unprojected SGD steps move Δ_i off Δ₀
    for i in range(5):
        st = model.sgd_step(st, p, jax.random.PRNGKey(i))
    return st


def test_projection_preserves_uc_and_pins_delta():
    for readout in ("gated_shared", "learned_shared", "task_heads"):
        st, p = state_with_task(seed=3, N=16, D=64, readout=readout, delta0=-0.5, eta=0.2)
        st = _perturb(st, p)
        act = np.asarray(st.g) == 1
        before = np.asarray(model.u_c(st, p))
        assert np.abs(np.asarray(model.imbalance(st, p)) - float(p.Delta0))[act].max() > 1e-6
        new = model.project(st, p)
        after = np.asarray(model.u_c(new, p))
        np.testing.assert_allclose(after, before, atol=1e-13, rtol=0)
        d = np.asarray(model.imbalance(new, p))
        assert np.abs(d - float(p.Delta0))[act].max() < 1e-13
        # inactive rows untouched
        assert np.array_equal(np.asarray(new.W)[~act], np.asarray(st.W)[~act])
        assert np.array_equal(np.asarray(new.a)[~act], np.asarray(st.a)[~act])


def test_projection_is_noop_when_disabled():
    st, p = state_with_task(seed=5, readout="gated_frozen")
    st = _perturb(st, p)
    new = model.project(st, p)
    assert np.array_equal(np.asarray(new.W), np.asarray(st.W))
    assert np.array_equal(np.asarray(new.a), np.asarray(st.a))


def test_stored_heads_keep_stored_uk():
    st, p = state_with_task(seed=8, N=16, D=64, readout="task_heads", delta0=2.0, eta=0.5)
    st = _perturb(st, p)
    ages = jnp.asarray([0])
    st = st._replace(t=jnp.int32(0))
    u_before = (st.probes_g[0] * st.heads[0]) @ st.W / p.pN
    new = model.project(st, p)
    u_after = (new.probes_g[0] * new.heads[0]) @ new.W / p.pN
    np.testing.assert_allclose(np.asarray(u_after), np.asarray(u_before), atol=1e-13, rtol=0)
    s0, e0, _ = evaluate_probes(st, p, ages)
    s1, e1, _ = evaluate_probes(new, p, ages)
    np.testing.assert_allclose(np.asarray(s1), np.asarray(s0), atol=1e-13, rtol=0)


def test_init_pins_delta_and_a_nonzero():
    st, p = state_with_task(seed=9, N=30, D=120, delta0=-0.75, eta=0.1)
    d = np.asarray(model.imbalance(st, p))
    np.testing.assert_allclose(d, float(p.Delta0), atol=1e-12, rtol=0)
    assert (np.asarray(st.a) != 0).all()


def test_projection_pins_delta_for_negative_delta0_with_tiny_a():
    """Δ₀ < 0 with a_i → 0 (learned shared at δ₀ < 0, Stage 1): the pin must hold to rounding, not to ε Δ₀²/a²."""
    st, p = state_with_task(seed=1, N=16, D=64, readout="learned_shared", delta0=-0.5, eta=1.0)
    for a_scale in (1e-3, 1e-5, 1e-8):
        tiny = st._replace(a=st.a * a_scale)
        before = np.asarray(model.u_c(tiny, p))
        new = model.project(tiny, p)
        np.testing.assert_allclose(np.asarray(model.u_c(new, p)), before, atol=1e-13, rtol=0)
        d = np.asarray(model.imbalance(new, p))
        assert np.abs(d - float(p.Delta0)).max() < 1e-12, (a_scale, np.abs(d - float(p.Delta0)).max())
