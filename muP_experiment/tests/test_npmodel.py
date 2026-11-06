"""``mup.npmodel.step_task`` (NumPy, in place on the active rows) against the JAX kernels of ``mup.model``."""
import numpy as np
import pytest

import mup._x64  # noqa: F401  (enables x64)
import jax
import jax.numpy as jnp

from mup import model, npmodel


class _FixedX:
    """Stand-in rng returning a given sequence of SGD inputs x."""
    def __init__(self, xs):
        self.xs = list(xs)

    def standard_normal(self, n):
        x = self.xs.pop(0)
        assert x.shape == (n,)
        return x


@pytest.mark.parametrize("solver", ["pop", "sgd"])
@pytest.mark.parametrize("Delta0", [2.0, 0.0, -0.3])
@pytest.mark.parametrize("gate", ["sparse", "full", "empty"])
@pytest.mark.parametrize("learn_a,do_proj", [(1.0, 1.0), (0.0, 0.0), (1.0, 0.0)])
def test_step_task_matches_model(solver, Delta0, gate, learn_a, do_proj):
    N, D, n_steps, eta, h, pN = 12, 40, 3, 0.7, 0.1, 3.0
    rng = np.random.default_rng(1)
    W0 = rng.standard_normal((N, D))
    a0 = rng.standard_normal(N) + 1.0        # |a| > 0 so the projection is well defined
    g = {"sparse": (rng.uniform(size=N) < 0.4).astype(float), "full": np.ones(N), "empty": np.zeros(N)}[gate]
    w = rng.standard_normal(D)
    xs = [rng.standard_normal(D) for _ in range(n_steps)]

    # reference: n_steps of model step + project_rows, as protocol.run_task does
    p = model.Params(*[jnp.asarray(v, dtype=jnp.float64)
                       for v in (Delta0, eta, h, pN, 4.0, 1.0, learn_a, do_proj, 0.0, 1.0)])
    st = model.State(W=jnp.asarray(W0), a=jnp.asarray(a0), g=jnp.asarray(g), w=jnp.asarray(w), heads=jnp.zeros((1, N)),
                     key=jax.random.PRNGKey(0), t=jnp.int32(0))
    fcum_ref = jnp.ones(N)
    for i in range(n_steps):
        if solver == "sgd":
            # same x as the NumPy side: model.sgd_step draws its own x, so re-implement it with the given x
            x = jnp.asarray(xs[i])
            hx = st.W @ x / jnp.sqrt(D)
            ga = st.g * st.a
            e = ga @ hx / p.pN - st.w @ x / jnp.sqrt(D)
            st = st._replace(W=st.W - p.eta * ga[:, None] * e * x[None, :] / jnp.sqrt(D),
                             a=st.a - p.learn_a * p.eta / D * st.g * e * hx)
        else:
            st = model.pop_step(st, p)
        st, f = model.project_rows(st, p)
        fcum_ref = fcum_ref * f

    W, a = W0.copy(), a0.copy()
    Q = npmodel.Q_diag(W0)
    fcum, u = npmodel.step_task(W, a, g, w, _FixedX(xs), Delta0, eta, h, pN, learn_a, do_proj == 1.0, n_steps, solver, Q)
    np.testing.assert_allclose(W, np.asarray(st.W), rtol=1e-12, atol=1e-13)
    # the tracked row norms and the effective weight match the full-matrix values
    np.testing.assert_allclose(Q, npmodel.Q_diag(W), rtol=1e-11, atol=1e-13)
    np.testing.assert_allclose(u, (g * a) @ W / pN, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(a, np.asarray(st.a), rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(fcum, np.asarray(fcum_ref), rtol=1e-12, atol=1e-13)
    # inactive rows are untouched bit for bit
    off = g == 0.0
    assert np.array_equal(W[off], W0[off]) and np.array_equal(a[off], a0[off]) and np.all(fcum[off] == 1.0)


def test_sgd_x_matches_model_draw():
    """The SGD step above reproduces model.sgd_step when given the x that the JAX key draws."""
    N, D = 6, 20
    rng = np.random.default_rng(2)
    W0 = rng.standard_normal((N, D)); a0 = rng.standard_normal(N); g = np.ones(N); w = rng.standard_normal(D)
    p = model.Params(*[jnp.asarray(v, dtype=jnp.float64) for v in (1.0, 0.5, 0.1, 6.0, 4.0, 0.0, 1.0, 0.0, 0.0, 1.0)])
    st = model.State(W=jnp.asarray(W0), a=jnp.asarray(a0), g=jnp.asarray(g), w=jnp.asarray(w), heads=jnp.zeros((1, N)),
                     key=jax.random.PRNGKey(0), t=jnp.int32(0))
    k = jax.random.PRNGKey(5)
    ref = model.sgd_step(st, p, k)
    x = np.asarray(jax.random.normal(k, (D,), dtype=jnp.float64))
    W, a = W0.copy(), a0.copy()
    npmodel.step_task(W, a, g, w, _FixedX([x]), 1.0, 0.5, 0.1, 6.0, 1.0, False, 1, "sgd")[0]
    np.testing.assert_allclose(W, np.asarray(ref.W), rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(a, np.asarray(ref.a), rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("readout", ["gated_shared", "learned_shared", "gated_frozen", "frozen", "task_heads"])
def test_tasks_tracks_Q_and_u(readout):
    """Over a stream of tasks the tracked Q and u of ``tasks`` equal the values recomputed from the full W."""
    from mup.config import Config, make_runs
    cfg = Config(N=[12], readout=[readout], delta0=[0.0], lam=[2.0], seeds=1, solver="pop", T_min=4)
    run = make_runs(cfg)[0]
    rng, W, a = npmodel.init_run(run)
    for t, W, a, g, w, fcum, Q, u in npmodel.tasks(run, rng, W, a, 25):
        np.testing.assert_allclose(Q, npmodel.Q_diag(W), rtol=1e-11, atol=1e-14)
        np.testing.assert_allclose(u, (g * a) @ W / run.pN, rtol=1e-11, atol=1e-14)
