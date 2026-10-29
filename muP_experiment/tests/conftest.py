import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import jax
import jax.numpy as jnp
import pytest

import mup  # noqa: F401  (enables x64)
from mup.config import Config, make_runs
from mup.model import Params, init_state


def params(readout="gated_shared", delta0=1.0, eta=0.1, h=0.05, pN=3.0):
    from mup.config import READOUTS
    gt, la, pr, uh = READOUTS[readout]
    f = lambda v: jnp.asarray(v, dtype=jnp.float64)
    return Params(Delta0=f(delta0 / eta), eta=f(eta), h=f(h), pN=f(pN), lam=f(1.0),
                  gated=f(gt), learn_a=f(la), do_proj=f(pr), use_heads=f(uh))


def state_with_task(seed=0, N=10, D=40, P=5, readout="gated_shared", **kw):
    """A state with a random teacher, gate and (for heads) one stored head."""
    from mup.model import draw_task, draw_readout, Q_diag
    p = params(readout=readout, **kw)
    key = jax.random.PRNGKey(seed)
    st = init_state(key, p, N, D, P)
    k1, k2 = jax.random.split(jax.random.fold_in(key, 7))
    w, g = draw_task(k1, N, D, p.pN, p.gated)
    st = st._replace(w=w, g=g)
    if readout == "task_heads":
        st = st._replace(heads=st.heads.at[0].set(st.a), a=draw_readout(k2, Q_diag(st.W), p))
        st = st._replace(probes_g=st.probes_g.at[0].set(g), probes_w=st.probes_w.at[0].set(w))
    return st, p


@pytest.fixture
def small_runs():
    from mup.config import READOUT_ORDER
    cfg = Config(N=[10], readout=list(READOUT_ORDER), delta0=[1.0], lam=[1.0], seeds=1, T_factor=0.5, T_min=5)
    return one_batch(make_runs(cfg))


def one_batch(runs):
    """Give the ungated runs the gated runs' κ and T so all five readouts share one vmap batch.

    κ does not enter an ungated run's dynamics (pN = N, g = 1); it only sets T and the age schedule.
    """
    import dataclasses
    g = next(r for r in runs if r.gated == 1.0)
    return [r if r.gated == 1.0 else dataclasses.replace(r, kappa=g.kappa, kappa_label=g.kappa_label,
                                                         T_burn=g.T_burn, T_meas=g.T_meas) for r in runs]
