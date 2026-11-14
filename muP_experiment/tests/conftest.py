import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.dirname(ROOT)):                  # the experiment, and the repo root that holds sscl_core/
    if _p not in sys.path:
        sys.path.insert(0, _p)

# On a GPU node the test process must not grab 75 % of every card: the GPU tests
# start one subprocess per card beside it.  Must precede the first ``import jax``.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import pytest

import sscl_core._x64  # noqa: F401  (enables x64)
from sscl_core.config import Config, make_runs
from sscl_core.model import Params, init_state


def params(readout="gated_shared", delta0=1.0, eta=0.1, h=0.05, pN=3.0, clock=0.0, lam=1.0):
    from sscl_core.config import READOUTS
    gt, la, pr, uh = READOUTS[readout]
    f = lambda v: jnp.asarray(v, dtype=jnp.float64)
    return Params(Delta0=f(delta0 / eta), eta=f(eta), h=f(h), pN=f(pN), lam=f(lam),
                  gated=f(gt), learn_a=f(la), do_proj=f(pr), use_heads=f(uh), sigma0=f(1.0), clock=f(clock))


def state_with_task(seed=0, N=10, D=40, P=5, readout="gated_shared", **kw):
    """A state with the teacher and gate of task 0 (so probe 0 regenerates to them) and, for heads, one stored head."""
    from sscl_core.model import draw_readout, probe_task, task_keys, Q_diag
    p = params(readout=readout, **kw)
    key = jax.random.PRNGKey(seed)
    st = init_state(key, p, N, D, P)
    w, g = probe_task(key, 0, N, D, p.pN, p.gated)
    st = st._replace(w=w, g=g)
    if readout == "task_heads":
        st = st._replace(heads=st.heads.at[0].set(st.a), a=draw_readout(task_keys(key, 0)[1], Q_diag(st.W), p))
    return st, p


@pytest.fixture
def small_runs():
    from sscl_core.config import READOUT_ORDER
    cfg = Config(solver="pop", N=[10], readout=list(READOUT_ORDER), delta0=[1.0], lam=[1.0], seeds=1, T_factor=0.5, T_min=5)
    return one_batch(make_runs(cfg))


def one_batch(runs):
    """Give the ungated runs the gated runs' κ and T so all five readouts share one vmap batch.

    κ does not enter an ungated run's dynamics (pN = N, g = 1); it only sets T and the age schedule.
    """
    import dataclasses
    g = next(r for r in runs if r.gated == 1.0)
    return [r if r.gated == 1.0 else dataclasses.replace(r, kappa=g.kappa, kappa_label=g.kappa_label, kappa_sched=g.kappa_sched,
                                                         T_burn=g.T_burn, T_meas=g.T_meas) for r in runs]
