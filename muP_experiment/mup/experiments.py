"""Batched kernels and job functions of the experiment scripts (PLAN.md §12.10–§12.13, §7.7).

Every script in ``scripts/`` that trains networks reduces to jobs over
``run_jobs`` (``mup.pool``): a job takes a list of ``config.Run`` sharing
``(N, D, solver, n_steps)`` — every readout, κ and seed of a cell or of an
(N, T) group together — and runs them as ONE ``vmap`` batch, scanning
``chunk`` tasks per device call (``device.default_chunk``: 16 on a GPU, 1 on
the CPU) exactly as ``protocol.run_stream`` does for the sweep.

``train_task`` is ``protocol.run_task`` without the observables: the same
draws, the same ``n_steps`` of step + row projection, the same head
rescaling and storing, so a state advanced here equals one advanced by the
sweep kernel (``tests/test_experiments.py``).  The per-task observables of
each experiment are evaluated inside the scan and come back stacked, one
host transfer per chunk.

Infeasible runs (δ₀ ≤ −min_i Q0_i, ``protocol.feasible``) are dropped from
the batch and come back as NaN columns, so every result keeps the order of
the job's runs.
"""
from __future__ import annotations

import functools
from typing import Callable, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

import mup._x64  # noqa: F401  (enables x64)
from mup import device, model
from mup.model import Params, State
from mup.observe import Moments, curves
from mup.protocol import _step_and_project, _step_fn, feasible, make_params, batched_init, run_stream


# ----------------------------------------------------------------------------
# one task without observables
# ----------------------------------------------------------------------------
def train_task(state: State, p: Params, *, n_steps: int, solver: str) -> State:
    """One task of one run — ``protocol.run_task``'s dynamics without the probes and norms.

    Parameters
    ----------
    state : State
    p : Params
    n_steps : int (static)
    solver : {"pop", "sgd"} (static)

    Returns
    -------
    State
        After the task: ``w``, ``g``, ``a`` of the task, the stored heads
        rescaled by the task's projection factors and the task's head
        stored at slot ``t`` (task heads), ``t + 1``.
    """
    N, D = state.W.shape
    k_teach, k_head, k_stream = model.task_keys(state.key, state.t)
    w_t, g_t = model.draw_task(k_teach, N, D, p.pN, p.gated)
    head = model.draw_readout(k_head, model.Q_diag(state.W), p)
    a = jnp.where(p.use_heads == 1.0, head, state.a)
    state = state._replace(w=w_t, g=g_t, a=a)
    t = state.t
    step = _step_fn(p, solver, k_stream)
    ones = jnp.ones((N,), dtype=state.W.dtype)

    def body(i, c):
        s, fcum = c
        s, f, _ = _step_and_project(s, p, step, i)
        return s, fcum * f

    state, fcum = jax.lax.fori_loop(0, n_steps - 1, body, (state, ones))
    state = step(state, n_steps - 1)
    state, f = model.project_rows(state, p)
    state = model.rescale_heads(state, p, fcum * f)
    state = state._replace(heads=state.heads.at[t].set(jnp.where(p.use_heads == 1.0, state.a, 0.0)))
    return state._replace(t=t + 1)


def probe_signal(state: State, p: Params, k) -> jnp.ndarray:
    """S_k = u·w_k*/|w_k*|² of probe ``k`` (its teacher, gate and stored head or the shared readout) against the current W."""
    N, D = state.W.shape
    wk, gk = model.probe_task(state.key, k, N, D, p.pN, p.gated)
    ak = jnp.where(p.use_heads == 1.0, state.heads[k], state.a)
    u = (gk * ak) @ state.W / p.pN
    return (u * wk).sum() / (wk * wk).sum()


#: Per-task norms of ``run_weight_norms.py`` in output order.
NORM_KEYS = ("Q_all", "Q_act", "a2_all", "a2_act", "u2")


def task_norms(state: State, p: Params) -> jnp.ndarray:
    """(5,) ``NORM_KEYS`` after a task: mean Q_ii and a_i² over all / active units, |u_t|²/D of the task just trained."""
    Q = model.Q_diag(state.W)
    a2 = state.a * state.a
    g = state.g
    n_act = g.sum()
    act = lambda x: jnp.where(n_act > 0, (g * x).sum() / jnp.maximum(n_act, 1.0), jnp.nan)
    u = (g * state.a) @ state.W / p.pN
    return jnp.stack([Q.mean(), act(Q), a2.mean(), act(a2), (u @ u) / state.W.shape[1]])


def imbalance_stats(state: State, p: Params) -> jnp.ndarray:
    """(4,) mean, min, max of Δ_i over the active units and max |Δ_i − Δ₀| (``run_gauge_comp.py``)."""
    g = state.g
    d = model.imbalance(state, p)
    n = jnp.maximum(g.sum(), 1.0)
    big = jnp.where(g == 1.0, d, -jnp.inf)
    small = jnp.where(g == 1.0, d, jnp.inf)
    return jnp.stack([(g * d).sum() / n, small.min(), big.max(), jnp.abs(jnp.where(g == 1.0, d - p.Delta0, 0.0)).max()])


def gauge_task(state: State, p: Params, *, n_steps: int, solver: str) -> tuple[State, jnp.ndarray]:
    """One task returning ``imbalance_stats`` after every step, (n_steps, 4); the loop of ``train_task``."""
    N, D = state.W.shape
    k_teach, k_head, k_stream = model.task_keys(state.key, state.t)
    w_t, g_t = model.draw_task(k_teach, N, D, p.pN, p.gated)
    head = model.draw_readout(k_head, model.Q_diag(state.W), p)
    a = jnp.where(p.use_heads == 1.0, head, state.a)
    state = state._replace(w=w_t, g=g_t, a=a)
    t = state.t
    step = _step_fn(p, solver, k_stream)

    def body(s, i):
        s, _, _ = _step_and_project(s, p, step, i)
        return s, imbalance_stats(s, p)

    state, rows = jax.lax.scan(body, state, jnp.arange(n_steps, dtype=jnp.int32))
    return state._replace(t=t + 1), rows


# ----------------------------------------------------------------------------
# chunked kernels: jit(scan over n_tasks of vmap(task + observable))
# ----------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def train_kernel(n_steps: int, solver: str, n_tasks: int) -> Callable:
    """``f(state, p) -> state`` advancing every run by ``n_tasks`` tasks (no outputs); state donated."""
    vf = jax.vmap(functools.partial(train_task, n_steps=n_steps, solver=solver))

    def run(state, p):
        return jax.lax.scan(lambda s, _: (vf(s, p), None), state, None, length=n_tasks)[0]

    return jax.jit(run, donate_argnums=(0,))


@functools.lru_cache(maxsize=None)
def signal_kernel(n_steps: int, solver: str, n_tasks: int) -> Callable:
    """``f(state, p, k) -> (state, S)`` with ``S`` (n_tasks, B) the signal of probe ``k`` after each task."""
    vf = jax.vmap(functools.partial(train_task, n_steps=n_steps, solver=solver))
    vs = jax.vmap(probe_signal, in_axes=(0, 0, None))

    def run(state, p, k):
        def body(s, _):
            s = vf(s, p)
            return s, vs(s, p, k)
        return jax.lax.scan(body, state, None, length=n_tasks)

    return jax.jit(run, donate_argnums=(0,))


@functools.lru_cache(maxsize=None)
def norms_kernel(n_steps: int, solver: str, n_tasks: int) -> Callable:
    """``f(state, p) -> (state, X)`` with ``X`` (n_tasks, B, 5) the ``NORM_KEYS`` after each task."""
    vf = jax.vmap(functools.partial(train_task, n_steps=n_steps, solver=solver))
    vn = jax.vmap(task_norms)

    def run(state, p):
        def body(s, _):
            s = vf(s, p)
            return s, vn(s, p)
        return jax.lax.scan(body, state, None, length=n_tasks)

    return jax.jit(run, donate_argnums=(0,))


@functools.lru_cache(maxsize=None)
def gauge_kernel(n_steps: int, solver: str, n_tasks: int) -> Callable:
    """``f(state, p) -> (state, R)`` with ``R`` (n_tasks, B, n_steps, 4) the per-step imbalance statistics."""
    vf = jax.vmap(functools.partial(gauge_task, n_steps=n_steps, solver=solver))

    def run(state, p):
        return jax.lax.scan(lambda s, _: vf(s, p), state, None, length=n_tasks)

    return jax.jit(run, donate_argnums=(0,))


def chunks(T: int, chunk: Optional[int] = None):
    """Yield the device-call sizes covering ``T`` tasks (``device.default_chunk``)."""
    c = max(1, min(device.default_chunk(chunk), T))
    t = 0
    while t < T:
        n = min(c, T - t)
        yield n
        t += n


def _batch(runs: Sequence, P: int) -> tuple[State, Params, np.ndarray]:
    """Feasible runs of ``runs`` as one batch with ``P`` head slots: (state, params, mask of kept runs)."""
    r0 = runs[0]
    for r in runs:
        assert (r.N, r.D, r.solver, r.n_steps) == (r0.N, r0.D, r0.solver, r0.n_steps), "job is not homogeneous"
    ok = feasible(runs)
    kept = [r for r, f in zip(runs, ok) if f]
    if not kept:
        return None, None, ok
    p = make_params(kept)
    keys = jnp.stack([jax.random.PRNGKey(r.rng_seed()) for r in kept])
    return batched_init(r0.N, r0.D, P)(keys, p), p, ok


def _expand(X: np.ndarray, ok: np.ndarray, axis: int = 1) -> np.ndarray:
    """Put the rows of the kept runs back at their positions (NaN for the dropped ones) along ``axis``."""
    shape = list(X.shape)
    shape[axis] = len(ok)
    out = np.full(shape, np.nan)
    idx = [slice(None)] * X.ndim
    idx[axis] = np.flatnonzero(ok)
    out[tuple(idx)] = X
    return out


# ----------------------------------------------------------------------------
# job functions (``"mup.experiments:<name>"`` for ``pool.run_jobs``)
# ----------------------------------------------------------------------------
def signal_after_tasks(runs: Sequence, T_pre: int, n_follow: int, k: int, chunk: Optional[int] = None) -> np.ndarray:
    """Train ``T_pre`` tasks, then ``n_follow`` more recording probe ``k``'s signal after each: (n_follow, B).

    ``run_szero.py``: ``T_pre = 0, k = 0``; ``run_stask.py`` / ``run_Nscaling.py``:
    ``T_pre = T, k = T``.  NaN columns for infeasible runs.
    """
    r0 = runs[0]
    state, p, ok = _batch(runs, T_pre + n_follow)
    if state is None:
        return np.full((n_follow, len(runs)), np.nan)
    for n in chunks(T_pre, chunk):
        state = train_kernel(r0.n_steps, r0.solver, n)(state, p)
    S = []
    kk = jnp.int32(k)
    for n in chunks(n_follow, chunk):
        state, s = signal_kernel(r0.n_steps, r0.solver, n)(state, p, kk)
        S.append(np.asarray(s))
    return _expand(np.concatenate(S, 0), ok)


def norms_over_tasks(runs: Sequence, T: int, chunk: Optional[int] = None) -> dict:
    """``{key: (B, T) array}`` of ``NORM_KEYS`` after each of ``T`` tasks (``run_weight_norms.py``)."""
    r0 = runs[0]
    state, p, ok = _batch(runs, T)
    if state is None:
        return {q: np.full((len(runs), T), np.nan) for q in NORM_KEYS}
    X = []
    for n in chunks(T, chunk):
        state, x = norms_kernel(r0.n_steps, r0.solver, n)(state, p)
        X.append(np.asarray(x))
    X = _expand(np.concatenate(X, 0), ok)                        # (T, B, 5)
    return {q: X[:, :, i].T for i, q in enumerate(NORM_KEYS)}


def imbalance_over_steps(runs: Sequence, do_proj: Sequence[float], n_tasks: int, chunk: Optional[int] = None) -> np.ndarray:
    """(len(runs), n_tasks · n_steps, 4) per-step imbalance statistics with ``Params.do_proj`` forced per run.

    ``run_gauge_comp.py``: the runs of a cell (one per readout) each listed
    with the projection switched off and on; ``do_proj[i]`` applies to ``runs[i]``.
    """
    r0 = runs[0]
    state, p, ok = _batch(runs, n_tasks)
    if state is None:
        return np.full((len(runs), n_tasks * r0.n_steps, 4), np.nan)
    p = p._replace(do_proj=jnp.asarray([float(d) for d, f in zip(do_proj, ok) if f], dtype=jnp.float64))
    R = []
    for n in chunks(n_tasks, chunk):
        state, r = gauge_kernel(r0.n_steps, r0.solver, n)(state, p)
        R.append(np.asarray(r))                                   # (n, B, n_steps, 4)
    R = np.concatenate(R, 0)
    R = np.transpose(R, (1, 0, 2, 3)).reshape(R.shape[1], -1, 4)
    return _expand(R, ok, axis=0)


def stream_batch(runs: Sequence, chunk: Optional[int] = None) -> dict:
    """``protocol.run_stream`` of one sweep batch (``validate.py``)."""
    return run_stream(runs, chunk=chunk)


def pooled_curves(batch: Sequence, res: dict) -> dict:
    """Pool a batch's moments over seeds: ``{(N, readout, eta): (ages, S, E, cv_s, S_se)}`` (``validate.py``)."""
    mom = {f: res[f] for f in Moments.FIELDS}
    per_seed = curves(mom)["S"]
    out = {}
    for key in {(r.N, r.readout, r.eta) for r in batch}:
        idx = [i for i, r in enumerate(batch) if (r.N, r.readout, r.eta) == key]
        s = {f: mom[f][idx].sum((0, 2)) for f in Moments.FIELDS}
        n = s["count"]
        with np.errstate(invalid="ignore", divide="ignore"):
            S, E = s["sum_s"] / n, s["sum_e"] / n
            cv = np.sqrt(np.maximum(s["sum_s2"] / n - S**2, 0)) / np.abs(S)
        se = per_seed[idx].std(0, ddof=1) / np.sqrt(len(idx)) if len(idx) > 1 else np.full_like(S, np.nan)
        out[key] = (res["ages"], S, E, cv, se)
    return out


# ----------------------------------------------------------------------------
# grouping runs into jobs
# ----------------------------------------------------------------------------
def group_key(r) -> tuple:
    """Runs that can share one vmap batch here: (N, D, solver, n_steps)."""
    return (r.N, r.D, r.solver, r.n_steps)


def run_cost(r, T: int) -> float:
    """Rough device time of one run over ``T`` tasks (N·D per step)."""
    return float(r.N) * r.D * r.n_steps * T


def split(runs: Sequence, pieces: int) -> list[list]:
    """Split ``runs`` into ≤ ``pieces`` nearly equal contiguous parts (none empty)."""
    pieces = max(1, min(pieces, len(runs)))
    bounds = np.linspace(0, len(runs), pieces + 1).round().astype(int)
    return [list(runs[a:b]) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]
