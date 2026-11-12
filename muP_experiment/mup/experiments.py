"""Batched kernels and job functions of the experiment scripts (PLAN.md §12.10–§12.13, §7.7).

Every script in ``scripts/`` that trains networks reduces to jobs over
``run_jobs`` (``mup.pool``): a job takes a list of ``config.Run`` sharing
``(N, D, solver, n_steps)`` — every readout, κ and seed of a cell or of an
(N, T) group together — and runs them as ONE ``vmap`` batch, scanning
``chunk`` tasks per device call (``device.default_chunk``: 16 on a GPU, 1 on
the CPU) exactly as ``protocol.run_stream`` does for the sweep.

``train_task`` is ``protocol.run_task`` without the observables: the same
draws, the same task clock (``model.task_h``: clock (ii) of the write-up by
default), the same ``n_steps`` of step + row projection, the same head
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
import warnings
from typing import Callable, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

import mup._x64  # noqa: F401  (enables x64)
from mup import device, model, pool
from mup.model import Params, State
from mup.observe import Moments, curves
from mup.protocol import _step_and_project, _step_fn, block_of, feasible, make_params, batched_init, run_stream


# ----------------------------------------------------------------------------
# one task without observables
# ----------------------------------------------------------------------------
def train_task(state: State, p: Params, *, n_steps: int, solver: str, K: Optional[int] = None,
               closed: bool = False) -> State:
    """One task of one run — ``protocol.run_task``'s dynamics without the probes and norms.

    Parameters
    ----------
    state : State
    p : Params
    n_steps : int (static)
    solver : {"pop", "sgd"} (static)
    K, closed : static, see ``protocol.run_task`` (``protocol.block_of``)

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
    t = state.t
    r = model.train_block(state.W, a, g_t, w_t, p, k_stream, n_steps=n_steps, solver=solver,
                          K=N if K is None else K, closed=closed)
    state = state._replace(W=r["W"], a=r["a"], g=g_t, w=w_t)
    state = model.rescale_heads(state, p, r["f"])
    state = state._replace(heads=state.heads.at[t].set(jnp.where(p.use_heads == 1.0, g_t * state.a, 0.0)))
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


#: Pre-training output of the next task (``leak_norms``), recorded by ``norms_kernel`` beside ``NORM_KEYS``.
LEAK_KEYS = ("ub2", "wub", "ub_diag")

#: Start-of-task quantities of the readout exponent (``clock_norms``), recorded by ``norms_kernel``.
CLOCK_KEYS = ("n_c", "abar_c", "Qbar_c", "qmax_c", "h_c")


def clock_norms(state: State, p: Params, *, n_steps: int, solver: str, K: int) -> jnp.ndarray:
    """(5,) ``CLOCK_KEYS`` of the task about to be trained, from the state BEFORE ``train_task`` (write-up,
    "Readout exponent" and "Decision: the task clock").

    With g the gate of task ``state.t`` and a its readout (the fresh head for task heads):

    ``n_c``      realized active count Σ_i g_i
    ``abar_c``   key norm ā²_c = Σ_i g_i a_i²/pN, the rate of the hidden channel (eq. rflow)
    ``Qbar_c``   mean Q_ii over the active units (NaN for an empty gate)
    ``qmax_c``   largest eigenvalue of the active block Q_c of Q; ‖M_c‖ = qmax_c/pN (eq. Mspec)
    ``h_c``      population step of the task under the run's clock (``model.task_h``; ``p.h`` for SGD)

    The host derives from them the task duration Λ_c = η h_c n, the exponents λ_c = Λ_c ā²_c and
    λᵃ_c = Λ_c qmax_c/pN, the bounds of eq. (lama) and the left side of eq. (popstab)
    (``scripts/run_weight_norms.py``).
    """
    N, D = state.W.shape
    k_teach, k_head, _ = model.task_keys(state.key, state.t)
    _, g = model.draw_task(k_teach, N, D, p.pN, p.gated)
    Q = model.Q_diag(state.W)
    a = jnp.where(p.use_heads == 1.0, model.draw_readout(k_head, Q, p), state.a)
    n_c = g.sum()
    Qbar = jnp.where(n_c > 0, (g * Q).sum() / jnp.maximum(n_c, 1.0), jnp.nan)
    h = model.task_h(a, g, p, n_steps) if solver == "pop" else p.h
    return jnp.stack([n_c, model.key_norm(a, g, p), Qbar, model.active_qmax(state.W, g, K), h])


def leak_norms(state: State, p: Params) -> jnp.ndarray:
    """(3,) ``LEAK_KEYS`` of the task about to be trained, from the state BEFORE ``train_task``.

    With w, g the teacher and gate of task ``state.t`` (the draws ``train_task`` makes), a its readout (the
    fresh head for task heads, else the shared ``state.a``) and u_b = (g ⊙ a) W / pN its output before any step:

    ``ub2``      |u_b|²/D — the leak norm; the frozen norm balance predicts f/(2 − f), f = 1 − e^{−λ}
    ``wub``      w·u_b/D — its overlap with the new teacher (|w|²/D ≈ 1); ≈ 0 for unrelated teachers
    ``ub_diag``  Σ_i g_i a_i² Q_ii / pN² — the single-unit (diagonal) part of ``ub2``; ``ub2 − ub_diag`` is the
                 sum over pairs i ≠ j of active units, a_i a_j W_i·W_j / (pN² D)

    All three are 0 when the gate drew no unit.
    """
    N, D = state.W.shape
    k_teach, k_head, _ = model.task_keys(state.key, state.t)
    w, g = model.draw_task(k_teach, N, D, p.pN, p.gated)
    Q = model.Q_diag(state.W)
    a = jnp.where(p.use_heads == 1.0, model.draw_readout(k_head, Q, p), state.a)
    u = (g * a) @ state.W / p.pN
    return jnp.stack([(u @ u) / D, (u @ w) / D, (g * a * a * Q).sum() / (p.pN * p.pN)])


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
    if solver == "pop":                                          # the task clock, as model.train_block
        p = p._replace(h=model.task_h(a, g_t, p, n_steps))
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
def train_kernel(n_steps: int, solver: str, n_tasks: int, K: Optional[int] = None, closed: bool = False) -> Callable:
    """``f(state, p) -> state`` advancing every run by ``n_tasks`` tasks (no outputs); state donated."""
    vf = jax.vmap(functools.partial(train_task, n_steps=n_steps, solver=solver, K=K, closed=closed))

    def run(state, p):
        return jax.lax.scan(lambda s, _: (vf(s, p), None), state, None, length=n_tasks)[0]

    return jax.jit(run, donate_argnums=(0,))


@functools.lru_cache(maxsize=None)
def signal_kernel(n_steps: int, solver: str, n_tasks: int, K: Optional[int] = None, closed: bool = False) -> Callable:
    """``f(state, p, k) -> (state, S)`` with ``S`` (n_tasks, B) the signal of probe ``k`` after each task."""
    vf = jax.vmap(functools.partial(train_task, n_steps=n_steps, solver=solver, K=K, closed=closed))
    vs = jax.vmap(probe_signal, in_axes=(0, 0, None))

    def run(state, p, k):
        def body(s, _):
            s = vf(s, p)
            return s, vs(s, p, k)
        return jax.lax.scan(body, state, None, length=n_tasks)

    return jax.jit(run, donate_argnums=(0,))


@functools.lru_cache(maxsize=None)
def norms_kernel(n_steps: int, solver: str, n_tasks: int, K: Optional[int] = None, closed: bool = False) -> Callable:
    """``f(state, p) -> (state, X)`` with ``X`` (n_tasks, B, 13): the ``NORM_KEYS`` after each task, then the
    ``LEAK_KEYS`` and the ``CLOCK_KEYS`` of that task from the state before it was trained."""
    vf = jax.vmap(functools.partial(train_task, n_steps=n_steps, solver=solver, K=K, closed=closed))
    vn = jax.vmap(task_norms)
    vl = jax.vmap(leak_norms)

    def run(state, p):
        vc = jax.vmap(functools.partial(clock_norms, n_steps=n_steps, solver=solver,
                                        K=state.W.shape[1] if K is None else K))
        def body(s, _):
            leak, clk = vl(s, p), vc(s, p)
            s = vf(s, p)
            return s, jnp.concatenate([vn(s, p), leak, clk], -1)
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


def _batch(runs: Sequence, P: int) -> tuple[State, Params, np.ndarray, tuple]:
    """Feasible runs of ``runs`` as one batch with ``P`` head slots: (state, params, mask of kept runs, (K, closed))."""
    r0 = runs[0]
    for r in runs:
        assert group_key(r)[:4] == group_key(r0)[:4], "job is not homogeneous"
    ok = feasible(runs)
    kept = [r for r, f in zip(runs, ok) if f]
    if not kept:
        return None, None, ok, (r0.N, False)
    p = make_params(kept)
    keys = jnp.stack([jax.random.PRNGKey(r.rng_seed()) for r in kept])
    return batched_init(r0.N, r0.D, P)(keys, p), p, ok, block_of(kept)


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
    state, p, ok, kc = _batch(runs, T_pre + n_follow)
    if state is None:
        return np.full((n_follow, len(runs)), np.nan)
    prog = pool.Progress(T_pre + n_follow)
    for n in chunks(T_pre, chunk):
        state = train_kernel(r0.n_steps, r0.solver, n, *kc)(state, p)
        prog.add(n)
    S = []
    kk = jnp.int32(k)
    for n in chunks(n_follow, chunk):
        state, s = signal_kernel(r0.n_steps, r0.solver, n, *kc)(state, p, kk)
        S.append(np.asarray(s))
        prog.add(n)
    return _expand(np.concatenate(S, 0), ok)


def first_and_steady_signal(runs: Sequence, T: int, n_follow: int, chunk: Optional[int] = None) -> dict:
    """``{"S0": (T, B), "ST": (n_follow, B)}`` from ONE stream of ``T + n_follow`` tasks (``run_halflife_lam.py``).

    ``S0`` is the signal of probe 0 after each of the first ``T`` tasks (``signal_after_tasks`` with
    ``T_pre = 0, k = 0``) and ``ST`` that of probe ``T`` after each of the next ``n_follow`` (``T_pre = T,
    k = T``): the first ``T`` tasks are both the life of the first memory and the burn-in of the steady-state
    one, so the pair costs ``T + n_follow`` tasks instead of ``2T + n_follow``.  NaN columns for infeasible runs.
    """
    r0 = runs[0]
    state, p, ok, kc = _batch(runs, T + n_follow)
    if state is None:
        return {"S0": np.full((T, len(runs)), np.nan), "ST": np.full((n_follow, len(runs)), np.nan)}
    prog = pool.Progress(T + n_follow)
    out = {}
    for name, k, length in (("S0", 0, T), ("ST", T, n_follow)):
        S = []
        kk = jnp.int32(k)
        for n in chunks(length, chunk):
            state, s = signal_kernel(r0.n_steps, r0.solver, n, *kc)(state, p, kk)
            S.append(np.asarray(s))
            prog.add(n)
        out[name] = _expand(np.concatenate(S, 0), ok)
    return out


def norms_over_tasks(runs: Sequence, T: int, chunk: Optional[int] = None) -> dict:
    """``{key: (B, T) array}`` of ``NORM_KEYS`` after each of ``T`` tasks and of ``LEAK_KEYS`` and
    ``CLOCK_KEYS`` before each (``run_weight_norms.py``)."""
    keys = NORM_KEYS + LEAK_KEYS + CLOCK_KEYS
    r0 = runs[0]
    state, p, ok, kc = _batch(runs, T)
    if state is None:
        return {q: np.full((len(runs), T), np.nan) for q in keys}
    X = []
    prog = pool.Progress(T)
    for n in chunks(T, chunk):
        state, x = norms_kernel(r0.n_steps, r0.solver, n, *kc)(state, p)
        X.append(np.asarray(x))
        prog.add(n)
    X = _expand(np.concatenate(X, 0), ok)                        # (T, B, 13)
    return {q: X[:, :, i].T for i, q in enumerate(keys)}


def imbalance_over_steps(runs: Sequence, do_proj: Sequence[float], n_tasks: int, chunk: Optional[int] = None) -> np.ndarray:
    """(len(runs), n_tasks · n_steps, 4) per-step imbalance statistics with ``Params.do_proj`` forced per run.

    ``run_gauge_comp.py``: the runs of a cell (one per readout) each listed
    with the projection switched off and on; ``do_proj[i]`` applies to ``runs[i]``.
    """
    r0 = runs[0]
    state, p, ok, _ = _batch(runs, n_tasks)
    if state is None:
        return np.full((len(runs), n_tasks * r0.n_steps, 4), np.nan)
    p = p._replace(do_proj=jnp.asarray([float(d) for d, f in zip(do_proj, ok) if f], dtype=jnp.float64))
    R = []
    prog = pool.Progress(n_tasks)
    for n in chunks(n_tasks, chunk):
        state, r = gauge_kernel(r0.n_steps, r0.solver, n)(state, p)
        R.append(np.asarray(r))                                   # (n, B, n_steps, 4)
        prog.add(n)
    R = np.concatenate(R, 0)
    R = np.transpose(R, (1, 0, 2, 3)).reshape(R.shape[1], -1, 4)
    return _expand(R, ok, axis=0)


def loss_task(state: State, p: Params, *, n_steps: int, solver: str, K: int) -> tuple[State, jnp.ndarray]:
    """``train_task`` returning the task's learning curve: the population loss L_c = |u_c − w_c|²/(2D) of the
    task being trained before its first step and after each of its ``n_steps`` steps, (n_steps + 1,).

    Always steps (never the closed form), since the curve is the point.
    """
    N, D = state.W.shape
    k_teach, k_head, k_stream = model.task_keys(state.key, state.t)
    w_t, g_t = model.draw_task(k_teach, N, D, p.pN, p.gated)
    head = model.draw_readout(k_head, model.Q_diag(state.W), p)
    a = jnp.where(p.use_heads == 1.0, head, state.a)
    t = state.t
    loss0 = model.loss(state._replace(w=w_t, g=g_t, a=a), p)
    r = model.train_block(state.W, a, g_t, w_t, p, k_stream, n_steps=n_steps, solver=solver, K=K, closed=False, traced=True)
    state = state._replace(W=r["W"], a=r["a"], g=g_t, w=w_t)
    state = model.rescale_heads(state, p, r["f"])
    state = state._replace(heads=state.heads.at[t].set(jnp.where(p.use_heads == 1.0, g_t * state.a, 0.0)))
    return state._replace(t=t + 1), jnp.concatenate([loss0[None], r["rows"][:, 6]])


@functools.lru_cache(maxsize=None)
def loss_kernel(n_steps: int, solver: str, K: int) -> Callable:
    """``f(state, p) -> (state, L)`` with ``L`` (B, n_steps + 1) the learning curve of the next task; state donated."""
    return jax.jit(jax.vmap(functools.partial(loss_task, n_steps=n_steps, solver=solver, K=K)), donate_argnums=(0,))


def loss_curves(runs: Sequence, tasks: Sequence[int], T: int, chunk: Optional[int] = None) -> np.ndarray:
    """(B, len(tasks), n_steps + 1) learning curves of the listed tasks of a ``T``-task stream (``run_training_verify.py``).

    The tasks in between are trained untraced in chunks; NaN rows for infeasible runs.
    """
    r0 = runs[0]
    tasks = sorted(set(int(t) for t in tasks))
    assert all(0 <= t < T for t in tasks), "tasks must lie in [0, T)"
    state, p, ok, (K, closed) = _batch(runs, T)
    if state is None:
        return np.full((len(runs), len(tasks), r0.n_steps + 1), np.nan)
    out = []
    prog = pool.Progress(T)
    t = 0
    for tt in tasks:
        for n in chunks(tt - t, chunk):
            state = train_kernel(r0.n_steps, r0.solver, n, K, closed)(state, p)
            prog.add(n)
        state, L = loss_kernel(r0.n_steps, r0.solver, K)(state, p)
        out.append(np.asarray(L))
        prog.add(1)
        t = tt + 1
    return _expand(np.stack(out, 1), ok, axis=0)


def stream_batch(runs: Sequence, chunk: Optional[int] = None) -> dict:
    """``protocol.run_stream`` of one sweep batch (``validate.py``, ``run_sigma0.py``), reporting every tenth of its tasks."""
    T = runs[0].T_burn + runs[0].T_meas
    return run_stream(runs, log=pool.report, log_every=max(1, T // 10), chunk=chunk)


def seed_curves(pairs: Sequence[tuple]) -> dict:
    """Each seed's own S(m), E(m) and CV of s from ``stream_batch`` results, by condition (``validate.py``, ``run_sigma0.py``).

    Parameters
    ----------
    pairs : sequence of (batch, res)
        The runs of each batch and its ``stream_batch`` result; a condition's
        seeds may be spread over several batches.

    Returns
    -------
    dict
        ``{(N, readout, eta): {"ages", "seeds", "S_seeds", "E_seeds", "cv_seeds", "S", "E", "cv_s", "S_se"}}``:
        the per-seed (n_seeds, n_ages) curves in seed order, their seed means
        (``S``, ``E``, ``cv_s``) and the s.e.m. of ``S`` over seeds (NaN for one seed).
    """
    rows = {}
    for batch, res in pairs:
        c = curves({f: res[f] for f in Moments.FIELDS})
        for i, r in enumerate(batch):
            e = rows.setdefault((r.N, r.readout, r.eta), {"ages": res["ages"], "rows": []})
            e["rows"].append((r.seed, c["S"][i], c["E"][i], c["cv_s"][i]))
    out = {}
    for key, e in rows.items():
        rs = sorted(e["rows"], key=lambda t: t[0])
        S, E, cv = (np.stack([t[k] for t in rs]) for k in (1, 2, 3))
        n = len(rs)
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out[key] = {"ages": e["ages"], "seeds": [t[0] for t in rs], "S_seeds": S, "E_seeds": E, "cv_seeds": cv,
                        "S": np.nanmean(S, 0), "E": np.nanmean(E, 0), "cv_s": np.nanmean(cv, 0),
                        "S_se": S.std(0, ddof=1) / np.sqrt(n) if n > 1 else np.full(S.shape[1], np.nan)}
    return out


# ----------------------------------------------------------------------------
# grouping runs into jobs
# ----------------------------------------------------------------------------
def group_key(r) -> tuple:
    """Runs that can share one vmap batch here: (N, D, solver, n_steps, closed, K) — every readout and seed of a
    width with the same block of active rows K (``Run.block_rows``: the ungated readouts have K = N, a gated
    readout K ≈ κ√N + 8 s.d., so every κ is its own group), the closed-form runs (frozen unprojected population)
    apart from the stepped ones.  A batch steps every run on the block of its largest gate, so mixing K would
    make the narrow-gate runs pay for the wide one."""
    return (r.N, r.D, r.solver, r.n_steps, r.closed, r.block_rows())


#: Per-step overhead of a device call in the units of ``run_cost`` (elements of the block streamed per step,
#: B·K·D): one launch of the vmap'd step costs about as much as streaming 3·10⁷ doubles.  Fitted to the GPU
#: log of 2026-11-06 (V100: ~0.8 ms per step at N ≤ 200, where the block part is a fraction of that), so
#: rough; the point is that a small batch is not free and the cost of a job does not vanish with its size.
STEP_OVERHEAD = 3.0e7


def run_cost(r, T: int) -> float:
    """Rough device time of one run over ``T`` tasks: K·D per step on its block of active rows (no overhead)."""
    return float(r.block_rows()) * r.D * r.n_steps * T


def job_cost(runs: Sequence, T: int) -> float:
    """Rough device time of a job of ``runs`` over ``T`` tasks: per step the block of the batch, B·K·D, plus the
    launch overhead ``STEP_OVERHEAD``; ``pool.run_jobs`` orders and places the jobs by it."""
    r = runs[0]
    K = max(rr.block_rows() for rr in runs)
    return float(r.n_steps) * T * (len(runs) * K * r.D + STEP_OVERHEAD)


def split(runs: Sequence, pieces: int) -> list[list]:
    """Split ``runs`` into ≤ ``pieces`` nearly equal contiguous parts (none empty)."""
    pieces = max(1, min(pieces, len(runs)))
    bounds = np.linspace(0, len(runs), pieces + 1).round().astype(int)
    return [list(runs[a:b]) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]
