"""``run_task`` (one task of one run, jitted and vmapped) and ``run_stream`` (the task loop).

PLAN.md §3.7, §5 and §12.  ``run_task`` is written for a single run and is
batched by ``batched_run_task`` with ``jax.vmap``; the Python loop over tasks
lives in ``run_stream``, which accumulates the running moments and the
per-task scalars on the host with one device-to-host copy per task.
"""
from __future__ import annotations

import functools
import time
from typing import Callable, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

import mup  # noqa: F401
from mup import model
from mup.model import Params, State
from mup.observe import Moments, age_schedule, evaluate_probes

#: Keys of PLAN.md §12.1 recorded both before (``_pre``) and after (``_post``) the last
#: projection of a task.
NORM_KEYS = ("Q_act", "Q_all", "W_fro2", "a2_act", "a2_all", "aQ_act", "aQ_all", "aQ_max", "heads_fro2")
#: Active-unit quantiles (evaluated on the pre-projection state; see ``run_task``).
QUANTILE_KEYS = ("Q_q05", "Q_q50", "Q_q95", "aQ_q05", "aQ_q95")
#: Per-task scalars returned by ``run_task`` and stored as (T, B) arrays in every batch file.
SCALARS = (
    ("abar2", "n_active", "L_end", "E0", "S0", "gauge_delta", "gauge_uc", "logf_sum", "logf_max")
    + tuple(k + "_pre" for k in NORM_KEYS) + tuple(k + "_post" for k in NORM_KEYS) + QUANTILE_KEYS
    + ("Q_mean", "a2_mean", "aQ_mean", "aQ_max")     # schema-1 names, kept for one version
)
#: Per-step quantities of a trace run (PLAN.md §12.2), in column order.
TRACE_QUANT = ("Q_act_pre", "Q_act_post", "Q_all", "a2_act", "aQ_act", "logf", "loss")


def _step_fn(p: Params, solver: str, k_stream: jnp.ndarray) -> Callable:
    """Return ``step(state, i) -> state`` for the solver; ``i`` is the step index of the task."""
    if solver == "sgd":
        return lambda s, i: model.sgd_step(s, p, jax.random.fold_in(k_stream, i))
    if solver == "pop":
        return lambda s, i: model.pop_step(s, p)
    raise ValueError(f"unknown solver {solver!r}")


def _step_and_project(state: State, p: Params, step: Callable, i) -> tuple[State, jnp.ndarray, jnp.ndarray]:
    """One step followed by the row projection (heads untouched).

    Returns
    -------
    (state, f, trace_row)
        ``f`` is the per-unit rescale factor of ``W_i`` (1 for untouched units),
        ``trace_row`` the (len(TRACE_QUANT),) per-step quantities (active-unit
        means are NaN when the gate is empty).
    """
    s1 = step(state, i)
    Q_pre = model.Q_diag(s1.W)
    s2, f = model.project_rows(s1, p)
    g = state.g
    n_act = g.sum()
    act_mean = lambda x: jnp.where(n_act > 0, (g * x).sum() / jnp.maximum(n_act, 1.0), jnp.nan)
    Q_post = model.Q_diag(s2.W)
    row = jnp.stack([
        act_mean(Q_pre), act_mean(Q_post), Q_post.mean(), act_mean(s2.a * s2.a),
        act_mean(jnp.abs(s2.a) * jnp.sqrt(Q_post)), act_mean(jnp.log(f)), model.loss(s2, p),
    ])
    return s2, f, row


def run_task(state: State, p: Params, ages: jnp.ndarray, trace_idx: Optional[jnp.ndarray] = None,
             *, n_steps: int, solver: str):
    """Train one task of ``n_steps`` steps, store it as a probe and evaluate all probes.

    Parameters
    ----------
    state : State
        One run's state at the start of the task (``state.t`` is the task index).
    p : Params
        The run's constants.
    ages : (n_ages,) int32 array
        Probe ages of the static schedule (``observe.age_schedule``).
    trace_idx : (n_trace,) int32 array, optional
        Step indices whose per-step quantities are returned (PLAN.md §12.2).
        ``None`` keeps the ``fori_loop`` and returns no trace.  The loop body
        is identical in both modes, so a traced run reproduces an untraced one.
    n_steps : int (static)
        Steps per task, ≥ 1.
    solver : {"pop", "sgd"} (static)

    Returns
    -------
    (new_state, out)
        ``out`` holds ``s``, ``eps``, ``valid`` of shape (n_ages,), every key of
        ``SCALARS`` as a 0-d array and, with ``trace_idx``, ``trace`` of shape
        (n_trace, len(TRACE_QUANT)).  ``E0``/``S0`` are ε and s of the task
        just learned (age 0).  ``*_pre`` scalars are evaluated after the
        last step and before the last projection (the ``uc_before`` point),
        ``*_post`` after it; the quantiles on the pre state.  ``logf_sum`` is
        Σ_steps mean_active log f, ``logf_max`` the largest single-step
        |log f| over active units.
    """
    assert n_steps >= 1
    N, D = state.W.shape
    kt = jax.random.fold_in(state.key, state.t)
    k_teach, k_head, k_stream = jax.random.split(kt, 3)

    w_t, g_t = model.draw_task(k_teach, N, D, p.pN, p.gated)
    head = model.draw_readout(k_head, model.Q_diag(state.W), p)
    a = jnp.where(p.use_heads == 1.0, head, state.a)
    state = state._replace(w=w_t, g=g_t, a=a)
    t = state.t

    n_act = g_t.sum()
    n_act1 = jnp.maximum(n_act, 1.0)
    abar2 = jnp.where(n_act > 0, (g_t * a * a).sum() / n_act1, jnp.nan)   # NaN: empty gate, nothing trained
    step = _step_fn(p, solver, k_stream)

    # The stored-head rescaling by 1/f (eq. proj) is a product over steps, so it is
    # accumulated in ``fcum`` and applied to the (P, N) head buffer once per task.
    ones = jnp.ones((N,), dtype=state.W.dtype)
    carry0 = (state, ones, jnp.zeros((), state.W.dtype), jnp.zeros((), state.W.dtype))

    def body(carry, i):
        """Loop body: step i, project, accumulate Πf, Σ mean log f and max |log f|; returns (carry, trace row)."""
        s, fcum, lsum, lmax = carry
        s, f, row = _step_and_project(s, p, step, i)
        logf = jnp.log(f)
        return (s, fcum * f, lsum + logf.sum() / n_act1, jnp.maximum(lmax, jnp.abs(logf).max())), row

    if trace_idx is None:
        carry = jax.lax.fori_loop(0, n_steps - 1, lambda i, c: body(c, i)[0], carry0)
        rows = None
    else:
        carry, rows = jax.lax.scan(body, carry0, jnp.arange(n_steps - 1, dtype=jnp.int32))
    state, fcum, logf_sum, logf_max = carry

    # last step with the gauge diagnostics (check 1) and the pre-projection norms
    state = step(state, n_steps - 1)
    uc_before = model.u_c(state, p)
    pre = model.norm_stats(state.W, state.a, g_t)
    pre["heads_fro2"] = model.heads_fro2(state.heads, t)
    state, f = model.project_rows(state, p)
    logf = jnp.log(f)
    logf_sum = logf_sum + logf.sum() / n_act1
    logf_max = jnp.maximum(logf_max, jnp.abs(logf).max())
    state = model.rescale_heads(state, p, fcum * f)
    uc_after = model.u_c(state, p)
    gauge_uc = jnp.max(jnp.abs(uc_after - uc_before))
    dres = jnp.abs(model.imbalance(state, p) - p.Delta0)
    gauge_delta = jnp.max(jnp.where(g_t == 1.0, dres, 0.0))

    # store the probe, then evaluate every probe of the age schedule
    state = state._replace(
        probes_w=state.probes_w.at[t].set(w_t),
        probes_g=state.probes_g.at[t].set(g_t),
        heads=state.heads.at[t].set(jnp.where(p.use_heads == 1.0, state.a, 0.0)),
    )
    s, eps, valid = evaluate_probes(state, p, ages)
    post = model.norm_stats(state.W, state.a, g_t)
    post["heads_fro2"] = model.heads_fro2(state.heads, t + 1)

    out = {
        "s": s, "eps": eps, "valid": valid,
        "abar2": abar2, "n_active": n_act, "L_end": model.loss(state, p), "E0": eps[0], "S0": s[0],
        "gauge_delta": gauge_delta, "gauge_uc": gauge_uc,
        "logf_sum": logf_sum, "logf_max": logf_max,
        **{k + "_pre": pre[k] for k in NORM_KEYS}, **{k + "_post": post[k] for k in NORM_KEYS},
        **{k: pre[k] for k in QUANTILE_KEYS},
        "Q_mean": post["Q_all"], "a2_mean": post["a2_all"], "aQ_mean": post["aQ_all"], "aQ_max": post["aQ_max"],
    }
    if rows is not None:
        last = jnp.stack([pre["Q_act"], post["Q_act"], post["Q_all"], post["a2_act"], post["aQ_act"],
                          logf.sum() / n_act1, out["L_end"]])
        trace = jnp.concatenate([rows, last[None, :]], axis=0)        # (n_steps, n_quant)
        out["trace"] = trace[trace_idx]
    return state._replace(t=t + 1), out


@functools.lru_cache(maxsize=None)
def batched_run_task(n_steps: int, solver: str, traced: bool = False) -> Callable:
    """``jit(vmap(run_task))`` for static ``(n_steps, solver, traced)``; cached.

    Parameters
    ----------
    n_steps : int
    solver : {"pop", "sgd"}
    traced : bool
        Whether the returned function takes a ``trace_idx`` array (shared by
        the batch) and returns ``out["trace"]``.

    Returns
    -------
    callable
        ``f(state, params, ages[, trace_idx]) -> (state, out)`` over a batch
        (leading axis B on ``state`` and ``params``).  The state is donated so
        the probe and head buffers are updated in place; callers must use the
        returned state and never the argument.
    """
    f = functools.partial(run_task, n_steps=n_steps, solver=solver)
    in_axes = (0, 0, None, None) if traced else (0, 0, None)
    return jax.jit(jax.vmap(f, in_axes=in_axes), donate_argnums=(0,))


def make_params(runs: Sequence) -> Params:
    """Stack the per-run constants of a batch of ``Run`` objects into a ``Params`` pytree.

    Parameters
    ----------
    runs : sequence of config.Run

    Returns
    -------
    Params
        Each field a (B,) float64 array in the order of ``runs``.
    """
    def col(name):
        """(B,) float64 column of one Run attribute."""
        return jnp.asarray([float(getattr(r, name)) for r in runs], dtype=jnp.float64)
    return Params(
        Delta0=col("Delta0"), eta=col("eta"), h=col("h"), pN=col("pN"), lam=col("lam"),
        gated=col("gated"), learn_a=col("learn_a"), do_proj=col("do_proj"), use_heads=col("use_heads"),
    )


@functools.lru_cache(maxsize=None)
def batched_init(N: int, D: int, P: int) -> Callable:
    """``jit(vmap(init_state))`` for static ``(N, D, P)``; cached so repeated calls do not re-trace.

    Returns
    -------
    callable
        ``f(keys (B, 2), params) -> State`` with a leading batch axis.
    """
    return jax.jit(jax.vmap(functools.partial(model.init_state, N=N, D=D, P=P), in_axes=(0, 0)))


def _init(runs: Sequence) -> tuple[State, Params]:
    """Initial batched state and params of ``runs`` (no feasibility check)."""
    r0 = runs[0]
    p = make_params(runs)
    keys = jnp.stack([jax.random.PRNGKey(r.rng_seed()) for r in runs])
    return batched_init(r0.N, r0.D, r0.T_burn + r0.T_meas)(keys, p), p


def feasible(runs: Sequence) -> np.ndarray:
    """Mask of runs whose initial readout is real, i.e. δ₀ > −min_i Q0_i (PLAN.md §3.5).

    At small D the row norms Q0_i fluctuate by ~√(2/D), so δ₀ = −0.75 fails
    for many seeds at N ≤ 20; such runs are refused, not clamped.

    Parameters
    ----------
    runs : sequence of config.Run sharing (N, D, T)

    Returns
    -------
    (B,) bool array
    """
    state, _ = _init(runs)
    return np.isfinite(np.asarray(state.a)).all(1)


def init_batch(runs: Sequence) -> tuple[State, Params]:
    """Initial states (leading axis B) for a batch sharing N, D and T.

    Parameters
    ----------
    runs : sequence of config.Run

    Returns
    -------
    (State, Params)

    Raises
    ------
    ValueError
        If any run's initial readout is not finite (infeasible δ₀); the
        message lists the runs.
    """
    state, p = _init(runs)
    bad = ~np.isfinite(np.asarray(state.a)).all(1)
    if bad.any():
        names = [runs[i].label() for i in np.where(bad)[0]]
        raise ValueError(
            "initial readout is not finite (δ₀ ≤ −min_i Q0_i); runs: " + ", ".join(names)
        )
    return state, p


def trace_indices(n_steps: int, solver: str, n_points: int) -> np.ndarray:
    """Step indices stored by a trace run (PLAN.md §12.2, §12.5 item 5).

    Parameters
    ----------
    n_steps : int
    solver : {"pop", "sgd"}
        Population steps are strided uniformly, SGD steps logarithmically
        (the interesting part is the first few hundred steps).
    n_points : int
        Target number of points; every step is kept when ``n_steps ≤ n_points``.

    Returns
    -------
    (n_trace,) int32 array, increasing, always containing 0 and n_steps − 1.
    """
    if n_steps <= n_points:
        return np.arange(n_steps, dtype=np.int32)
    if solver == "sgd":
        idx = np.unique(np.round(np.geomspace(1, n_steps - 1, n_points - 1)).astype(np.int64))
        idx = np.concatenate([[0], idx])
    else:
        idx = np.unique(np.round(np.linspace(0, n_steps - 1, n_points)).astype(np.int64))
    return idx.astype(np.int32)


def run_stream(
    runs: Sequence,
    log: Optional[Callable[[str], None]] = None,
    log_every: int = 0,
) -> dict:
    """Run a batch of runs through the full task stream (PLAN.md §3.7, §5, §12).

    Parameters
    ----------
    runs : sequence of config.Run
        Must share ``(N, D, kappa, solver, n_steps, T_burn, T_meas, trace)``.
    log : callable, optional
        Receives progress strings.
    log_every : int
        Log every this many tasks (0: only at the end).

    Returns
    -------
    dict of numpy arrays
        ``ages`` (n_ages,); the running moments ``sum_s``, ``sum_s2``, ``sum_e``,
        ``sum_e2``, ``count`` each (B, n_ages, 2) over the two measurement
        halves; every key of ``SCALARS`` as (T, B); ``alpha_task`` (B,);
        ``T_burn``, ``T_meas``, ``n_steps`` as ints.  For trace runs also
        ``trace`` (B, T, n_trace, len(TRACE_QUANT)) float32 and ``trace_idx``.
    """
    r0 = runs[0]
    for r in runs:
        assert r.batch_key() == r0.batch_key(), "batch is not homogeneous"
    B = len(runs)
    T_burn, T_meas = r0.T_burn, r0.T_meas
    T = T_burn + T_meas
    ages = age_schedule(r0.N, r0.kappa_s, T, r0.n_ages_geom, r0.m_max_factor)   # schedule κ (PLAN.md §5)
    ages_j = jnp.asarray(ages)
    state, p = init_batch(runs)
    traced = bool(r0.trace)
    step_fn = batched_run_task(r0.n_steps, r0.solver, traced)
    if traced:
        tidx = trace_indices(r0.n_steps, r0.solver, r0.trace_points)
        tidx_j = jnp.asarray(tidx)
        traces = np.zeros((B, T, len(tidx), len(TRACE_QUANT)), dtype=np.float32)

    mom = Moments(B, len(ages))
    scalars = {k: np.zeros((T, B)) for k in SCALARS}
    t0 = time.time()
    for t in range(T):
        state, out = step_fn(state, p, ages_j, tidx_j) if traced else step_fn(state, p, ages_j)
        if t >= T_burn:
            half = 0 if t < T_burn + T_meas // 2 else 1
            mom.add(np.asarray(out["s"]), np.asarray(out["eps"]), np.asarray(out["valid"]), half)
        for k in SCALARS:
            scalars[k][t] = np.asarray(out[k])
        if traced:
            traces[:, t] = np.asarray(out["trace"], dtype=np.float32)
        if log is not None and log_every and (t + 1) % log_every == 0:
            log(f"task {t + 1}/{T}  {time.time() - t0:.1f}s")
    if log is not None:
        log(f"batch of {B} runs, {T} tasks × {r0.n_steps} steps done in {time.time() - t0:.1f}s")

    res = {
        "ages": ages, "alpha_task": np.array([r.alpha_task for r in runs]),
        "T_burn": T_burn, "T_meas": T_meas, "n_steps": r0.n_steps,
    }
    res.update(mom.as_dict())
    res.update(scalars)
    if traced:
        res["trace"] = traces
        res["trace_idx"] = tidx
    return res


def results_moments(res: dict) -> dict:
    """The five running-moment arrays of a result dict, keyed as in ``Moments.FIELDS``."""
    return {f: res[f] for f in Moments.FIELDS}
