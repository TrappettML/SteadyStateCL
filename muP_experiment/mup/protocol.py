"""run_task (jit, single run, vmapped over a batch) and run_stream (Python loop over tasks)."""
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

SCALARS = (
    "abar2", "n_active", "Q_mean", "a2_mean", "aQ_mean", "aQ_max",
    "L_end", "E0", "gauge_delta", "gauge_uc",
)


def run_task(state: State, p: Params, ages: jnp.ndarray, *, n_steps: int, solver: str):
    """Train one task of ``n_steps`` steps and evaluate the probes (PLAN.md §3.7).

    Returns ``(new_state, out)`` where ``out`` holds ``s``, ``eps``, ``valid``
    (n_ages,) and the per-task scalars in ``SCALARS``.
    """
    assert n_steps >= 1
    N, D = state.W.shape
    kt = jax.random.fold_in(state.key, state.t)
    k_teach, k_head, k_stream = jax.random.split(kt, 3)

    w_t, g_t = model.draw_task(k_teach, N, D, p.pN)
    head = model.draw_readout(k_head, model.Q_diag(state.W), p)
    a = jnp.where(p.use_heads == 1.0, head, state.a)
    state = state._replace(w=w_t, g=g_t, a=a)

    n_act = g_t.sum()
    abar2 = (g_t * a * a).sum() / jnp.maximum(n_act, 1.0)

    if solver == "sgd":
        def step(s, i):
            return model.sgd_step(s, p, jax.random.fold_in(k_stream, i))
    elif solver == "pop":
        def step(s, i):
            return model.pop_step(s, p)
    else:
        raise ValueError(f"unknown solver {solver!r}")

    def body(i, s):
        return model.project(step(s, i), p)

    state = jax.lax.fori_loop(0, n_steps - 1, body, state)
    # last step with the gauge diagnostics (check 1)
    state = step(state, n_steps - 1)
    uc_before = model.u_c(state, p)
    state = model.project(state, p)
    uc_after = model.u_c(state, p)
    gauge_uc = jnp.max(jnp.abs(uc_after - uc_before))
    dres = jnp.abs(model.imbalance(state, p) - p.Delta0)
    gauge_delta = jnp.max(jnp.where(g_t == 1.0, dres, 0.0))

    # store the probe
    t = state.t
    state = state._replace(
        probes_w=state.probes_w.at[t].set(w_t),
        probes_g=state.probes_g.at[t].set(g_t),
        heads=state.heads.at[t].set(jnp.where(p.use_heads == 1.0, state.a, 0.0)),
    )
    s, eps, valid = evaluate_probes(state, p, ages)

    Q = model.Q_diag(state.W)
    aQ = jnp.abs(state.a) * jnp.sqrt(Q)
    out = {
        "s": s, "eps": eps, "valid": valid,
        "abar2": abar2, "n_active": n_act,
        "Q_mean": Q.mean(), "a2_mean": (state.a**2).mean(),
        "aQ_mean": aQ.mean(), "aQ_max": aQ.max(),
        "L_end": model.loss(state, p), "E0": eps[0],
        "gauge_delta": gauge_delta, "gauge_uc": gauge_uc,
    }
    return state._replace(t=t + 1), out


@functools.lru_cache(maxsize=None)
def batched_run_task(n_steps: int, solver: str) -> Callable:
    """jit(vmap(run_task)) for a given static (n_steps, solver); cached.

    The state is donated so the probe/head buffers are updated in place
    instead of copied on every task; callers must use the returned state.
    """
    f = functools.partial(run_task, n_steps=n_steps, solver=solver)
    return jax.jit(jax.vmap(f, in_axes=(0, 0, None)), donate_argnums=(0,))


def make_params(runs: Sequence) -> Params:
    """Stack the per-run constants of a batch of ``Run`` objects into a Params pytree."""
    def col(name):
        return jnp.asarray([float(getattr(r, name)) for r in runs], dtype=jnp.float64)
    return Params(
        Delta0=col("Delta0"), eta=col("eta"), h=col("h"), pN=col("pN"), lam=col("lam"),
        learn_a=col("learn_a"), do_proj=col("do_proj"), use_heads=col("use_heads"),
    )


def _init(runs: Sequence) -> tuple[State, Params]:
    r0 = runs[0]
    P = r0.T_burn + r0.T_meas
    p = make_params(runs)
    keys = jnp.stack([jax.random.PRNGKey(int(r.seed)) for r in runs])
    init = jax.jit(jax.vmap(functools.partial(model.init_state, N=r0.N, D=r0.D, P=P), in_axes=(0, 0)))
    return init(keys, p), p


def feasible(runs: Sequence) -> np.ndarray:
    """Mask of runs whose initial readout is real, i.e. δ₀ > −min_i Q0_i (PLAN.md §3.5).

    At small D the row norms Q0_i fluctuate by ~√(2/D), so δ₀ = −0.75 fails
    for many seeds at N ≤ 20; such runs are refused, not clamped.
    """
    state, _ = _init(runs)
    return np.isfinite(np.asarray(state.a)).all(1)


def init_batch(runs: Sequence) -> tuple[State, Params]:
    """Initial states (B leading axis) for a batch sharing N, D and T; raises on infeasible runs."""
    state, p = _init(runs)
    bad = ~np.isfinite(np.asarray(state.a)).all(1)
    if bad.any():
        names = [runs[i].label() for i in np.where(bad)[0]]
        raise ValueError(
            "initial readout is not finite (δ₀ ≤ −min_i Q0_i); runs: " + ", ".join(names)
        )
    return state, p


def run_stream(
    runs: Sequence,
    log: Optional[Callable[[str], None]] = None,
    log_every: int = 0,
) -> dict:
    """Run a batch of runs through the full task stream (PLAN.md §3.7, §5).

    All runs must share (N, D, kappa, solver, n_steps, T_burn, T_meas).  Returns a
    dict of numpy arrays: ``ages`` (n_ages,), the running moments
    (``sum_s`` … ``count``, each (B, n_ages, 2)), every scalar in ``SCALARS``
    as (T, B), and ``alpha_task`` (B,).
    """
    r0 = runs[0]
    for r in runs:
        assert (r.N, r.D, r.kappa, r.solver, r.n_steps, r.T_burn, r.T_meas) == \
            (r0.N, r0.D, r0.kappa, r0.solver, r0.n_steps, r0.T_burn, r0.T_meas), "batch is not homogeneous"
    B = len(runs)
    T_burn, T_meas = r0.T_burn, r0.T_meas
    T = T_burn + T_meas
    ages = age_schedule(r0.N, r0.kappa, T)
    ages_j = jnp.asarray(ages)
    state, p = init_batch(runs)
    step_fn = batched_run_task(r0.n_steps, r0.solver)

    mom = Moments(B, len(ages))
    scalars = {k: np.zeros((T, B)) for k in SCALARS}
    t0 = time.time()
    for t in range(T):
        state, out = step_fn(state, p, ages_j)
        if t >= T_burn:
            half = 0 if t < T_burn + T_meas // 2 else 1
            mom.add(np.asarray(out["s"]), np.asarray(out["eps"]), np.asarray(out["valid"]), half)
        for k in SCALARS:
            scalars[k][t] = np.asarray(out[k])
        if log is not None and log_every and (t + 1) % log_every == 0:
            log(f"task {t + 1}/{T}  {time.time() - t0:.1f}s")
    if log is not None:
        log(f"batch of {B} runs, {T} tasks × {r0.n_steps} steps done in {time.time() - t0:.1f}s")

    alpha = np.array([r.alpha_task for r in runs])
    res = {"ages": ages, "alpha_task": alpha, "T_burn": T_burn, "T_meas": T_meas, "n_steps": r0.n_steps}
    res.update(mom.as_dict())
    res.update(scalars)
    return res


def results_moments(res: dict) -> dict:
    return {f: res[f] for f in Moments.FIELDS}
