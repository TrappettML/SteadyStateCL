"""Synaptic gating (write-up Section "Synaptic gating"): every synapse gated on its own, at the density of the unit gate.

Task c has a mask Γ_c ∈ {0, 1}^{N×D} of i.i.d. indicators with Pr = p = pN/N applied entrywise to W,

    u_{c,j} = Σ_i γ_cij a_i W_ij / pN,

where unit ("hidden-layer") gating has γ_cij = g_ci for every j.  The population step is eq. (syn_pop),

    W_ij ← W_ij − hη a_i γ_cij r_cj,        a_i ← a_i − hη (γ_ci ⊙ W_i)·r_c / D,

and the projection onto Δ_i = Δ₀ (``model.project_rows``, with the norm of the whole row) runs on all N units
after every step.  Everything else is the unit-gated run of the same ``config.Run``: the same ``State`` and
``Params``, the same run key, hence the same W⁽⁰⁾, initial readout, teachers (``draw_mask`` draws w_c from the
key ``model.draw_task`` uses) and fresh heads, and the same step count n.  The task clock is the run's
(``task_h``):

clock (i), ``nominal``    the fixed duration Λ = λ/(1 + δ₀) of the unit-gated twin (its h), so the pair differs in
    the mask alone and is compared at equal task duration: the condition of eq. (syn_Sigma), under which the
    mean signals of the two gatings are equal at every age.
clock (ii), ``realized``  (the default) Λ_c = λ/ā²_c at the MEAN rate over the input coordinates,
    ā²_c = Σ_ij γ_cij a_i² / (pN D) (write-up, "Residual flow and learning per task" of the synaptic section).
    Coordinate j then has the exponent λ ā²_cj/ā²_c ≈ λ n_cj/pN, a spread no clock removes, so S(0) = θ̄ of
    eq. (syn_theta) and the heads decay as eq. (syn_heads) (``theory.syn_theta_bar``, ``theory.sigma_syn_heads``).

Only the gated readouts have a synaptic form (``gated_shared``,
``gated_frozen``, ``task_heads``); ``State.g`` is unused and a stored head is the whole (N,) head, its mask
being regenerated from the key like the teacher.  Population solver only.

A step touches all N·D synapses where unit gating touches the block of ≈ pN active rows, so there is no block
of active rows here and every κ of a width shares one ``vmap`` batch (``group_key``).  Within a task the rows
are kept as diag(s)·W̃ and only the open synapses are advanced (``train_task``): two matrix–vector products,
one update and one reduction per step.  The frozen, unprojected
readout has the closed form of eq. (syn_rfrozen), one relaxation rate per input coordinate.

``signal_after_tasks`` is the job function of ``scripts/run_synapse_comp.py`` and mirrors
``experiments.signal_after_tasks``.
"""
from __future__ import annotations

import functools
from typing import Callable, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

import sscl_core._x64  # noqa: F401  (enables x64)
from sscl_core import experiments as X
from sscl_core import model, pool
from sscl_core.model import Params, State


def draw_mask(k_teach: jnp.ndarray, N: int, D: int, pN: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Teacher w ~ N(0, I_D), bit for bit that of ``model.draw_task``, and mask γ_ij ~ Bernoulli(pN/N), (N, D) 0/1 floats."""
    kw, kg = jax.random.split(k_teach)
    w = jax.random.normal(kw, (D,), dtype=jnp.float64)
    # 32 random bits per synapse against the threshold p·2³² (half the generator work of a float64 uniform;
    # p is resolved to 2⁻³²); p ≥ 1 opens every synapse exactly
    dens = pN / N
    thresh = jnp.clip(dens, 0.0, 1.0) * 4294967296.0
    bits = jax.random.bits(jax.random.fold_in(kg, 1), (N, D), dtype=jnp.uint32)
    M = ((bits.astype(jnp.float64) < thresh) | (dens >= 1.0)).astype(jnp.float64)
    return w, M


def probe_mask(key: jnp.ndarray, k, N: int, D: int, pN: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Teacher and mask of task ``k`` of a run, regenerated from the run key (the draws of ``train_task``)."""
    return draw_mask(model.task_keys(key, k)[0], N, D, pN)


def _project(a: jnp.ndarray, Q: jnp.ndarray, p: Params) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Eq. (proj) on every unit from (a, Q) (``model.project_rows`` with the gate all ones): the projected a, the
    projected Q = Q f² and the factor f the rows of W take; a no-op with f = 1 when ``do_proj == 0``."""
    AB = (a * a / p.eta) * (Q / p.eta)
    big = 0.5 * (jnp.abs(p.Delta0) + jnp.hypot(p.Delta0, 2.0 * jnp.sqrt(AB)))
    small = AB / big
    Ap = jnp.where(p.Delta0 >= 0.0, big, small)
    Bp = jnp.where(p.Delta0 >= 0.0, small, big)
    on = p.do_proj == 1.0
    f = jnp.where(on, jnp.sqrt(p.eta * Bp / Q), 1.0)
    return jnp.where(on, jnp.sign(a) * jnp.sqrt(p.eta * Ap), a), Q * f * f, f


def task_h(a: jnp.ndarray, M: jnp.ndarray, p: Params, n_steps: int) -> jnp.ndarray:
    """Population step of one task under the run's clock — ``model.task_h`` with the mask on the synapse.

    Clock (i): ``p.h``.  Clock (ii): h_c = λ/(η n ā²_c) with the mean rate over the input coordinates,
    ā²_c = Σ_ij γ_ij a_i² / (pN D) (0 when no synapse is open); for a unit gate, γ_ij = g_i, this is
    ``model.task_h``.
    """
    abar = (M * (a * a)[:, None]).sum() / (p.pN * M.shape[1])
    h_real = jnp.where(abar > 0.0, p.lam / (p.eta * n_steps * jnp.where(abar > 0.0, abar, 1.0)), 0.0)
    return jnp.where(p.clock == 1.0, h_real, p.h)


def train_task(state: State, p: Params, *, n_steps: int, closed: bool = False) -> State:
    """One task of one run under synaptic gating — ``experiments.train_task`` with the mask on the synapse.

    Parameters
    ----------
    state : State
    p : Params
    n_steps : int (static)
    closed : bool (static)
        Frozen, unprojected readout: the steps in closed form, coordinate j relaxing by q_j = 1 − hη ā_j per
        step, ā_j = Σ_i γ_ij a_i² / pN (eq. syn_rfrozen; the step loop to rounding).

    Returns
    -------
    State
        After the task: ``w`` of the task, the stored heads rescaled by the task's projection factors and the
        task's head stored whole at slot ``t`` (task heads), ``t + 1``.  ``g`` is untouched.
    """
    N, D = state.W.shape
    k_teach, k_head, _ = model.task_keys(state.key, state.t)
    w, M = draw_mask(k_teach, N, D, p.pN)
    Q0 = model.Q_diag(state.W)
    head = model.draw_readout(k_head, Q0, p)
    a = jnp.where(p.use_heads == 1.0, head, state.a)
    W = state.W
    c = task_h(a, M, p, n_steps) * p.eta
    t = state.t
    if closed:
        abar = (M * (a * a)[:, None]).sum(0) / p.pN                          # (D,) key norm of every coordinate
        r0 = (a @ (M * W)) / p.pN - w
        q = 1.0 - c * abar
        fac = jnp.where(abar > 0.0, (1.0 - q ** n_steps) / jnp.where(abar > 0.0, abar, 1.0), n_steps * c)
        W = W - a[:, None] * M * (fac * r0)[None, :]
        F = jnp.ones((N,), W.dtype)
    else:
        # The steps of eq. (syn_pop) + projection, arranged to pass over the N × D arrays as few times as
        # possible (the step is memory-bound: this form measured 1.3–1.5× the direct one on a CPU at N ≥ 200).
        # Within the task W = diag(s)·W̃ with the projection's row factors folded into s lazily (the projection
        # then costs N numbers, not a pass over W), and only the open synapses V = Γ ⊙ W̃ are advanced, the
        # closed ones not moving.  The row norms follow from V alone, Q_i = s_i² (Q_i^closed + Σ_j V_ij² / D),
        # with Q^closed the closed synapses' share, fixed over the task.  One step: two matrix–vector products
        # with V (the residual, the readout gradient), the update of V and one reduction over it.
        Mb = M == 1.0
        V0 = M * W
        Q_closed = Q0 - (V0 * V0).sum(1) / D

        def step(_, carry):
            V, a, sc = carry
            r = ((a * sc) @ V) / p.pN - w
            a_n = a - p.learn_a * c * sc * (V @ r) / D                       # (γ_i ⊙ W_i)·r = s_i (V r)_i
            V_n = jnp.where(Mb, V - (c * a / sc)[:, None] * r[None, :], V)
            Q_n = sc * sc * (Q_closed + (V_n * V_n).sum(1) / D)
            a_n, _, f = _project(a_n, Q_n, p)
            return V_n, a_n, sc * f

        V, a, F = jax.lax.fori_loop(0, n_steps, step, (V0, a, jnp.ones((N,), W.dtype)))
        W = F[:, None] * jnp.where(Mb, V, W)
    state = state._replace(W=W, a=a, w=w)
    state = model.rescale_heads(state, p, F)
    state = state._replace(heads=state.heads.at[t].set(jnp.where(p.use_heads == 1.0, state.a, 0.0)))
    return state._replace(t=t + 1)


def probe_signal(state: State, p: Params, wk: jnp.ndarray, Mk: jnp.ndarray, k) -> jnp.ndarray:
    """S_k = u·w_k*/|w_k*|² with u_j = Σ_i γ_kij a_i W_ij / pN: probe ``k``'s teacher ``wk``, mask ``Mk`` and
    stored head (task heads) or the shared readout, against the current W."""
    ak = jnp.where(p.use_heads == 1.0, state.heads[k], state.a)
    u = (ak @ (Mk * state.W)) / p.pN
    return (u * wk).sum() / (wk * wk).sum()


# ----------------------------------------------------------------------------
# chunked kernels: jit(scan over n_tasks of vmap(task + observable))
# ----------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def train_kernel(n_steps: int, n_tasks: int, closed: bool = False) -> Callable:
    """``f(state, p) -> state`` advancing every run by ``n_tasks`` tasks (no outputs); state donated."""
    vf = jax.vmap(functools.partial(train_task, n_steps=n_steps, closed=closed))

    def run(state, p):
        return jax.lax.scan(lambda s, _: (vf(s, p), None), state, None, length=n_tasks)[0]

    return jax.jit(run, donate_argnums=(0,))


@functools.lru_cache(maxsize=None)
def signal_kernel(n_steps: int, n_tasks: int, closed: bool = False) -> Callable:
    """``f(state, p, k) -> (state, S)`` with ``S`` (n_tasks, B) the signal of probe ``k`` after each task.

    The probe's teacher and mask do not change while it is followed, so they are regenerated once per device
    call, outside the scan."""
    vf = jax.vmap(functools.partial(train_task, n_steps=n_steps, closed=closed))
    vs = jax.vmap(probe_signal, in_axes=(0, 0, 0, 0, None))

    def run(state, p, k):
        N, D = state.W.shape[1:]
        wk, Mk = jax.vmap(lambda key, pN: probe_mask(key, k, N, D, pN))(state.key, p.pN)

        def body(s, _):
            s = vf(s, p)
            return s, vs(s, p, wk, Mk, k)
        return jax.lax.scan(body, state, None, length=n_tasks)

    return jax.jit(run, donate_argnums=(0,))


# ----------------------------------------------------------------------------
# job function and grouping
# ----------------------------------------------------------------------------
def signal_after_tasks(runs: Sequence, T_pre: int, n_follow: int, k: int, chunk: Optional[int] = None) -> np.ndarray:
    """``experiments.signal_after_tasks`` under synaptic gating: train ``T_pre`` tasks, then ``n_follow`` more
    recording probe ``k``'s signal after each, (n_follow, B); NaN columns for infeasible runs."""
    r0 = runs[0]
    assert all(r.solver == "pop" and r.gated == 1.0 for r in runs), "synaptic gating: population solver, gated readouts"
    state, p, ok, _ = X._batch(runs, T_pre + n_follow)
    if state is None:
        return np.full((n_follow, len(runs)), np.nan)
    closed = all(r.closed for r in runs)
    prog = pool.Progress(T_pre + n_follow)
    for n in X.chunks(T_pre, chunk):
        state = train_kernel(r0.n_steps, n, closed)(state, p)
        prog.add(n)
    S = []
    kk = jnp.int32(k)
    for n in X.chunks(n_follow, chunk):
        state, s = signal_kernel(r0.n_steps, n, closed)(state, p, kk)
        S.append(np.asarray(s))
        prog.add(n)
    return X._expand(np.concatenate(S, 0), ok)


def first_and_steady_signal(runs: Sequence, T: int, n_follow: int, chunk: Optional[int] = None) -> dict:
    """``experiments.first_and_steady_signal`` under synaptic gating: ``{"S0": (T, B), "ST": (n_follow, B)}`` from
    one stream of ``T + n_follow`` tasks, probe 0 followed over the first ``T`` and probe ``T`` over the rest."""
    r0 = runs[0]
    assert all(r.solver == "pop" and r.gated == 1.0 for r in runs), "synaptic gating: population solver, gated readouts"
    state, p, ok, _ = X._batch(runs, T + n_follow)
    if state is None:
        return {"S0": np.full((T, len(runs)), np.nan), "ST": np.full((n_follow, len(runs)), np.nan)}
    closed = all(r.closed for r in runs)
    prog = pool.Progress(T + n_follow)
    out = {}
    for name, k, length in (("S0", 0, T), ("ST", T, n_follow)):
        S = []
        kk = jnp.int32(k)
        for n in X.chunks(length, chunk):
            state, s = signal_kernel(r0.n_steps, n, closed)(state, p, kk)
            S.append(np.asarray(s))
            prog.add(n)
        out[name] = X._expand(np.concatenate(S, 0), ok)
    return out


def all_open(r) -> bool:
    """Whether the run's gate density is 1 (κ = √N): every mask is all ones and every unit gate open, so the
    synaptic run IS the unit-gated run (``tests/test_synapse.py``) and need not be trained again."""
    return r.pN >= r.N


def group_key(r) -> tuple:
    """Runs that can share one vmap batch here: (N, D, solver, n_steps, closed).  No block of active rows, so
    every κ of a width batches together (cf. ``experiments.group_key``)."""
    return (r.N, r.D, r.solver, r.n_steps, r.closed)


def job_cost(runs: Sequence, T: int) -> float:
    """Rough device time of a job over ``T`` tasks: per step all B·N·D synapses plus the launch overhead
    (``experiments.STEP_OVERHEAD``); a closed-form batch takes one update per task."""
    r = runs[0]
    steps = 1 if all(rr.closed for rr in runs) else r.n_steps
    return float(steps) * T * (len(runs) * r.N * r.D + X.STEP_OVERHEAD)
