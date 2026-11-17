"""Sample replay for the learned shared readout (``replay_experiment/``): kernels and job functions.

The network is the μP student with both layers learned and one shared readout (projection on, no heads),
trained by SGD on the task stream of ``sscl_core.model``.  A replay run adds a buffer of ``C`` stored samples
(x, y), y = w_k·x/√D the output of the teacher the sample came from, and differs from the μP run in two places
only:

replay (``mode``)
    ``sleep`` (default): the task trains exactly as in the μP experiment (``n_steps`` steps of B fresh samples),
    then ``n_sleep = round(ρ n_steps)`` further steps each sum the updates of B samples drawn uniformly from the
    buffer.  ``interleaved``: every one of the task's ``n_steps`` steps sums the updates of its B fresh samples
    and of ``B_rep = round(ρ B)`` buffer samples.  ρ is ``replay_ratio``, replayed samples per fresh sample, so
    both modes replay ρ n_steps B samples per task at the same η.  A replay sample enters a step exactly as a
    fresh one does (eqs. sgdW–sgda with its stored target), and the row projection follows every step.

write (``method``), after the task (and its sleep): ``m`` fresh samples of the task just trained are stored.
    Empty slots are filled first; in a full buffer they overwrite the ``m`` slots with the lowest score:

    ``fifo``      score = insertion order: the oldest samples go (the buffer holds the last C/m tasks)
    ``random``    score = a fresh uniform draw: m slots chosen at random go
    ``priority``  score = the norm of the gradient of the sample's loss ½e² w.r.t. (W, a) at the current
                  weights, under the sample's own gate (``grad_norm``), recomputed for the whole buffer at every
                  write: the samples the network currently fits best go, the ones with a large gradient are kept

    ``none`` has no buffer and no replay: the μP run of the same readout (``tests/test_replay.py`` checks it
    against ``experiments.train_task``).

gating (``gating``, static), the context gate every sample is trained and read out through:

    ``none``     every unit active (readout ``learned_shared``, pN = N): the replay runs before gating existed
    ``unit``     κ-sparse unit gates (readout ``gated_shared``): task k draws g_k ~ Bernoulli(κ/√N) per hidden
                 unit (``model.draw_task``) and its output is u_k = (g_k ⊙ a) W / pN, pN = κ√N
    ``synapse``  synaptic gating (``sscl_core.synapse``, readout ``gated_shared``): task k draws a mask
                 Γ_k ∈ {0,1}^{N×D} of i.i.d. Bernoulli(κ/√N) entries and u_k,j = Σ_i γ_kij a_i W_ij / pN

    A sample is always trained under the gate of the task it came from: a fresh sample under the current task's,
    a replayed sample under that of its stored task (the unit gate is stored in the slot, the synaptic mask is
    regenerated from the run key and the slot's task index, bit for bit ``synapse.draw_mask``).  The step is
    eqs. (sgdW)–(sgda) with the gate on, per sample: W_i ← W_i − (η/√D) a_i Σ_j γ_ji e_j x_j (γ_ji the unit gate
    g_ji, or the mask row Γ_j,i· entrywise) and a_i ← a_i − (η/D) Σ_j e_j h_ij with h_ij = (γ_ji ⊙ W_i)·x_j/√D
    and e_j = a·h_j/pN − y_j.  Inactive units (unit) and closed synapses (synapse) of a sample get exactly zero
    from it.  The projection follows every step on the units the step touched (unit: the union of the step's
    gates; synapse: every unit, as ``synapse.train_task``).  The traced loss, the end-of-task loss and the
    probe signals read the task's output through its own gate.

The buffer is read (replayed) before it is written, so in both modes a task is trained against samples of
EARLIER tasks only, and the two modes differ in when the replay happens, not in what is replayed.

Every draw is a function of the run key and the task index: methods and capacities at one seed see the same
initial weights, teachers, gates and fresh inputs (``RRun.rng_seed`` is ``Run.rng_seed``; the unit- and
synapse-gated runs of a seed share it, as in ``sscl_core.synapse``).

Job functions (``"sscl_core.replay:<name>"`` for ``pool.run_jobs``) mirror ``sscl_core.experiments``:
``signal_after_tasks``, ``norms_over_tasks``, ``loss_curves``, ``loss_slopes``.  A job is a list of ``RRun``
sharing ``group_key`` — every method and seed of one (N, capacity, mode, gating) — run as ONE ``vmap`` batch
with the method a per-run constant, ``chunk`` tasks per device call.
"""
from __future__ import annotations

import dataclasses
import functools
from dataclasses import dataclass
from typing import Callable, NamedTuple, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

import sscl_core._x64  # noqa: F401  (enables x64)
from sscl_core import experiments as X
from sscl_core import model, pool, synapse
from sscl_core.config import Run
from sscl_core.model import Params, State

#: Buffer write rules in display order; ``none`` = no buffer, no replay.
METHODS = ("none", "fifo", "random", "priority")
METHOD_ID = {m: i for i, m in enumerate(METHODS)}
#: Replay schedules.
MODES = ("sleep", "interleaved")
#: Context gating of the network (module docstring); ``none`` = every unit active.
GATINGS = ("none", "unit", "synapse")
#: Readout (``config.READOUTS``) of each gating: the shared learned readout, ungated or κ-sparse.
GATING_READOUT = {"none": "learned_shared", "unit": "gated_shared", "synapse": "gated_shared"}


# ----------------------------------------------------------------------------
# runs
# ----------------------------------------------------------------------------
@dataclass
class RRun(Run):
    """A ``config.Run`` of the learned shared readout (ungated or gated) with a replay buffer.

    Attributes
    ----------
    method : one of ``METHODS``
    capacity : buffer size C in samples (0 for ``none``)
    cap_label : C as typed (``"4D"``, ``"400"``); names the condition in figures and files
    write : samples of each task stored after it, m ≤ C
    replay_ratio : ρ, replayed samples per fresh sample
    mode : one of ``MODES`` (``"none"`` for method ``none``)
    gating : one of ``GATINGS``; the run's readout is ``GATING_READOUT[gating]``
    """
    method: str = "none"
    capacity: int = 0
    cap_label: str = "0"
    write: int = 0
    replay_ratio: float = 1.0
    mode: str = "none"
    gating: str = "none"

    @property
    def spec(self) -> tuple:
        """Static shape of the task kernel: (n_steps, solver, C, m, B_rep, n_sleep, gating); zeros without a buffer."""
        if self.method == "none":
            return (self.n_steps, self.solver, 0, 0, 0, 0, self.gating)
        B = model.sgd_batch(self.solver)
        B_rep = max(1, int(round(self.replay_ratio * B))) if self.mode == "interleaved" else 0
        n_sleep = max(1, int(round(self.replay_ratio * self.n_steps))) if self.mode == "sleep" else 0
        return (self.n_steps, self.solver, int(self.capacity), int(min(self.write, self.capacity)), B_rep, n_sleep,
                self.gating)

    @property
    def extra_state_bytes(self) -> int:
        """Device bytes beside the μP state (``pool.job_bytes``): the buffer with its stored unit gates, and under
        synaptic gating the per-sample masks of one step and of the priority pass."""
        n = 8 * int(self.capacity) * (self.D + self.N + 3)
        if self.gating == "synapse":
            b = max(model.sgd_batch(self.solver), 1) + self.spec[4]
            n += 8 * (b + 2) * self.N * self.D
        return n

    def kappa_text(self) -> str:
        """κ as typed, shortest form: ``1``, ``2.5``, ``sqrtN``."""
        lab = self.kappa_label or format(self.kappa, "g")
        try:
            return format(float(lab), "g")
        except ValueError:
            return lab

    def gate_label(self) -> str:
        """The gating as it names files and titles: ``ungated``, ``unit_k1``, ``synapse_k2``."""
        return "ungated" if self.gating == "none" else f"{self.gating}_k{self.kappa_text()}"

    def label(self) -> str:
        return f"{super().label()} {self.method} C={self.cap_label} {self.mode} gating={self.gating}"


def group_key(r: RRun) -> tuple:
    """Runs that share one vmap batch: (N, D, κ) and the kernel's static shape — every method and seed of one
    capacity, mode and gating; the ``none`` runs (no buffer) form their own group."""
    return (r.N, r.D, r.kappa) + r.spec


def job_cost(runs: Sequence[RRun], T: int) -> float:
    """Rough device time of a job over ``T`` tasks, in the units of ``experiments.job_cost``."""
    r = runs[0]
    n_steps, _, C, _, B_rep, n_sleep, gating = r.spec
    B = max(model.sgd_batch(r.solver), 1)
    steps = n_steps * (1.0 + B_rep / B) + n_sleep
    per = r.N * r.D * (1.0 if gating == "none" else 2.0 if gating == "unit" else 3.0 * B)   # masked steps: no lazy block
    return float(steps) * T * (len(runs) * per + X.STEP_OVERHEAD)


# ----------------------------------------------------------------------------
# state
# ----------------------------------------------------------------------------
class Buffer(NamedTuple):
    """Stored samples of one run.  The filled slots are always 0 … n − 1 (empty slots fill in slot order)."""
    X: jnp.ndarray         # (C, D) inputs
    y: jnp.ndarray         # (C,)   targets w_k·x/√D
    stamp: jnp.ndarray     # (C,)   insertion counter of the sample in the slot
    task: jnp.ndarray      # (C,)   int32 task the sample came from
    G: jnp.ndarray         # (C, N) unit gate of that task (all ones ungated; unused under synaptic gating)
    n: jnp.ndarray         # int32  filled slots
    count: jnp.ndarray     # float  samples written so far


class RState(NamedTuple):
    s: State
    buf: Buffer


class RParams(NamedTuple):
    p: Params
    method: jnp.ndarray    # int32 ``METHOD_ID``


def empty_buffer(C: int, D: int, N: int) -> Buffer:
    """A buffer of ``C`` empty slots (one dummy slot when C = 0, never read)."""
    C = max(int(C), 1)
    z = jnp.zeros
    return Buffer(X=z((C, D)), y=z((C,)), stamp=z((C,)), task=z((C,), jnp.int32), G=z((C, N)), n=jnp.int32(0),
                  count=z(()))


# ----------------------------------------------------------------------------
# steps
# ----------------------------------------------------------------------------
def sgd_phase(W, a, p: Params, sample: Callable, n_steps: int, w_trace=None):
    """``n_steps`` SGD steps on every row, each followed by the row projection, on samples from ``sample``.

    ``sample(i) -> (X (b, D), y (b,), wgt (b,))`` gives step i's inputs, targets and 0/1 weights; the step is the
    sum of the b single-sample updates of eqs. (sgdW)–(sgda) at the weights the step starts with, with the error
    e_j = a·Wx_j/(√D pN) − y_j.  The block is kept as diag(s)·W̃ and the row norms Q_i advanced analytically, as
    ``model.train_block`` does (the ungated case, K = N), so with fresh samples of one teacher this is
    ``train_block``'s SGD loop.

    Returns
    -------
    (W, a, rows)
        ``rows`` (n_steps,) the population loss |u − w_trace|²/(2D) after each step when ``w_trace`` is given,
        else None.
    """
    N, D = W.shape
    sqD = jnp.sqrt(D)
    proj = p.do_proj == 1.0

    def solve(Wb, ab, Qb, s, i):
        Xs, y, wgt = sample(i)
        hx = s[:, None] * (Wb @ Xs.T) / sqD                         # (N, b)
        e = (ab @ hx / p.pN - y) * wgt                              # (b,)
        c = p.eta / sqD
        v = e @ Xs                                                  # (D,): W_i ← W_i − c·a_i v
        he = hx @ e                                                 # (N,): W_i·v = √D (hx e)_i
        a_new = ab - p.learn_a * p.eta / D * he
        Q_new = Qb - (2.0 * c / D) * ab * (he * sqD) + (c * c / D) * ab * ab * (v @ v)
        return Wb - c * (ab / s)[:, None] * v[None, :], a_new, Q_new

    def project(ab, Qb, s):
        """Eq. (proj) from (a, Q), as ``model.train_block``."""
        A = ab * ab / p.eta
        Bq = Qb / p.eta
        AB = A * Bq
        big = 0.5 * (jnp.abs(p.Delta0) + jnp.hypot(p.Delta0, 2.0 * jnp.sqrt(AB)))
        small = AB / big
        Ap = jnp.where(p.Delta0 >= 0.0, big, small)
        Bp = jnp.where(p.Delta0 >= 0.0, small, big)
        f = jnp.where(proj, jnp.sqrt(p.eta * Bp / Qb), 1.0)
        return jnp.where(proj, jnp.sign(ab) * jnp.sqrt(p.eta * Ap), ab), Qb * f * f, s * f

    def step(c, i):
        Wb, ab, Qb, s = c
        Wb, ab, Qb = solve(Wb, ab, Qb, s, i)
        ab, Qb, s = project(ab, Qb, s)
        if w_trace is None:
            return (Wb, ab, Qb, s), None
        r = (ab * s) @ Wb / p.pN - w_trace
        return (Wb, ab, Qb, s), r @ r / (2.0 * D)

    c0 = (W, a, model.Q_diag(W), jnp.ones((N,), W.dtype))
    idx = jnp.arange(n_steps, dtype=jnp.int32)
    if w_trace is None:
        Wb, ab, _, s = jax.lax.fori_loop(0, n_steps, lambda i, c: step(c, i)[0], c0)
        rows = None
    else:
        (Wb, ab, _, s), rows = jax.lax.scan(step, c0, idx)
    return Wb * s[:, None], ab, rows


def gated_phase(W, a, p: Params, sample: Callable, n_steps: int, gating: str, trace=None):
    """``sgd_phase`` with a gate per sample (module docstring, "gating"): ``n_steps`` SGD steps, each followed by
    the row projection of the units it touched.

    ``sample(i) -> (X (b, D), y (b,), wgt (b,), G)`` with ``G`` the samples' gates: (b, N) unit gates under
    ``gating = "unit"``, (b, N, D) synaptic masks under ``"synapse"``.  The step sums the b single-sample updates,
    each at the weights the step starts with and through its own gate.  The weights are updated directly (no
    lazy row scaling: the rows a step touches differ from sample to sample).

    ``trace = (w, M)``: also return the population loss |u − w|²/(2D) after each step, u = a (M ⊙ W)/pN with M
    the traced task's gate, (N, 1) or (N, D).

    Returns
    -------
    (W, a, rows)
    """
    N, D = W.shape
    sqD = jnp.sqrt(D)
    c = p.eta / sqD
    on = p.do_proj == 1.0

    def step(carry, i):
        W, a = carry
        Xs, y, wgt, G = sample(i)
        if gating == "unit":
            Gt = G.T                                                # (N, b)
            hx = W @ Xs.T / sqD                                     # (N, b): W_i·x_j/√D (the gate is on a_i)
            e = ((Gt * a[:, None]) * hx).sum(0) / p.pN - y          # (b,)
            Ge = Gt * (e * wgt)[None, :]                            # (N, b): γ_ji e_j
            dW = c * a[:, None] * (Ge @ Xs)
            da = p.learn_a * p.eta / D * (Ge * hx).sum(1)
            act = ((Gt * (wgt > 0.0)[None, :]).max(1) > 0.0) & on   # units some weighted sample of the step opened
        elif gating == "synapse":
            hx = jnp.einsum("jid,id,jd->ij", G, W, Xs) / sqD        # (N, b): (Γ_j,i ⊙ W_i)·x_j/√D
            e = (a @ hx / p.pN - y) * wgt
            dW = c * a[:, None] * jnp.einsum("jid,j,jd->id", G, e, Xs)
            da = p.learn_a * p.eta / D * (hx @ e)
            act = jnp.broadcast_to(on, (N,))                        # every unit, as ``synapse.train_task``
        else:
            raise ValueError(f"gated_phase needs gating unit or synapse, not {gating!r}")
        W, a = W - dW, a - da
        a_p, _, f = synapse._project(a, model.Q_diag(W), p)
        a = jnp.where(act, a_p, a)
        W = jnp.where(act[:, None], W * f[:, None], W)
        if trace is None:
            return (W, a), None
        w_tr, M_tr = trace
        r = a @ (M_tr * W) / p.pN - w_tr
        return (W, a), r @ r / (2.0 * D)

    idx = jnp.arange(n_steps, dtype=jnp.int32)
    if trace is None:
        W, a = jax.lax.fori_loop(0, n_steps, lambda i, cw: step(cw, i)[0], (W, a))
        return W, a, None
    (W, a), rows = jax.lax.scan(step, (W, a), idx)
    return W, a, rows


def task_gate(run_key, k, N: int, D: int, p: Params, gating: str) -> jnp.ndarray:
    """The gate task ``k`` reads its output through, shaped to multiply W: (N, 1) unit gate (ones ungated) or
    (N, D) synaptic mask, regenerated from the run key."""
    if gating == "synapse":
        return synapse.task_mask(run_key, k, N, D, p.pN)
    if gating == "unit":
        return model.probe_task(run_key, k, N, D, p.pN, p.gated, p.shared)[1][:, None]
    return jnp.ones((N, 1), jnp.float64)


def grad_norm(W, a, p: Params, Xs, y, G=None, tasks=None, run_key=None, gating: str = "none") -> jnp.ndarray:
    """(C,) norm of the gradient of each sample's loss ½e² w.r.t. (W, a) under the sample's own gate — the
    priority of ``method = priority``.  With h = (γ ⊙ W) x/√D and e = a·h/pN − y:

        ungated / unit gate g:  |e| √(|g ⊙ a|²|x|²/D + Σ_i g_i h_i²) / pN
        synaptic mask Γ:        |e| √(Σ_ij a_i² Γ_ij x_j²/D + |h|²) / pN

    ``G`` (C, N) the stored unit gates (``gating = "unit"``); ``tasks`` (C,) and ``run_key`` regenerate the
    masks (``"synapse"``, one sample at a time)."""
    D = W.shape[1]
    sqD = jnp.sqrt(D)
    if gating == "synapse":
        def one(args):
            x, yy, k = args
            M = synapse.task_mask(run_key, k, W.shape[0], D, p.pN)
            h = (M * W) @ x / sqD
            e = a @ h / p.pN - yy
            return jnp.abs(e) * jnp.sqrt((a * a) @ (M @ (x * x)) / D + h @ h) / p.pN
        return jax.lax.map(one, (Xs, y, tasks))
    hx = W @ Xs.T / sqD                                             # (N, C)
    if gating == "unit":
        ga = G * a[None, :]                                         # (C, N)
        e = (ga * hx.T).sum(1) / p.pN - y
        return jnp.abs(e) * jnp.sqrt((ga * ga).sum(1) * (Xs * Xs).sum(1) / D + (G * hx.T * hx.T).sum(1)) / p.pN
    e = a @ hx / p.pN - y
    return jnp.abs(e) * jnp.sqrt((a @ a) * (Xs * Xs).sum(1) / D + (hx * hx).sum(0)) / p.pN


def write_buffer(buf: Buffer, state: State, p: Params, method, key, m: int, gating: str = "none") -> Buffer:
    """Store ``m`` fresh samples of the task ``state`` was just trained on (teacher ``state.w``, gate ``state.g``,
    task ``state.t − 1``): into the empty slots first (in slot order), else over the ``m`` slots with the lowest
    score of the run's ``method`` (module docstring)."""
    C, D = buf.X.shape
    k_x, k_u = jax.random.split(key)
    Xn = jax.random.normal(k_x, (m, D), dtype=buf.X.dtype)
    yn = Xn @ state.w / jnp.sqrt(D)
    slot = jnp.arange(C)
    prio = grad_norm(state.W, state.a, p, buf.X, buf.y, buf.G, buf.task, state.key, gating)
    score = jnp.select([method == METHOD_ID["fifo"], method == METHOD_ID["random"]],
                       [buf.stamp, jax.random.uniform(k_u, (C,), dtype=buf.X.dtype)], prio)
    score = jnp.where(slot < buf.n, score, -(C - slot).astype(buf.X.dtype))   # empty slots first, lowest slot first
    idx = jnp.argsort(score)[:m]
    return Buffer(X=buf.X.at[idx].set(Xn), y=buf.y.at[idx].set(yn),
                  stamp=buf.stamp.at[idx].set(buf.count + jnp.arange(m, dtype=buf.X.dtype)),
                  task=buf.task.at[idx].set(state.t - 1), G=buf.G.at[idx].set(state.g),
                  n=jnp.minimum(buf.n + m, C).astype(jnp.int32), count=buf.count + m)


def replay_task(rs: RState, rp: RParams, *, spec: tuple, traced: bool = False):
    """One task of one run: train (with interleaved replay), sleep, write (module docstring).

    Parameters
    ----------
    rs : RState
    rp : RParams
    spec : (n_steps, solver, C, m, B_rep, n_sleep, gating), static (``RRun.spec``)
    traced : bool (static)
        Also return the learning curve of the task: its population loss |u − w|²/(2D), u read through the task's
        gate, before the first step and after each of the ``n_steps`` training and ``n_sleep`` sleep steps,
        (n_steps + n_sleep + 1,).

    Returns
    -------
    RState, or (RState, curve) with ``traced``
    """
    n_steps, solver, C, m, B_rep, n_sleep, gating = spec
    B = model.sgd_batch(solver)
    assert B >= 1, "replay runs train by SGD (solver: sgd)"
    state, buf, p = rs.s, rs.buf, rp.p
    N, D = state.W.shape
    sqD = jnp.sqrt(D)
    k_teach, k_head, k_stream = model.task_keys(state.key, state.t)
    w_t, g_t = model.draw_task(k_teach, N, D, p.pN, p.gated, state.key, p.shared)
    if gating == "synapse":                                         # the mask of ``synapse.draw_mask``, same key
        M_t = synapse.mask_from_key(jax.random.split(k_teach)[1], N, D, p.pN)
    elif gating == "unit":
        M_t = g_t[:, None]
    else:
        M_t = jnp.ones((N, 1), state.W.dtype)
    k_rep, k_sleep, k_write = jax.random.split(jax.random.fold_in(k_head, 1), 3)
    n_fill = jnp.maximum(buf.n, 1)
    has = (buf.n > 0).astype(state.W.dtype)                         # an empty buffer replays nothing

    def fresh_gate(n):
        if gating == "synapse":
            return jnp.broadcast_to(M_t, (n, N, D))
        return jnp.broadcast_to(g_t, (n, N))

    def fresh(i):                                                   # the draws of ``model.train_block``
        k = jax.random.fold_in(k_stream, i)
        Xs = jax.random.normal(k, (B, D), dtype=state.W.dtype) if B > 1 else jax.random.normal(k, (D,), dtype=state.W.dtype)[None]
        return Xs, Xs @ w_t / sqD, jnp.ones((B,), state.W.dtype), fresh_gate(B)

    def stored(key, i, n):
        idx = jax.random.randint(jax.random.fold_in(key, i), (n,), 0, n_fill)
        if gating == "synapse":                                     # each sample under its own task's mask
            G = jax.vmap(lambda k: synapse.task_mask(state.key, k, N, D, p.pN))(buf.task[idx])
        else:
            G = buf.G[idx]                                          # each sample under its own task's unit gate
        return buf.X[idx], buf.y[idx], jnp.full((n,), has), G

    def mixed(i):
        (Xa, ya, wa, Ga), (Xb, yb, wb, Gb) = fresh(i), stored(k_rep, i, B_rep)
        return (jnp.concatenate([Xa, Xb]), jnp.concatenate([ya, yb]), jnp.concatenate([wa, wb]),
                jnp.concatenate([Ga, Gb]))

    train = mixed if B_rep else fresh
    sleep = lambda i: stored(k_sleep, i, B)
    if gating == "none":                                            # the ungated kernel, bit for bit as before
        wtr = w_t if traced else None
        phase = lambda W, a, smp, n: sgd_phase(W, a, p, lambda i: smp(i)[:3], n, wtr)
    else:
        tr = (w_t, M_t) if traced else None
        phase = lambda W, a, smp, n: gated_phase(W, a, p, smp, n, gating, tr)
    if traced:
        r0 = state.a @ (M_t * state.W) / p.pN - w_t
        curve = [(r0 @ r0 / (2.0 * D))[None]]
    W, a, rows = phase(state.W, state.a, train, n_steps)
    if traced:
        curve.append(rows)
    if n_sleep:
        W, a, rows = phase(W, a, sleep, n_sleep)
        if traced:
            curve.append(rows)
    state = state._replace(W=W, a=a, g=g_t, w=w_t, t=state.t + 1)
    if C:
        buf = write_buffer(buf, state, p, rp.method, k_write, m, gating)
    rs = RState(state, buf)
    return (rs, jnp.concatenate(curve)) if traced else rs


# ----------------------------------------------------------------------------
# observables
# ----------------------------------------------------------------------------
#: Per-task quantities of ``run_weight_norms.py`` in output order.
NORM_KEYS = ("Q_all", "a2_all", "u2", "loss_end", "buf_age")


def task_norms(rs: RState, rp: RParams, gating: str = "none") -> jnp.ndarray:
    """(5,) ``NORM_KEYS`` after a task: mean Q_ii, mean a_i², |u|²/D and the population loss |u − w|²/(2D) of the
    task just trained (after its sleep), u read through that task's gate, and the mean age in tasks of the
    buffered samples (0 = the task just trained; NaN without a buffer)."""
    s, p, buf = rs.s, rp.p, rs.buf
    N, D = s.W.shape
    u = s.a @ (task_gate(s.key, s.t - 1, N, D, p, gating) * s.W) / p.pN
    r = u - s.w
    filled = jnp.arange(buf.X.shape[0]) < buf.n
    age = jnp.where(filled, (s.t - 1 - buf.task).astype(u.dtype), 0.0).sum() / jnp.maximum(buf.n, 1)
    age = jnp.where((buf.n > 0) & (rp.method != 0), age, jnp.nan)
    return jnp.stack([model.Q_diag(s.W).mean(), (s.a * s.a).mean(), (u @ u) / D, (r @ r) / (2.0 * D), age])


#: Per-task summary of the learning curve (``loss_slopes``), in output order.
SLOPE_KEYS = ("area", "slope", "loss_start", "loss_end", "loss_sleep")


def slope_task(rs: RState, rp: RParams, *, spec: tuple):
    """``replay_task`` reduced to (5,) ``SLOPE_KEYS``: the area under the task's learning curve over its
    ``n_steps`` training steps and the fitted slope per step (``experiments.curve_area``, ``fit_slope``), the loss
    before the first step, after the last training step, and after the sleep (= ``loss_end`` without one)."""
    rs, L = replay_task(rs, rp, spec=spec, traced=True)
    Lt = L[: spec[0] + 1]
    return rs, jnp.stack([X.curve_area(Lt), X.fit_slope(Lt), L[0], Lt[-1], L[-1]])


# ----------------------------------------------------------------------------
# chunked kernels: jit(scan over n_tasks of vmap(task + observable))
# ----------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def train_kernel(spec: tuple, n_tasks: int) -> Callable:
    """``f(rs, rp) -> rs`` advancing every run by ``n_tasks`` tasks; state donated."""
    vf = jax.vmap(functools.partial(replay_task, spec=spec))
    return jax.jit(lambda rs, rp: jax.lax.scan(lambda s, _: (vf(s, rp), None), rs, None, length=n_tasks)[0],
                   donate_argnums=(0,))


def probe_signal(state: State, p: Params, k, gating: str = "none") -> jnp.ndarray:
    """S_k = u_k·w_k/|w_k|² of probe ``k`` against the current W, u_k read through task k's own gate
    (``experiments.probe_signal``; ``synapse.probe_signal`` under synaptic gating)."""
    if gating == "synapse":
        N, D = state.W.shape
        wk, Mk = synapse.probe_mask(state.key, k, N, D, p.pN, p.shared)
        return synapse.probe_signal(state, p, wk, Mk, k)
    return X.probe_signal(state, p, k)


@functools.lru_cache(maxsize=None)
def signal_kernel(spec: tuple, n_tasks: int) -> Callable:
    """``f(rs, rp, k) -> (rs, S)`` with ``S`` (n_tasks, B) the signal of probe ``k`` after each task."""
    vf = jax.vmap(functools.partial(replay_task, spec=spec))
    vs = jax.vmap(functools.partial(probe_signal, gating=spec[6]), in_axes=(0, 0, None))

    def run(rs, rp, k):
        def body(s, _):
            s = vf(s, rp)
            return s, vs(s.s, rp.p, k)
        return jax.lax.scan(body, rs, None, length=n_tasks)

    return jax.jit(run, donate_argnums=(0,))


@functools.lru_cache(maxsize=None)
def norms_kernel(spec: tuple, n_tasks: int) -> Callable:
    """``f(rs, rp) -> (rs, Y)`` with ``Y`` (n_tasks, B, 5) the ``NORM_KEYS`` after each task."""
    vf = jax.vmap(functools.partial(replay_task, spec=spec))
    vn = jax.vmap(functools.partial(task_norms, gating=spec[6]))

    def run(rs, rp):
        def body(s, _):
            s = vf(s, rp)
            return s, vn(s, rp)
        return jax.lax.scan(body, rs, None, length=n_tasks)

    return jax.jit(run, donate_argnums=(0,))


@functools.lru_cache(maxsize=None)
def loss_kernel(spec: tuple) -> Callable:
    """``f(rs, rp) -> (rs, L)`` with ``L`` (B, n_steps + n_sleep + 1) the learning curve of the next task."""
    return jax.jit(jax.vmap(functools.partial(replay_task, spec=spec, traced=True)), donate_argnums=(0,))


@functools.lru_cache(maxsize=None)
def slope_kernel(spec: tuple, n_tasks: int) -> Callable:
    """``f(rs, rp) -> (rs, Y)`` with ``Y`` (n_tasks, B, 5) the ``SLOPE_KEYS`` of each task."""
    vf = jax.vmap(functools.partial(slope_task, spec=spec))
    return jax.jit(lambda rs, rp: jax.lax.scan(lambda s, _: vf(s, rp), rs, None, length=n_tasks), donate_argnums=(0,))


def init_batch(runs: Sequence[RRun]):
    """Feasible runs of ``runs`` as one batch: (rs, rp, mask of kept runs); (None, None, mask) when none is."""
    r0 = runs[0]
    for r in runs:
        assert group_key(r) == group_key(r0), "job is not homogeneous"
        assert r.use_heads == 0.0 and r.readout == GATING_READOUT[r.gating], \
            f"a {r.gating!r}-gated replay run has readout {GATING_READOUT[r.gating]!r}, not {r.readout!r}"
    state, p, ok, _ = X._batch(runs, 1)
    if state is None:
        return None, None, ok
    kept = [r for r, f in zip(runs, ok) if f]
    one = empty_buffer(r0.spec[2], r0.D, r0.N)
    buf = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (len(kept),) + x.shape) + jnp.zeros((), x.dtype), one)
    return RState(state, buf), RParams(p, jnp.asarray([METHOD_ID[r.method] for r in kept], jnp.int32)), ok


# ----------------------------------------------------------------------------
# job functions
# ----------------------------------------------------------------------------
def signal_after_tasks(runs: Sequence[RRun], T_pre: int, n_follow: int, k: int, chunk: Optional[int] = None) -> np.ndarray:
    """Train ``T_pre`` tasks, then ``n_follow`` more recording probe ``k``'s signal S_k = u·w_k/|w_k|² after each
    (after its sleep): (n_follow, B), NaN columns for infeasible runs (``experiments.signal_after_tasks``)."""
    spec = runs[0].spec
    rs, rp, ok = init_batch(runs)
    if rs is None:
        return np.full((n_follow, len(runs)), np.nan)
    prog = pool.Progress(T_pre + n_follow)
    for n in X.chunks(T_pre, chunk):
        rs = train_kernel(spec, n)(rs, rp)
        prog.add(n)
    S = []
    kk = jnp.int32(k)
    for n in X.chunks(n_follow, chunk):
        rs, s = signal_kernel(spec, n)(rs, rp, kk)
        S.append(np.asarray(s))
        prog.add(n)
    return X._expand(np.concatenate(S, 0), ok)


def norms_over_tasks(runs: Sequence[RRun], T: int, chunk: Optional[int] = None) -> dict:
    """``{key: (B, T)}`` of ``NORM_KEYS`` after each of ``T`` tasks."""
    spec = runs[0].spec
    rs, rp, ok = init_batch(runs)
    if rs is None:
        return {q: np.full((len(runs), T), np.nan) for q in NORM_KEYS}
    Y = []
    prog = pool.Progress(T)
    for n in X.chunks(T, chunk):
        rs, y = norms_kernel(spec, n)(rs, rp)
        Y.append(np.asarray(y))
        prog.add(n)
    Y = X._expand(np.concatenate(Y, 0), ok)                         # (T, B, 5)
    return {q: Y[:, :, i].T for i, q in enumerate(NORM_KEYS)}


def loss_curves(runs: Sequence[RRun], tasks: Sequence[int], T: int, chunk: Optional[int] = None) -> np.ndarray:
    """(B, len(tasks), n_steps + n_sleep + 1) learning curves of the listed tasks of a ``T``-task stream; the
    tasks in between train untraced (``experiments.loss_curves``)."""
    r0 = runs[0]
    spec = r0.spec
    tasks = sorted(set(int(t) for t in tasks))
    assert all(0 <= t < T for t in tasks), "tasks must lie in [0, T)"
    rs, rp, ok = init_batch(runs)
    if rs is None:
        return np.full((len(runs), len(tasks), spec[0] + spec[5] + 1), np.nan)
    out = []
    prog = pool.Progress(T)
    t = 0
    for tt in tasks:
        for n in X.chunks(tt - t, chunk):
            rs = train_kernel(spec, n)(rs, rp)
            prog.add(n)
        rs, L = loss_kernel(spec)(rs, rp)
        out.append(np.asarray(L))
        prog.add(1)
        t = tt + 1
    return X._expand(np.stack(out, 1), ok, axis=0)


def loss_slopes(runs: Sequence[RRun], T: int, chunk: Optional[int] = None) -> dict:
    """``{key: (T, B)}`` over ``SLOPE_KEYS`` for each of the ``T`` tasks of a stream."""
    spec = runs[0].spec
    rs, rp, ok = init_batch(runs)
    if rs is None:
        return {q: np.full((T, len(runs)), np.nan) for q in SLOPE_KEYS}
    Y = []
    prog = pool.Progress(T)
    for n in X.chunks(T, chunk):
        rs, y = slope_kernel(spec, n)(rs, rp)
        Y.append(np.asarray(y))
        prog.add(n)
    Y = X._expand(np.concatenate(Y, 0), ok)                         # (T, B, 5)
    return {q: Y[:, :, i] for i, q in enumerate(SLOPE_KEYS)}


# ----------------------------------------------------------------------------
# grid → runs
# ----------------------------------------------------------------------------
def parse_size(v, D: int) -> tuple[int, str]:
    """A buffer size as typed → (samples, label): an integer is a number of samples, ``"4D"`` / ``"0.5D"`` a
    multiple of the input dimension D (D samples determine one teacher)."""
    s = str(v).strip()
    if s.upper().endswith("D"):
        f = float(s[:-1] or 1)
        return max(1, int(round(f * D))), f"{f:g}D"
    return max(1, int(float(s))), f"{int(float(s))}"


def expand(base: Sequence[Run], methods: Sequence[str], capacities: Sequence, write, replay_ratio: float,
           mode: str, gating: str = "none") -> list[RRun]:
    """Every base run (``config.make_runs`` at readout ``GATING_READOUT[gating]``) × method × capacity as ``RRun``.

    ``none`` has no capacity and is listed once per base run (``cap_label = "0"``, ``mode = "none"``).
    """
    for m in methods:
        if m not in METHODS:
            raise ValueError(f"unknown replay method {m!r}; known: {METHODS}")
    if mode not in MODES:
        raise ValueError(f"unknown replay mode {mode!r}; known: {MODES}")
    if gating not in GATINGS:
        raise ValueError(f"unknown gating {gating!r}; known: {GATINGS}")
    out = []
    for r in base:
        if not r.solver.startswith("sgd"):
            raise ValueError("replay stores samples, so it needs solver: sgd")
        if r.readout != GATING_READOUT[gating]:
            raise ValueError(f"{gating!r} gating runs readout {GATING_READOUT[gating]!r}, not {r.readout!r}")
        d = dataclasses.asdict(r)
        for method in METHODS:
            if method not in methods:
                continue
            if method == "none":
                out.append(RRun(**d, gating=gating))
                continue
            for cap in capacities:
                C, lab = parse_size(cap, r.D)
                m, _ = parse_size(write, r.D)
                out.append(RRun(**d, method=method, capacity=C, cap_label=lab, write=min(m, C),
                                replay_ratio=float(replay_ratio), mode=mode, gating=gating))
    return out
