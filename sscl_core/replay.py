"""Sample replay for the ungated learned shared readout (``replay_experiment/``): kernels and job functions.

The network is the μP student with every unit active and both layers learned (readout ``learned_shared``:
projection on, no heads), trained by SGD on the task stream of ``sscl_core.model``.  A replay run adds a buffer
of ``C`` stored samples (x, y), y = w_k·x/√D the output of the teacher the sample came from, and differs from
the μP run in two places only:

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
                  weights, |e| √(|a|²|x|²/D + |Wx|²/D) / N, recomputed for the whole buffer at every write: the
                  samples the network currently fits best go, the ones with a large gradient are kept

    ``none`` has no buffer and no replay: the μP run of ``learned_shared`` (``tests/test_replay.py`` checks it
    against ``experiments.train_task``).

The buffer is read (replayed) before it is written, so in both modes a task is trained against samples of
EARLIER tasks only, and the two modes differ in when the replay happens, not in what is replayed.

Every draw is a function of the run key and the task index: methods and capacities at one seed see the same
initial weights, teachers and fresh inputs (``RRun.rng_seed`` is ``Run.rng_seed``).

Job functions (``"sscl_core.replay:<name>"`` for ``pool.run_jobs``) mirror ``sscl_core.experiments``:
``signal_after_tasks``, ``norms_over_tasks``, ``loss_curves``, ``loss_slopes``.  A job is a list of ``RRun``
sharing ``group_key`` — every method and seed of one (N, capacity, mode) — run as ONE ``vmap`` batch with the
method a per-run constant, ``chunk`` tasks per device call.
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
from sscl_core import model, pool
from sscl_core.config import Run
from sscl_core.model import Params, State

#: Buffer write rules in display order; ``none`` = no buffer, no replay.
METHODS = ("none", "fifo", "random", "priority")
METHOD_ID = {m: i for i, m in enumerate(METHODS)}
#: Replay schedules.
MODES = ("sleep", "interleaved")


# ----------------------------------------------------------------------------
# runs
# ----------------------------------------------------------------------------
@dataclass
class RRun(Run):
    """A ``config.Run`` of the ungated learned shared readout with a replay buffer.

    Attributes
    ----------
    method : one of ``METHODS``
    capacity : buffer size C in samples (0 for ``none``)
    cap_label : C as typed (``"4D"``, ``"400"``); names the condition in figures and files
    write : samples of each task stored after it, m ≤ C
    replay_ratio : ρ, replayed samples per fresh sample
    mode : one of ``MODES`` (``"none"`` for method ``none``)
    """
    method: str = "none"
    capacity: int = 0
    cap_label: str = "0"
    write: int = 0
    replay_ratio: float = 1.0
    mode: str = "none"

    @property
    def spec(self) -> tuple:
        """Static shape of the task kernel: (n_steps, solver, C, m, B_rep, n_sleep); zeros without a buffer."""
        if self.method == "none":
            return (self.n_steps, self.solver, 0, 0, 0, 0)
        B = model.sgd_batch(self.solver)
        B_rep = max(1, int(round(self.replay_ratio * B))) if self.mode == "interleaved" else 0
        n_sleep = max(1, int(round(self.replay_ratio * self.n_steps))) if self.mode == "sleep" else 0
        return (self.n_steps, self.solver, int(self.capacity), int(min(self.write, self.capacity)), B_rep, n_sleep)

    @property
    def extra_state_bytes(self) -> int:
        """Device bytes of the buffer beside the μP state (``pool.job_bytes``)."""
        return 8 * int(self.capacity) * (self.D + 3)

    def label(self) -> str:
        return f"{super().label()} {self.method} C={self.cap_label} {self.mode}"


def group_key(r: RRun) -> tuple:
    """Runs that share one vmap batch: (N, D) and the kernel's static shape — every method and seed of one
    capacity and mode; the ``none`` runs (no buffer) form their own group."""
    return (r.N, r.D) + r.spec


def job_cost(runs: Sequence[RRun], T: int) -> float:
    """Rough device time of a job over ``T`` tasks, in the units of ``experiments.job_cost``."""
    r = runs[0]
    n_steps, _, C, _, B_rep, n_sleep = r.spec
    B = max(model.sgd_batch(r.solver), 1)
    steps = n_steps * (1.0 + B_rep / B) + n_sleep
    return float(steps) * T * (len(runs) * r.N * r.D + X.STEP_OVERHEAD)


# ----------------------------------------------------------------------------
# state
# ----------------------------------------------------------------------------
class Buffer(NamedTuple):
    """Stored samples of one run.  The filled slots are always 0 … n − 1 (empty slots fill in slot order)."""
    X: jnp.ndarray         # (C, D) inputs
    y: jnp.ndarray         # (C,)   targets w_k·x/√D
    stamp: jnp.ndarray     # (C,)   insertion counter of the sample in the slot
    task: jnp.ndarray      # (C,)   int32 task the sample came from
    n: jnp.ndarray         # int32  filled slots
    count: jnp.ndarray     # float  samples written so far


class RState(NamedTuple):
    s: State
    buf: Buffer


class RParams(NamedTuple):
    p: Params
    method: jnp.ndarray    # int32 ``METHOD_ID``


def empty_buffer(C: int, D: int) -> Buffer:
    """A buffer of ``C`` empty slots (one dummy slot when C = 0, never read)."""
    C = max(int(C), 1)
    z = jnp.zeros
    return Buffer(X=z((C, D)), y=z((C,)), stamp=z((C,)), task=z((C,), jnp.int32), n=jnp.int32(0), count=z(()))


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


def grad_norm(W, a, p: Params, Xs, y) -> jnp.ndarray:
    """(C,) norm of the gradient of each sample's loss ½e², e = a·Wx/(√D pN) − y, w.r.t. (W, a):
    |e| √(|a|²|x|²/D + |Wx|²/D) / pN — the priority of ``method = priority``."""
    D = W.shape[1]
    hx = W @ Xs.T / jnp.sqrt(D)                                     # (N, C)
    e = a @ hx / p.pN - y
    return jnp.abs(e) * jnp.sqrt((a @ a) * (Xs * Xs).sum(1) / D + (hx * hx).sum(0)) / p.pN


def write_buffer(buf: Buffer, state: State, p: Params, method, key, m: int) -> Buffer:
    """Store ``m`` fresh samples of the task ``state`` was just trained on (teacher ``state.w``, task
    ``state.t − 1``): into the empty slots first (in slot order), else over the ``m`` slots with the lowest score
    of the run's ``method`` (module docstring)."""
    C, D = buf.X.shape
    k_x, k_u = jax.random.split(key)
    Xn = jax.random.normal(k_x, (m, D), dtype=buf.X.dtype)
    yn = Xn @ state.w / jnp.sqrt(D)
    slot = jnp.arange(C)
    score = jnp.select([method == METHOD_ID["fifo"], method == METHOD_ID["random"]],
                       [buf.stamp, jax.random.uniform(k_u, (C,), dtype=buf.X.dtype)],
                       grad_norm(state.W, state.a, p, buf.X, buf.y))
    score = jnp.where(slot < buf.n, score, -(C - slot).astype(buf.X.dtype))   # empty slots first, lowest slot first
    idx = jnp.argsort(score)[:m]
    return Buffer(X=buf.X.at[idx].set(Xn), y=buf.y.at[idx].set(yn),
                  stamp=buf.stamp.at[idx].set(buf.count + jnp.arange(m, dtype=buf.X.dtype)),
                  task=buf.task.at[idx].set(state.t - 1), n=jnp.minimum(buf.n + m, C).astype(jnp.int32),
                  count=buf.count + m)


def replay_task(rs: RState, rp: RParams, *, spec: tuple, traced: bool = False):
    """One task of one run: train (with interleaved replay), sleep, write (module docstring).

    Parameters
    ----------
    rs : RState
    rp : RParams
    spec : (n_steps, solver, C, m, B_rep, n_sleep), static (``RRun.spec``)
    traced : bool (static)
        Also return the learning curve of the task: its population loss |u − w|²/(2D) before the first step and
        after each of the ``n_steps`` training and ``n_sleep`` sleep steps, (n_steps + n_sleep + 1,).

    Returns
    -------
    RState, or (RState, curve) with ``traced``
    """
    n_steps, solver, C, m, B_rep, n_sleep = spec
    B = model.sgd_batch(solver)
    assert B >= 1, "replay runs train by SGD (solver: sgd)"
    state, buf, p = rs.s, rs.buf, rp.p
    N, D = state.W.shape
    sqD = jnp.sqrt(D)
    k_teach, k_head, k_stream = model.task_keys(state.key, state.t)
    w_t, g_t = model.draw_task(k_teach, N, D, p.pN, p.gated)
    k_rep, k_sleep, k_write = jax.random.split(jax.random.fold_in(k_head, 1), 3)
    n_fill = jnp.maximum(buf.n, 1)
    has = (buf.n > 0).astype(state.W.dtype)                         # an empty buffer replays nothing

    def fresh(i):                                                   # the draws of ``model.train_block``
        k = jax.random.fold_in(k_stream, i)
        Xs = jax.random.normal(k, (B, D), dtype=state.W.dtype) if B > 1 else jax.random.normal(k, (D,), dtype=state.W.dtype)[None]
        return Xs, Xs @ w_t / sqD, jnp.ones((B,), state.W.dtype)

    def stored(key, i, n):
        idx = jax.random.randint(jax.random.fold_in(key, i), (n,), 0, n_fill)
        return buf.X[idx], buf.y[idx], jnp.full((n,), has)

    def mixed(i):
        (Xa, ya, wa), (Xb, yb, wb) = fresh(i), stored(k_rep, i, B_rep)
        return jnp.concatenate([Xa, Xb]), jnp.concatenate([ya, yb]), jnp.concatenate([wa, wb])

    wtr = w_t if traced else None
    if traced:
        r0 = state.a @ state.W / p.pN - w_t
        curve = [(r0 @ r0 / (2.0 * D))[None]]
    W, a, rows = sgd_phase(state.W, state.a, p, mixed if B_rep else fresh, n_steps, wtr)
    if traced:
        curve.append(rows)
    if n_sleep:
        W, a, rows = sgd_phase(W, a, p, lambda i: stored(k_sleep, i, B), n_sleep, wtr)
        if traced:
            curve.append(rows)
    state = state._replace(W=W, a=a, g=g_t, w=w_t, t=state.t + 1)
    if C:
        buf = write_buffer(buf, state, p, rp.method, k_write, m)
    rs = RState(state, buf)
    return (rs, jnp.concatenate(curve)) if traced else rs


# ----------------------------------------------------------------------------
# observables
# ----------------------------------------------------------------------------
#: Per-task quantities of ``run_weight_norms.py`` in output order.
NORM_KEYS = ("Q_all", "a2_all", "u2", "loss_end", "buf_age")


def task_norms(rs: RState, rp: RParams) -> jnp.ndarray:
    """(5,) ``NORM_KEYS`` after a task: mean Q_ii, mean a_i², |u|²/D and the population loss |u − w|²/(2D) of the
    task just trained (after its sleep), and the mean age in tasks of the buffered samples (0 = the task just
    trained; NaN without a buffer)."""
    s, p, buf = rs.s, rp.p, rs.buf
    D = s.W.shape[1]
    u = s.a @ s.W / p.pN
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


@functools.lru_cache(maxsize=None)
def signal_kernel(spec: tuple, n_tasks: int) -> Callable:
    """``f(rs, rp, k) -> (rs, S)`` with ``S`` (n_tasks, B) the signal of probe ``k`` after each task."""
    vf = jax.vmap(functools.partial(replay_task, spec=spec))
    vs = jax.vmap(X.probe_signal, in_axes=(0, 0, None))

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
    vn = jax.vmap(task_norms)

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
        assert r.gated == 0.0 and r.use_heads == 0.0, "replay runs are ungated, without task heads"
    state, p, ok, _ = X._batch(runs, 1)
    if state is None:
        return None, None, ok
    kept = [r for r, f in zip(runs, ok) if f]
    one = empty_buffer(r0.spec[2], r0.D)
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
           mode: str) -> list[RRun]:
    """Every base run (``config.make_runs`` at readout ``learned_shared``) × method × capacity as ``RRun``.

    ``none`` has no capacity and is listed once per base run (``cap_label = "0"``, ``mode = "none"``).
    """
    for m in methods:
        if m not in METHODS:
            raise ValueError(f"unknown replay method {m!r}; known: {METHODS}")
    if mode not in MODES:
        raise ValueError(f"unknown replay mode {mode!r}; known: {MODES}")
    out = []
    for r in base:
        if not r.solver.startswith("sgd"):
            raise ValueError("replay stores samples, so it needs solver: sgd")
        if r.gated == 1.0 or r.use_heads == 1.0 or r.learn_a != 1.0:
            raise ValueError(f"replay runs use the ungated learned shared readout, not {r.readout!r}")
        d = dataclasses.asdict(r)
        for method in METHODS:
            if method not in methods:
                continue
            if method == "none":
                out.append(RRun(**d))
                continue
            for cap in capacities:
                C, lab = parse_size(cap, r.D)
                m, _ = parse_size(write, r.D)
                out.append(RRun(**d, method=method, capacity=C, cap_label=lab, write=min(m, C),
                                replay_ratio=float(replay_ratio), mode=mode))
    return out
