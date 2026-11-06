"""Single-run kernels on a ``State`` pytree (PLAN.md §2–§3).

Every function here acts on ONE run; the caller adds the batch axis with
``jax.vmap``.  Per-run constants live in ``Params``; shapes (N, D, P) are
static and come from the arrays themselves.
"""
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp

import mup._x64  # noqa: F401  (enables x64)


class Params(NamedTuple):
    """
    Per-run constants (0-d float64 arrays; vmapped over a leading axis).

    Attributes
    ----------
    Delta0 : raw imbalance Δ₀ = δ₀/η
    eta : η_a = η_W = η
    h : population step in α (unused for SGD)
    pN : nominal active count κ√N (exactly N for an ungated run)
    lam : λ = Λ ā² (bookkeeping only)
    gated : 1 draws the gate ~ Bernoulli(pN/N); 0 activates every unit
    learn_a : 1 if the readout learns, 0 if frozen
    do_proj : 1 projects onto Δ_i = Δ₀ after every step
    use_heads : 1 for task-specific heads
    sigma0 : initial scale, W⁽⁰⁾_ij ~ N(0, σ₀²) (check 5; 1 in every sweep)
    """

    Delta0: jnp.ndarray    # raw imbalance Δ₀ = δ₀/η
    eta: jnp.ndarray       # η_a = η_W = η
    h: jnp.ndarray         # population step in α (unused for SGD)
    pN: jnp.ndarray        # nominal active count κ√N
    lam: jnp.ndarray       # λ = Λ ā²  (bookkeeping only)
    gated: jnp.ndarray     # 1: gate ~ Bernoulli(pN/N); 0: all units active (pN = N)
    learn_a: jnp.ndarray   # 1: readout learns, 0: frozen
    do_proj: jnp.ndarray   # 1: project onto Δ_i = Δ₀ after every step
    use_heads: jnp.ndarray  # 1: task-specific heads
    sigma0: jnp.ndarray    # initial scale of W (1 unless check 5 sweeps it)


class State(NamedTuple):
    """
    One run's state (PLAN.md §2).

    Attributes
    ----------
    W : (N, D) hidden rows
    a : (N,) shared readout, or the current task head
    g : (N,) current gate, 0/1 floats (all ones when ungated)
    w : (D,) current teacher
    heads : (P, N) stored task heads, masked by their gate (zeros unless use_heads)
    key : PRNGKey of the run
    t : int32 task index

    A probe's teacher and gate are not stored: task k's draws are a pure
    function of ``(key, k)`` (``task_keys``), so ``probe_task`` regenerates
    them bit for bit when the probe is evaluated.  Only the heads, which
    depend on the state at the task they were drawn, are kept.
    """
    W: jnp.ndarray         # (N, D) hidden rows
    a: jnp.ndarray         # (N,)   shared readout, or the current task head
    g: jnp.ndarray         # (N,)   current gate, 0/1 floats (all ones when ungated)
    w: jnp.ndarray         # (D,)   current teacher
    heads: jnp.ndarray     # (P, N) stored task heads (zeros unless use_heads)
    key: jnp.ndarray       # PRNGKey of the run
    t: jnp.ndarray         # int32 task index


# ----------------------------------------------------------------------------
# derived quantities
# ----------------------------------------------------------------------------
def key_vec(state: State) -> jnp.ndarray:
    """
    Key of the current task, g ⊙ a.

    Parameters
    ----------
    state : State

    Returns
    -------
    (N,) array
    """
    return state.g * state.a


def u_c(state: State, p: Params) -> jnp.ndarray:
    """
    u_c = (g⊙a)ᵀW / pN, eq. (student): the function the student computes on task c.

    Parameters
    ----------
    state : State
    p : Params

    Returns
    -------
    (D,) array
    """
    return key_vec(state) @ state.W / p.pN


def residual(state: State, p: Params) -> jnp.ndarray:
    """
    r_c = u_c − w_c.

    Parameters
    ----------
    state : State
    p : Params

    Returns
    -------
    (D,) array
    """
    return u_c(state, p) - state.w


def loss(state: State, p: Params) -> jnp.ndarray:
    """
    Population loss L_c = |u_c − w_c|² / (2D).

    Parameters
    ----------
    state : State
    p : Params

    Returns
    -------
    0-d array
    """
    r = residual(state, p)
    return r @ r / (2.0 * state.W.shape[1])


def forward(state: State, p: Params, x: jnp.ndarray) -> jnp.ndarray:
    """
    ŷ_c(x) = (g⊙a)ᵀ(Wx) / (pN √D).

    Parameters
    ----------
    state : State
    p : Params
    x : (D,) input

    Returns
    -------
    0-d array
    """
    D = state.W.shape[1]
    return key_vec(state) @ (state.W @ x) / (p.pN * jnp.sqrt(D))


def sample_loss(state: State, p: Params, x: jnp.ndarray) -> jnp.ndarray:
    """
    ℓ = ½(ŷ − y)² for one input, with y = w·x/√D (used by the gradient test).

    Parameters
    ----------
    state : State
    p : Params
    x : (D,) input

    Returns
    -------
    0-d array
    """
    D = state.W.shape[1]
    y = state.w @ x / jnp.sqrt(D)
    return 0.5 * (forward(state, p, x) - y) ** 2


def Q_diag(W: jnp.ndarray) -> jnp.ndarray:
    """
    Q_ii = |W_i|² / D for every row.

    Parameters
    ----------
    W : (N, D) array

    Returns
    -------
    (N,) array
    """
    return (W * W).sum(-1) / W.shape[1]


def imbalance(state: State, p: Params) -> jnp.ndarray:
    """
    Δ_i = a_i²/η − Q_ii/η, eq. (imb), the per-unit conserved quantity of gradient flow.

    Parameters
    ----------
    state : State
    p : Params

    Returns
    -------
    (N,) array
    """
    return state.a * state.a / p.eta - Q_diag(state.W) / p.eta


# ----------------------------------------------------------------------------
# steps
# ----------------------------------------------------------------------------
def sgd_step(state: State, p: Params, k: jnp.ndarray) -> State:
    """
    One online-SGD step, eqs. (sgdW)–(sgda), simultaneous update on the active units.

    Parameters
    ----------
    state : State
    p : Params
    k : PRNGKey
        Key of this step's input x ~ N(0, I_D).

    Returns
    -------
    State
        With ``W`` and ``a`` updated (``a`` only when ``learn_a``); inactive
        units receive exactly zero.
    """
    W, a, g, w = state.W, state.a, state.g, state.w
    D = W.shape[1]
    sqD = jnp.sqrt(D)
    x = jax.random.normal(k, (D,), dtype=W.dtype)
    hx = W @ x / sqD
    ga = g * a
    e = ga @ hx / p.pN - w @ x / sqD
    W_n = W - p.eta * ga[:, None] * e * x[None, :] / sqD
    a_n = a - p.learn_a * p.eta / D * g * e * hx
    return state._replace(W=W_n, a=a_n)


def pop_step(state: State, p: Params) -> State:
    """
    One population-gradient step, eq. (pop): the expected SGD step with step h in α.

    Parameters
    ----------
    state : State
    p : Params

    Returns
    -------
    State
    """
    W, a, g, w = state.W, state.a, state.g, state.w
    D = W.shape[1]
    ga = g * a
    r = ga @ W / p.pN - w
    W_n = W - p.h * p.eta * ga[:, None] * r[None, :]
    a_n = a - p.learn_a * p.h * p.eta * g * (W @ r) / D
    return state._replace(W=W_n, a=a_n)


def project_rows(state: State, p: Params) -> tuple[State, jnp.ndarray]:
    """
    Rescale every active unit onto Δ_i = Δ₀, eq. (proj), without touching the stored heads.

    Uses the cancellation-free product form: the larger root directly,
    A′ = ½(Δ₀ + √(Δ₀² + 4AB)) when Δ₀ ≥ 0 or B′ = ½(−Δ₀ + √(Δ₀² + 4AB))
    when Δ₀ < 0, and the other as AB divided by it, which preserves u_c
    exactly (A′B′ = AB) and pins Δ′ = A′ − B′ to Δ₀ to rounding of the
    larger root.  Taking A′ from the formula for Δ₀ < 0 would subtract two
    nearly equal numbers, with an error in Δ′ of order ε Δ₀²/A′: with
    the learned shared readout at δ₀ < 0, where a_i → 0, that reached
    0.05 at N = 100 (Stage 1, 2026-10-30).  A no-op when ``do_proj == 0``.

    Parameters
    ----------
    state : State
    p : Params

    Returns
    -------
    (State, f)
        ``f`` is the (N,) factor applied to ``W_i`` (1 for untouched units),
        so the caller can rescale the stored heads by 1/f — once per step
        (``project``) or once per task with the accumulated product
        (``run_task``), which is the same map at a fraction of the cost.
    """
    W, a, g = state.W, state.a, state.g
    Q = Q_diag(W)
    A = a * a / p.eta
    B = Q / p.eta
    AB = A * B
    big = 0.5 * (jnp.abs(p.Delta0) + jnp.hypot(p.Delta0, 2.0 * jnp.sqrt(AB)))   # the larger of A′, B′
    small = AB / big
    Ap = jnp.where(p.Delta0 >= 0.0, big, small)
    Bp = jnp.where(p.Delta0 >= 0.0, small, big)
    a_p = jnp.sign(a) * jnp.sqrt(p.eta * Ap)
    f = jnp.sqrt(p.eta * Bp / Q)
    act = (g == 1.0) & (p.do_proj == 1.0)
    a_new = jnp.where(act, a_p, a)
    W_new = jnp.where(act[:, None], W * f[:, None], W)
    f_safe = jnp.where(act, f, 1.0)
    return state._replace(W=W_new, a=a_new), f_safe


def rescale_heads(state: State, p: Params, f: jnp.ndarray) -> State:
    """
    Stored heads ← heads / f so that every stored u_k is unchanged (task-heads only).

    Parameters
    ----------
    state : State
    p : Params
    f : (N,) factor applied to the rows of W

    Returns
    -------
    State
    """
    heads = jnp.where(p.use_heads == 1.0, state.heads / f[None, :], state.heads)
    return state._replace(heads=heads)


def project(state: State, p: Params) -> State:
    """
    Projection eq. (proj) on the active units plus the stored-head rescaling.

    Preserves u_c exactly (A′B′ = AB) and every stored u_k.  A no-op when
    ``do_proj == 0``.

    Parameters
    ----------
    state : State
    p : Params

    Returns
    -------
    State
    """
    state, f = project_rows(state, p)
    return rescale_heads(state, p, f)


# ----------------------------------------------------------------------------
# random draws
# ----------------------------------------------------------------------------
def draw_readout(key: jnp.ndarray, Q: jnp.ndarray, p: Params) -> jnp.ndarray:
    """
    a_i = ε_i √(η (Δ₀ + Q_ii/η)), ε_i = ±1 equiprobable — the initial readout and fresh task heads.

    Parameters
    ----------
    key : PRNGKey
    Q : (N,) current Q_ii
    p : Params

    Returns
    -------
    (N,) array
        NaN where Δ₀ + Q_ii/η < 0 (infeasible δ₀; see ``protocol.feasible``).
    """
    eps = jax.random.rademacher(key, Q.shape, dtype=Q.dtype)
    return eps * jnp.sqrt(p.eta * (p.Delta0 + Q / p.eta))


def draw_task(key: jnp.ndarray, N: int, D: int, pN: jnp.ndarray, gated=1.0):
    """
    Teacher w ~ N(0, I_D) and gate g_i ~ Bernoulli(pN/N), independent; g = 1 when ungated.

    Parameters
    ----------
    key : PRNGKey
    N, D : int
    pN : 0-d array
        Nominal active count κ√N.
    gated : 0-d array or float
        1 draws the gate, 0 sets every unit active.

    Returns
    -------
    (w, g)
        (D,) teacher and (N,) 0/1 gate.
    """
    kw, kg = jax.random.split(key)
    w = jax.random.normal(kw, (D,), dtype=jnp.float64)
    g = (jax.random.uniform(kg, (N,), dtype=jnp.float64) < pN / N).astype(jnp.float64)
    g = jnp.where(gated == 1.0, g, jnp.ones_like(g))
    return w, g


def task_keys(key: jnp.ndarray, t) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    The three keys of task ``t`` of a run: (teacher and gate, head, input stream).

    Parameters
    ----------
    key : PRNGKey of the run
    t : int or 0-d int array
        Task index.

    Returns
    -------
    (k_teach, k_head, k_stream)
    """
    kt = jax.random.fold_in(key, t)
    k_teach, k_head, k_stream = jax.random.split(kt, 3)
    return k_teach, k_head, k_stream


def probe_task(key: jnp.ndarray, k, N: int, D: int, pN: jnp.ndarray, gated=1.0):
    """
    Teacher and gate of task ``k`` of a run, regenerated from the run key (bit for bit the draws of ``run_task``).

    Parameters
    ----------
    key : PRNGKey of the run
    k : int or 0-d int array
    N, D : int
    pN, gated : see ``draw_task``

    Returns
    -------
    (w_k, g_k)
        (D,) teacher and (N,) 0/1 gate.
    """
    return draw_task(task_keys(key, k)[0], N, D, pN, gated)


#: Attempts at redrawing the rows of W⁽⁰⁾ that make the initial readout imaginary.
MAX_REDRAW = 32


def draw_W0(key: jnp.ndarray, p: Params, N: int, D: int) -> jnp.ndarray:
    """
    W⁽⁰⁾ with rows ~ N(0, σ₀² I_D), the infeasible rows redrawn.

    A row is infeasible when Δ₀ + Q_ii/η < 0, i.e. ``draw_readout`` would give
    an imaginary a_i (δ₀ ≤ −Q_ii; at small D the row norms fluctuate by
    ~√(2/D), so δ₀ = −0.75 fails for about a third of the seeds at N ≤ 20).
    Only those rows are redrawn (up to ``MAX_REDRAW`` times, each from a
    fresh fold of the key), so a feasible first draw is unchanged and no seed
    is lost; a row still infeasible after the last attempt is left as drawn
    and shows up as NaN in ``a`` (``protocol.feasible``).

    Parameters
    ----------
    key : PRNGKey (the W part of the run key)
    p : Params
    N, D : int

    Returns
    -------
    (N, D) float64 array
    """
    def bad_rows(W):
        return p.Delta0 + Q_diag(W) / p.eta < 0.0

    def cond(c):
        W, i = c
        return jnp.logical_and(bad_rows(W).any(), i < MAX_REDRAW)

    def body(c):
        W, i = c
        Wn = p.sigma0 * jax.random.normal(jax.random.fold_in(key, i + 1), (N, D), dtype=jnp.float64)
        return jnp.where(bad_rows(W)[:, None], Wn, W), i + 1

    W0 = p.sigma0 * jax.random.normal(key, (N, D), dtype=jnp.float64)
    W0, _ = jax.lax.while_loop(cond, body, (W0, jnp.int32(0)))
    return W0


def init_state(key: jnp.ndarray, p: Params, N: int, D: int, P: int) -> State:
    """
    Fresh run: W ~ N(0, σ₀²) with infeasible rows redrawn (``draw_W0``), a from ``draw_readout`` at Δ_i = Δ₀, empty heads.

    Parameters
    ----------
    key : PRNGKey of the run
    p : Params
    N, D, P : int
        Width, input dimension and number of head slots (= number of tasks).

    Returns
    -------
    State
    """
    kW, ka = jax.random.split(key)
    W0 = draw_W0(kW, p, N, D)
    a0 = draw_readout(ka, Q_diag(W0), p)
    z = jnp.zeros
    return State(
        W=W0,
        a=a0,
        g=z((N,)),
        w=z((D,)),
        heads=z((P, N)),
        key=key,
        t=jnp.int32(0),
    )


# ----------------------------------------------------------------------------
# norm statistics (PLAN.md §12.1)
# ----------------------------------------------------------------------------
def masked_quantile(x: jnp.ndarray, mask: jnp.ndarray, q: float) -> jnp.ndarray:
    """Nearest-rank quantile of ``x`` over the entries where ``mask`` is true.

    Parameters
    ----------
    x : (N,) array
    mask : (N,) bool array
        Entries to include (the active units).
    q : float in [0, 1]
        Quantile; 0.5 is the median.

    Returns
    -------
    0-d array
        The value of ``x`` at rank ``round(q·(n−1))`` among the masked entries,
        ``NaN`` when no entry is masked in.  Shape-static, so it traces under
        ``vmap``/``jit`` (a sort plus one gather).
    """
    n = mask.sum()
    xs = jnp.sort(jnp.where(mask, x, jnp.inf))
    idx = jnp.clip(jnp.floor(q * (n - 1) + 0.5), 0, x.shape[0] - 1).astype(jnp.int32)
    return jnp.where(n > 0, xs[idx], jnp.nan)


def norm_stats(W: jnp.ndarray, a: jnp.ndarray, g: jnp.ndarray) -> dict:
    """``norm_stats_Q`` of ``Q_diag(W)``."""
    return norm_stats_Q(Q_diag(W), a, g)


def norm_stats_Q(Q: jnp.ndarray, a: jnp.ndarray, g: jnp.ndarray) -> dict:
    """Per-unit norm summaries of one state (the keys of PLAN.md §12.1 without a suffix).

    Parameters
    ----------
    Q : (N,) row norms Q_ii = |W_i|²/D
    a : (N,) readout (shared, or the current task head)
    g : (N,) gate, 0/1 floats

    Returns
    -------
    dict of 0-d arrays
        ``Q_act``, ``Q_all`` (mean of Q_i over active / all units), ``W_fro2``
        (Σ_i Q_i), ``a2_act``, ``a2_all``, ``aQ_act`` (mean over active units of
        |a_i|√Q_i), ``aQ_all``, ``aQ_max`` (over all units), and the active-unit
        quantiles ``Q_q05``, ``Q_q50``, ``Q_q95``, ``aQ_q05``, ``aQ_q95``.
        Active-unit means and quantiles are NaN when the gate is empty (a
        task that trained nothing), so host-side averages must use ``nanmean``.
    """
    act = g == 1.0
    n_act = g.sum()
    act_mean = lambda x: jnp.where(n_act > 0, (g * x).sum() / jnp.maximum(n_act, 1.0), jnp.nan)
    a2 = a * a
    aQ = jnp.abs(a) * jnp.sqrt(Q)
    return {
        "Q_act": act_mean(Q), "Q_all": Q.mean(), "W_fro2": Q.sum(),
        "a2_act": act_mean(a2), "a2_all": a2.mean(),
        "aQ_act": act_mean(aQ), "aQ_all": aQ.mean(), "aQ_max": aQ.max(),
        "Q_q05": masked_quantile(Q, act, 0.05), "Q_q50": masked_quantile(Q, act, 0.5),
        "Q_q95": masked_quantile(Q, act, 0.95),
        "aQ_q05": masked_quantile(aQ, act, 0.05), "aQ_q95": masked_quantile(aQ, act, 0.95),
    }


def heads_fro2(heads: jnp.ndarray, n_stored) -> jnp.ndarray:
    """Mean squared norm of the first ``n_stored`` stored heads (0 when none are stored).

    The heads are stored masked by their gate (``g_k ⊙ a_k``), so this is the
    norm of what the probes read; before 2026-11-06 the unmasked head was
    stored and about 85 % of this value came from entries outside the gate,
    which are never trained or read.

    Parameters
    ----------
    heads : (P, N) stored task heads; rows beyond ``n_stored`` are zero.
    n_stored : int or 0-d int array

    Returns
    -------
    0-d array
        ``Σ_k |heads[k]|² / max(n_stored, 1)``.
    """
    return (heads * heads).sum() / jnp.maximum(n_stored, 1)


# ----------------------------------------------------------------------------
# one task on the block of active rows (PLAN.md §3.2–§3.4 in the form of npmodel.step_task)
# ----------------------------------------------------------------------------
def block_rows(N: int, pN: float, gated: bool, n_sd: float = 8.0) -> int:
    """
    Rows of the active block a task's steps run on: N when ungated, else the
    gate count's mean plus ``n_sd`` standard deviations (capped at N).

    The gate count is Binomial(N, pN/N); with ``n_sd = 8`` the chance of a
    task exceeding the block is below 1e-11, and such a task is not run
    silently: ``train_block`` returns NaN for it.

    Parameters
    ----------
    N : int
    pN : float
        Nominal active count κ√N (the largest of the batch).
    gated : bool
    n_sd : float

    Returns
    -------
    int
    """
    if not gated:
        return N
    p = min(pN / N, 1.0)
    return int(min(N, math.ceil(pN + n_sd * math.sqrt(N * p * (1.0 - p))) + 1))


def _closed_form(Wb, ab, m, w, p, n_steps: int):
    """Population steps of a frozen, unprojected readout in closed form.

    With a fixed key the residual obeys r ← (1 − hηA) r, A = Σ_i (m_i a_i)²/pN,
    so after n steps W = W⁽⁰⁾ − (m⊙a) r₀ᵀ (1 − qⁿ)/A with q = 1 − hηA (the
    step loop to rounding; n·hη when A = 0).
    """
    ma = m * ab
    A = (ma * ma).sum() / p.pN
    r0 = ma @ Wb / p.pN - w
    c = p.h * p.eta
    q = 1.0 - c * A
    fac = jnp.where(A > 0.0, (1.0 - q ** n_steps) / jnp.where(A > 0.0, A, 1.0), n_steps * c)
    return Wb - fac * ma[:, None] * r0[None, :]


def train_block(W, a, g, w, p: Params, k_stream, *, n_steps: int, solver: str, K: int,
                closed: bool = False, traced: bool = False) -> dict:
    """
    ``n_steps`` solver steps, each followed by the row projection, on the block of the active rows.

    The steps change ``W`` only through g⊙a and ``a`` only through g, so the
    inactive rows receive exactly zero and the projection leaves them alone:
    only the first ``K`` rows of ``argsort(−g)`` (the active units first, then
    padding) are gathered, advanced and scattered back.  Within the task the
    block is kept as diag(s)·W̃ with the projection's row factors folded into
    ``s`` lazily, and the row norms Q_i are advanced analytically from the
    quantities the step already computes (for W_i ← W_i − c·a_i r:
    Q_i ← Q_i − 2c·a_i (W r)_i/D + c² a_i² |r|²/D), so one step reads the
    block twice (the two matrix–vector products) and writes it once —
    ``npmodel.step_task`` in JAX.  ``closed`` (population solver, frozen and
    unprojected readout) replaces the loop by ``_closed_form``.

    Parameters
    ----------
    W : (N, D)
    a, g : (N,)
    w : (D,)
    p : Params
    k_stream : PRNGKey of the task's input stream (SGD)
    n_steps : int (static), ≥ 1
    solver : {"pop", "sgd"} (static)
    K : int (static)
        Block rows (``block_rows``); N runs every row without a gather.
    closed : bool (static)
    traced : bool (static)
        Also return the per-step rows of ``protocol.TRACE_QUANT``.

    Returns
    -------
    dict
        ``W``, ``a`` after the task (the projection's last factors applied);
        ``f`` (N,) the product of the projection factors over the task (1 for
        untouched units), ``f_last`` those of the last step alone; ``Q_pre``, ``a_pre`` (N,) after the last step and
        before its projection; ``logf_sum`` = Σ_steps mean_active log f,
        ``logf_max`` the largest |log f| over the steps; ``uc_before``,
        ``uc_after`` the student function before and after the last
        projection; ``n_act``; and with ``traced`` ``rows`` (n_steps, 7).
        Every array is NaN when more than ``K`` units are active.
    """
    assert n_steps >= 1
    N, D = W.shape
    sqD = jnp.sqrt(D)
    n_act = g.sum()
    n_act1 = jnp.maximum(n_act, 1.0)
    full = K >= N
    if full:
        Wb, ab, m = W, a, g
    else:
        idx = jnp.argsort(-g)[:K]
        Wb, ab, m = W[idx], a[idx], g[idx]
    Q_all = Q_diag(W)
    Qb = Q_all if full else Q_all[idx]
    Q_inact = ((1.0 - g) * Q_all).sum()                         # fixed over the task
    act_mean = lambda x: jnp.where(n_act > 0, (m * x).sum() / n_act1, jnp.nan)
    proj = (m == 1.0) & (p.do_proj == 1.0)

    def solve(Wb, ab, Qb, s, i):
        """One solver step on the lazy block: (W̃, a, Q) → (W̃, a, Q), s unchanged."""
        ma = m * ab
        if solver == "sgd":
            x = jax.random.normal(jax.random.fold_in(k_stream, i), (D,), dtype=W.dtype)
            hx = s * (Wb @ x) / sqD
            e = ma @ hx / p.pN - w @ x / sqD
            c = p.eta * e / sqD
            a_new = ab - p.learn_a * p.eta / D * m * e * hx
            Q_new = Qb - (2.0 * c / D) * ma * (hx * sqD) + (c * c / D) * ma * ma * (x @ x)
            v = x
        else:
            r = (ma * s) @ Wb / p.pN - w
            Wr = s * (Wb @ r)
            c = p.h * p.eta
            a_new = ab - p.learn_a * c * m * Wr / D
            Q_new = Qb - (2.0 * c / D) * ma * Wr + (c * c / D) * ma * ma * (r @ r)
            v = r
        return Wb - c * (ma / s)[:, None] * v[None, :], a_new, Q_new

    def project(ab, Qb, s):
        """Eq. (proj) on the active rows from (a, Q); returns (a, Q, s, f)."""
        A = ab * ab / p.eta
        B = Qb / p.eta
        AB = A * B
        big = 0.5 * (jnp.abs(p.Delta0) + jnp.hypot(p.Delta0, 2.0 * jnp.sqrt(AB)))
        small = AB / big
        Ap = jnp.where(p.Delta0 >= 0.0, big, small)
        Bp = jnp.where(p.Delta0 >= 0.0, small, big)
        f = jnp.where(proj, jnp.sqrt(p.eta * Bp / Qb), 1.0)
        return jnp.where(proj, jnp.sign(ab) * jnp.sqrt(p.eta * Ap), ab), Qb * f * f, s * f, f

    def trace_row(Wb, a_p, Q_pre, Q_p, s, logf):
        r_post = (m * a_p * s) @ Wb / p.pN - w
        return jnp.stack([act_mean(Q_pre), act_mean(Q_p), (Q_inact + (m * Q_p).sum()) / N, act_mean(a_p * a_p),
                          act_mean(jnp.abs(a_p) * jnp.sqrt(Q_p)), act_mean(logf), r_post @ r_post / (2.0 * D)])

    def step(c, i):
        Wb, ab, Qb, s, lsum, lmax = c
        Wb, a_new, Q_new = solve(Wb, ab, Qb, s, i)
        a_p, Q_p, s, f = project(a_new, Q_new, s)
        logf = jnp.log(f)
        out = (Wb, a_p, Q_p, s, lsum + (m * logf).sum() / n_act1, jnp.maximum(lmax, jnp.abs(logf).max()))
        return out, (trace_row(Wb, a_p, Q_new, Q_p, s, logf) if traced else None)

    ones = jnp.ones(m.shape, W.dtype)
    zero = jnp.zeros((), W.dtype)
    if closed:
        Wb = _closed_form(Wb, ab, m, w, p, n_steps)
        Qb = Q_diag(Wb)
        a_pre_b, Q_pre_b, s, lsum, lmax = ab, Qb, ones, zero, zero
        uc_before = uc_after = (m * ab) @ Wb / p.pN
        f = ones
        rows = jnp.stack([trace_row(Wb, ab, Qb, Qb, ones, jnp.zeros_like(ones))] * n_steps) if traced else None
    else:
        c0 = (Wb, ab, Qb, ones, zero, zero)
        steps = jnp.arange(n_steps - 1, dtype=jnp.int32)
        if traced:
            (Wb, ab, Qb, s, lsum, lmax), rows = jax.lax.scan(step, c0, steps)
        else:
            Wb, ab, Qb, s, lsum, lmax = jax.lax.fori_loop(0, n_steps - 1, lambda i, c: step(c, i)[0], c0)
            rows = None
        # the last step: its pre-projection state is recorded and u_c checked across the projection
        Wb, a_pre_b, Q_pre_b = solve(Wb, ab, Qb, s, n_steps - 1)
        uc_before = (m * a_pre_b * s) @ Wb / p.pN
        ab, Qb, s, f = project(a_pre_b, Q_pre_b, s)
        logf = jnp.log(f)
        lsum = lsum + (m * logf).sum() / n_act1
        lmax = jnp.maximum(lmax, jnp.abs(logf).max())
        uc_after = (m * ab * s) @ Wb / p.pN
        if traced:
            rows = jnp.concatenate([rows, trace_row(Wb, ab, Q_pre_b, Qb, s, logf)[None, :]], 0)
    Wb = Wb * s[:, None]                                          # fold the row scale in
    if full:
        W_out, a_out, f_out, fl_out, Q_pre, a_pre = Wb, ab, s, f, Q_pre_b, a_pre_b
    else:
        W_out = W.at[idx].set(Wb)
        a_out = a.at[idx].set(ab)
        f_out = jnp.ones((N,), W.dtype).at[idx].set(s)
        fl_out = jnp.ones((N,), W.dtype).at[idx].set(f)
        Q_pre = Q_all.at[idx].set(Q_pre_b)
        a_pre = a.at[idx].set(a_pre_b)
    bad = n_act > K
    nan = lambda x: jnp.where(bad, jnp.nan, x)
    out = dict(W=nan(W_out), a=nan(a_out), f=nan(f_out), f_last=nan(fl_out), Q_pre=nan(Q_pre), a_pre=nan(a_pre),
               logf_sum=nan(lsum), logf_max=nan(lmax), n_act=n_act,
               uc_before=nan(uc_before), uc_after=nan(uc_after))
    if traced:
        out["rows"] = nan(rows)
    return out
