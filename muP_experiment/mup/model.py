"""Single-run kernels on a ``State`` pytree (PLAN.md §2–§3).

Every function here acts on ONE run; the caller adds the batch axis with
``jax.vmap``.  Per-run constants live in ``Params``; shapes (N, D, P) are
static and come from the arrays themselves.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp

import mup  # noqa: F401  (enables x64)


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


class State(NamedTuple):
    """
    One run's state (PLAN.md §2).

    Attributes
    ----------
    W : (N, D) hidden rows
    a : (N,) shared readout, or the current task head
    g : (N,) current gate, 0/1 floats (all ones when ungated)
    w : (D,) current teacher
    heads : (P, N) stored task heads (zeros unless use_heads)
    probes_w : (P, D) stored teachers
    probes_g : (P, N) stored gates
    key : PRNGKey of the run
    t : int32 task index
    """
    W: jnp.ndarray         # (N, D) hidden rows
    a: jnp.ndarray         # (N,)   shared readout, or the current task head
    g: jnp.ndarray         # (N,)   current gate, 0/1 floats (all ones when ungated)
    w: jnp.ndarray         # (D,)   current teacher
    heads: jnp.ndarray     # (P, N) stored task heads (zeros unless use_heads)
    probes_w: jnp.ndarray  # (P, D)
    probes_g: jnp.ndarray  # (P, N)
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


def init_state(key: jnp.ndarray, p: Params, N: int, D: int, P: int) -> State:
    """
    Fresh run: W ~ N(0, 1), a from ``draw_readout`` at Δ_i = Δ₀, empty probes and heads.

    Parameters
    ----------
    key : PRNGKey of the run
    p : Params
    N, D, P : int
        Width, input dimension and number of probe slots (= number of tasks).

    Returns
    -------
    State
    """
    kW, ka = jax.random.split(key)
    W0 = jax.random.normal(kW, (N, D), dtype=jnp.float64)
    a0 = draw_readout(ka, Q_diag(W0), p)
    z = jnp.zeros
    return State(
        W=W0,
        a=a0,
        g=z((N,)),
        w=z((D,)),
        heads=z((P, N)),
        probes_w=z((P, D)),
        probes_g=z((P, N)),
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
    """Per-unit norm summaries of one state (the keys of PLAN.md §12.1 without a suffix).

    Parameters
    ----------
    W : (N, D) hidden rows
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
    Q = Q_diag(W)
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
