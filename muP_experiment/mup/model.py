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
    """Per-run constants (0-d float64 arrays; vmapped over a leading axis)."""

    Delta0: jnp.ndarray    # raw imbalance Δ₀ = δ₀/η
    eta: jnp.ndarray       # η_a = η_W = η
    h: jnp.ndarray         # population step in α (unused for SGD)
    pN: jnp.ndarray        # nominal active count κ√N
    lam: jnp.ndarray       # λ = Λ ā²  (bookkeeping only)
    learn_a: jnp.ndarray   # 1: readout learns, 0: frozen
    do_proj: jnp.ndarray   # 1: project onto Δ_i = Δ₀ after every step
    use_heads: jnp.ndarray  # 1: task-specific heads


class State(NamedTuple):
    W: jnp.ndarray         # (N, D) hidden rows
    a: jnp.ndarray         # (N,)   shared readout, or the current head
    g: jnp.ndarray         # (N,)   current gate, 0/1 floats
    w: jnp.ndarray         # (D,)   current teacher
    heads: jnp.ndarray     # (P, N) stored heads (zeros unless use_heads)
    probes_w: jnp.ndarray  # (P, D)
    probes_g: jnp.ndarray  # (P, N)
    key: jnp.ndarray       # PRNGKey of the run
    t: jnp.ndarray         # int32 task index


# ----------------------------------------------------------------------------
# derived quantities
# ----------------------------------------------------------------------------
def key_vec(state: State) -> jnp.ndarray:
    """Key of the current task, g ⊙ a."""
    return state.g * state.a


def u_c(state: State, p: Params) -> jnp.ndarray:
    """u_c = (g⊙a)ᵀW / pN, eq. (student)."""
    return key_vec(state) @ state.W / p.pN


def residual(state: State, p: Params) -> jnp.ndarray:
    return u_c(state, p) - state.w


def loss(state: State, p: Params) -> jnp.ndarray:
    """Population loss L_c = |u_c − w_c|² / (2D)."""
    r = residual(state, p)
    return r @ r / (2.0 * state.W.shape[1])


def forward(state: State, p: Params, x: jnp.ndarray) -> jnp.ndarray:
    """ŷ_c(x) = (g⊙a)ᵀ(Wx) / (pN √D)."""
    D = state.W.shape[1]
    return key_vec(state) @ (state.W @ x) / (p.pN * jnp.sqrt(D))


def sample_loss(state: State, p: Params, x: jnp.ndarray) -> jnp.ndarray:
    """ℓ = ½(ŷ − y)² for one input (used by the gradient test)."""
    D = state.W.shape[1]
    y = state.w @ x / jnp.sqrt(D)
    return 0.5 * (forward(state, p, x) - y) ** 2


def Q_diag(W: jnp.ndarray) -> jnp.ndarray:
    """Q_ii = |W_i|² / D."""
    return (W * W).sum(-1) / W.shape[1]


def imbalance(state: State, p: Params) -> jnp.ndarray:
    """Δ_i = a_i²/η − Q_ii/η, eq. (imb)."""
    return state.a * state.a / p.eta - Q_diag(state.W) / p.eta


# ----------------------------------------------------------------------------
# steps
# ----------------------------------------------------------------------------
def sgd_step(state: State, p: Params, k: jnp.ndarray) -> State:
    """One online-SGD step, eqs. (sgdW)–(sgda), simultaneous update."""
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
    """One population-gradient step, eq. (pop)."""
    W, a, g, w = state.W, state.a, state.g, state.w
    D = W.shape[1]
    ga = g * a
    r = ga @ W / p.pN - w
    W_n = W - p.h * p.eta * ga[:, None] * r[None, :]
    a_n = a - p.learn_a * p.h * p.eta * g * (W @ r) / D
    return state._replace(W=W_n, a=a_n)


def project(state: State, p: Params) -> State:
    """Rescale every active unit onto Δ_i = Δ₀, eq. (proj).

    Preserves u_c exactly (A′B′ = AB).  With ``use_heads`` the stored heads are
    multiplied by 1/f so that every stored u_k is unchanged as well.  A no-op
    when ``do_proj == 0``.
    """
    W, a, g = state.W, state.a, state.g
    Q = Q_diag(W)
    A = a * a / p.eta
    B = Q / p.eta
    Ap = 0.5 * (p.Delta0 + jnp.hypot(p.Delta0, 2.0 * jnp.sqrt(A * B)))
    Bp = A * B / Ap
    a_p = jnp.sign(a) * jnp.sqrt(p.eta * Ap)
    f = jnp.sqrt(p.eta * Bp / Q)
    act = (g == 1.0) & (p.do_proj == 1.0)
    a_new = jnp.where(act, a_p, a)
    W_new = jnp.where(act[:, None], W * f[:, None], W)
    f_safe = jnp.where(act, f, 1.0)
    heads_new = jnp.where(p.use_heads == 1.0, state.heads / f_safe[None, :], state.heads)
    return state._replace(W=W_new, a=a_new, heads=heads_new)


# ----------------------------------------------------------------------------
# random draws
# ----------------------------------------------------------------------------
def draw_readout(key: jnp.ndarray, Q: jnp.ndarray, p: Params) -> jnp.ndarray:
    """a_i = ε_i √(η (Δ₀ + Q_ii/η)), ε_i = ±1 — initial readout and fresh heads."""
    eps = jax.random.rademacher(key, Q.shape, dtype=Q.dtype)
    return eps * jnp.sqrt(p.eta * (p.Delta0 + Q / p.eta))


def draw_task(key: jnp.ndarray, N: int, D: int, pN: jnp.ndarray):
    """Teacher w ~ N(0, I_D) and gate g_i ~ Bernoulli(pN/N), independent."""
    kw, kg = jax.random.split(key)
    w = jax.random.normal(kw, (D,), dtype=jnp.float64)
    g = (jax.random.uniform(kg, (N,), dtype=jnp.float64) < pN / N).astype(jnp.float64)
    return w, g


def init_state(key: jnp.ndarray, p: Params, N: int, D: int, P: int) -> State:
    """Fresh run: W ~ N(0,1), a from ``draw_readout`` at Δ_i = Δ₀, empty probes."""
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
