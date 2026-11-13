"""Nested-model test: is the Stage-1 student more than a palimpsest?  (``scripts/run_nested.py``)

A ladder of five models, each removing one ingredient from the one above it, run on the same task stream
(same teachers, gates, order and seeds) from the same forked weights:

``sgd``            the project's model: online SGD on the gated shared readout (``npmodel.step_task``,
                   solver "sgd", projection on), n = round(λD/c) steps per task, c = η ā² per step.
``gf``             the same network on the population loss (solver "pop" at a fine step): removes sampling noise.
``delta_matched``  readout frozen at the forked value, so the keys k_c = g_c ⊙ a are fixed; each task is one
                   relaxed projection (Kaczmarz) u_c ← u_c − θ_c e_c with θ_c = 1 − exp(−ϱ Λ_c ā²_c).  The one
                   number ϱ is gradient flow's measured rate over its bare hidden-channel rate on the same
                   tasks (``rate_ratio``): removes the second layer except for its effect on the learning rate.
``delta_raw``      the same with ϱ = 1, θ_c = 1 − exp(−Λ_c ā²_c): nothing fitted; removes the second layer.
``hebb``           fixed-size writes with decay, W ← (1 − γ)W + bΛ_c k_c w_cᵀ: removes error correction.
                   The write is a Hebbian rate b integrated over the task duration Λ_c; (b, γ) are tuned once at
                   the reference condition and held fixed (``tune_hebb``).

Scaling and clock are the project's: u_c = k_cᵀW/pN, error E_c = |u_c − w_c|²/D, key norm ā²_c = |k_c|²/pN.
A task lasts Λ_c and removes 1 − e^{−Λ_c ā²_c} of its residual through the hidden layer.  Under the default
clock (ii) (``Settings.clock = "realized"``) Λ_c = λ/ā²_c with the key norm the task starts with, so that
fraction is θ = 1 − e^{−λ} for every task and every δ₀; ``"nominal"`` is clock (i), Λ = λ/(1 + δ₀).  The
project's SGD runs clock (i) only (a fixed η).  For clock (ii) SGD here keeps its step count n = round(λD/c)
and takes η_c = c/ā²_c per task (c = ``sgd_contraction``), which is the same flow time Λ_c = η_c n/D as
gradient flow's and a per-step contraction η_c ā²_c = c that does not depend on the task.  With isotropic
inputs the population error is exact, so nothing here draws test samples.

Protocol (``run_seed``): gradient flow is burned in from the initialisation at δ₀ until the errors of recent
tasks are stationary (``burn_in``); (W, a) is copied; ``sgd``, ``gf`` and the two delta models start from the
copy and train the same next T tasks.  ``hebb`` is burned in separately on the same stream with the forked
readout as its (fixed) key vector (``run_hebb``).  Every task learned in the first ``T_learn`` tasks after
the fork is probed at every age of ``age_grid``, so each age averages the same tasks.

Everything a model stores about a probe is three quadratic forms, uu = |u|²/D, uw = u·w/D, ww = |w|²/D, so
E = uu − 2uw + ww, the project's signal is uw/ww, and the Hebb field (linear in b) is rescaled without a rerun.

Teachers (write-up, "Teachers"): w_k = √ρ U z_k + √(1 − ρ) ξ_k.  ``latent="common"`` takes z_k = e₁ for every
task, so two teachers have cosine ρ (the reading of "teacher similarity ρ" this experiment's predictions use);
``latent="sphere"`` draws z_k uniformly on the unit sphere of R^r as the write-up does.

The gate overlap of the experiment's design is the gate density p here: two independent gates share a
fraction p of their units.  The project's gate scale is κ = p√N.
"""
from __future__ import annotations

import dataclasses
import functools
import hashlib
import math
import os
import pickle
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from mup import theory

MODELS = ("sgd", "gf", "delta_matched", "delta_raw", "hebb")
#: models whose readout moves, for which the key-drift split is defined
LEARNED = ("sgd", "gf")
#: the pre-registered comparisons, (A, B): deviations are normalised by M_A(0)
COMPARISONS = (("sgd", "gf"), ("gf", "delta_matched"), ("delta_matched", "delta_raw"), ("delta_matched", "hebb"))
#: decision thresholds, fixed before running: max |ΔM|/M_A(0), relative τ½ difference, SEM multiple
TOL_M, TOL_TAU, N_SEM = 0.05, 0.10, 2.0
_OFFSET = 1 << 32          # stream indices may be negative (the Hebb burn-in reaches back before task 0)


@dataclass(frozen=True)
class Cond:
    """One condition of the sweep.  ``p`` is the gate density (the design's gate overlap), κ = p√N."""
    N: int
    D: int
    p: float
    rho: float
    delta0: float
    lam: float
    latent: str = "common"
    rank: int = 1

    @property
    def pN(self) -> float:
        return self.p * self.N

    @property
    def kappa(self) -> float:
        return self.p * math.sqrt(self.N)

    def unit(self) -> int:
        """The lifetime unit of the schedule: the half-life of eq. S_learned at this condition (ρ = 0 theory).
        It tracks the measured τ½ across p and N to within a factor of about two (150 against 257 tasks at
        N = 256, p = 0.1, δ₀ = 1; 10 against 8 at p = 0.5), which the defaults of ``Settings`` allow for."""
        return max(8, int(math.ceil(theory.mhalf("gated_shared", self.N, self.kappa, self.lam, self.delta0))))

    def stem(self) -> str:
        return f"N{self.N}_D{self.D}_p{self.p:g}_rho{self.rho:g}_d0{self.delta0:g}_lam{self.lam:g}"

    def label(self) -> str:
        return f"N={self.N} D={self.D} p={self.p:g} ρ={self.rho:g} δ₀={self.delta0:g} λ={self.lam:g}"


@dataclass(frozen=True)
class Settings:
    """Solver and schedule settings shared by every condition.

    gf_step : h η ā²_nom of a gradient-flow step (the project's sweeps use 0.1; finer here so that the Euler
        bias on θ, ≈ gf_step/2 relative, stays far below the 5 % decision threshold)
    sgd_contraction : η ā²_nom of SGD (≤ the project's 0.05); n = round(λD/this) steps per task
    burn, burn_max : burn-in in lifetime units: at least ``burn``, extended one unit at a time while the mean
        error of the last unit differs from the one before by more than ``burn_tol``, up to ``burn_max``
    learn, tau_max : tasks probed (T_learn) and the largest age, in lifetime units; T = T_learn + τ_max
    n_ages : geometric ages beyond the dense first 32
    clock : "realized" (clock (ii), Λ_c = λ/ā²_c, the project's default) or "nominal" (clock (i), Λ = λ/(1 + δ₀))
    backend : how SGD and gradient flow are stepped.  "numpy": ``npmodel.step_task``, one run at a time (CPU
        workers).  "jax": ``model.train_block`` vmapped over the seeds of a condition, one device call per task
        (``JaxTrainer``; the GPU path).  The population steps of the two agree to rounding; their SGD input
        streams are different draws of the same distribution.
    """
    gf_step: float = 0.02
    sgd_contraction: float = 0.05
    burn: float = 10.0
    burn_max: float = 15.0
    burn_tol: float = 0.05
    learn: float = 1.0
    tau_max: float = 3.0
    n_ages: int = 96
    clock: str = "realized"
    backend: str = "numpy"
    models: tuple = MODELS

    def schedule(self, cond: Cond) -> tuple[int, int, np.ndarray]:
        """(T_learn, T, ages) of a condition."""
        unit = cond.unit()
        T_learn = max(2, int(math.ceil(self.learn * unit)))
        tau_max = max(2, int(math.ceil(self.tau_max * unit)))
        return T_learn, T_learn + tau_max, age_grid(tau_max, self.n_ages)


def age_grid(tau_max: int, n_geom: int) -> np.ndarray:
    """Ages 0 … 32 densely, then ``n_geom`` geometrically spaced up to ``tau_max`` (integers, increasing)."""
    dense = np.arange(0, min(tau_max, 32) + 1)
    if tau_max <= 32:
        return dense
    geo = np.round(np.geomspace(32, tau_max, n_geom)).astype(int)
    return np.unique(np.concatenate([dense, geo]))


# ----------------------------------------------------------------------------
# the task stream
# ----------------------------------------------------------------------------
class Stream:
    """Teachers, gates and the initial weights of one (condition, seed): task ``t`` is a pure function of ``t``.

    The stream does not depend on δ₀ or λ, so those axes compare the same tasks; every model of a run reads it.
    """

    def __init__(self, cond: Cond, seed: int):
        self.cond, self.seed = cond, seed
        tag = f"nested|{seed}|{cond.N}|{cond.D}|{cond.p}|{cond.rho}|{cond.latent}|{cond.rank}"
        self.base = int.from_bytes(hashlib.sha256(tag.encode()).digest()[:4], "little")
        if cond.latent not in ("common", "sphere"):
            raise ValueError(f"unknown latent {cond.latent!r}")
        r = 1 if cond.latent == "common" else cond.rank
        q, _ = np.linalg.qr(np.random.default_rng([self.base, 0]).standard_normal((cond.D, r)))
        self.U = math.sqrt(cond.D) * q                                 # U = √D Ũ, Ũ orthonormal

    def task(self, t: int) -> tuple[np.ndarray, np.ndarray]:
        """(gate (N,) bool, teacher (D,)) of task ``t`` (any integer)."""
        c = self.cond
        rng = np.random.default_rng([self.base, 1, t + _OFFSET])
        w = rng.standard_normal(c.D)
        g = rng.random(c.N) < c.p
        if c.rho > 0.0:
            if c.latent == "common":
                z = np.ones(1)
            else:
                z = rng.standard_normal(c.rank)
                z /= np.linalg.norm(z)
            w = math.sqrt(c.rho) * self.U @ z + math.sqrt(1.0 - c.rho) * w
        return g, w

    def init(self) -> tuple[np.ndarray, np.ndarray]:
        """W⁽⁰⁾ ~ N(0, 1) and a_i = ±√(δ₀ + Q_ii) (write-up, "Initialization"; δ₀ ≥ 0 here)."""
        c = self.cond
        rng = np.random.default_rng([self.base, 2])
        W = rng.standard_normal((c.N, c.D))
        arg = c.delta0 + np.einsum("ij,ij->i", W, W) / c.D
        if (arg <= 0.0).any():
            raise ValueError(f"δ₀ = {c.delta0} makes the initial readout imaginary")
        return W, rng.choice(np.array([-1.0, 1.0]), size=c.N) * np.sqrt(arg)


# ----------------------------------------------------------------------------
# the five models: ``W``, ``a``, ``train(t, g, w)`` and ``u(K)`` = K W / pN for rows of keys K
# ----------------------------------------------------------------------------
class _Clocked:
    """Task duration of a model from the key it starts the task with."""
    cond: Cond
    clock: str

    def Lam(self, kk: float) -> float:
        """Λ_c for a key of squared norm ``kk`` = |k_c|²: λ/ā²_c (realized; 0 for an empty key) or λ/(1 + δ₀)."""
        c = self.cond
        if self.clock == "nominal":
            return c.lam / (1.0 + c.delta0)
        return c.lam * c.pN / kk if kk > 0.0 else 0.0


class FlowModel(_Clocked):
    """The gated shared readout of the project (both layers learn, projection onto Δ_i = Δ₀ after every step)
    under online SGD (``solver="sgd"``) or the population flow (``"pop"``).

    The steps are ``npmodel.step_task``, the NumPy port of ``model.sgd_step`` / ``pop_step`` and the projection
    (checked against them in ``tests/test_npmodel.py``); on the CPU it is several times faster per SGD step
    than the jitted ``model.train_block`` at these sizes and needs no compilation in the workers.
    """

    def __init__(self, cond: Cond, W: np.ndarray, a: np.ndarray, solver: str, st: Settings, base: int = 0):
        self.cond, self.solver, self.clock, self.base = cond, solver, st.clock, base
        self.W, self.a = np.array(W, dtype=np.float64, order="C"), np.array(a, dtype=np.float64)
        if st.clock not in ("realized", "nominal"):
            raise ValueError(f"unknown clock {st.clock!r}")
        if solver == "sgd":
            self.c = st.sgd_contraction                                # η ā² per step
            self.n_steps = max(1, int(round(cond.lam * cond.D / self.c)))
        elif solver == "pop":
            self.n_steps = max(1, int(math.ceil(cond.lam / st.gf_step - 1e-12)))
        else:
            raise ValueError(f"unknown solver {solver!r}")

    def task_rates(self, g: np.ndarray) -> tuple[float, float]:
        """(η, h) of a task with gate ``g`` from the key it starts with: the flow time is Λ_c = ``Lam``.
        SGD: Λ_c = η n/D, so η_c = c/ā²_c under clock (ii); population: η = 1, h = Λ_c/n.  (1, 0) for an empty key."""
        Lam = self.Lam(float((g * self.a * self.a).sum()))
        if Lam == 0.0:
            return 1.0, 0.0
        return (Lam * self.cond.D / self.n_steps, 0.0) if self.solver == "sgd" else (1.0, Lam / self.n_steps)

    def train(self, t: int, g: np.ndarray, w: np.ndarray) -> None:
        from mup import npmodel
        c = self.cond
        if not (g * self.a * self.a).sum() > 0.0:
            return
        eta, h = self.task_rates(g)
        rng = np.random.default_rng([self.base, 3, t + _OFFSET]) if self.solver == "sgd" else None
        npmodel.step_task(self.W, self.a, g.astype(np.float64), w, rng, c.delta0 / eta, eta, h, c.pN, 1.0, True,
                          self.n_steps, self.solver)

    def u(self, K: np.ndarray) -> np.ndarray:
        return K @ self.W / self.cond.pN


class DeltaModel(_Clocked):
    """Frozen readout; each task is one relaxed projection, W ← W − θ_c pN k_c e_cᵀ/|k_c|², i.e.
    u_c ← u_c − θ_c e_c, with θ_c = 1 − exp(−``rate``·Λ_c ā²_c) (``rate`` = 1: the raw delta rule; under the
    realized clock Λ_c ā²_c = λ, so θ_c = 1 − e^{−rate·λ} for every task)."""

    def __init__(self, cond: Cond, W: np.ndarray, a: np.ndarray, rate: float = 1.0, clock: str = "realized"):
        self.cond, self.W, self.a, self.rate, self.clock = cond, np.array(W), np.array(a), float(rate), clock

    def theta(self, kk: float) -> float:
        return -math.expm1(-self.rate * self.Lam(kk) * kk / self.cond.pN)

    def train(self, t: int, g: np.ndarray, w: np.ndarray) -> None:
        act = np.flatnonzero(g)
        k = self.a[act]
        kk = float(k @ k)
        if kk == 0.0:
            return
        pN = self.cond.pN
        e = k @ self.W[act] / pN - w
        self.W[act] -= (self.theta(kk) * pN / kk) * np.outer(k, e)

    def u(self, K: np.ndarray) -> np.ndarray:
        return K @ self.W / self.cond.pN


class HebbModel(_Clocked):
    """Palimpsest with frozen keys: W ← (1 − γ)W + bΛ_c k_c w_cᵀ, started from W = 0 (the write adds
    bΛ_c ā²_c w_c to u_c: bλ w_c under the realized clock).

    The field is linear in b, so it is stored for b = 1 and ``u`` returns the unit-b field; the decay is a
    scalar factor folded in lazily (W = ``s``·``Wt``), so a task writes its active rows only.
    """

    def __init__(self, cond: Cond, a: np.ndarray, gamma: float, clock: str = "realized"):
        self.cond, self.a, self.gamma, self.clock = cond, np.array(a), float(gamma), clock
        self.Wt, self.s = np.zeros((cond.N, cond.D)), 1.0

    @property
    def W(self) -> np.ndarray:
        return self.s * self.Wt

    def train(self, t: int, g: np.ndarray, w: np.ndarray) -> None:
        self.s *= 1.0 - self.gamma
        if self.s < 1e-100:
            self.Wt *= self.s
            self.s = 1.0
        act = np.flatnonzero(g)
        k = self.a[act]
        self.Wt[act] += (self.Lam(float(k @ k)) / self.s) * np.outer(k, w)

    def u(self, K: np.ndarray) -> np.ndarray:
        return self.s * (K @ self.Wt) / self.cond.pN


@functools.lru_cache(maxsize=None)
def _jax_kernel(n_steps: int, solver: str, K: int):
    """Jitted, vmapped ``model.train_block`` for one task of a batch of runs: (W, a, g, w, params, key) → (W, a)."""
    import jax
    from mup.model import train_block

    def one(W, a, g, w, p, key):
        out = train_block(W, a, g, w, p, key, n_steps=n_steps, solver=solver, K=K)
        return out["W"], out["a"]
    return jax.jit(jax.vmap(one))


class JaxTrainer:
    """Trains a batch of ``FlowModel`` (one condition and solver, one per seed) on the device.

    One call per task runs all of its steps for every seed inside one jitted, vmapped ``model.train_block``
    (a ``fori_loop`` on the device), so on a GPU the batch costs about what one run does.  The weights stay on
    the device between tasks; after each task they are copied to the models' ``W`` and ``a`` (NumPy), which is
    what the probes read.  The task duration is ``FlowModel.task_rates``, as on the NumPy path.
    """

    def __init__(self, models: Sequence["FlowModel"]):
        import jax
        import jax.numpy as jnp
        import mup._x64  # noqa: F401
        from mup.model import Params, block_rows
        self.models, self._jnp, self._Params = list(models), jnp, Params
        m, c = self.models[0], self.models[0].cond
        self.K = block_rows(c.N, c.pN, True)
        self.kernel = _jax_kernel(m.n_steps, m.solver, self.K)
        self.W = jnp.asarray(np.stack([x.W for x in self.models]))
        self.a = jnp.asarray(np.stack([x.a for x in self.models]))
        keys = jnp.stack([jax.random.PRNGKey(x.base) for x in self.models])
        self._fold = jax.jit(jax.vmap(jax.random.fold_in, in_axes=(0, None)))
        self._keys = keys

    def __call__(self, t: int, gs: Sequence[np.ndarray], ws: Sequence[np.ndarray]) -> None:
        jnp, c = self._jnp, self.models[0].cond
        G = np.stack(gs).astype(np.float64)
        if G.sum(1).max() > self.K:
            raise RuntimeError(f"{int(G.sum(1).max())} active units exceed the block of {self.K} rows")
        eta, h = np.array([m.task_rates(g) for m, g in zip(self.models, gs)]).T
        B = len(self.models)
        f = lambda v: jnp.asarray(np.broadcast_to(np.asarray(v, np.float64), (B,)))
        p = self._Params(Delta0=f(c.delta0 / eta), eta=f(eta), h=f(h), pN=f(c.pN), lam=f(c.lam), gated=f(1.0),
                         learn_a=f(1.0), do_proj=f(1.0), use_heads=f(0.0), sigma0=f(1.0), clock=f(0.0))
        self.W, self.a = self.kernel(self.W, self.a, jnp.asarray(G), jnp.asarray(np.stack(ws)), p,
                                     self._fold(self._keys, t % (1 << 31)))
        W, a = np.asarray(self.W), np.asarray(self.a)
        if not np.isfinite(a).all():
            raise RuntimeError(f"non-finite readout after task {t}")
        for b, m in enumerate(self.models):
            m.W, m.a = W[b], a[b]


# ----------------------------------------------------------------------------
# training with probes
# ----------------------------------------------------------------------------
def _quad(U: np.ndarray, Wt: np.ndarray) -> np.ndarray:
    """(3, n): |u|²/D, u·w/D, |w|²/D for the rows of ``U`` and ``Wt``."""
    D = U.shape[1]
    return np.stack([np.einsum("ij,ij->i", U, U), np.einsum("ij,ij->i", U, Wt), np.einsum("ij,ij->i", Wt, Wt)]) / D


def errors(q: np.ndarray, b: float = 1.0) -> np.ndarray:
    """E = b²uu − 2b uw + ww from stacked quadratics ``q`` (3, …); ``b`` rescales a unit-b Hebb field."""
    return b * b * q[0] - 2.0 * b * q[1] + q[2]


class _Probe:
    """The bookkeeping of ``follow`` for one model on one stream."""

    def __init__(self, model, stream: Stream, t0: int, T: int, T_learn: int, ages: np.ndarray, value: bool):
        c = stream.cond
        self.model, self.stream, self.t0, self.T_learn, self.ages, self.value = model, stream, t0, T_learn, ages, value
        self.G, self.Wst = np.zeros((T_learn, c.N)), np.zeros((T_learn, c.D))
        self.A_learn = np.zeros((T_learn, c.N)) if value else None
        self.pre, self.post, self.lam_raw = np.zeros((3, T)), np.zeros((3, T)), np.zeros(T)
        self.acc, self.acc_v = np.zeros((3, len(ages))), np.zeros((3, len(ages)))
        self.count = np.zeros(len(ages), int)

    def before(self, j: int) -> tuple[np.ndarray, np.ndarray]:
        m = self.model
        self.g, self.w = g, w = self.stream.task(self.t0 + j)
        k = g * m.a
        self.pre[:, j] = _quad(m.u(k[None]), w[None])[:, 0]
        self.lam_raw[j] = m.Lam(float(k @ k)) * float(k @ k) / m.cond.pN
        return g, w

    def after(self, j: int) -> None:
        m, g, w = self.model, self.g, self.w
        self.post[:, j] = _quad(m.u((g * m.a)[None]), w[None])[:, 0]
        if j < self.T_learn:
            self.G[j], self.Wst[j] = g, w
            if self.value:
                self.A_learn[j] = m.a
        ci = j - self.ages
        ok = (ci >= 0) & (ci < self.T_learn)
        if ok.any():
            idx = ci[ok]
            self.acc[:, ok] += _quad(m.u(self.G[idx] * m.a), self.Wst[idx])
            self.count[ok] += 1
            if self.value:
                self.acc_v[:, ok] += _quad(m.u(self.G[idx] * self.A_learn[idx]), self.Wst[idx])

    def result(self) -> dict:
        n = np.maximum(self.count, 1)
        return {"pre": self.pre, "post": self.post, "lam_raw": self.lam_raw, "age": self.acc / n,
                "age_value": self.acc_v / n if self.value else None, "count": self.count}


def follow(model, stream, t0: int, T: int, T_learn: int = 0, ages: Optional[np.ndarray] = None,
           value: bool = False, trainer=None, progress=None):
    """Train ``model`` on tasks t0 … t0 + T − 1 of ``stream`` and probe what it stores.

    Per task: the quadratics of the task's own function before (``pre``) and after (``post``) training, and
    the raw hidden-channel exponent Λ ā²_c of the key the task starts with (``lam_raw``).  Each of the first
    ``T_learn`` tasks is probed at every age of ``ages`` after it was learned, through its gate and the
    CURRENT readout (``age``, the mean over those tasks; needs T ≥ T_learn + max age for a full average), and
    with ``value`` also through the readout it had when learned and the current W (``age_value``: the error
    had the key not moved).

    ``model`` and ``stream`` may be lists of equal length (the seeds of a condition), advanced task by task
    together; ``trainer(t, gates, teachers)`` then trains all of them at once (``JaxTrainer``) instead of
    each model's own ``train``.  ``progress(j, T)`` is called at every tenth of the tasks.

    Returns
    -------
    dict, or a list of them for lists
        ``pre``, ``post`` (3, T); ``lam_raw`` (T,); ``age`` and ``age_value`` (3, n_ages); ``count`` (n_ages,).
    """
    single = not isinstance(model, (list, tuple))
    models, streams = ([model], [stream]) if single else (list(model), list(stream))
    ages = np.zeros(0, int) if ages is None else np.asarray(ages, int)
    probes = [_Probe(m, s, t0, T, T_learn, ages, value) for m, s in zip(models, streams)]
    for j in range(T):
        tasks = [p.before(j) for p in probes]
        if trainer is None:
            for m, (g, w) in zip(models, tasks):
                m.train(t0 + j, g, w)
        else:
            trainer(t0 + j, [g for g, _ in tasks], [w for _, w in tasks])
        for p in probes:
            p.after(j)
        if progress is not None and (j + 1) % max(1, T // 10) == 0:
            progress(j + 1, T)
    res = [p.result() for p in probes]
    return res[0] if single else res


def burn_in(model, stream, st: Settings, trainer=None) -> dict:
    """Train from task 0 until the errors of recent tasks are stationary; returns ``{"t": tasks run, "drift"}``.

    At least ``st.burn`` lifetime units; then one unit at a time while the mean error before or after
    training over the last unit differs from the unit before by more than ``st.burn_tol`` (relative), up to
    ``st.burn_max`` units.  ``drift`` is the last such difference.  For lists (the seeds of a condition,
    ``follow``) the largest drift over the seeds decides, so every seed forks at the same task.
    """
    single = not isinstance(model, (list, tuple))
    models, streams = ([model], [stream]) if single else (list(model), list(stream))
    unit = streams[0].cond.unit()
    T, T_max = int(math.ceil(st.burn * unit)), int(math.ceil(max(st.burn, st.burn_max) * unit))
    pre, post = np.zeros((len(models), 0)), np.zeros((len(models), 0))
    while True:
        rs = follow(models, streams, pre.shape[1], T - pre.shape[1], trainer=trainer)
        pre = np.concatenate([pre, np.stack([errors(r["pre"]) for r in rs])], 1)
        post = np.concatenate([post, np.stack([errors(r["post"]) for r in rs])], 1)
        drift = float("nan")
        if T >= 2 * unit:
            drift = max(float(np.abs(x[:, -unit:].mean(1) / x[:, -2 * unit:-unit].mean(1) - 1.0).max()) for x in (pre, post))
        if not drift > st.burn_tol or T >= T_max:
            return {"t": T, "drift": drift, "E_pre": float(pre[:, -unit:].mean()), "E_post": float(post[:, -unit:].mean())}
        T = min(T + unit, T_max)


def rate_ratio(r: dict) -> float:
    """ϱ: the exponent a learned-readout run realizes over its bare hidden-channel exponent, pooled over tasks,
    ϱ = Σ_c ½ ln(E_pre/E_post) / Σ_c Λ ā²_c.  ϱ = 1 for a frozen readout; the readout channel adds to it."""
    E0, E1 = errors(r["pre"]), errors(r["post"])
    ok = (E0 > 0) & (E1 > 0) & (r["lam_raw"] > 0)
    return float(0.5 * np.log(E0[ok] / E1[ok]).sum() / r["lam_raw"][ok].sum())


def _summ(r: dict) -> dict:
    """What a model's ``follow`` result keeps: the per-age quadratics, the mean pre/post quadratics and the rates."""
    E0, E1 = errors(r["pre"]), errors(r["post"])
    out = {"age": r["age"], "pre": r["pre"].mean(1), "post": r["post"].mean(1),
           "theta": float(np.mean(1.0 - np.sqrt(E1 / E0))), "theta_raw": float(np.mean(-np.expm1(-r["lam_raw"]))),
           "rate": rate_ratio(r)}
    if r["age_value"] is not None:
        out["age_value"] = r["age_value"]
    return out


def run_seeds(cond: dict, seeds: Sequence[int], settings: dict, save: Optional[Sequence[str]] = None) -> list:
    """Job: for every seed of ``seeds``, burn in gradient flow, fork, and run ``sgd``, ``gf`` and the delta
    models on the next T tasks.

    The seeds advance together.  With ``backend="jax"`` SGD and gradient flow are one vmapped batch on the
    device (``JaxTrainer``); with ``"numpy"`` each seed steps on the host.  The delta models and the probes are
    NumPy on the host in both.

    Parameters
    ----------
    cond, settings : dict
        Fields of ``Cond`` and ``Settings`` (plain dicts travel through ``mup.pool``).
    seeds : sequence of int
    save : sequence of str, optional
        One path per seed: its result is pickled there (so a long sweep can be resumed;
        ``scripts/run_nested.py --reuse``).

    Returns
    -------
    list of dict, one per seed
        ``models``: {name: ``_summ``}; ``ages``; ``T_learn``; ``t_fork``; ``a_fork`` (the readout every
        frozen-key model uses); ``burn`` (``burn_in``); ``a2``, ``Q`` mean a_i² and Q_ii at the fork.
    """
    from mup import pool
    c, st = Cond(**cond), Settings(**{**settings, "models": tuple(settings.get("models", MODELS))})
    if st.backend not in ("numpy", "jax"):
        raise ValueError(f"unknown backend {st.backend!r}")
    trainer = (lambda ms: JaxTrainer(ms)) if st.backend == "jax" else (lambda ms: None)
    tag = f"{c.label()} seeds {list(seeds)}"
    streams = [Stream(c, s) for s in seeds]
    T_learn, T, ages = st.schedule(c)
    gfs = [FlowModel(c, *s.init(), "pop", st) for s in streams]
    tr = trainer(gfs)
    burn = burn_in(gfs, streams, st, tr)
    t0 = burn["t"]
    forks = [(m.W.copy(), m.a.copy()) for m in gfs]
    outs = [{"cond": cond, "seed": seed, "ages": ages, "T_learn": T_learn, "t_fork": t0, "a_fork": af, "burn": burn,
             "a2": float((af * af).mean()), "Q": float(np.einsum("ij,ij->i", Wf, Wf).mean() / c.D), "models": {}}
            for seed, (Wf, af) in zip(seeds, forks)]
    pool.report(f"{tag}: forked after {t0} tasks (drift {burn['drift']:.3f})")
    for o, r in zip(outs, follow(gfs, streams, t0, T, T_learn, ages, value=True, trainer=tr)):
        o["models"]["gf"] = _summ(r)
    del tr, gfs
    for o, s, (Wf, af) in zip(outs, streams, forks):
        for name, rate in (("delta_matched", o["models"]["gf"]["rate"]), ("delta_raw", 1.0)):
            if name in st.models:
                o["models"][name] = _summ(follow(DeltaModel(c, Wf, af, rate, st.clock), s, t0, T, T_learn, ages))
    if "sgd" in st.models:
        ms = [FlowModel(c, Wf, af, "sgd", st, s.base) for s, (Wf, af) in zip(streams, forks)]
        rs = follow(ms, streams, t0, T, T_learn, ages, value=True, trainer=trainer(ms),
                    progress=lambda j, n: pool.report(f"{tag}: SGD task {j}/{n}"))
        for o, r in zip(outs, rs):
            o["models"]["sgd"] = _summ(r)
    for o, path in zip(outs, save or []):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path + ".tmp", "wb") as f:
            pickle.dump(o, f)
        os.replace(path + ".tmp", path)
    return outs


def run_seed(cond: dict, seed: int, settings: dict, save: Optional[str] = None) -> dict:
    """``run_seeds`` for one seed."""
    return run_seeds(cond, [seed], settings, [save] if save else None)[0]


#: longest Hebb burn-in, in tasks
HEBB_BURN_CAP = 50_000


def run_hebb(cond: dict, seed: int, settings: dict, a_fork: np.ndarray, t_fork: int, gamma: float) -> dict:
    """Job: the palimpsest at decay ``gamma`` on the stream of ``run_seed``, with the forked readout as keys.

    It is burned in on the tasks before the fork, max(t_fork, 6/γ) of them (the stream extends to negative
    indices; capped at ``HEBB_BURN_CAP``), then follows the same T tasks.  Returns the unit-b ``_summ``:
    ``errors(q, b)`` gives the errors of any write rate b.
    """
    c, st = Cond(**cond), Settings(**{**settings, "models": tuple(settings.get("models", MODELS))})
    stream = Stream(c, seed)
    T_learn, T, ages = st.schedule(c)
    m = HebbModel(c, a_fork, gamma, st.clock)
    for t in range(t_fork - min(HEBB_BURN_CAP, max(t_fork, int(math.ceil(6.0 / gamma)))), t_fork):
        g, w = stream.task(t)
        m.train(t, g, w)
    return _summ(follow(m, stream, t_fork, T, T_learn, ages))


# ----------------------------------------------------------------------------
# statistics and the decision rules
# ----------------------------------------------------------------------------
def memory(E_age: np.ndarray, E_inf: np.ndarray) -> np.ndarray:
    """M(τ) = 1 − E(τ)/E_∞ per seed: ``E_age`` (n_ages, n_seeds), ``E_inf`` (n_seeds,) the error of the same
    steady-state network on a task it has not seen (its mean error before training)."""
    return 1.0 - np.asarray(E_age, float) / np.asarray(E_inf, float)[None, :]


def half_age(curve: np.ndarray, ages: np.ndarray) -> float:
    """The age at which ``curve`` first falls to curve[0]/2, linear between grid ages; NaN if it never does
    or if there is no memory to halve (curve[0] ≤ 0)."""
    if not curve[0] > 0.0:
        return float("nan")
    i = theory.first_half_crossing(curve)
    return float(np.interp(i, np.arange(len(ages)), ages)) if np.isfinite(i) else float("nan")


def tau_half(M: np.ndarray, ages: np.ndarray) -> tuple[float, float]:
    """(τ½, jackknife s.e.) of the seed-mean curve of ``M`` (n_ages, n_seeds); s.e. NaN with one seed or when
    a leave-one-out curve never reaches half."""
    n = M.shape[1]
    t = half_age(M.mean(1), ages)
    if n < 2 or not np.isfinite(t):
        return t, float("nan")
    tot = M.sum(1)
    loo = np.array([half_age((tot - M[:, j]) / (n - 1), ages) for j in range(n)])
    return t, float(np.sqrt((n - 1) / n * ((loo - loo.mean()) ** 2).sum()))


def coincide(MA: np.ndarray, MB: np.ndarray, ages: np.ndarray) -> dict:
    """The pre-registered test that two forgetting curves coincide.  ``MA``, ``MB``: (n_ages, n_seeds).

    All three must hold:
      1. max_τ |M̄_A − M̄_B| / M̄_A(0) < ``TOL_M``;
      2. τ½ agrees within ``TOL_TAU`` (relative to A; two curves that both never reach half agree);
      3. the difference is within ``N_SEM`` s.e.m. across seeds, tested on two scalars chosen beforehand:
         the curve averaged over the age grid, |⟨M̄_A⟩ − ⟨M̄_B⟩| ≤ N_SEM·√(s.e.m._A² + s.e.m._B²), and τ½ with
         the jackknife errors.  (Testing at the age of the largest deviation instead would reject two
         identical noisy curves: the largest of ~100 deviations exceeds 2 s.e.m. by selection.)  The s.e.m.
         is the unpaired one, the seed-to-seed spread of each curve.  The models share their streams, so the
         paired z of the age-averaged difference is also returned (``z_paired``); it is not part of the rule.
         With one seed criterion 3 cannot be evaluated (``sem_ok`` None) and is left out.
    """
    n = MA.shape[1]
    mA, mB = MA.mean(1), MB.mean(1)
    d = mA - mB
    j = int(np.argmax(np.abs(d)))
    dev = float(abs(d[j]) / abs(mA[0]))
    (tA, sA), (tB, sB) = tau_half(MA, ages), tau_half(MB, ages)
    both_nan = not np.isfinite(tA) and not np.isfinite(tB)
    dtau = 0.0 if both_nan else (abs(tA - tB) / tA if np.isfinite(tA) and np.isfinite(tB) else float("inf"))
    sem_ok, z_paired = None, float("nan")
    if n >= 2:
        a, b = MA.mean(0), MB.mean(0)                                  # (n_seeds,) age-averaged memory
        diff = float(a.mean() - b.mean())
        sem_ok = bool(abs(diff) <= N_SEM * math.sqrt((a.var(ddof=1) + b.var(ddof=1)) / n))
        if not np.isfinite(dtau):
            sem_ok = False
        elif not both_nan and np.isfinite(sA) and np.isfinite(sB):
            sem_ok = sem_ok and bool(abs(tA - tB) <= N_SEM * math.hypot(sA, sB))
        sd = (a - b).std(ddof=1) / math.sqrt(n)
        z_paired = diff / sd if sd > 0 else (0.0 if diff == 0 else math.copysign(float("inf"), diff))
    m_ok, tau_ok = dev < TOL_M, dtau < TOL_TAU
    return {"coincide": bool(m_ok and tau_ok and sem_ok is not False), "max_dev": dev, "age_dev": int(ages[j]),
            "m_ok": bool(m_ok), "tau_A": tA, "tau_B": tB, "dtau": float(dtau), "tau_ok": bool(tau_ok),
            "sem_ok": sem_ok, "z_paired": float(z_paired)}


def key_fraction(E: np.ndarray, Ev: np.ndarray) -> np.ndarray:
    """φ_key(τ) = (E_full − E_value)/(E_full − E(0)) per seed, the share of forgetting that comes from the keys
    moving: ``E`` the error through the current key, ``Ev`` through the key the task had when learned and the
    current W, both (n_ages, n_seeds).  NaN at age 0."""
    with np.errstate(invalid="ignore", divide="ignore"):
        phi = (E - Ev) / (E - E[:1])
    phi[0] = np.nan
    return phi


SHAPES = {
    "exponential": (lambda t, m0, t0: m0 * np.exp(-t / t0), 2),
    "stretched": (lambda t, m0, t0, b: m0 * np.exp(-(t / t0) ** b), 3),
    "power": (lambda t, m0, t0, al: m0 * (1.0 + t / t0) ** (-al), 3),
}


def fit_shapes(ages: np.ndarray, M: np.ndarray) -> dict:
    """Least-squares fits of the mean curve ``M`` (n_ages,) by the three shapes: ``{name: {"params", "rss",
    "aic"}}`` plus ``"best"`` (lowest AIC = n ln(RSS/n) + 2k).  None has a plateau: a curve that levels off
    above zero (ρ > 0) is fitted badly by all three, which the RSS shows."""
    from scipy.optimize import curve_fit
    t, M = np.asarray(ages, float), np.asarray(M, float)
    t0 = half_age(M, ages)
    t0 = (t0 if np.isfinite(t0) else t[-1]) / math.log(2.0)
    p0 = {"exponential": (M[0], t0), "stretched": (M[0], t0, 1.0), "power": (M[0], t0, 1.0)}
    out = {}
    for name, (f, k) in SHAPES.items():
        try:
            lo = [-np.inf] + [1e-9] * (k - 1)
            par, _ = curve_fit(f, t, M, p0=p0[name], bounds=(lo, np.inf), maxfev=20000)
            rss = float(((f(t, *par) - M) ** 2).sum())
            out[name] = {"params": [float(x) for x in par], "rss": rss,
                         "aic": float(len(t) * math.log(max(rss, 1e-300) / len(t)) + 2 * k)}
        except Exception:                                              # no convergence: leave the shape out
            out[name] = {"params": [], "rss": float("nan"), "aic": float("inf")}
    out["best"] = min(SHAPES, key=lambda s: out[s]["aic"])
    return out


def stack(results: Sequence[dict], model: str, key: str = "age") -> np.ndarray:
    """Per-seed quadratics of one condition stacked on a last axis: (3, n_ages, n_seeds) for ``age`` and
    ``age_value``, (3, n_seeds) for ``pre`` and ``post``.  ``results`` are ``run_seed`` outputs in seed order
    (for ``hebb``: dicts with ``models["hebb"]``)."""
    return np.stack([r["models"][model][key] for r in results], -1)


def curves(results: Sequence[dict], model: str, b: float = 1.0) -> dict:
    """The observables of one model at one condition from its per-seed results.

    Returns ``{"E": (n_ages, n_seeds), "E_inf": (n_seeds,), "M": (n_ages, n_seeds), "eps0": (n_seeds,),
    "S": (n_ages, n_seeds) the project's signal u·w/|w|², "Ev": … or None}``; ``b`` is the Hebb write rate.
    """
    q = stack(results, model)
    E, E_inf = errors(q, b), errors(stack(results, model, "pre"), b)
    Ev = errors(stack(results, model, "age_value"), b) if "age_value" in results[0]["models"][model] else None
    return {"E": E, "E_inf": E_inf, "M": memory(E, E_inf), "eps0": E[0], "S": b * q[1] / q[2], "Ev": Ev}


def tune_hebb(by_gamma: dict, ages: np.ndarray, lam: float, eps0: float, tau: float) -> dict:
    """Choose the palimpsest's (b, γ) at the reference condition to match gradient flow's ε(0) and τ½.

    Parameters
    ----------
    by_gamma : {γ: list of unit-b ``run_hebb`` results over seeds}
    ages : (n_ages,)
    lam : float
        λ of the reference condition (b is searched over 0.01/λ … 10/λ, the write bΛā² ≈ bλ of order one).
    eps0, tau : float
        Gradient flow's ε(0) and τ½ there.

    Returns
    -------
    dict
        ``b``, ``gamma``, the achieved ``eps0`` and ``tau``, and ``loss`` = ln²(ε(0)/target) + ln²(τ½/target).
        A palimpsest need not reach both targets (its crosstalk floor on ε(0) grows as γ falls): the loss
        says how far off the best one is.
    """
    bs = np.geomspace(0.01, 10.0, 241) / lam
    best = {"loss": float("inf")}
    for gamma, res in by_gamma.items():
        wrapped = [{"models": {"hebb": r}} for r in res]
        for b in bs:
            cv = curves(wrapped, "hebb", b)
            e0, t = float(cv["eps0"].mean()), half_age(cv["M"].mean(1), ages)
            if not (np.isfinite(t) and t > 0 and e0 > 0 and cv["M"].mean(1)[0] > 0):
                continue
            loss = math.log(e0 / eps0) ** 2 + math.log(t / tau) ** 2
            if loss < best["loss"]:
                best = {"loss": loss, "b": float(b), "gamma": float(gamma), "eps0": e0, "tau": t}
    return best
