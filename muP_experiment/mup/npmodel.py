"""float64 NumPy port of the single-run kernels of ``mup.model`` (PLAN.md §2–§3), JAX-free.

Used by the scripts that run one run per process (``run_Nscaling.py``,
``run_weight_norms.py``): a run is a plain Python loop over tasks with no
compilation, and the package's JAX modules are never imported.  A run's
random stream is ``numpy.random.default_rng(run.rng_seed())``, so its draws
match the JAX sweep in distribution, not bit for bit.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.linalg import blas

from mup.config import Run


def Q_diag(W: np.ndarray) -> np.ndarray:
    """Q_ii = |W_i|² / D for every row (one pass over ``W``, no N×D temporary)."""
    return np.einsum("ij,ij->i", W, W) / W.shape[1]


def draw_readout(rng: np.random.Generator, Q: np.ndarray, eta: float, Delta0: float) -> np.ndarray:
    """a_i = ε_i √(η (Δ₀ + Q_ii/η)), ε_i = ±1; NaN where Δ₀ + Q_ii/η < 0 (infeasible δ₀)."""
    eps = rng.choice(np.array([-1.0, 1.0]), size=Q.shape)
    with np.errstate(invalid="ignore"):
        return eps * np.sqrt(eta * (Delta0 + Q / eta))


def init_run(run: Run) -> tuple:
    """Fresh run: ``(rng, W, a)`` with W ~ N(0, 1) and a from ``draw_readout`` at Δ_i = Δ₀.

    ``a`` holds NaN when the run is infeasible (δ₀ ≤ −min_i Q0_i, ``protocol.feasible``).
    """
    rng = np.random.default_rng(run.rng_seed())
    W = rng.standard_normal((run.N, run.D))
    a = draw_readout(rng, Q_diag(W), float(run.eta), float(run.Delta0))
    return rng, W, a


def _rank1_update(W: np.ndarray, c: float, u: np.ndarray, v: np.ndarray) -> None:
    """W ← W + c·u vᵀ in place with BLAS ``dger`` (one read-and-write pass, no temporary).

    ``W`` must be C-contiguous (so ``W.T`` is Fortran-contiguous, which is what
    ``dger`` updates in place); the transpose identity Wᵀ ← Wᵀ + c·v uᵀ is used.
    """
    assert W.flags.c_contiguous
    out = blas.dger(c, v, u, a=W.T, overwrite_a=1)
    assert np.shares_memory(out, W), "dger copied: W is not C-contiguous float64"


def step_task(W, a, g, w, rng, Delta0, eta, h, pN, learn_a, do_proj, n_steps, solver, Q=None):
    """``n_steps`` solver steps, each followed by the row projection, in place on the active rows of ``W``.

    Reference: ``model.sgd_step`` / ``model.pop_step`` and ``model.project_rows``
    (PLAN.md §3.2–§3.4; tested against them in ``tests/test_npmodel.py``).

    The steps change W only through g⊙a and a only through g, so inactive
    rows and entries receive exactly zero and the projection leaves them
    alone (f = 1): only the active rows are touched.  Within a task the
    active block is kept as diag(s)·W̃ (the projection's row factors ``s``
    are folded into W̃ once at the end), its rank-1 updates run in place
    with BLAS ``dger``, and the row norms Q_i are advanced analytically
    from the quantities the step already computes — for W_i ← W_i − c·a_i r:
    Q_i ← Q_i − 2c·a_i (W r)_i / D + c² a_i² |r|² / D — so one step reads
    the block twice (the two matrix–vector products) and writes it once.
    Q is recomputed from W at the start of every task, so rounding does
    not accumulate across tasks.  An ungated frozen run (every unit
    active, ``learn_a = do_proj = 0``) moves W only along the fixed
    direction a, W(step) = W(start) + a·δvᵀ, and is advanced in the
    D-vector δv alone (one rank-1 update of W per task).

    ``W`` and ``a`` are modified in place; returns ``(fcum, u)``: ``fcum``
    is the product of the projection factors over the steps (1 for
    inactive units) and ``u = (g ⊙ a) W / pN`` the effective weight of
    the task after its last step (zeros for an empty gate), computed from
    the active block, so nothing here reads an inactive row.

    ``Q`` (N,), optional: the tracked row norms Q_ii = |W_i|²/D, updated in
    place for the active rows (refreshed from the gathered block at the
    start of the task, so rounding does not accumulate across tasks;
    inactive rows are untouched and stay exact).  With it no caller needs
    ``Q_diag`` on the full matrix after ``init_run``.
    """
    N, D = W.shape
    sqD = math.sqrt(D)
    act = np.flatnonzero(g == 1.0)
    fcum = np.ones(N)
    if len(act) == 0:
        return fcum, np.zeros(D)                              # empty gate: nothing trains
    full = len(act) == N

    if full and learn_a == 0.0 and not do_proj:               # ungated frozen: W = W_start + a δvᵀ
        u0 = a @ W / pN
        a2 = a @ a
        dv = np.zeros(D)
        for _ in range(n_steps):
            if solver == "sgd":
                x = rng.standard_normal(D)
                hx = (W @ x + a * (dv @ x)) / sqD
                e = a @ hx / pN - w @ x / sqD
                dv -= (eta * e / sqD) * x
            else:
                r = u0 + (a2 / pN) * dv - w
                dv -= (h * eta) * r
        _rank1_update(W, 1.0, a, dv)
        if Q is not None:
            Q[:] = Q_diag(W)                                  # every row moved: one pass, exact
        return fcum, u0 + (a2 / pN) * dv

    Wa = W if full else np.ascontiguousarray(W[act])          # (n_act, D) block, a copy unless full
    aa = a[act].copy()
    s = np.ones(len(act))                                     # lazy row scale: block = diag(s)·Wa
    Qb = Q_diag(Wa)                                           # row norms of the block (s = 1 here)
    for _ in range(n_steps):
        if solver == "sgd":
            x = rng.standard_normal(D)
            hx = s * (Wa @ x) / sqD                           # (W x)/√D on the active rows
            e = aa @ hx / pN - w @ x / sqD
            c = eta * e / sqD                                 # W_i ← W_i − c·a_i x
            a_new = aa - learn_a * eta / D * e * hx
            Qb = Qb - (2.0 * c / D) * aa * (hx * sqD) + (c * c / D) * (aa * aa) * (x @ x)
            _rank1_update(Wa, -c, aa / s, x)
        else:
            r = (aa * s) @ Wa / pN - w
            Wr = s * (Wa @ r)                                 # (W r) on the active rows
            c = h * eta                                       # W_i ← W_i − c·a_i r
            a_new = aa - learn_a * c * Wr / D
            Qb = Qb - (2.0 * c / D) * aa * Wr + (c * c / D) * (aa * aa) * (r @ r)
            _rank1_update(Wa, -c, aa / s, r)
        aa = a_new
        if do_proj:
            A = aa * aa / eta
            B = Qb / eta
            AB = A * B
            big = 0.5 * (abs(Delta0) + np.hypot(Delta0, 2.0 * np.sqrt(AB)))
            small = AB / big
            Ap, Bp = (big, small) if Delta0 >= 0.0 else (small, big)
            aa = np.sign(aa) * np.sqrt(eta * Ap)
            f = np.sqrt(eta * Bp / Qb)
            s *= f
            Qb = Qb * (f * f)
    Wa *= s[:, None]                                          # fold the row scale in
    u = aa @ Wa / pN                                          # (g ⊙ a) W / pN from the block alone
    if full:
        a[:] = aa
        fcum[:] = s
        if Q is not None:
            Q[:] = Qb
    else:
        W[act] = Wa
        a[act] = aa
        fcum[act] = s
        if Q is not None:
            Q[act] = Qb
    return fcum, u


def tasks(run: Run, rng: np.random.Generator, W: np.ndarray, a: np.ndarray, n_tasks: int):
    """Generator over ``n_tasks`` tasks of one run: yields ``(t, W, a, g, w, fcum, Q, u)`` after each task.

    One task draws a teacher w and gate g (every unit active when
    ungated), a fresh head when ``use_heads`` (else keeps ``a``), then runs
    ``n_steps`` solver steps each followed by the row projection, exactly as
    ``protocol.run_task`` (``step_task``, in place on the active rows).
    ``fcum`` is the product over the task's steps of the projection
    factors, by which a stored head of an earlier task must be divided
    (``model.rescale_heads``) to keep its output unchanged.  ``W`` and
    ``a`` are the live buffers, updated in place by the next task: copy
    them to keep them.  ``g``, ``w``, ``fcum`` and ``u`` are fresh each task.
    ``Q`` (N,) holds the row norms Q_ii = |W_i|²/D of the live ``W``,
    tracked by ``step_task`` (a live buffer too) so that no task reads the
    whole matrix: the only pass over ``W`` is the one in ``init_run``.
    ``u = (g ⊙ a) W / pN`` is the effective weight of the task just trained.
    """
    N, D = run.N, run.D
    eta, h, pN, Delta0 = float(run.eta), float(run.h), float(run.pN), float(run.Delta0)
    gated, learn_a = run.gated == 1.0, float(run.learn_a)
    do_proj, use_heads = run.do_proj == 1.0, run.use_heads == 1.0
    n_steps, solver = int(run.n_steps), run.solver
    assert n_steps >= 1
    if solver not in ("pop", "sgd"):
        raise ValueError(f"unknown solver {solver!r}")
    W = np.array(W, dtype=np.float64, copy=True)
    a = np.array(a, dtype=np.float64, copy=True)
    Q = Q_diag(W)                                             # the one full pass; tracked from here on
    for t in range(n_tasks):
        w = rng.standard_normal(D)
        g = (rng.uniform(size=N) < pN / N).astype(np.float64) if gated else np.ones(N)
        head = draw_readout(rng, Q, eta, Delta0)              # drawn every task to keep the rng stream fixed
        if use_heads:
            a = head
        fcum, u = step_task(W, a, g, w, rng, Delta0, eta, h, pN, learn_a, do_proj, n_steps, solver, Q)
        yield t, W, a, g, w, fcum, Q, u
