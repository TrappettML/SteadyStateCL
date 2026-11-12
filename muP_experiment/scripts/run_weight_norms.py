#!/usr/bin/env python
"""Weight norms against tasks and against width (PLAN.md §12.1, §12.3).

    python scripts/run_weight_norms.py [--N 5 10 20 50 100] [--kappa 1 2 4] [--delta0 0] [--lam 4] [--seeds 4]
                                       [--clock realized|nominal] [--gpus auto]

For every (N, κ, readout, seed) the network trains the Stage-1 stream of
T = T_burn + T_meas tasks (14N under ``configs/stage1.yaml``) and, after
every task, records the hidden-layer norm mean_i Q_ii = mean_i |W_i|²/D
(= tr(WWᵀ)/(N D)) and the output-layer norm mean_i a_i², both over all
N units (``_all``) and over the task's active units (``_act``), and the
effective weight norm |u_t|²/D of the task just trained, u_t = (g ⊙ a) W / pN
(the gate g of that task; for task heads ``a`` is its head;
``experiments.task_norms``).  Before every task it also records the output of
that task's active set before any step, u_b = (g ⊙ a) W / pN with the new
task's gate (and fresh head) on the weights left by the previous tasks
(``experiments.leak_norms``): its norm |u_b|²/D, its overlap w·u_b/D with the
new teacher, and the single-unit part Σ_i g_i a_i² Q_ii / pN² of the norm.
These test the incoherence behind the scaling Q̄ ∝ pN: in steady state the
frozen norm balance needs |u_b|²/D = f/(2 − f), f = 1 − e^{−λ}, with
w·u_b ≈ 0, and the pairs i ≠ j of active units should carry a vanishing
fraction of |u_b|².  Figures in ``results/weight_norms/figures/``:

``norms_vs_task_k{κ}_…``   rows = (mean Q_ii, mean a_i², |u_t|²/D, the scaled Q̄
    of ``norms_vs_N``, |u_b|²/D, w·u_b/D, the pair fraction of |u_b|², per task), columns = N,
    traces = readouts at the first κ listed (ungated readouts have no κ),
    x = task; seed mean with a ±1 s.e. band, burn-in shaded.
``norms_vs_N_…``           rows = the same quantities: the three norms, the scaling
    collapse Q̄/pN for the frozen readouts and Q̄²/pN for the learned ones
    (Q̄ the steady-state mean_i Q_ii of the top row, pN = κ√N, or N when
    ungated; a constant in N confirms the mechanism, and a frozen constant
    with a drifting learned one points at extra dynamics in a), then the
    three leak rows (the pair fraction here is the ratio of the window
    means, 1 − ⟨diagonal⟩/⟨|u_b|²⟩, not the mean of the per-task ratio), columns = κ,
    x = N (log), y = the steady-state value (mean over the measurement
    window t ≥ T_burn, then seed mean ± 1 s.e. bars), traces = readouts;
    the ungated readouts repeat in every column.

Both figures overlay, dotted in the readout's colour, the write-up's closed-form steady state for the gated
readouts (``norm_theory``; nothing measured goes into it): eq. (abarss) for the projected ones,
ā_act = ½[δ₀ + σ₀² + √((δ₀ + σ₀²)² + 4θ pN/(2 − θ))] and Q̄_act = ā_act − δ₀, and eq. (Qss) for the frozen
one, Q̄ = σ₀² + θ/(2 − θ)·pN/(δ₀ + σ₀²).  The recorded norms are a check of these, not an input to anything:
the learned-readout decay curve is drawn from the closed form (``theory.sigma_learned``).  The estimate is
leading order and was not derived for the ungated readouts, which get no overlay.

``exponents_vs_N_…``       the learning per task of the two channels and the step's stability (write-up, "Step
    sizes and learning per task"), from the state each task starts with (``experiments.clock_norms``): rows =
    the hidden-layer exponent λ_c = Λ_c ā²_c the task realizes (λ under clock (ii); under clock (i) it is eq.
    lamreal and settles at λ ā_act/(1 + δ₀), dotted), the readout exponent λᵃ_c = Λ_c‖M_c‖ from the largest
    eigenvalue of the active block of Q with the band of its bounds eq. (lama), Λ_c Q̄_c/pN ≤ λᵃ_c ≤
    Λ_c n_c Q̄_c/pN, and the left side of the contraction condition eq. (popstab), hη (n_c/pN)(δ₀ + 2Q̄_c)
    (the hidden channel alone, hη ā²_c, for the frozen readouts), against its limit 2; columns = κ, x = N, steady-state means, traces = readouts (the frozen readouts have no
    readout channel and no λᵃ).  The console prints how many tasks leave the bounds (none should).

The per-run curves are written to ``norms_….npz`` beside the figures.

Placement (``mup.pool``): the runs of one width — every κ, readout and
seed, which share (N, D, n_steps, T) — form one ``vmap`` batch advanced
``--chunk`` tasks per device call with the norms evaluated inside the scan
(``experiments.norms_over_tasks``, the JAX kernels of the sweep) and runs
as a job, largest N first, over the GPUs (``--gpus``: the four cards,
several processes each, a width split only as far as needed to fill them
— ``pool.plan``), over ``--workers`` CPU processes, or in this process.
"""
import argparse
import math
import os
import sys
import time
import warnings
from typing import Callable, Optional, Sequence

import numpy as np

import _path  # noqa: F401

from mup import experiments as X
from mup import plotting as P
from mup import pool
from mup import theory as TH
from mup.config import Config, READOUT_ORDER, Run, add_clock_arg, clock_override, is_gated, make_runs

_T0 = time.perf_counter()        # wall clock from after the imports to the end of the script

#: Recorded per task: (key, y-axis label)
QUANT = (("Q_all", r"$\bar Q = \mathrm{mean}_i\,|W_i|^2/D$"), ("a2_all", r"mean $a_i^2$"), ("u2", r"$|u_t|^2/D$"))
#: ("Q_all", "Q_act", "a2_all", "a2_act", "u2", "ub2", "wub", "ub_diag", "n_c", "abar_c", "Qbar_c", "qmax_c", "h_c")
ALL_KEYS = X.NORM_KEYS + X.LEAK_KEYS + X.CLOCK_KEYS
#: Fourth row of ``norms_vs_N``: the scaling collapse of Q̄ = mean Q_ii (derived from ``Q_all`` per seed)
Q_SCALED = ("Q_scaled", r"$\bar Q/pN$ (frozen),  $\bar Q^2/pN$ (learned)")


def q_scaled(ss_Q: np.ndarray, run: Run) -> np.ndarray:
    """Collapse variable of one condition: Q̄/pN when the readout is frozen, Q̄²/pN when it learns.

    Parameters
    ----------
    ss_Q : (n_seeds,) per-seed steady-state mean Q_ii
    run : config.Run of the condition (supplies ``pN`` and ``learn_a``)

    Returns
    -------
    np.ndarray
        Same shape as ``ss_Q``.
    """
    pN = float(run.pN)
    return ss_Q / pN if run.learn_a == 0.0 else ss_Q * ss_Q / pN


#: Leak rows of both figures: the output u_b of a task's active set before it is trained (``X.leak_norms``).
#: ``cross`` is derived: the share of |u_b|² carried by pairs i ≠ j of active units, (ub2 − ub_diag)/ub2.
LEAK = (("ub2", r"$|u^{\mathrm{b}}_t|^2/D$"), ("wub", r"$w_t\cdot u^{\mathrm{b}}_t/D$"),
        ("cross", r"pair fraction of $|u^{\mathrm{b}}_t|^2$"))
ROWS = QUANT + (Q_SCALED,) + LEAK


def _ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """``num/den`` with NaN where ``den`` is 0 or not finite (an empty gate has u_b = 0)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(np.isfinite(den) & (den > 0), num / np.where(den > 0, den, 1.0), np.nan)


def per_task(d: dict[str, np.ndarray], q: str, run: Run) -> np.ndarray:
    """(n_seeds, T) values of row ``q`` of ``ROWS`` task by task, from a condition's recorded curves ``d``."""
    if q == Q_SCALED[0]:
        return q_scaled(d["Q_all"], run)
    if q == "cross":
        return _ratio(d["ub2"] - d["ub_diag"], d["ub2"])
    return d[q]


def steady(d: dict[str, np.ndarray], q: str, run: Run) -> np.ndarray:
    """(n_seeds,) steady-state value of row ``q``: its mean over the measurement window t ≥ T_burn.

    The two derived rows are functions of window means, not window means of the per-task value: the collapse
    variable of the mean Q̄, and for ``cross`` the ratio 1 − ⟨ub_diag⟩/⟨ub2⟩ (the per-task ratio is noisy when
    few units are active and undefined when none is)."""
    win = lambda k: np.nanmean(d[k][:, run.T_burn:], 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if q == Q_SCALED[0]:
            return q_scaled(win("Q_all"), run)
        if q == "cross":
            return _ratio(win("ub2") - win("ub_diag"), win("ub2"))
        return win(q)


def norm_theory(q: str, run: Run) -> float:
    """Closed-form steady-state value of row ``q`` for a gated readout, NaN where the write-up gives none.

    ``Q_all`` → Q̄ and ``a2_all`` → ā² of ``theory.steady_norms``: eq. (abarss) for a projected readout
    (ā_act, Q̄_act = ā_act − δ₀), eq. (Qss) for the frozen one (ā² = δ₀ + σ₀², Q̄ = σ₀² + θ/(2 − θ)·pN/ā²);
    ``Q_scaled`` → the collapse variable of that Q̄ (``q_scaled``).  The mean-field estimate does not
    distinguish the mean over all units from the mean over the active ones.  Ungated readouts: NaN (the
    estimate's loss rate θ(2 − θ)(ν − p)/(N − 1) vanishes at p = 1; it was not derived for them).
    """
    if run.gated != 1.0 or q not in ("Q_all", "a2_all", Q_SCALED[0]):
        return float("nan")
    abar, Q = TH.steady_norms(run.do_proj == 1.0, run.delta0, run.pN, run.lam, run.sigma0)
    if q == "a2_all":
        return abar
    return Q if q == "Q_all" else float(q_scaled(np.array(Q), run))


#: Rows of ``exponents_vs_N``: (key of ``exponents``, y-axis label)
EXP_ROWS = (("lam_c", r"hidden-layer exponent $\lambda_c=\Lambda_c\,\bar a^2_c$"),
            ("lam_a", r"readout exponent $\lambda^a_c=\Lambda_c\|M_c\|$"),
            ("popstab", r"$h\eta\,(\bar a^2_c+\mathrm{tr}\,M_c)$"))


def exponents(d: dict[str, np.ndarray], run: Run) -> dict[str, np.ndarray]:
    """Per-task learning of the two channels and the step's stability, (n_seeds, T) each, from a condition's
    recorded ``experiments.CLOCK_KEYS`` (write-up, "Step sizes and learning per task").

    With Λ_c = η h_c n the duration the task ran for (α_task η under SGD):
    ``lam_c`` = Λ_c ā²_c, the hidden-layer exponent (eq. lam; eq. lamreal under clock (i), λ under clock (ii));
    ``lam_a`` = Λ_c qmax_c/pN = Λ_c‖M_c‖, the readout exponent (0 where the readout is frozen);
    ``lam_a_lo``, ``lam_a_hi`` = Λ_c Q̄_c/pN and Λ_c n_c Q̄_c/pN, its bounds, eq. (lama);
    ``popstab`` = hη(ā²_c + tr M_c), tr M_c = (n_c/pN) Q̄_c: the left side of eq. (popstab),
    hη (n_c/pN)(δ₀ + 2Q̄_c), for a projected readout and hη ā²_c for a frozen one (NaN under SGD).
    Tasks with an empty gate are NaN throughout.
    """
    pN = float(run.pN)
    pop = run.solver == "pop"
    h_eta = run.eta * d["h_c"] if pop else np.full_like(d["h_c"], run.eta / run.D)
    Lam = run.n_steps * h_eta
    empty = d["n_c"] == 0
    nan = lambda x: np.where(empty, np.nan, x)
    learn = float(run.learn_a)
    return {"lam_c": nan(Lam * d["abar_c"]), "lam_a": nan(learn * Lam * d["qmax_c"] / pN),
            "lam_a_lo": nan(learn * Lam * d["Qbar_c"] / pN), "lam_a_hi": nan(learn * Lam * d["n_c"] * d["Qbar_c"] / pN),
            "popstab": (nan(h_eta * (d["abar_c"] + learn * d["n_c"] / pN * np.nan_to_num(d["Qbar_c"]))) if pop
                        else np.full_like(Lam, np.nan))}


def bound_violations(e: dict[str, np.ndarray], rtol: float = 1e-9) -> tuple[int, int]:
    """(tasks whose λᵃ_c lies outside the bounds of eq. (lama), tasks compared) over a condition's ``exponents``."""
    ok = np.isfinite(e["lam_a"]) & np.isfinite(e["lam_a_lo"])
    lo, hi, v = e["lam_a_lo"][ok], e["lam_a_hi"][ok], e["lam_a"][ok]
    return int(((v < lo * (1 - rtol)) | (v > hi * (1 + rtol))).sum()), int(ok.sum())


def leak_reference(ax, q: str, lam: float) -> None:
    """Dotted reference of a leak row: f/(2 − f), f = 1 − e^{−λ}, for |u_b|²/D (frozen norm balance); 0 for the others."""
    if q not in dict(LEAK):
        return
    f = 1.0 - math.exp(-lam)
    ax.axhline(f / (2.0 - f) if q == "ub2" else 0.0, color="0.35", lw=0.8, ls=":", zorder=1)


def groups(runs: list[Run]) -> dict[tuple, list[Run]]:
    """``{(N, D, solver, n_steps, T): runs}``, largest N first."""
    g = {}
    for r in runs:
        g.setdefault(X.group_key(r) + (r.T_burn + r.T_meas,), []).append(r)
    return {k: g[k] for k in sorted(g, key=lambda k: -k[0])}


def make_jobs(groups_: dict[tuple, list[Run]], pieces: Sequence[int]) -> list[tuple]:
    """One job per piece of a (N, D, n_steps, T) group: the norms after each of its T tasks."""
    jobs = []
    for (key, runs), k in zip(groups_.items(), pieces):
        T = key[-1]
        for i, part in enumerate(X.split(runs, k)):
            jobs.append(((key, i), {"runs": part, "T": T}, X.job_cost(part, T)))
    return jobs


def collect(runs: list[Run], plan: pool.Plan, log: Callable[[str], None] = print) -> dict[tuple, dict[str, np.ndarray]]:
    """``{(N, kappa_label, readout): {key: (n_seeds, T) array}}`` for every condition of ``runs``."""
    jobs = make_jobs(groups(runs), plan.pieces)
    res = pool.run_jobs("mup.experiments:norms_over_tasks", jobs, log=log, **plan.place)
    per_run = {}
    for (_, kw, _), d in zip(jobs, res):
        for j, r in enumerate(kw["runs"]):
            per_run[(r.N, r.kappa_label, r.readout, r.seed)] = {q: d[q][j] for q in ALL_KEYS}
    cond = {}
    for r in runs:
        cond.setdefault((r.N, r.kappa_label, r.readout), []).append(r)
    out = {}
    for key in sorted(cond, key=lambda k: (k[0], READOUT_ORDER.index(k[2]), k[1])):
        b = cond[key]
        out[key] = {q: np.stack([per_run[key + (r.seed,)][q] for r in b]) for q in ALL_KEYS}
        ss = {q: steady(out[key], q, b[0]) for q, _ in QUANT + LEAK}
        n_ok = int(np.isfinite(ss["Q_all"]).sum())
        print(f"  N={key[0]:4d} κ={key[1]:5s} {key[2]:15s} T={out[key]['Q_all'].shape[1]:5d}  "
              f"steady mean Q = {np.nanmean(ss['Q_all']):7.3f}  mean a² = {np.nanmean(ss['a2_all']):7.3f}  "
              f"|u|²/D = {np.nanmean(ss['u2']):7.3f}  |u_b|²/D = {np.nanmean(ss['ub2']):7.3f}  "
              f"w·u_b/D = {np.nanmean(ss['wub']):+7.3f}  pairs = {np.nanmean(ss['cross']):+7.3f}  "
              f"infeasible {len(b) - n_ok}/{len(b)}", flush=True)
        r = b[0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            e = exponents(out[key], r)
            w = {q: np.nanmean(v[:, r.T_burn:]) for q, v in e.items()}
            bad, n = bound_violations(e)
            th = "" if not np.isfinite(norm_theory("Q_all", r)) else (
                f"  closed form Q̄ = {norm_theory('Q_all', r):.3f}, ā² = {norm_theory('a2_all', r):.3f}")
            print(f"        λ_c = {w['lam_c']:6.3f} (λ = {r.lam:g})  λᵃ_c = {w['lam_a']:6.3f} in [{w['lam_a_lo']:.3f}, "
                  f"{w['lam_a_hi']:.3f}], outside eq. (lama) in {bad}/{n} tasks  popstab = {w['popstab']:5.3f} "
                  f"(max {np.nanmax(e['popstab']) if r.solver == 'pop' else float('nan'):5.3f}, limit 2){th}", flush=True)
    return out


def _mean_se(Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Seed mean and s.e. along axis 0 over the finite entries (NaN where none, s.e. 0 where one)."""
    n = np.isfinite(Y).sum(0)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = np.nanmean(Y, 0)
        sd = np.nanstd(Y, 0, ddof=1)
    se = np.where(n > 1, sd / np.sqrt(np.maximum(n, 1)), 0.0)
    return m, se


def fig_vs_task(res: dict[tuple, dict[str, np.ndarray]], Ns: list[int], klabel: str, run0: list[Run], path: str) -> str:
    """Rows = quantities, columns = N, traces = readouts at κ = ``klabel``; seed mean ± 1 s.e. band.

    ``run0`` is the full run list (a run per condition is looked up for ``pN``, ``learn_a``, ``T_burn``)."""
    P.apply_style()
    Ns = [N for N in Ns if any(k[0] == N for k in res)]
    rows = ROWS
    fig, axes = P.grid(len(rows), len(Ns), sharex="col", sharey="row", panel=(2.6, 2.0))
    figdata = {}
    readouts = []
    n_inf = 0
    for c, N in enumerate(Ns):
        for r, (q, lab) in enumerate(rows):
            ax = axes[r, c]
            for ro in READOUT_ORDER:
                k = klabel if is_gated(ro) else "sqrtN"
                if (N, k, ro) not in res:
                    continue
                if ro not in readouts:
                    readouts.append(ro)
                run = next(rr for rr in run0 if rr.N == N and rr.readout == ro and rr.kappa_label == k)
                Y = per_task(res[(N, k, ro)], q, run)                 # per task, per seed
                if r == 0:
                    n_inf += int((~np.isfinite(Y).any(1)).sum())
                m, se = _mean_se(Y)
                x = np.arange(1, Y.shape[1] + 1)
                P.band(ax, x, m - se, m + se, P.READOUT_COLOR[ro])
                P.plot_readout(ax, x, m, ro, zorder=4)
                figdata[f"{q}_N{N}_{ro}"] = np.stack([x, m, se])
                th = norm_theory(q, run)                              # closed-form steady state, eqs. (abarss), (Qss)
                if np.isfinite(th):
                    ax.axhline(th, color=P.READOUT_COLOR[ro], ls=":", lw=P.LW_REF, zorder=2)
                    figdata[f"theory_{q}_N{N}_{ro}"] = np.array(th)
            run = next(rr for rr in run0 if rr.N == N)
            leak_reference(ax, q, run.lam)
            P.shade_burnin(ax, run.T_burn, run.T_meas)
            if r == len(rows) - 1:
                ax.set_xlabel("task")
        axes[0, c].set_title(f"N = {N}")
    for r, (q, lab) in enumerate(rows):
        axes[r, 0].set_ylabel(lab)
    r0 = run0[0]
    P.cell_title(fig, kappa=klabel, d0=r0.delta0, lam=r0.lam, solver=r0.solver, eta=r0.eta)
    P.caption(fig, f"after each task: mean over all N units of Q_ii = |W_i|²/D (top) and of a_i² (middle; the head "
                   f"just trained for task heads), and the effective weight norm |u_t|²/D of the task just trained, "
                   f"u_t = (g ⊙ a) W / pN (third), and the scaling collapse of the top row, Q̄/pN for the frozen "
                   f"readouts and Q̄²/pN for the learned ones, pN = κ√N or N when ungated (fourth; flat across the "
                   f"columns confirms the mechanism); then, before each task, the output u_b of its active set on "
                   f"the old weights: |u_b|²/D (dotted: f/(2−f), f = 1−e^(−λ), the frozen norm balance), its overlap "
                   f"w·u_b/D with the new teacher, and the fraction of |u_b|² carried by pairs i ≠ j of active units "
                   f"(0 = units add incoherently); seed mean, band ±1 s.e. over seeds; dotted horizontal in a gated "
                   f"readout's colour (rows 1, 2, 4): the closed-form steady state, eq. (abarss) for the projected "
                   f"readouts and eq. (Qss) for the frozen one; grey: burn-in, dotted vertical: "
                   f"half of the measurement window; the ungated readouts have no κ; {n_inf} infeasible seeds dropped.")
    P.fig_legend(fig, readouts)
    return P.save(fig, path, figdata=figdata)


def fig_vs_N(res: dict[tuple, dict[str, np.ndarray]], Ns: list[int], klabels: list[str], run0: list[Run], path: str) -> str:
    """Rows = quantities, columns = κ, x = N (log), y = steady-state value (seed mean ± 1 s.e.), traces = readouts.

    ``run0`` is the full run list, as in ``fig_vs_task``."""
    P.apply_style()
    rows = ROWS
    fig, axes = P.grid(len(rows), len(klabels), sharex="all", sharey="row", panel=(3.0, 2.4))
    figdata = {}
    readouts = []
    for c, klabel in enumerate(klabels):
        for r, (q, lab) in enumerate(rows):
            ax = axes[r, c]
            for ro in READOUT_ORDER:
                k = klabel if is_gated(ro) else "sqrtN"
                pts = []
                for N in Ns:
                    if (N, k, ro) not in res:
                        continue
                    run = next(rr for rr in run0 if rr.N == N and rr.readout == ro and rr.kappa_label == k)
                    ss = steady(res[(N, k, ro)], q, run)                           # per-seed steady-state value
                    v = ss[np.isfinite(ss)]
                    if len(v):
                        pts.append((N, v.mean(), v.std(ddof=1) / math.sqrt(len(v)) if len(v) > 1 else 0.0))
                if not pts:
                    continue
                if ro not in readouts:
                    readouts.append(ro)
                x, m, se = map(np.array, zip(*pts))
                P.plot_readout(ax, x, m, ro, marker="o", ms=3)
                ax.errorbar(x, m, yerr=se, fmt="none", ecolor=P.READOUT_COLOR[ro], elinewidth=0.8, capsize=2, zorder=3)
                figdata[f"{q}_k{klabel}_{ro}"] = np.stack([x, m, se])
                th = np.array([norm_theory(q, next(rr for rr in run0 if rr.N == N and rr.readout == ro
                                                   and rr.kappa_label == k)) for N in x])
                if np.isfinite(th).all():                             # closed-form steady state, eqs. (abarss), (Qss)
                    ax.plot(x, th, color=P.READOUT_COLOR[ro], ls=":", lw=P.LW_REF, zorder=2)
                    figdata[f"theory_{q}_k{klabel}_{ro}"] = np.stack([x, th])
            leak_reference(ax, q, run0[0].lam)
            P.logaxis(ax, "x")
            P.discrete_ticks(ax, Ns, "x")
            if r == len(rows) - 1:
                ax.set_xlabel("$N$")
        axes[0, c].set_title(f"κ = {klabel}")
    for r, (q, lab) in enumerate(rows):
        axes[r, 0].set_ylabel(lab)
    r0 = run0[0]
    P.cell_title(fig, d0=r0.delta0, lam=r0.lam, solver=r0.solver, eta=r0.eta)
    P.caption(fig, "steady-state norms: mean over the measurement window (t ≥ T_burn) of mean_i Q_ii (top), "
                   "mean_i a_i² (second) and the effective weight norm |u_t|²/D, u_t = (g ⊙ a) W / pN (third), "
                   "then seed mean with ±1 s.e. bars; fourth: the scaling collapse of the top row, Q̄/pN for the "
                   "frozen readouts and Q̄²/pN for the learned ones (pN = κ√N, or N when ungated) — flat in N "
                   "confirms the mechanism; last three: the output u_b of each task's active set before it is "
                   "trained — |u_b|²/D (dotted: f/(2−f), f = 1−e^(−λ), the frozen norm balance), its overlap w·u_b/D "
                   "with the new teacher, and the fraction of |u_b|² carried by pairs i ≠ j of active units, "
                   "1 − ⟨Σ_i g_i a_i² Q_ii/pN²⟩/⟨|u_b|²/D⟩ over the window (0 = units add incoherently); "
                   "dotted in a gated readout's colour (rows 1, 2, 4): the closed-form steady state, eq. (abarss) "
                   "for the projected readouts and eq. (Qss) for the frozen one, no measured input; "
                   "the ungated readouts have no κ and repeat in every column.")
    P.fig_legend(fig, readouts, marker=True)
    return P.save(fig, path, figdata=figdata)


def fig_exponents(res: dict[tuple, dict[str, np.ndarray]], Ns: list[int], klabels: list[str], run0: list[Run], path: str) -> str:
    """Rows = ``EXP_ROWS`` (hidden-layer exponent, readout exponent with the band of eq. (lama), eq. (popstab)),
    columns = κ, x = N (log), y = the mean over the measurement window (seed mean ± 1 s.e.), traces = readouts.

    ``run0`` is the full run list, as in ``fig_vs_task``."""
    P.apply_style()
    fig, axes = P.grid(len(EXP_ROWS), len(klabels), sharex="all", sharey="row", panel=(3.0, 2.4))
    figdata = {}
    readouts = []
    r0 = run0[0]
    for c, klabel in enumerate(klabels):
        for r, (q, lab) in enumerate(EXP_ROWS):
            ax = axes[r, c]
            for ro in READOUT_ORDER:
                k = klabel if is_gated(ro) else "sqrtN"
                pts, th = [], []
                for N in Ns:
                    if (N, k, ro) not in res:
                        continue
                    run = next(rr for rr in run0 if rr.N == N and rr.readout == ro and rr.kappa_label == k)
                    if q == "lam_a" and run.learn_a == 0.0:           # a frozen readout has no readout channel
                        continue
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        e = exponents(res[(N, k, ro)], run)
                        win = {kk: np.nanmean(v[:, run.T_burn:], 1) for kk, v in e.items()}
                    v = win[q][np.isfinite(win[q])]
                    if not len(v):
                        continue
                    pts.append((N, v.mean(), v.std(ddof=1) / math.sqrt(len(v)) if len(v) > 1 else 0.0,
                                np.nanmean(win["lam_a_lo"]), np.nanmean(win["lam_a_hi"])))
                    # clock (i): λ_c settles at λ ā_act/(1 + δ₀) for a gated projected readout (eq. lamreal with abarss)
                    th.append(run.lam * TH.abar_ss(run.delta0, run.pN, run.lam, run.sigma0) / (1.0 + run.delta0)
                              if run.clock == "nominal" and run.gated == 1.0 and run.do_proj == 1.0 else np.nan)
                if not pts:
                    continue
                if ro not in readouts:
                    readouts.append(ro)
                x, m, se, lo, hi = map(np.array, zip(*pts))
                P.plot_readout(ax, x, m, ro, marker="o", ms=3)
                ax.errorbar(x, m, yerr=se, fmt="none", ecolor=P.READOUT_COLOR[ro], elinewidth=0.8, capsize=2, zorder=3)
                figdata[f"{q}_k{klabel}_{ro}"] = np.stack([x, m, se])
                if q == "lam_a":
                    P.band(ax, x, lo, hi, P.READOUT_COLOR[ro])
                    figdata[f"lam_a_bounds_k{klabel}_{ro}"] = np.stack([x, lo, hi])
                if q == "lam_c" and np.isfinite(th).all():
                    ax.plot(x, th, color=P.READOUT_COLOR[ro], ls=":", lw=P.LW_REF, zorder=2)
                    figdata[f"theory_lam_c_k{klabel}_{ro}"] = np.stack([x, np.array(th)])
            if q == "lam_c":
                ax.axhline(r0.lam, color=P.GREY_REF, ls=":", lw=P.LW_REF, zorder=1)
            if q == "popstab":
                ax.axhline(2.0, color=P.GREY_REF, ls=":", lw=P.LW_REF, zorder=1)
            if q != "lam_c":
                P.logaxis(ax, "y")
            P.logaxis(ax, "x")
            P.discrete_ticks(ax, Ns, "x")
            if r == len(EXP_ROWS) - 1:
                ax.set_xlabel("$N$")
        axes[0, c].set_title(f"κ = {klabel}")
    for r, (q, lab) in enumerate(EXP_ROWS):
        axes[r, 0].set_ylabel(lab)
    P.cell_title(fig, d0=r0.delta0, lam=r0.lam, solver=r0.solver, eta=r0.eta, extra=f"{r0.clock} clock")
    P.caption(fig, "learning per task of the two channels, from the state each task starts with, mean over the "
                   "measurement window then seed mean ± 1 s.e. Top: the hidden-layer exponent the task realizes, "
                   "Λ_c ā²_c (dotted grey: λ; clock (ii) sets it to λ, under clock (i) it follows eq. (lamreal), "
                   "dotted in colour: λ ā_act/(1+δ₀) with eq. (abarss)). Middle: the readout exponent Λ_c‖M_c‖ from "
                   "the largest eigenvalue of the active block of Q; band: its bounds Λ_c Q̄_c/pN and Λ_c n_c Q̄_c/pN, "
                   "eq. (lama); the frozen readouts have none. Bottom: the left side of the sufficient contraction "
                   "condition eq. (popstab), hη(n_c/pN)(δ₀+2Q̄_c) (hη ā²_c for the frozen readouts; dotted grey: its limit 2). The ungated readouts repeat in every column.")
    P.fig_legend(fig, readouts, marker=True)
    return P.save(fig, path, figdata=figdata)


def main(argv: Optional[list[str]] = None) -> int:
    """Command-line entry point (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(_path.ROOT, "configs", "stage1.yaml"))
    ap.add_argument("--N", type=int, nargs="+", default=[5, 10, 20, 50, 100])
    ap.add_argument("--kappa", nargs="+", default=["1", "2", "4"])
    ap.add_argument("--delta0", type=float, default=0.0)
    ap.add_argument("--lam", type=float, default=None, help="default: the config's first λ")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--readout", nargs="*", default=list(READOUT_ORDER))
    ap.add_argument("--out", default=os.path.join(_path.ROOT, "results", "weight_norms"))
    add_clock_arg(ap)
    pool.add_args(ap)
    args = ap.parse_args(argv)

    cfg = Config.from_yaml(args.config)
    over = {"name": "weight_norms", "N": list(args.N), "kappa": list(args.kappa), "readout": list(args.readout),
            "delta0": [args.delta0], "lam": [args.lam if args.lam is not None else cfg.lam[0]],
            "seeds": args.seeds, "extra": (), "trace": False, **clock_override(args)}
    cfg = Config(**{**cfg.__dict__, **over})
    runs = make_runs(cfg)
    if not runs:
        print("no runs left after skipping κ outside [1, √N]; nothing to do", flush=True)
        return 0
    run0 = runs[0]
    plan = pool.plan(args, [(g, key[-1]) for key, g in groups(runs).items()], out_per_task=8 * len(ALL_KEYS))
    print(f"weight norms: N={args.N} κ={args.kappa} δ₀={run0.delta0} λ={run0.lam} {run0.solver} {run0.clock} clock seeds={args.seeds}  "
          f"[{plan.describe()}]", flush=True)
    res = collect(runs, plan)

    klabels = [str(k) for k in args.kappa]              # the runs' kappa_label is the argument as typed
    fdir = os.path.join(args.out, "figures")
    tag = f"d0{P.fmt(run0.delta0)}_lam{P.fmt(run0.lam)}"
    f1 = fig_vs_task(res, list(args.N), klabels[0], runs, os.path.join(fdir, f"norms_vs_task_k{P.fmt(klabels[0])}_{tag}"))
    f2 = fig_vs_N(res, list(args.N), klabels, runs, os.path.join(fdir, f"norms_vs_N_{tag}"))
    f3 = fig_exponents(res, list(args.N), klabels, runs, os.path.join(fdir, f"exponents_vs_N_{tag}"))
    np.savez_compressed(os.path.join(fdir, f"norms_{tag}.npz"),
                        **{f"N{N}_k{k}_{ro}_{q}": v for (N, k, ro), d in res.items() for q, v in d.items()})
    print(f"wrote {f1}\n      {f2}\n      {f3}")
    return 0


if __name__ == "__main__":
    _rc = main()
    _dt = time.perf_counter() - _T0
    print(f"run_weight_norms: total {_dt:.1f} s ({_dt / 60:.2f} min) from after the imports to the end", flush=True)
    sys.exit(_rc)
