#!/usr/bin/env python
"""Training verification: the per-step loss of the task being trained at the start, middle and end of the stream.

    python scripts/run_training_verify.py [--cell N,kappa,delta0 ...] [--seeds 4] [--lam 4] [--clock realized|nominal] [--gpus]
                                          [--comp_opts] [--eta_sgd η] [--tasks T]

What the write-up predicts for these curves ("Decision: the task clock"): with ρ_c = |r_c(end)|²/|r_c(0)|² the
loss ratio over a task, the frozen readouts end every task at ρ_c = e^{−2λ} exactly under the default clock (ii)
(up to the forward step's over-contraction, eq. lameff: e^{−2λ_eff}, λ_eff = −n ln(1 − λ/n)); for the learned
ones the write-up states ρ_c ≤ e^{−2λ}, the surplus being the readout channel M_c.  That bound is eq. (rdecay)
with ā²_c held at the value the task starts with; the exact statement is ρ_c ≤ e^{−2Λ_c⟨ā²_c⟩} with the time
average over the task, so a learned readout whose key norm falls during the task can end a few per cent above
e^{−2λ}.  Under ``--clock nominal`` the frozen readouts end at e^{−2λ_c} with λ_c of eq. (lamreal), which
drifts with the settled norms.  The console prints the measured ρ_c of the last traced task beside e^{−2λ_eff}.

For every cell (N, κ, δ₀) — by default the four
``100,2,1``, ``50,1,0``, ``20,1,4``, ``200,2,1`` — every readout × seed is
trained through the Stage-1 stream (``configs/stage1.yaml``: λ = 4,
SGD with 16 samples per step, T = T_burn + T_meas = 14N tasks × 20N steps; ``solver: pop`` in the config: × 40 steps) and the
population loss L_c = |u_c − w_c|²/(2D) of the task being trained is
recorded before its first step and after each step for six tasks: the
first two (0, 1), the middle two (T/2 − 1, T/2) and the last two (T − 2,
T − 1) (``experiments.loss_curves``; the other tasks train untraced).

One figure per cell, ``results/training_verify/loss_N{N}_k{κ}_d0{δ₀}.png``:
three panels on a broken x axis (steps of the first two tasks | of the
middle two | of the last two), y = the loss on a log axis (seed mean,
±1 s.e. band), one trace per readout drawn continuously across the task
switch, which is marked by a faint dotted vertical line; the loss jumps
up when a new task starts and decays as it is learned.  The curves are in
``loss_….npz`` as ``{readout}`` (n_seeds, 6, n_steps + 1) with ``tasks``.

``--comp_opts`` compares the optimization methods: every cell is trained twice, with the population solver
(the config's η) and with online SGD (eqs. sgdW–sgda, one fresh sample per step), from the same seeds, so the
two share W⁽⁰⁾, the readout, the teachers and the gates.  The figure, ``loss_comp_N{N}_k{κ}_d0{δ₀}.png``, has
the same three panels with one colour per readout, population solid and SGD dotted.  The two take different
numbers of steps per task (population: 40; SGD: n = λD/(η ā²_nom), 80·D at the default rate), so x is the
fraction of the task's steps.  SGD's rate is ``--eta_sgd``, by default the largest the contraction rule allows,
η = sgd_contraction/(1 + δ₀) (0.05/(1 + δ₀)); SGD always runs the nominal clock (i), the population solver the
clock of ``--clock``.  Both curves are the population loss L_c (SGD's is evaluated, not estimated from its
samples).  The ``.npz`` holds ``{readout}`` (population) and ``sgd_{readout}`` (n_seeds, 6, n_sgd + 1).  SGD
costs n/40 times the population stream (≈ 800× at N = 100): shorten the stream with ``--tasks`` on a laptop.

``--tasks T`` runs a stream of T tasks instead of the config's T_burn + T_meas (the traced tasks are still the
first, middle and last two).

Placement (``sscl_core.pool``): the runs of a cell are one ``vmap`` batch per
closed-form class, split over the GPUs (``--gpus``), ``--workers`` CPU
processes or this process.
"""
import argparse
import math
import os
import sys
import time
from typing import Optional

import numpy as np
from matplotlib.ticker import LogLocator, NullLocator

import _path  # noqa: F401

from sscl_core import experiments as X
from sscl_core import plotting as P
from sscl_core import pool
from sscl_core import theory as TH
from sscl_core.config import Config, READOUT_ORDER, Run, add_clock_arg, clock_override, make_runs

_T0 = time.perf_counter()        # wall clock from after the imports to the end of the script

DEFAULT_CELLS = ("100,2,1", "50,1,0", "20,1,4", "200,2,1")


def parse_cell(spec: str) -> tuple[int, str, float]:
    """``"N,kappa,delta0"`` → (N, κ as typed, δ₀); ``sqrtN`` is accepted for κ."""
    N, k, d0 = spec.split(",")
    return int(N), k, float(d0)


def runs_of(cells: list[tuple], cfg: Config, seeds: int, lam: float, readouts: list[str],
            solver: Optional[str] = None, eta: Optional[float] = None) -> dict[tuple, list[Run]]:
    """``{cell: runs}``: every readout × seed of each cell under the config's solver and stream rules.

    ``solver`` replaces the config's, and with it the rate: ``eta`` for ``"sgd"`` when given, else ``"auto"``
    (``config.resolve_eta``: 1 for the population solver, sgd_contraction/(1 + δ₀) for SGD).
    """
    out = {}
    for N, k, d0 in cells:
        over = {}
        if solver is not None:
            over["solver"] = solver
            over["eta"] = [eta if (solver == "sgd" and eta is not None) else "auto"]
        c = Config(**{**cfg.__dict__, "name": "training_verify", "N": [N], "kappa": [k], "readout": readouts,
                      "delta0": [d0], "lam": [lam], "seeds": seeds, "extra": (), "trace": False, **over})
        runs = make_runs(c)
        if runs:                                            # a cell with only gated readouts and κ outside [1, √N] is empty
            out[(N, k, d0)] = runs
    return out


def windows(T: int) -> list[int]:
    """The six traced tasks: first two, middle two, last two."""
    return [0, 1, T // 2 - 1, T // 2, T - 2, T - 1]


def stream_T(r: Run, tasks: Optional[int] = None) -> int:
    """Tasks in the stream of a run: ``tasks`` (``--tasks``) or the config's T_burn + T_meas."""
    return int(tasks) if tasks else r.T_burn + r.T_meas


def groups(cells_: dict, tasks: Optional[int] = None) -> dict[tuple, list[Run]]:
    """``{group_key + (T,): runs}`` over every cell (a cell's closed-form and stepped runs apart; the two solvers
    of ``--comp_opts`` differ in ``group_key``)."""
    g = {}
    for runs in cells_.values():
        for r in runs:
            g.setdefault(X.group_key(r) + (stream_T(r, tasks),), []).append(r)
    return {k: g[k] for k in sorted(g, key=lambda k: -k[0])}


def make_jobs(groups_: dict, pieces) -> list:
    """One job per piece of a group: ``loss_curves(runs, windows(T), T)``."""
    jobs = []
    for (key, runs), k in zip(groups_.items(), pieces):
        T = key[-1]
        for i, part in enumerate(X.split(runs, k)):
            jobs.append(((key, i), {"runs": part, "tasks": windows(T), "T": T}, X.job_cost(part, T)))
    return jobs


def collect(cells_: dict, plan: pool.Plan, log=print, tasks: Optional[int] = None) -> dict[tuple, dict[str, np.ndarray]]:
    """``{cell: {readout: (n_seeds, 6, n_steps + 1)}}`` of the loss curves through ``pool.run_jobs``."""
    jobs = make_jobs(groups(cells_, tasks), plan.pieces)
    res = pool.run_jobs("sscl_core.experiments:loss_curves", jobs, log=log, **plan.place)
    per_run = {}
    for (_, kw, _), L in zip(jobs, res):
        for j, r in enumerate(kw["runs"]):
            per_run[r.label()] = L[j]
    out = {}
    for cell, runs in cells_.items():
        out[cell] = {}
        for ro in READOUT_ORDER:
            rows = [per_run[r.label()] for r in runs if r.readout == ro]
            rows = [L for L in rows if np.isfinite(L).all()]       # infeasible seeds dropped
            if rows:
                out[cell][ro] = np.stack(rows)
    return out


def figure(curves: dict[str, np.ndarray], run0: Run, T: int, path: str) -> str:
    """Three panels (first two | middle two | last two tasks) on a broken x axis; y = loss (log); traces = readouts."""
    P.apply_style()
    n = run0.n_steps
    tasks = windows(T)
    fig, axes = P.grid(1, 3, sharex=False, sharey="row", panel=(2.9, 2.6))
    figdata = {"tasks": np.array(tasks), "step": np.arange(n + 1)}
    readouts = [ro for ro in READOUT_ORDER if ro in curves]
    n_seeds = max(Y.shape[0] for Y in curves.values())
    for w in range(3):
        ax = axes[0, w]
        ta, tb = tasks[2 * w], tasks[2 * w + 1]
        x = np.arange(2 * (n + 1))                                  # task ta: steps 0..n, then task tb
        for ro in readouts:
            Y = np.concatenate([curves[ro][:, 2 * w], curves[ro][:, 2 * w + 1]], 1)   # (n_seeds, 2(n+1))
            m = Y.mean(0)
            se = Y.std(0, ddof=1) / np.sqrt(Y.shape[0]) if Y.shape[0] > 1 else np.zeros_like(m)
            P.band(ax, x, np.maximum(m - se, 1e-300), m + se, P.READOUT_COLOR[ro])
            P.plot_readout(ax, x, m, ro)
            figdata[ro] = curves[ro]
        ax.axvline(n + 0.5, color=P.GREY_REF, ls=":", lw=0.9, alpha=0.5, zorder=1)
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(numticks=6))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_xticks([0, n, n + 1, 2 * n + 1])
        ax.set_xticklabels(["0", str(n), "0", str(n)])
        ax.set_title(f"tasks {ta} and {tb}", fontsize=P.FS_MIN + 1)
        ax.set_xlabel("step within task")
        if w > 0:                                                   # the broken axis: no spine between the panels
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="y", length=0)
        if w < 2:
            ax.spines["right"].set_visible(False)
    axes[0, 0].set_ylabel(r"$L_c = |u_c - w_c|^2 / 2D$ of the task being trained")
    for w in range(2):                                              # break marks between the panels
        for ax, xx in ((axes[0, w], 1.0), (axes[0, w + 1], 0.0)):
            ax.plot([xx, xx], [-0.02, 0.02], transform=ax.transAxes, color="k", lw=0.8, clip_on=False)
    fig.get_layout_engine().set(wspace=0.02)
    P.cell_title(fig, kappa=run0.kappa_label, d0=run0.delta0, lam=run0.lam, solver=run0.solver, eta=run0.eta,
                 extra=f"N = {run0.N}, {n_seeds} seeds")
    P.caption(fig, f"population loss of the task being trained, before its first step and after each of its {n} steps, "
                   f"for the first two, the middle two and the last two of the stream's {T} tasks (broken x axis); "
                   f"line: seed mean ({n_seeds} seeds), band: ±1 s.e.; dotted: the task switch, where the loss jumps "
                   f"to the new task's initial error.  Every task is learned to the same residual at steady state.")
    P.fig_legend(fig, readouts)
    return P.save(fig, path, figdata=figdata)


def task_pair(Y: np.ndarray, w: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Window ``w`` (two consecutive traced tasks) of (n_seeds, 6, n + 1) curves on the fraction-of-task axis:
    (x, seed mean, s.e.), each (2(n + 1),), the first task on [0, 1] and the second on [1, 2]."""
    n = Y.shape[2] - 1
    frac = np.arange(n + 1) / n
    y = np.concatenate([Y[:, 2 * w], Y[:, 2 * w + 1]], 1)
    se = y.std(0, ddof=1) / np.sqrt(y.shape[0]) if y.shape[0] > 1 else np.zeros(y.shape[1])
    return np.concatenate([frac, 1.0 + frac]), y.mean(0), se


#: Line style of each solver in the comparison figure.
SOLVER_LS = {"pop": "-", "sgd": ":"}


def figure_comp(curves: dict[str, dict[str, np.ndarray]], run0: dict[str, Run], T: int, path: str) -> str:
    """``--comp_opts``: the three panels of ``figure`` with both solvers, population solid and SGD dotted.

    Parameters
    ----------
    curves : dict
        ``{"pop" | "sgd": {readout: (n_seeds, 6, n_steps + 1)}}``; the two solvers differ in ``n_steps``.
    run0 : dict
        ``{"pop" | "sgd": any run of the cell}``.
    T : int
        Tasks in the stream.
    path : str
        Figure stem.

    Returns
    -------
    str
        PNG path.
    """
    P.apply_style()
    tasks = windows(T)
    fig, axes = P.grid(1, 3, sharex=False, sharey="row", panel=(2.9, 2.6))
    figdata = {"tasks": np.array(tasks)}
    readouts = [ro for ro in READOUT_ORDER if any(ro in curves[s] for s in curves)]
    n_seeds = max(Y.shape[0] for c in curves.values() for Y in c.values())
    for s, c in curves.items():
        figdata["step" if s == "pop" else f"step_{s}"] = np.arange(run0[s].n_steps + 1)
        for ro, Y in c.items():
            figdata[ro if s == "pop" else f"{s}_{ro}"] = Y
    for w in range(3):
        ax = axes[0, w]
        for s in ("pop", "sgd"):
            for ro in readouts:
                if ro not in curves[s]:
                    continue
                x, m, se = task_pair(curves[s][ro], w)
                P.band(ax, x, np.maximum(m - se, 1e-300), m + se, P.READOUT_COLOR[ro])
                P.plot_readout(ax, x, m, ro, ls=SOLVER_LS[s])
        ax.axvline(1.0, color=P.GREY_REF, ls=":", lw=0.9, alpha=0.5, zorder=1)
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(numticks=6))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["0", "1 | 0", "1"])
        ax.set_title(f"tasks {tasks[2 * w]} and {tasks[2 * w + 1]}", fontsize=P.FS_MIN + 1)
        ax.set_xlabel("fraction of the task's steps")
        if w > 0:                                                   # the broken axis: no spine between the panels
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="y", length=0)
        if w < 2:
            ax.spines["right"].set_visible(False)
    axes[0, 0].set_ylabel(r"$L_c = |u_c - w_c|^2 / 2D$ of the task being trained")
    for w in range(2):                                              # break marks between the panels
        for ax, xx in ((axes[0, w], 1.0), (axes[0, w + 1], 0.0)):
            ax.plot([xx, xx], [-0.02, 0.02], transform=ax.transAxes, color="k", lw=0.8, clip_on=False)
    fig.get_layout_engine().set(wspace=0.02)
    rp, rs = run0["pop"], run0["sgd"]
    P.cell_title(fig, kappa=rp.kappa_label, d0=rp.delta0, lam=rp.lam, extra=f"N = {rp.N}, population vs SGD")
    P.caption(fig, f"population loss of the task being trained against the fraction of its steps, for the first two, "
                   f"the middle two and the last two of the stream's {T} tasks (broken x axis); solid: population "
                   f"solver ({rp.n_steps} steps per task, η = {P.fmt(rp.eta)}, {rp.clock} clock), dotted: online SGD "
                   f"({rs.n_steps} steps{f' of {rs.sgd_batch} samples' if rs.sgd_batch > 1 else ''}, η = {P.fmt(rs.eta)}, {rs.clock} clock), same seeds; line: seed mean "
                   f"({n_seeds} seeds), band: ±1 s.e.; grey dotted: the task switch.")
    P.fig_legend(fig, readouts, styles=[(SOLVER_LS["pop"], "population"), (SOLVER_LS["sgd"], "SGD")])
    return P.save(fig, path, figdata=figdata)


def report(N: int, k: str, d0: float, lam: float, run0: Run, curves: dict[str, np.ndarray], tag: str = "") -> None:
    """Print, per readout, the loss at the start and end of the first and last traced task and ρ_c of the last."""
    pred = math.exp(-2.0 * TH.lam_eff(lam, run0.n_steps))           # frozen readout, clock (ii): e^{−2λ_eff}, eq. (lameff)
    for ro, Y in curves.items():
        with np.errstate(invalid="ignore", divide="ignore"):
            rho = np.nanmean(Y[:, -1, -1] / Y[:, -1, 0])                 # ρ_c of the last traced task, seed mean
        print(f"  N={N:4d} κ={k:5s} δ₀={d0:4g} {tag}{ro:15s} loss at task start → end: first {Y[:, 0, 0].mean():.3f} → "
              f"{Y[:, 0, -1].mean():.2e}, last {Y[:, -1, 0].mean():.3f} → {Y[:, -1, -1].mean():.2e}  "
              f"ρ_c(last) = {rho:.2e} (e^(−2λ_eff) = {pred:.2e}, {run0.clock} clock)  ({Y.shape[0]} seeds)",
              flush=True)


def main(argv=None) -> int:
    """Command-line entry point (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(_path.ROOT, "configs", "stage1.yaml"))
    ap.add_argument("--cell", nargs="+", default=list(DEFAULT_CELLS), help="N,kappa,delta0 per cell")
    ap.add_argument("--lam", type=float, default=None, help="default: the config's first λ")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--readout", nargs="*", default=list(READOUT_ORDER))
    ap.add_argument("--out", default=os.path.join(_path.ROOT, "results", "training_verify"))
    ap.add_argument("--tasks", type=int, default=None, help="tasks in the stream (default: the config's T_burn + T_meas)")
    ap.add_argument("--comp_opts", action="store_true",
                    help="compare optimization methods: train every cell with the population solver and with SGD, "
                         "one figure per cell (population solid, SGD dotted)")
    ap.add_argument("--eta_sgd", type=float, default=None,
                    help="SGD rate of --comp_opts (default: sgd_contraction/(1 + δ₀), the largest allowed)")
    add_clock_arg(ap)
    pool.add_args(ap)
    args = ap.parse_args(argv)

    cfg = Config.from_yaml(args.config)
    lam = args.lam if args.lam is not None else cfg.lam[0]
    cfg = Config(**{**cfg.__dict__, **clock_override(args)})
    if args.tasks is not None and args.tasks < 6:
        raise SystemExit("--tasks must be at least 6 (the first, middle and last two tasks are traced)")
    specs = [parse_cell(c) for c in args.cell]
    if args.comp_opts:                                              # both solvers; the SGD cells carry a fourth key
        cells_ = runs_of(specs, cfg, args.seeds, lam, list(args.readout), solver="pop")
        cells_.update({key + ("sgd",): runs for key, runs in
                       runs_of(specs, cfg, args.seeds, lam, list(args.readout), solver="sgd", eta=args.eta_sgd).items()})
    else:
        cells_ = runs_of(specs, cfg, args.seeds, lam, list(args.readout))
    if not cells_:
        print("no runs left after skipping κ outside [1, √N]; nothing to do", flush=True)
        return 0
    g = groups(cells_, args.tasks)
    plan = pool.plan(args, [(v, k[-1]) for k, v in g.items()])
    r0 = next(iter(cells_.values()))[0]
    what = "population vs SGD" if args.comp_opts else r0.solver
    print(f"training verification: cells={[c for c in cells_ if len(c) == 3]} λ={lam} {what} {r0.clock} clock "
          f"n_steps={sorted({r.n_steps for runs in cells_.values() for r in runs})} seeds={args.seeds}  "
          f"[{plan.describe()}]", flush=True)
    res = collect(cells_, plan, tasks=args.tasks)

    outs = []
    for (N, k, d0) in [c for c in res if len(c) == 3]:
        curves = res[(N, k, d0)]
        run0 = cells_[(N, k, d0)][0]
        T = stream_T(run0, args.tasks)
        if not args.comp_opts:
            report(N, k, d0, lam, run0, curves)
            if curves:
                outs.append(figure(curves, run0, T, os.path.join(args.out, f"loss_N{N}_k{P.fmt(k)}_d0{P.fmt(d0)}")))
            continue
        both = {"pop": curves, "sgd": res[(N, k, d0, "sgd")]}
        runs0 = {"pop": run0, "sgd": cells_[(N, k, d0, "sgd")][0]}
        for s in both:
            report(N, k, d0, lam, runs0[s], both[s], tag=f"{s} ")
        if both["pop"] or both["sgd"]:
            outs.append(figure_comp(both, runs0, T, os.path.join(args.out, f"loss_comp_N{N}_k{P.fmt(k)}_d0{P.fmt(d0)}")))
    print("wrote\n  " + "\n  ".join(outs))
    return 0

if __name__ == "__main__":
    _rc = main()
    _dt = time.perf_counter() - _T0
    print(f"run_training_verify: total {_dt:.1f} s ({_dt / 60:.2f} min) from after the imports to the end", flush=True)
    sys.exit(_rc)
