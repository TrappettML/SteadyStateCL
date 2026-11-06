#!/usr/bin/env python
"""Training verification: the per-step loss of the task being trained at the start, middle and end of the stream.

    python scripts/run_training_verify.py [--cell N,kappa,delta0 ...] [--seeds 4] [--lam 4] [--gpus]

For every cell (N, κ, δ₀) — by default the four
``100,2,1``, ``50,1,0``, ``20,1,4``, ``200,2,1`` — every readout × seed is
trained through the Stage-1 stream (``configs/stage1.yaml``: λ = 4,
population solver, T = T_burn + T_meas = 14N tasks × 40 steps) and the
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

Placement (``mup.pool``): the runs of a cell are one ``vmap`` batch per
closed-form class, split over the GPUs (``--gpus``), ``--workers`` CPU
processes or this process.
"""
import argparse
import os
import sys
import time

import numpy as np
from matplotlib.ticker import LogLocator, NullLocator

import _path  # noqa: F401

from mup import experiments as X
from mup import plotting as P
from mup import pool
from mup.config import Config, READOUT_ORDER, Run, make_runs

_T0 = time.perf_counter()        # wall clock from after the imports to the end of the script

DEFAULT_CELLS = ("100,2,1", "50,1,0", "20,1,4", "200,2,1")


def parse_cell(spec: str) -> tuple[int, str, float]:
    """``"N,kappa,delta0"`` → (N, κ as typed, δ₀); ``sqrtN`` is accepted for κ."""
    N, k, d0 = spec.split(",")
    return int(N), k, float(d0)


def runs_of(cells: list[tuple], cfg: Config, seeds: int, lam: float, readouts: list[str]) -> dict[tuple, list[Run]]:
    """``{cell: runs}``: every readout × seed of each cell under the config's solver and stream rules."""
    out = {}
    for N, k, d0 in cells:
        c = Config(**{**cfg.__dict__, "name": "training_verify", "N": [N], "kappa": [k], "readout": readouts,
                      "delta0": [d0], "lam": [lam], "seeds": seeds, "extra": (), "trace": False})
        runs = make_runs(c)
        if runs:                                            # a cell with only gated readouts and κ outside [1, √N] is empty
            out[(N, k, d0)] = runs
    return out


def windows(T: int) -> list[int]:
    """The six traced tasks: first two, middle two, last two."""
    return [0, 1, T // 2 - 1, T // 2, T - 2, T - 1]


def groups(cells_: dict) -> dict[tuple, list[Run]]:
    """``{group_key + (T,): runs}`` over every cell (a cell's closed-form and stepped runs apart)."""
    g = {}
    for runs in cells_.values():
        for r in runs:
            g.setdefault(X.group_key(r) + (r.T_burn + r.T_meas,), []).append(r)
    return {k: g[k] for k in sorted(g, key=lambda k: -k[0])}


def make_jobs(groups_: dict, pieces) -> list:
    """One job per piece of a group: ``loss_curves(runs, windows(T), T)``."""
    jobs = []
    for (key, runs), k in zip(groups_.items(), pieces):
        T = key[-1]
        for i, part in enumerate(X.split(runs, k)):
            jobs.append(((key, i), {"runs": part, "tasks": windows(T), "T": T}, X.job_cost(part, T)))
    return jobs


def collect(cells_: dict, plan: pool.Plan, log=print) -> dict[tuple, dict[str, np.ndarray]]:
    """``{cell: {readout: (n_seeds, 6, n_steps + 1)}}`` of the loss curves through ``pool.run_jobs``."""
    jobs = make_jobs(groups(cells_), plan.pieces)
    res = pool.run_jobs("mup.experiments:loss_curves", jobs, log=log, **plan.place)
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


def main(argv=None) -> int:
    """Command-line entry point (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(_path.ROOT, "configs", "stage1.yaml"))
    ap.add_argument("--cell", nargs="+", default=list(DEFAULT_CELLS), help="N,kappa,delta0 per cell")
    ap.add_argument("--lam", type=float, default=None, help="default: the config's first λ")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--readout", nargs="*", default=list(READOUT_ORDER))
    ap.add_argument("--out", default=os.path.join(_path.ROOT, "results", "training_verify"))
    pool.add_args(ap)
    args = ap.parse_args(argv)

    cfg = Config.from_yaml(args.config)
    lam = args.lam if args.lam is not None else cfg.lam[0]
    cells_ = runs_of([parse_cell(c) for c in args.cell], cfg, args.seeds, lam, list(args.readout))
    if not cells_:
        print("no runs left after skipping κ outside [1, √N]; nothing to do", flush=True)
        return 0
    g = groups(cells_)
    plan = pool.plan(args, [(v, k[-1]) for k, v in g.items()])
    r0 = next(iter(cells_.values()))[0]
    print(f"training verification: cells={[c for c in cells_]} λ={lam} {r0.solver} n_steps={r0.n_steps} "
          f"seeds={args.seeds}  [{plan.describe()}]", flush=True)
    res = collect(cells_, plan)

    outs = []
    for (N, k, d0), curves in res.items():
        run0 = cells_[(N, k, d0)][0]
        T = run0.T_burn + run0.T_meas
        for ro, Y in curves.items():
            print(f"  N={N:4d} κ={k:5s} δ₀={d0:4g} {ro:15s} loss at task start → end: first {Y[:, 0, 0].mean():.3f} → "
                  f"{Y[:, 0, -1].mean():.2e}, last {Y[:, -1, 0].mean():.3f} → {Y[:, -1, -1].mean():.2e}  ({Y.shape[0]} seeds)",
                  flush=True)
        if curves:
            outs.append(figure(curves, run0, T, os.path.join(args.out, f"loss_N{N}_k{P.fmt(k)}_d0{P.fmt(d0)}")))
    print("wrote\n  " + "\n  ".join(outs))
    return 0


if __name__ == "__main__":
    _rc = main()
    _dt = time.perf_counter() - _T0
    print(f"run_training_verify: total {_dt:.1f} s ({_dt / 60:.2f} min) from after the imports to the end", flush=True)
    sys.exit(_rc)
