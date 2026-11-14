#!/usr/bin/env python
"""Area under the loss curve of the task being trained, against task: one trace per readout, one column per κ.

    python scripts/run_loss_slope.py [--N 100] [--kappa 1 2] [--delta0 0] [--lam 4] [--seeds 4] [--tasks T]
                                     [--clock realized|nominal] [--gpus auto]

For every task t of the stream the population loss L_c = |u_c − w_c|²/(2D) of the task being trained is
recorded before its first step and after each of its n steps (``experiments.loss_task``, the curve of
``run_training_verify.py``), and reduced to one number: the area between that curve and the axis L = 0, from
the start of training on the task to its end, by the trapezoid rule in loss × steps
(``experiments.curve_area``).  The loss is non-negative, so the area is positive; it is small when the task
starts with a small loss and is learned quickly, and equals n · L_start when nothing is learned (a gated task
whose gate drew no unit).

One figure per (N, δ₀), ``results/loss_slope/figures/loss_slope_N{N}_d0{δ₀}_lam{λ}.png``: one column per κ,
x = t + 1 on a log axis (task 0 at x = 1), y = the area, seed mean (band: ±1 s.e. over seeds), one trace per
readout of §4; the ungated readouts (no κ) are trained once and drawn in every column.  N = 100, δ₀ = 0, 4
seeds by default; the κ columns, solver, λ, step-size rule and stream length (T_burn + T_meas tasks, 1400 at
N = 100, unless ``--tasks``) come from ``configs/stage1.yaml``.

The arrays are written next to the figure as ``loss_slope_….npz``: per column and readout
``area_k{κ}_{readout}``, ``slope_k{κ}_{readout}`` (the least-squares slope of the curve per step,
``experiments.fit_slope``: the quantity this script first plotted), ``loss_start_k{κ}_{readout}`` and
``loss_end_k{κ}_{readout}``, each (T, n_seeds), with ``task``.

Placement (``sscl_core.pool``): every readout and seed of a κ column is one ``vmap`` batch advanced ``--chunk``
tasks per device call with the area evaluated inside the scan (``experiments.loss_slopes``), over the GPUs
(``--gpus``), ``--workers`` CPU processes, or this process.
"""
import argparse
import os
import sys
import time
from typing import Callable, Optional, Sequence

import numpy as np

import _path  # noqa: F401

from sscl_core import experiments as X
from sscl_core import plotting as P
from sscl_core import pool
from sscl_core.config import Config, READOUT_ORDER, Run, add_clock_arg, clock_override, make_runs
from run_szero import cells, groups

_T0 = time.perf_counter()        # wall clock from after the imports to the end of the script


def make_jobs(groups_: dict[tuple, list[Run]], T: int, pieces: Sequence[int]) -> list[tuple]:
    """One job per piece of a group: ``(key, kwargs of experiments.loss_slopes, cost)``; ``pieces`` per group."""
    jobs = []
    for (key, runs), k in zip(groups_.items(), pieces):
        for i, part in enumerate(X.split(runs, k)):
            jobs.append(((key, i), {"runs": part, "T": T}, X.job_cost(part, T)))
    return jobs


def collect(cells_: dict[tuple, list[Run]], T: int, plan: pool.Plan,
            log: Callable[[str], None] = print) -> dict[tuple, dict[str, dict[str, np.ndarray]]]:
    """``{cell: {readout: {key: (T, n_seeds)}}}`` over ``experiments.SLOPE_KEYS``, infeasible seeds dropped.

    The distinct runs of every cell go through ``pool.run_jobs`` once (the ungated readouts are listed in
    every κ cell and trained once).
    """
    jobs = make_jobs(groups(cells_), T, plan.pieces)
    res = pool.run_jobs("sscl_core.experiments:loss_slopes", jobs, log=log, **plan.place)
    cols = {}
    for (_, kw, _), out in zip(jobs, res):
        for j, r in enumerate(kw["runs"]):
            cols[r.label()] = {q: out[q][:, j] for q in X.SLOPE_KEYS}
    data = {}
    for key, cell in cells_.items():
        data[key] = {}
        for ro in READOUT_ORDER:
            runs = [r for r in cell if r.readout == ro]
            runs = [r for r in runs if all(np.isfinite(cols[r.label()][q]).all() for q in X.SLOPE_KEYS)]
            if runs:
                data[key][ro] = {q: np.stack([cols[r.label()][q] for r in runs], 1) for q in X.SLOPE_KEYS}
    return data


def seed_mean(Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and s.e. over the seeds (axis 1) of a (T, n_seeds) array; the s.e. is 0 for one seed."""
    n = Y.shape[1]
    return Y.mean(1), (Y.std(1, ddof=1) / np.sqrt(n) if n > 1 else np.zeros(Y.shape[0]))


def figure(cols: dict[str, dict[str, dict[str, np.ndarray]]], run0: Run, path: str) -> str:
    """Area under the current task's loss curve against task: one column per κ, one trace per readout (± 1 s.e.).

    Parameters
    ----------
    cols : dict
        ``{κ label: {readout: {key: (T, n_seeds)}}}`` in column order (``collect``).
    run0 : config.Run
        Any run of the figure (N, δ₀, λ, solver, n_steps).
    path : str
        Figure stem.

    Returns
    -------
    str
        PNG path.
    """
    P.apply_style()
    fig, axes = P.grid(1, len(cols), sharey="row", panel=(3.4, 2.6))
    figdata = {}
    readouts = [ro for ro in READOUT_ORDER if any(ro in c for c in cols.values())]
    n_seeds, T = 0, 0
    for ax, (k, by_ro) in zip(axes[0], cols.items()):
        for ro, d in by_ro.items():
            Y = d["area"]
            T, n_seeds = max(T, Y.shape[0]), max(n_seeds, Y.shape[1])
            x = np.arange(Y.shape[0]) + 1                               # log axis: task 0 is drawn at x = 1
            m, se = seed_mean(Y)
            P.band(ax, x, m - se, m + se, P.READOUT_COLOR[ro])
            P.plot_readout(ax, x, m, ro)
            for q in X.SLOPE_KEYS:
                figdata[f"{q}_k{P.fmt(k)}_{ro}"] = d[q]
        P.ref_line(ax, 0.0)
        P.logaxis(ax, "x")
        ax.set_title(rf"$\kappa$ = {'√N' if k == 'sqrtN' else P.fmt(k)}", fontsize=P.FS_MIN + 1)
        ax.set_xlabel("task $t + 1$")
    axes[0, 0].set_ylabel("area under $L_c$ of the current task\n(loss × steps)")
    figdata["task"] = np.arange(T)
    P.cell_title(fig, d0=run0.delta0, lam=run0.lam, solver=run0.solver, eta=run0.eta, extra=f"N = {run0.N}")
    P.caption(fig, f"area between the population loss $L_c = |u_c - w_c|^2/2D$ of the task being trained and 0, from "
                   f"its first step to its last ({run0.n_steps} steps, trapezoid rule; smaller = lower loss "
                   f"during the task), drawn at t + 1 so "
                   f"task 0 is at 1; line: seed mean ({n_seeds} seeds), band: ±1 s.e.; the ungated readouts are "
                   f"the same runs in every column.")
    P.fig_legend(fig, readouts)
    return P.save(fig, path, figdata=figdata)


def main(argv: Optional[list[str]] = None) -> int:
    """Command-line entry point (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(_path.ROOT, "configs", "stage1.yaml"),
                    help="config supplying the κ columns, solver, λ, step-size and stream-length rules (default: Stage 1)")
    ap.add_argument("--N", type=int, nargs="+", default=[100])
    ap.add_argument("--kappa", nargs="+", default=None, help="one column each (default: the config's κ)")
    ap.add_argument("--delta0", type=float, nargs="+", default=[0.0])
    ap.add_argument("--lam", type=float, default=None, help="default: the config's first λ")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--tasks", type=int, default=None, help="tasks to run (default: the config's T_burn + T_meas)")
    ap.add_argument("--readout", nargs="*", default=list(READOUT_ORDER))
    ap.add_argument("--out", default=os.path.join(_path.ROOT, "results", "loss_slope"))
    add_clock_arg(ap)
    pool.add_args(ap)
    args = ap.parse_args(argv)

    cfg = Config.from_yaml(args.config)
    kappas = [str(k) for k in (args.kappa if args.kappa is not None else cfg.kappa)]
    over = {"name": "loss_slope", "N": list(args.N), "kappa": kappas, "readout": list(args.readout),
            "delta0": list(args.delta0), "lam": [args.lam if args.lam is not None else cfg.lam[0]],
            "seeds": args.seeds, "extra": (), "trace": False, **clock_override(args)}
    cfg = Config(**{**cfg.__dict__, **over})
    runs = make_runs(cfg)
    if not runs:
        print("no runs left after skipping κ outside [1, √N]; nothing to do", flush=True)
        return 0
    cells_ = cells(runs)
    T_of = {key: args.tasks or (cell[0].T_burn + cell[0].T_meas) for key, cell in cells_.items()}
    if len(set(T_of.values())) != 1:
        raise SystemExit("cells differ in T; pass --tasks")
    T = next(iter(T_of.values()))
    plan = pool.plan(args, [(g, T) for g in groups(cells_).values()])
    r0 = runs[0]
    print(f"loss area vs task: N={args.N} κ={kappas} δ₀={args.delta0} λ={r0.lam} {r0.solver} n_steps={r0.n_steps} "
          f"{r0.clock} clock T={T} seeds={args.seeds}  [{plan.describe()}]", flush=True)
    data = collect(cells_, T, plan)

    figs = {}                                                       # (N, δ₀, λ) → {κ label: {readout: {key: (T, n_seeds)}}}
    for (N, k, d0, lam), by_ro in data.items():
        print(f"cell N={N} κ={k} δ₀={d0} λ={lam}:")
        n = cells_[(N, k, d0, lam)][0].n_steps
        half = T // 2
        for ro, d in by_ro.items():
            A, s, L0 = d["area"].mean(1), d["slope"].mean(1), d["loss_start"].mean(1)
            print(f"  {ro:15s} area under the loss (loss × steps): task 0 {A[0]:.3f}, mean of the last {T - half} "
                  f"tasks {A[half:].mean():.3f}  (of n·L_start = {n * L0[0]:.2f}, {n * L0[half:].mean():.2f}; fitted "
                  f"slope per step {s[0]:+.3e}, {s[half:].mean():+.3e})  ({d['area'].shape[1]} seeds)", flush=True)
        if by_ro:
            figs.setdefault((N, d0, lam), {})[k] = by_ro
    for (N, d0, lam), cols in figs.items():
        run0 = next(r for r in runs if r.N == N and r.delta0 == d0 and r.lam == lam)
        stem = os.path.join(args.out, "figures", f"loss_slope_N{N}_d0{P.fmt(d0)}_lam{P.fmt(lam)}")
        print(f"wrote {figure(cols, run0, stem)} and {stem}.npz")
    return 0


if __name__ == "__main__":
    _rc = main()
    _dt = time.perf_counter() - _T0
    print(f"run_loss_slope: total {_dt:.1f} s ({_dt / 60:.2f} min) from after the imports to the end", flush=True)
    sys.exit(_rc)
