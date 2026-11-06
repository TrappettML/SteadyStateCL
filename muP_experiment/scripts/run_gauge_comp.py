#!/usr/bin/env python
"""Gauge comparison (PLAN.md §12.10): Δ_i per step with and without the projection, one figure per cell.

    python scripts/run_gauge_comp.py [--N 100] [--kappa 1] [--delta0 0] [--lam 4] [--seed 0] [--tasks T] [--gpus auto]

One figure per cell (N, κ, δ₀, seed), ``results/gauge_comp/figures/gauge_comp_N{N}_k{κ}_d0{δ₀}_lam{λ}[_s{seed}].png``:
two rows × one column, x = global training step (log), y = the per-unit
imbalance Δ_i = (a_i² − Q_ii)/η averaged over the active units of the
current task (min–max over active units as a band).  Row 1 runs every
readout with the projection switched off (``do_proj = 0``), row 2 with it
switched on (``do_proj = 1``); every other setting of the readout (gating,
learned/frozen a, task heads) is what PLAN.md §4 defines.  One trace per
readout, one seed, N = 100 and κ = 1 by default; the task stream is the
Stage-1 one (``configs/stage1.yaml``: λ = 4, population solver, T_burn =
T_meas = 7N) unless ``--tasks`` shortens it.  The per-step arrays are
written next to the figure as ``gauge_comp_….npz``.

Placement (``mup.pool``): the readouts of a cell, each listed with the
projection off and on, are one ``vmap`` batch advanced ``--chunk`` tasks per
device call with the per-step statistics returned from the scan
(``experiments.imbalance_over_steps``); every cell is one job over the
GPUs (``--gpus auto``: one process per card), ``--workers`` CPU processes
or this process — list several ``--N``, ``--kappa``, ``--delta0`` or
``--seed`` to fill the cards.
"""
import argparse
import os
import sys
import time

import numpy as np

import _path  # noqa: F401

from mup import experiments as X
from mup import plotting as P
from mup import pool
from mup.config import Config, READOUT_ORDER, make_runs

_T0 = time.perf_counter()        # wall clock from after the imports to the end of the script

#: the two rows: (do_proj value, row title)
ROWS = ((0.0, "no gauge projection"), (1.0, "gauge projection after every step"))


def cells(runs: list) -> dict:
    """``{(N, κ label, δ₀, λ, seed): [run per readout]}``; the ungated readouts join every κ cell."""
    out = {}
    kappas = sorted({r.kappa_label for r in runs if r.gated == 1.0}, key=float) or [None]
    for N in sorted({r.N for r in runs}):
        for k in kappas:
            for d0 in sorted({r.delta0 for r in runs}):
                for lam in sorted({r.lam for r in runs}):
                    for seed in sorted({r.seed for r in runs}):
                        cell = [r for r in runs if (r.N, r.delta0, r.lam, r.seed) == (N, d0, lam, seed)
                                and (r.kappa_label == k if r.gated == 1.0 else True)]
                        if cell:
                            out[(N, k, d0, lam, seed)] = cell
    return out


def make_jobs(cells_: dict, n_tasks: int) -> list:
    """One job per cell: every readout with ``do_proj`` 0 then 1 (``experiments.imbalance_over_steps``)."""
    jobs = []
    for key, cell in cells_.items():
        runs = [r for do_proj, _ in ROWS for r in cell]
        do_proj = [d for d, _ in ROWS for _ in cell]
        jobs.append((key, {"runs": runs, "do_proj": do_proj, "n_tasks": n_tasks}, X.run_cost(cell[0], n_tasks) * len(runs)))
    return jobs


def traces_of(cell: list, R: np.ndarray) -> dict:
    """``{(do_proj, readout): {"mean", "min", "max", "abs_max"}}`` of a cell from its job result."""
    out = {}
    i = 0
    for do_proj, _ in ROWS:
        for r in cell:
            rows = R[i]
            i += 1
            if not np.isfinite(rows).any():
                continue                                             # infeasible run
            out[(do_proj, r.readout)] = {"mean": rows[:, 0], "min": rows[:, 1], "max": rows[:, 2], "abs_max": rows[:, 3]}
    return out


def collect(cells_: dict, n_tasks: int, place: dict, log=print) -> dict:
    """``{cell: traces}`` through ``pool.run_jobs``."""
    jobs = make_jobs(cells_, n_tasks)
    res = pool.run_jobs("mup.experiments:imbalance_over_steps", jobs, log=log, **place)
    return {key: traces_of(cells_[key], R) for (key, _, _), R in zip(jobs, res)}


def figure(traces: dict, run0, n_tasks: int, path: str) -> str:
    """Two rows (no projection / projection), one trace per readout, x = step (log).

    Parameters
    ----------
    traces : dict
        ``traces[(do_proj, readout)]`` from ``trace_imbalance``.
    run0 : config.Run
        Any run of the cell (N, κ, δ₀, λ, n_steps).
    n_tasks : int
    path : str
        Figure stem.

    Returns
    -------
    str
        PNG path.
    """
    P.apply_style()
    fig, axes = P.grid(len(ROWS), 1, sharex="col", sharey=False, panel=(5.2, 2.2))
    readouts = [ro for ro in READOUT_ORDER if any(k[1] == ro for k in traces)]
    figdata = {}
    for r, (do_proj, title) in enumerate(ROWS):
        ax = axes[r, 0]
        for ro in readouts:
            tr = traces[(do_proj, ro)]
            x = np.arange(1, len(tr["mean"]) + 1)
            P.band(ax, x, tr["min"], tr["max"], P.READOUT_COLOR[ro])
            P.plot_readout(ax, x, tr["mean"], ro)
            tag = f"{'proj' if do_proj else 'noproj'}_{ro}"
            figdata.update({f"{tag}_mean": tr["mean"], f"{tag}_min": tr["min"], f"{tag}_max": tr["max"]})
        P.ref_line(ax, run0.Delta0)
        P.logaxis(ax, "x")
        ax.set_title(title, fontsize=P.FS_MIN + 1, loc="left")
        ax.set_ylabel(r"$\Delta_i$ (active-unit mean)")
    axes[-1, 0].set_xlabel("training step")
    figdata["step"] = np.arange(1, n_tasks * run0.n_steps + 1)
    P.cell_title(fig, kappa=run0.kappa_label, d0=run0.delta0, lam=run0.lam, solver=run0.solver, eta=run0.eta,
                 extra=f"N = {run0.N}, seed {run0.seed}")
    P.caption(fig, f"line: mean of Δ_i = (a_i² − Q_ii)/η over the active units after each step; band: min–max over "
                   f"active units; grey line: Δ₀ = {P.fmt(run0.Delta0)}.  {n_tasks} tasks × {run0.n_steps} steps.")
    P.fig_legend(fig, readouts)
    return P.save(fig, path, figdata=figdata)


def main(argv=None):
    """Command-line entry point (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(_path.ROOT, "configs", "stage1.yaml"),
                    help="config supplying the solver, λ and step-size rules (default: Stage 1)")
    ap.add_argument("--N", type=int, nargs="+", default=[100])
    ap.add_argument("--kappa", nargs="+", default=["1.0"])
    ap.add_argument("--delta0", type=float, nargs="+", default=[0.0])
    ap.add_argument("--lam", type=float, default=None, help="default: the config's first λ")
    ap.add_argument("--seed", type=int, nargs="+", default=[0])
    ap.add_argument("--tasks", type=int, default=None, help="tasks to run (default: the config's T_burn + T_meas)")
    ap.add_argument("--readout", nargs="*", default=list(READOUT_ORDER))
    ap.add_argument("--out", default=os.path.join(_path.ROOT, "results", "gauge_comp"))
    pool.add_args(ap)
    args = ap.parse_args(argv)

    cfg = Config.from_yaml(args.config)
    cfg = Config(**{**cfg.__dict__, "name": "gauge_comp", "N": list(args.N), "kappa": list(args.kappa),
                    "readout": list(args.readout), "delta0": list(args.delta0),
                    "lam": [args.lam if args.lam is not None else cfg.lam[0]], "seeds": list(args.seed),
                    "extra": (), "trace": False})
    runs = make_runs(cfg)
    run0 = runs[0]
    n_tasks = args.tasks or (run0.T_burn + run0.T_meas)
    place = pool.placement(args)
    cells_ = cells(runs)
    print(f"gauge comparison: N={args.N} κ={args.kappa} δ₀={args.delta0} λ={run0.lam} seeds={args.seed} {run0.solver} "
          f"n_steps={run0.n_steps} tasks={n_tasks} ({n_tasks * run0.n_steps} steps)  "
          f"[{pool.describe(place)}, {len(cells_)} cell(s)]", flush=True)
    res = collect(cells_, n_tasks, place)

    for (N, k, d0, lam, seed), traces in res.items():
        cell = cells_[(N, k, d0, lam, seed)]
        run0 = next((r for r in cell if r.gated == 1.0), cell[0])
        print(f"cell N={N} κ={k} δ₀={d0} λ={lam} seed={seed}:")
        for do_proj, title in ROWS:
            for ro in [r.readout for r in cell]:
                if (do_proj, ro) not in traces:
                    print(f"  {title:36s} {ro:15s} infeasible, skipped", flush=True)
                    continue
                tr = traces[(do_proj, ro)]
                print(f"  {title:36s} {ro:15s} final mean Δ = {tr['mean'][-1]:+.3e}  max|Δ−Δ₀| = {tr['abs_max'].max():.3e}", flush=True)
        if not traces:
            continue
        tag = f"gauge_comp_N{N}_k{P.fmt(k)}_d0{P.fmt(d0)}_lam{P.fmt(lam)}" + (f"_s{seed}" if len(args.seed) > 1 else "")
        stem = os.path.join(args.out, "figures", tag)
        out = figure(traces, run0, n_tasks, stem)
        print(f"wrote {out} and {stem}.npz")
    return 0


if __name__ == "__main__":
    _rc = main()
    _dt = time.perf_counter() - _T0
    print(f"run_gauge_comp: total {_dt:.1f} s ({_dt / 60:.2f} min) from after the imports to the end", flush=True)
    sys.exit(_rc)
