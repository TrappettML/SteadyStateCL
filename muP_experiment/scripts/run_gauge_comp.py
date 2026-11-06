#!/usr/bin/env python
"""Gauge comparison (PLAN.md §12.10): Δ_i per step with and without the projection, one figure per cell.

    python scripts/run_gauge_comp.py [--N 100] [--kappa 1] [--delta0 0] [--lam 4] [--seeds 1] [--tasks T] [--gpus]

One figure per cell (N, κ, δ₀, λ), ``results/gauge_comp/figures/gauge_comp_N{N}_k{κ}_d0{δ₀}_lam{λ}.png``:
two rows × one column, x = global training step (log), y = the per-unit
imbalance Δ_i = (a_i² − Q_ii)/η averaged over the active units of the
current task, then averaged over the seeds (band: ±1 s.e. over seeds; the
seed means of the per-run min and max over active units go to the .npz).
Row 1 runs every
readout with the projection switched off (``do_proj = 0``), row 2 with it
switched on (``do_proj = 1``); every other setting of the readout (gating,
learned/frozen a, task heads) is what PLAN.md §4 defines.  One trace per
readout; ``--seeds n`` runs seeds 0 … n − 1 as the other ``run_*`` scripts
(default 1); N = 100 and κ = 1 by default; the task stream is the
Stage-1 one (``configs/stage1.yaml``: λ = 4, population solver, T_burn =
T_meas = 7N) unless ``--tasks`` shortens it.  The per-step arrays are
written next to the figure as ``gauge_comp_….npz``.

Placement (``mup.pool``): the readouts of a cell, each listed with the
projection off and on, are one ``vmap`` batch advanced ``--chunk`` tasks per
device call with the per-step statistics returned from the scan
(``experiments.imbalance_over_steps``); the distinct (run, projection)
pairs of every cell — an ungated run joins every κ cell but is trained
once — are the jobs over the
GPUs (``--gpus``: the four cards, several processes each — ``pool.plan``),
``--workers`` CPU processes or this process — list several ``--N``, ``--kappa``, ``--delta0`` or
raise ``--seeds`` to fill the cards.
"""
import argparse
import os
import sys
import time
from typing import Callable, Optional

import numpy as np

import _path  # noqa: F401

from mup import experiments as X
from mup import plotting as P
from mup import pool
from mup.config import Config, READOUT_ORDER, Run, make_runs

_T0 = time.perf_counter()        # wall clock from after the imports to the end of the script

#: the two rows: (do_proj value, row title)
ROWS = ((0.0, "no gauge projection"), (1.0, "gauge projection after every step"))


def cells(runs: list[Run]) -> dict[tuple, list[Run]]:
    """``{(N, κ label, δ₀, λ): [runs: every readout × seed]}``; the ungated readouts join every κ cell."""
    out = {}
    kappas = sorted({r.kappa_label for r in runs if r.gated == 1.0}, key=float) or [None]
    for N in sorted({r.N for r in runs}):
        for k in kappas:
            for d0 in sorted({r.delta0 for r in runs}):
                for lam in sorted({r.lam for r in runs}):
                    cell = [r for r in runs if (r.N, r.delta0, r.lam) == (N, d0, lam)
                            and (r.kappa_label == k if r.gated == 1.0 else True)]
                    if cell:
                        out[(N, k, d0, lam)] = cell
    return out


def groups(cells_: dict[tuple, list[Run]]) -> dict[tuple, list[tuple[Run, float]]]:
    """``{group_key: [(run, do_proj)]}`` over the distinct (run, projection) pairs of every cell: an ungated run,
    which joins every κ cell, is trained once per projection setting."""
    seen, out = set(), {}
    for cell in cells_.values():
        for do_proj, _ in ROWS:
            for r in cell:
                if (r.label(), do_proj) not in seen:
                    seen.add((r.label(), do_proj))
                    out.setdefault(X.group_key(r), []).append((r, do_proj))
    return out


def make_jobs(groups_: dict[tuple, list[tuple[Run, float]]], n_tasks: int) -> list[tuple]:
    """One job per group: its runs with their ``do_proj`` flags (``experiments.imbalance_over_steps``)."""
    jobs = []
    for key, pairs in groups_.items():
        runs, do_proj = [r for r, _ in pairs], [d for _, d in pairs]
        jobs.append((key, {"runs": runs, "do_proj": do_proj, "n_tasks": n_tasks}, X.run_cost(runs[0], n_tasks) * len(runs)))
    return jobs


def traces_of(cell: list[Run], rows_of: dict[tuple, np.ndarray]) -> dict[tuple, dict[str, np.ndarray]]:
    """``{(do_proj, readout): {"mean", "min", "max", "abs_max"}}`` of a cell from the per-(run, do_proj) rows,
    each an (n_seeds, n_steps) array over the feasible seeds of that readout (the readout is absent when none is)."""
    out = {}
    for do_proj, _ in ROWS:
        for ro in READOUT_ORDER:
            rows = [rows_of[(r.label(), do_proj)] for r in cell if r.readout == ro]
            rows = [R for R in rows if np.isfinite(R).any()]         # infeasible seeds dropped
            if rows:
                R = np.stack(rows)                                   # (n_seeds, n_steps, 4)
                out[(do_proj, ro)] = {"mean": R[:, :, 0], "min": R[:, :, 1], "max": R[:, :, 2], "abs_max": R[:, :, 3]}
    return out


def seed_stats(Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Seed mean and standard error along axis 0 of an (n_seeds, n_steps) array (s.e. 0 for one seed)."""
    m = Y.mean(0)
    se = Y.std(0, ddof=1) / np.sqrt(Y.shape[0]) if Y.shape[0] > 1 else np.zeros_like(m)
    return m, se


def collect(cells_: dict[tuple, list[Run]], n_tasks: int, plan: pool.Plan,
            log: Callable[[str], None] = print) -> dict[tuple, dict[tuple, dict[str, np.ndarray]]]:
    """``{cell: traces}`` through ``pool.run_jobs`` over the distinct (run, do_proj) pairs."""
    jobs = make_jobs(groups(cells_), n_tasks)
    res = pool.run_jobs("mup.experiments:imbalance_over_steps", jobs, log=log, **plan.place)
    rows_of = {}
    for (_, kw, _), R in zip(jobs, res):
        for j, (r, d) in enumerate(zip(kw["runs"], kw["do_proj"])):
            rows_of[(r.label(), d)] = R[j]
    return {key: traces_of(cell, rows_of) for key, cell in cells_.items()}


def figure(traces: dict[tuple, dict[str, np.ndarray]], run0: Run, n_tasks: int, path: str) -> str:
    """Two rows (no projection / projection), one trace per readout (seed mean ± 1 s.e.), x = step (log).

    Parameters
    ----------
    traces : dict
        ``traces[(do_proj, readout)]`` from ``traces_of`` (per-seed arrays).
    run0 : config.Run
        Any run of the cell (N, κ, δ₀, λ, n_steps).
    n_tasks : int
        Tasks run (the x axis spans ``n_tasks * run0.n_steps`` steps).
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
    n_seeds = max(tr["mean"].shape[0] for tr in traces.values())
    for r, (do_proj, title) in enumerate(ROWS):
        ax = axes[r, 0]
        for ro in readouts:
            if (do_proj, ro) not in traces:
                continue
            tr = traces[(do_proj, ro)]
            m, se = seed_stats(tr["mean"])                           # seed mean ± s.e. of the active-unit mean
            x = np.arange(1, len(m) + 1)
            P.band(ax, x, m - se, m + se, P.READOUT_COLOR[ro])
            P.plot_readout(ax, x, m, ro)
            tag = f"{'proj' if do_proj else 'noproj'}_{ro}"
            figdata.update({f"{tag}_mean": m, f"{tag}_se": se, f"{tag}_n_seeds": np.int64(tr["mean"].shape[0]),
                            f"{tag}_min": tr["min"].mean(0), f"{tag}_max": tr["max"].mean(0)})
        P.ref_line(ax, run0.Delta0)
        P.logaxis(ax, "x")
        ax.set_title(title, fontsize=P.FS_MIN + 1, loc="left")
        ax.set_ylabel(r"$\Delta_i$ (active-unit mean)")
    axes[-1, 0].set_xlabel("training step")
    figdata["step"] = np.arange(1, n_tasks * run0.n_steps + 1)
    P.cell_title(fig, kappa=run0.kappa_label, d0=run0.delta0, lam=run0.lam, solver=run0.solver, eta=run0.eta,
                 extra=f"N = {run0.N}, {n_seeds} seed{'s' if n_seeds != 1 else ''}")
    P.caption(fig, f"line: seed mean of the active-unit mean of Δ_i = (a_i² − Q_ii)/η after each step ({n_seeds} "
                   f"seeds); band: ±1 s.e. over seeds; grey line: Δ₀ = {P.fmt(run0.Delta0)}.  {n_tasks} tasks × "
                   f"{run0.n_steps} steps.  The .npz also holds the seed means of the per-run min and max over active units.")
    P.fig_legend(fig, readouts)
    return P.save(fig, path, figdata=figdata)


def main(argv: Optional[list[str]] = None) -> int:
    """Command-line entry point (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(_path.ROOT, "configs", "stage1.yaml"),
                    help="config supplying the solver, λ and step-size rules (default: Stage 1)")
    ap.add_argument("--N", type=int, nargs="+", default=[100])
    ap.add_argument("--kappa", nargs="+", default=["1.0"])
    ap.add_argument("--delta0", type=float, nargs="+", default=[0.0])
    ap.add_argument("--lam", type=float, default=None, help="default: the config's first λ")
    ap.add_argument("--seeds", type=int, default=1, help="number of seeds (0 … n − 1), one figure each")
    ap.add_argument("--tasks", type=int, default=None, help="tasks to run (default: the config's T_burn + T_meas)")
    ap.add_argument("--readout", nargs="*", default=list(READOUT_ORDER))
    ap.add_argument("--out", default=os.path.join(_path.ROOT, "results", "gauge_comp"))
    pool.add_args(ap)
    args = ap.parse_args(argv)

    cfg = Config.from_yaml(args.config)
    cfg = Config(**{**cfg.__dict__, "name": "gauge_comp", "N": list(args.N), "kappa": list(args.kappa),
                    "readout": list(args.readout), "delta0": list(args.delta0),
                    "lam": [args.lam if args.lam is not None else cfg.lam[0]], "seeds": args.seeds,
                    "extra": (), "trace": False})
    runs = make_runs(cfg)
    run0 = runs[0]
    n_tasks = args.tasks or (run0.T_burn + run0.T_meas)
    cells_ = cells(runs)
    plan = pool.plan(args, [([r for r, _ in g], n_tasks) for g in groups(cells_).values()], out_per_task=32 * run0.n_steps, split=False)
    print(f"gauge comparison: N={args.N} κ={args.kappa} δ₀={args.delta0} λ={run0.lam} seeds={args.seeds} {run0.solver} "
          f"n_steps={run0.n_steps} tasks={n_tasks} ({n_tasks * run0.n_steps} steps)  "
          f"[{plan.describe()}, {len(cells_)} cell(s)]", flush=True)
    res = collect(cells_, n_tasks, plan)

    for (N, k, d0, lam), traces in res.items():
        cell = cells_[(N, k, d0, lam)]
        run0 = next((r for r in cell if r.gated == 1.0), cell[0])
        print(f"cell N={N} κ={k} δ₀={d0} λ={lam} ({args.seeds} seeds):")
        for do_proj, title in ROWS:
            for ro in [r for r in READOUT_ORDER if any(rr.readout == r for rr in cell)]:
                if (do_proj, ro) not in traces:
                    print(f"  {title:36s} {ro:15s} every seed infeasible, skipped", flush=True)
                    continue
                tr = traces[(do_proj, ro)]
                m, se = seed_stats(tr["mean"][:, -1:])
                am, ase = seed_stats(tr["abs_max"].max(1)[:, None])
                print(f"  {title:36s} {ro:15s} final mean Δ = {m[0]:+.3e} ± {se[0]:.1e}  "
                      f"max|Δ−Δ₀| = {am[0]:.3e} ± {ase[0]:.1e}  ({tr['mean'].shape[0]}/{args.seeds} seeds)", flush=True)
        if not traces:
            continue
        tag = f"gauge_comp_N{N}_k{P.fmt(k)}_d0{P.fmt(d0)}_lam{P.fmt(lam)}"
        stem = os.path.join(args.out, "figures", tag)
        out = figure(traces, run0, n_tasks, stem)
        print(f"wrote {out} and {stem}.npz")
    return 0


if __name__ == "__main__":
    _rc = main()
    _dt = time.perf_counter() - _T0
    print(f"run_gauge_comp: total {_dt:.1f} s ({_dt / 60:.2f} min) from after the imports to the end", flush=True)
    sys.exit(_rc)
