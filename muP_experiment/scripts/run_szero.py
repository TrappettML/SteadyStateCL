#!/usr/bin/env python
"""First-teacher signal against task (PLAN.md §12.11): S₀(t) = u_t·w₀*/|w₀*|², one trace per readout, one figure per cell.

    python scripts/run_szero.py [--N 100 200] [--kappa 1 2] [--delta0 0] [--lam 4] [--seeds 4] [--tasks T] [--gpus auto]

One figure per cell (N, κ, δ₀), ``results/szero/figures/szero_N{N}_k{κ}_d0{δ₀}_lam{λ}.png``:
x = t + 1 on a log axis (task 0 at x = 1), y = S₀(t)/S₀(0) (every trace starts at 1), with S₀(t) how well the student after task t
still aligns with the FIRST teacher w₀*: u_t = (g₀ ⊙ a) W_t / pN with the
gate g₀ of task 0 and its stored head (task heads) or the current shared
readout, projected on w₀* (PLAN.md §5 with the probe fixed at k = 0 and
age m = t).  S₀(0) is the signal just after learning task 0 and the
decay of S₀(t) from there is the memory of that task; the half-life m½
is where the seed mean crosses S₀(0)/2 and is given per readout in the caption.  One
trace per readout of §4 as the seed mean with a ±1 s.e. band over seeds.
N = 100, κ = 1, δ₀ = 0 by default; solver, λ and step-size rule come from
``configs/stage1.yaml``; the stream is the
Stage-1 one, T_burn + T_meas tasks (1400 at N = 100), unless ``--tasks``.  The
per-task arrays are written next to the figure as ``szero_….npz``.

Placement (``mup.pool``): every readout and seed of a cell is one ``vmap``
batch advanced ``--chunk`` tasks per device call with S₀ evaluated inside
the scan (``experiments.signal_after_tasks``) and runs as a job over the GPUs
(``--gpus``: the four cards, several processes each, the cells split only
as far as needed to fill them — ``pool.plan``), over ``--workers`` CPU
processes, or in this process.
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


def cell_key(r) -> tuple:
    """(N, κ label, δ₀, λ) of a run: one figure per cell."""
    return (r.N, r.kappa_label if r.gated == 1.0 else None, r.delta0, r.lam)


def cells(runs: list) -> dict:
    """``{(N, κ label, δ₀, λ): runs}``; the ungated readouts (no κ) are listed in every κ cell."""
    out = {}
    kappas = sorted({r.kappa_label for r in runs if r.gated == 1.0}, key=float) or [None]
    for N in sorted({r.N for r in runs}):
        for k in kappas:
            for d0 in sorted({r.delta0 for r in runs}):
                for lam in sorted({r.lam for r in runs}):
                    cell = [r for r in runs if r.N == N and r.delta0 == d0 and r.lam == lam
                            and (r.kappa_label == k if r.gated == 1.0 else True)]
                    if cell:
                        out[(N, k, d0, lam)] = cell
    return out


def make_jobs(cells_: dict, T: int, pieces) -> list:
    """One job per piece of a cell: ``(key, kwargs of experiments.signal_after_tasks, cost)``; ``pieces`` per cell."""
    jobs = []
    for (key, cell), k in zip(cells_.items(), pieces):
        for i, part in enumerate(X.split(cell, k)):
            jobs.append(((key, i), {"runs": part, "T_pre": 0, "n_follow": T, "k": 0}, X.run_cost(part[0], T) * len(part)))
    return jobs


def by_readout(cell: list, S: np.ndarray) -> dict:
    """``{readout: (T, n_seeds) array}`` of a cell from its stacked job results, infeasible seeds dropped."""
    out = {}
    for ro in READOUT_ORDER:
        cols = [j for j, r in enumerate(cell) if r.readout == ro]
        if not cols:
            continue
        Y = S[:, cols]
        Y = Y[:, np.isfinite(Y).all(0)]
        if Y.shape[1] == 0:
            print(f"  {ro}: every seed infeasible (δ₀ ≤ −min_i Q0_i), skipped", flush=True)
            continue
        out[ro] = Y
    return out


def collect(cells_: dict, T: int, plan: pool.Plan, log=print) -> dict:
    """Run every cell's jobs through ``pool.run_jobs`` and return ``{cell: {readout: (T, n_seeds)}}``."""
    jobs = make_jobs(cells_, T, plan.pieces)
    res = pool.run_jobs("mup.experiments:signal_after_tasks", jobs, log=log, **plan.place)
    parts = {}
    for (key, i), r in zip([j[0] for j in jobs], res):
        parts.setdefault(key, []).append((i, r))
    return {key: by_readout(cells_[key], np.concatenate([r for _, r in sorted(parts[key])], 1)) for key in cells_}


def figure(S0: dict, run0, path: str) -> str:
    """S₀(t) vs task, one trace per readout (seed mean ± 1 s.e.), half-life in the caption.

    Parameters
    ----------
    S0 : dict
        From ``szero_by_readout``.
    run0 : config.Run
        Any run of the cell.
    path : str
        Figure stem.

    Returns
    -------
    str
        PNG path.
    """
    P.apply_style()
    fig, axes = P.grid(1, 1, panel=(5.2, 2.6))
    ax = axes[0, 0]
    figdata = {}
    mh = {}
    for ro, Y in S0.items():
        T, n = Y.shape
        t = np.arange(T)
        m = Y.mean(1)
        se = Y.std(1, ddof=1) / np.sqrt(n) if n > 1 else np.zeros(T)
        x = t + 1                                                       # log axis: task 0 is drawn at x = 1
        s0 = m[0]                                                       # y is S₀(t)/S₀(0): every trace starts at 1
        P.band(ax, x, (m - se) / s0, (m + se) / s0, P.READOUT_COLOR[ro])
        P.plot_readout(ax, x, m / s0, ro)
        mh[ro] = half_life(m)
        figdata[f"S0_{ro}"] = Y
        figdata[f"S0norm_{ro}"] = m / s0
    figdata["task"] = np.arange(max(Y.shape[0] for Y in S0.values()))
    P.ref_line(ax, 0.0)
    P.ref_line(ax, 0.5)
    P.logaxis(ax, "x")
    ax.set_xlabel("task $t + 1$")
    ax.set_ylabel(r"$S_0(t)\,/\,S_0(0)$,  $S_0(t) = u_t\cdot w_0^*/|w_0^*|^2$")
    n_seeds = max(Y.shape[1] for Y in S0.values())
    P.cell_title(fig, kappa=run0.kappa_label, d0=run0.delta0, lam=run0.lam, solver=run0.solver, eta=run0.eta,
                 extra=f"N = {run0.N}")
    def _m(ro):
        v = mh[ro]
        return "> T" if not np.isfinite(v) else f"{v:.1f}"
    mtxt = ", ".join(f"{P.READOUT_LABEL[ro]} {_m(ro)}" for ro in S0)
    P.caption(fig, f"line: seed mean of the first teacher's signal after task t divided by its value after task 0, "
                   f"drawn at t + 1 so task 0 is at 1 ({n_seeds} seeds); band: ±1 s.e. over seeds; half-life m½ "
                   f"(tasks) where the mean crosses ½ (grey line): {mtxt}.")
    P.fig_legend(fig, list(S0))
    return P.save(fig, path, figdata=figdata)


def half_life(m: np.ndarray) -> float:
    """First task at which ``m`` falls to ``m[0]/2``, linearly interpolated; NaN if never."""
    half = m[0] / 2
    below = np.nonzero(m <= half)[0]
    if len(below) == 0:
        return float("nan")
    j = int(below[0])
    if j == 0:
        return 0.0
    return j - 1 + (m[j - 1] - half) / (m[j - 1] - m[j])


def main(argv=None):
    """Command-line entry point (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(_path.ROOT, "configs", "stage1.yaml"),
                    help="config supplying the solver, λ, step-size and stream-length rules (default: Stage 1)")
    ap.add_argument("--N", type=int, nargs="+", default=[100])
    ap.add_argument("--kappa", nargs="+", default=["1.0"])
    ap.add_argument("--delta0", type=float, nargs="+", default=[0.0])
    ap.add_argument("--lam", type=float, default=None, help="default: the config's first λ")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--tasks", type=int, default=None, help="tasks to run (default: the config's T_burn + T_meas)")
    ap.add_argument("--readout", nargs="*", default=list(READOUT_ORDER))
    ap.add_argument("--out", default=os.path.join(_path.ROOT, "results", "szero"))
    pool.add_args(ap)
    args = ap.parse_args(argv)

    cfg = Config.from_yaml(args.config)
    over = {"name": "szero", "N": list(args.N), "kappa": list(args.kappa), "readout": list(args.readout),
            "delta0": list(args.delta0), "lam": [args.lam if args.lam is not None else cfg.lam[0]],
            "seeds": args.seeds, "extra": (), "trace": False}
    cfg = Config(**{**cfg.__dict__, **over})
    runs = make_runs(cfg)
    cells_ = cells(runs)
    T_of = {key: args.tasks or (cell[0].T_burn + cell[0].T_meas) for key, cell in cells_.items()}
    if len(set(T_of.values())) != 1:
        raise SystemExit("cells differ in T; pass --tasks")
    T = next(iter(T_of.values()))
    plan = pool.plan(args, [(cell, T) for cell in cells_.values()])
    r0 = runs[0]
    print(f"S₀(t) vs task: N={args.N} κ={args.kappa} δ₀={args.delta0} λ={r0.lam} {r0.solver} n_steps={r0.n_steps} "
          f"T={T} seeds={args.seeds}  [{plan.describe()}]", flush=True)
    res = collect(cells_, T, plan)

    outs = []
    for (N, k, d0, lam), S0 in res.items():
        print(f"cell N={N} κ={k} δ₀={d0} λ={lam}:")
        for ro, Y in S0.items():
            m = Y.mean(1)
            print(f"  {ro:15s} S₀(0) = {m[0]:.4f}  m½ = {half_life(m):.1f}  S₀(T−1) = {m[-1]:+.4f}", flush=True)
        if not S0:
            continue
        cell = cells_[(N, k, d0, lam)]
        run0 = next((r for r in cell if r.gated == 1.0), cell[0])
        stem = os.path.join(args.out, "figures", f"szero_N{N}_k{P.fmt(k)}_d0{P.fmt(d0)}_lam{P.fmt(lam)}")
        outs.append(figure(S0, run0, stem))
        print(f"wrote {outs[-1]} and {stem}.npz")
    return 0


if __name__ == "__main__":
    _rc = main()
    _dt = time.perf_counter() - _T0
    print(f"run_szero: total {_dt:.1f} s ({_dt / 60:.2f} min) from after the imports to the end", flush=True)
    sys.exit(_rc)
