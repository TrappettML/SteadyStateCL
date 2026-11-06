#!/usr/bin/env python
"""Steady-state teacher signal against age (PLAN.md §12.12): S_T(τ) = u_{T+τ}·w_T*/|w_T*|², one trace per readout, one figure per cell.

    python scripts/run_stask.py [--N 100] [--kappa 1] [--delta0 0] [--lam 4] [--seeds 4] [--T 1400] [--tau 1000] [--gpus auto]

The companion of ``run_szero.py`` with the probe taken after the network
has reached its steady state: train T tasks (default the Stage-1 stream
length, T_burn + T_meas = 1400 at N = 100), then follow the teacher of
task T for τ = 0 … ``--tau`` more tasks.  S_T(τ) = u_{T+τ}·w_T*/|w_T*|²
with u = (g_T ⊙ a) W / pN, the gate of task T and its stored head (task
heads) or the current shared readout, evaluated after every task T + τ.
One figure per cell (N, κ, δ₀), ``results/stask/figures/stask_N{N}_k{κ}_d0{δ₀}_lam{λ}.png``:
x = τ + 1 on a log axis (τ = 0, the task just learned, at x = 1), y =
S_T(τ)/S_T(0) so every trace starts at 1, one trace per readout of §4 as the seed mean with a ±1 s.e.
band over seeds; the half-life m½ (first crossing of S_T(0)/2) is in
the caption.  N = 100, κ = 1, δ₀ = 0 by default; solver, λ and step-size
rule come from ``configs/stage1.yaml``.  The per-τ arrays are written
next to the figure as ``stask_….npz``.

Placement as ``run_szero.py``: every readout and seed of a cell is one
``vmap`` batch (``experiments.signal_after_tasks`` with ``T_pre = T, k = T``),
run as jobs over the GPUs (``--gpus``, planned by ``pool.plan``), CPU
processes or in this process (``mup.pool``).
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
from run_szero import by_readout, cells, half_life

_T0 = time.perf_counter()        # wall clock from after the imports to the end of the script


def make_jobs(cells_: dict, T: int, tau: int, pieces) -> list:
    """One job per piece of a cell: train ``T`` tasks, follow probe ``T`` for ``tau + 1`` more; ``pieces`` per cell."""
    jobs = []
    for (key, cell), k in zip(cells_.items(), pieces):
        for i, part in enumerate(X.split(cell, k)):
            jobs.append(((key, i), {"runs": part, "T_pre": T, "n_follow": tau + 1, "k": T},
                         X.run_cost(part[0], T + tau + 1) * len(part)))
    return jobs


def collect(cells_: dict, T: int, tau: int, plan: pool.Plan, log=print) -> dict:
    """``{cell: {readout: (tau + 1, n_seeds) array}}``; row τ is S_T after task T + τ."""
    jobs = make_jobs(cells_, T, tau, plan.pieces)
    res = pool.run_jobs("mup.experiments:signal_after_tasks", jobs, log=log, **plan.place)
    parts = {}
    for (key, i), r in zip([j[0] for j in jobs], res):
        parts.setdefault(key, []).append((i, r))
    return {key: by_readout(cells_[key], np.concatenate([r for _, r in sorted(parts[key])], 1)) for key in cells_}


def figure(S: dict, run0, T: int, path: str) -> str:
    """S_T(τ) vs τ + 1 (log), one trace per readout (seed mean ± 1 s.e.), half-life in the caption.

    Parameters
    ----------
    S : dict
        From ``stask_by_readout``.
    run0 : config.Run
        Any run of the cell.
    T : int
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
    for ro, Y in S.items():
        n_tau, n = Y.shape
        x = np.arange(n_tau) + 1                                        # log axis: τ = 0 at x = 1
        m = Y.mean(1)
        se = Y.std(1, ddof=1) / np.sqrt(n) if n > 1 else np.zeros(n_tau)
        s0 = m[0]                                                       # y is S_T(τ)/S_T(0): every trace starts at 1
        P.band(ax, x, (m - se) / s0, (m + se) / s0, P.READOUT_COLOR[ro])
        P.plot_readout(ax, x, m / s0, ro)
        mh[ro] = half_life(m)
        figdata[f"ST_{ro}"] = Y
        figdata[f"STnorm_{ro}"] = m / s0
    figdata["tau"] = np.arange(max(Y.shape[0] for Y in S.values()))
    P.ref_line(ax, 0.0)
    P.ref_line(ax, 0.5)
    P.logaxis(ax, "x")
    ax.set_xlabel(r"$\tau + 1$ (tasks after task $T$)")
    ax.set_ylabel(r"$S_T(\tau)\,/\,S_T(0)$,  $S_T(\tau) = u_{T+\tau}\cdot w_T^*/|w_T^*|^2$")
    n_seeds = max(Y.shape[1] for Y in S.values())
    mtxt = ", ".join(f"{P.READOUT_LABEL[ro]} {mh[ro]:.1f}" if np.isfinite(mh[ro]) else f"{P.READOUT_LABEL[ro]} > τ"
                     for ro in S)
    P.cell_title(fig, kappa=run0.kappa_label, d0=run0.delta0, lam=run0.lam, solver=run0.solver, eta=run0.eta,
                 extra=f"N = {run0.N}, T = {T}")
    P.caption(fig, f"line: seed mean of the signal of teacher T = {T} after task T + τ divided by its value at τ = 0, "
                   f"drawn at τ + 1 so τ = 0 is at 1 ({n_seeds} seeds); band: ±1 s.e. over seeds; half-life m½ "
                   f"(tasks) where the mean crosses ½ (grey line): {mtxt}.")
    P.fig_legend(fig, list(S))
    return P.save(fig, path, figdata=figdata)


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
    ap.add_argument("--T", type=int, default=None, help="tasks before the probed one (default: the config's T_burn + T_meas)")
    ap.add_argument("--tau", type=int, default=1000, help="tasks followed after task T")
    ap.add_argument("--readout", nargs="*", default=list(READOUT_ORDER))
    ap.add_argument("--out", default=os.path.join(_path.ROOT, "results", "stask"))
    pool.add_args(ap)
    args = ap.parse_args(argv)

    cfg = Config.from_yaml(args.config)
    over = {"name": "stask", "N": list(args.N), "kappa": list(args.kappa), "readout": list(args.readout),
            "delta0": list(args.delta0), "lam": [args.lam if args.lam is not None else cfg.lam[0]],
            "seeds": args.seeds, "extra": (), "trace": False}
    cfg = Config(**{**cfg.__dict__, **over})
    runs = make_runs(cfg)
    cells_ = cells(runs)
    T_of = {key: args.T or (cell[0].T_burn + cell[0].T_meas) for key, cell in cells_.items()}
    if len(set(T_of.values())) != 1:
        raise SystemExit("cells differ in T; pass --T")
    T = next(iter(T_of.values()))
    plan = pool.plan(args, [(cell, T + args.tau + 1) for cell in cells_.values()])
    r0 = runs[0]
    print(f"S_T(τ) vs τ: N={args.N} κ={args.kappa} δ₀={args.delta0} λ={r0.lam} {r0.solver} n_steps={r0.n_steps} "
          f"T={T} τ≤{args.tau} seeds={args.seeds}  [{plan.describe()}]", flush=True)
    res = collect(cells_, T, args.tau, plan)

    outs = []
    for (N, k, d0, lam), S in res.items():
        print(f"cell N={N} κ={k} δ₀={d0} λ={lam}:")
        for ro, Y in S.items():
            m = Y.mean(1)
            print(f"  {ro:15s} S_T(0) = {m[0]:.4f}  m½ = {half_life(m):.1f}  S_T(τ_max) = {m[-1]:+.4f}", flush=True)
        if not S:
            continue
        cell = cells_[(N, k, d0, lam)]
        run0 = next((r for r in cell if r.gated == 1.0), cell[0])
        stem = os.path.join(args.out, "figures", f"stask_N{N}_k{P.fmt(k)}_d0{P.fmt(d0)}_lam{P.fmt(lam)}")
        outs.append(figure(S, run0, T, stem))
        print(f"wrote {outs[-1]} and {stem}.npz")
    return 0


if __name__ == "__main__":
    _rc = main()
    _dt = time.perf_counter() - _T0
    print(f"run_stask: total {_dt:.1f} s ({_dt / 60:.2f} min) from after the imports to the end", flush=True)
    sys.exit(_rc)
