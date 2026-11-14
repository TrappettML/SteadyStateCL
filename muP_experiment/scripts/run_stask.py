#!/usr/bin/env python
"""Steady-state teacher signal against age (PLAN.md §12.12): S_T(τ) = u_{T+τ}·w_T*/|w_T*|², one trace per readout, one figure per cell.

    python scripts/run_stask.py [--N 100] [--kappa 1] [--delta0 0] [--lam 4] [--seeds 4] [--T 1400] [--tau 1000]
                                [--stat mean|seed] [--clock realized|nominal] [--gpus auto]

The companion of ``run_szero.py`` with the probe taken after the network
has reached its steady state: train T tasks (default the Stage-1 stream
length, T_burn + T_meas = 1400 at N = 100), then follow the teacher of
task T for τ = 0 … ``--tau`` more tasks.  S_T(τ) = u_{T+τ}·w_T*/|w_T*|²
with u = (g_T ⊙ a) W / pN, the gate of task T and its stored head (task
heads) or the current shared readout, evaluated after every task T + τ.
One figure per cell (N, κ, δ₀), ``results/stask/figures/stask_N{N}_k{κ}_d0{δ₀}_lam{λ}.png``:
x = τ + 1 on a log axis (τ = 0, the task just learned, at x = 1), y =
S_T(τ)/S_T(0) so every trace starts at 1, one trace per readout of §4.  With ``--stat mean`` (default) the
trace is the write-up's Σ(τ) = ⟨S_T(τ)⟩/⟨S_T(0)⟩ over seeds (eq. Sigma; band: jackknife ±1 s.e.) and the
half-life m½ in the caption its first crossing of ½; with ``--stat seed`` the seed mean of each seed's own
ratio and the mean of the per-seed half-lives (``run_szero.seed_signal``, ``sscl_core.halflife``).  Dotted, in the
readout's colour: the equation the write-up says to plot — eq. (S_heads), eq. (S_frozen) and eq. (S_learned)
with the key drift of eqs. (delta_est) and (abarss), i.e. from the closed-form steady-state norms and no
measured input (``theory.sigma``).  N = 100, κ = 1, δ₀ = 0 by default; solver, λ and step-size
rule come from ``configs/stage1.yaml``.  The per-τ arrays are written
next to the figure as ``stask_….npz``.

Placement as ``run_szero.py``: every readout and seed of a cell is one
``vmap`` batch (``experiments.signal_after_tasks`` with ``T_pre = T, k = T``),
run as jobs over the GPUs (``--gpus``, planned by ``pool.plan``), CPU
processes or in this process (``sscl_core.pool``).
"""
import argparse
import os
import sys
import time
from typing import Callable, Optional, Sequence

import numpy as np

import _path  # noqa: F401

from sscl_core import experiments as X
from sscl_core import halflife as H
from sscl_core import plotting as P
from sscl_core import pool
from sscl_core import theory as TH
from sscl_core.config import Config, READOUT_ORDER, Run, add_clock_arg, clock_override, make_runs
from run_szero import (LINE_TEXT, MHALF_TEXT, by_readout, cells, columns, groups, mhalf_text, per_cell, seed_signal,
                       theory_curve)

_T0 = time.perf_counter()        # wall clock from after the imports to the end of the script


def make_jobs(groups_: dict[tuple, list[Run]], T: int, tau: int, pieces: Sequence[int]) -> list[tuple]:
    """One job per piece of a group of distinct runs (``run_szero.groups``): train ``T`` tasks, follow probe ``T``
    for ``tau + 1`` more; ``pieces`` per group."""
    jobs = []
    for (key, runs), k in zip(groups_.items(), pieces):
        for i, part in enumerate(X.split(runs, k)):
            jobs.append(((key, i), {"runs": part, "T_pre": T, "n_follow": tau + 1, "k": T},
                         X.job_cost(part, T + tau + 1)))
    return jobs


def collect(cells_: dict[tuple, list[Run]], T: int, tau: int, plan: pool.Plan,
            log: Callable[[str], None] = print) -> dict[tuple, dict[str, np.ndarray]]:
    """``{cell: {readout: (tau + 1, n_seeds) array}}``; row τ is S_T after task T + τ."""
    jobs = make_jobs(groups(cells_), T, tau, plan.pieces)
    res = pool.run_jobs("sscl_core.experiments:signal_after_tasks", jobs, log=log, **plan.place)
    return per_cell(cells_, columns(jobs, res))


def figure(S: dict[str, np.ndarray], run0: Run, T: int, path: str, stat: str = "mean") -> str:
    """S_T(τ) vs τ + 1 (log), one trace per readout (± 1 s.e.), its equation dotted, half-life in the caption.

    Parameters
    ----------
    S : dict
        ``{readout: (tau + 1, n_seeds)}`` from ``run_szero.by_readout``.
    run0 : config.Run
        Any run of the cell.
    T : int
        Tasks trained before the probed one (printed in the title).
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
        m, se, mh[ro] = seed_signal(Y, stat)                            # normalized signal and its half-life
        P.band(ax, x, m - se, m + se, P.READOUT_COLOR[ro])
        P.plot_readout(ax, x, m, ro)
        theory_curve(ax, ro, run0, n_tau, True, figdata)
        figdata[f"ST_{ro}"] = Y
        figdata[f"STnorm_{ro}"] = m
        figdata[f"mhalf_{ro}"] = mh[ro]
        figdata[f"mhalf_seed_{ro}"] = H.per_seed(Y)
    figdata["tau"] = np.arange(max(Y.shape[0] for Y in S.values()))
    P.ref_line(ax, 0.0)
    P.ref_line(ax, 0.5)
    P.logaxis(ax, "x")
    ax.set_xlabel(r"$\tau + 1$ (tasks after task $T$)")
    ax.set_ylabel(r"$S_T(\tau)\,/\,S_T(0)$,  $S_T(\tau) = u_{T+\tau}\cdot w_T^*/|w_T^*|^2$")
    n_seeds = max(Y.shape[1] for Y in S.values())
    n_tau_all = max(Y.shape[0] for Y in S.values())
    mtxt = ", ".join(f"{P.READOUT_LABEL[ro]} {mhalf_text(mh[ro], n_tau_all - 1)}" for ro in S)
    P.cell_title(fig, kappa=run0.kappa_label, d0=run0.delta0, lam=run0.lam, solver=run0.solver, eta=run0.eta,
                 extra=f"N = {run0.N}, T = {T}")
    P.caption(fig, f"line: {LINE_TEXT[stat].format(what=f'signal of teacher T = {T} after task T + τ', ref='at τ = 0')}, "
                   f"drawn at τ + 1 so τ = 0 is at 1 ({n_seeds} seeds); band: ±1 s.e.; dotted in the readout's "
                   f"colour: its equation, (S_heads), (S_frozen) or (S_learned) with the closed-form norms of "
                   f"eq. (abarss); half-life m½ (tasks), first crossing of ½ (grey line), {MHALF_TEXT[stat]}: {mtxt}.")
    P.fig_legend(fig, list(S), styles=[(":", "equation")])
    return P.save(fig, path, figdata=figdata)


def main(argv: Optional[list[str]] = None) -> int:
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
    H.add_arg(ap)
    add_clock_arg(ap)
    pool.add_args(ap)
    args = ap.parse_args(argv)

    cfg = Config.from_yaml(args.config)
    over = {"name": "stask", "N": list(args.N), "kappa": list(args.kappa), "readout": list(args.readout),
            "delta0": list(args.delta0), "lam": [args.lam if args.lam is not None else cfg.lam[0]],
            "seeds": args.seeds, "extra": (), "trace": False, **clock_override(args)}
    cfg = Config(**{**cfg.__dict__, **over})
    runs = make_runs(cfg)
    if not runs:
        print("no runs left after skipping κ outside [1, √N]; nothing to do", flush=True)
        return 0
    cells_ = cells(runs)
    T_of = {key: args.T or (cell[0].T_burn + cell[0].T_meas) for key, cell in cells_.items()}
    if len(set(T_of.values())) != 1:
        raise SystemExit("cells differ in T; pass --T")
    T = next(iter(T_of.values()))
    plan = pool.plan(args, [(g, T + args.tau + 1) for g in groups(cells_).values()])
    r0 = runs[0]
    print(f"S_T(τ) vs τ: N={args.N} κ={args.kappa} δ₀={args.delta0} λ={r0.lam} {r0.solver} n_steps={r0.n_steps} "
          f"{r0.clock} clock T={T} τ≤{args.tau} seeds={args.seeds} statistic={args.stat}  [{plan.describe()}]", flush=True)
    res = collect(cells_, T, args.tau, plan)

    outs = []
    for (N, k, d0, lam), S in res.items():
        print(f"cell N={N} κ={k} δ₀={d0} λ={lam}:")
        for ro, Y in S.items():
            m = Y.mean(1)
            kap = cells_[(N, k, d0, lam)][0].kappa if k else 1.0
            print(f"  {ro:15s} S_T(0) = {m[0]:.4f} (θ = {TH.theta(lam):.4f})  m½ = "
                  f"{mhalf_text(seed_signal(Y, args.stat)[2], Y.shape[0] - 1)}  S_T(τ_max) = {m[-1]:+.4f}  "
                  f"equation m½ = {TH.mhalf(ro, N, kap, lam, d0):.1f}", flush=True)
        if not S:
            continue
        cell = cells_[(N, k, d0, lam)]
        run0 = next((r for r in cell if r.gated == 1.0), cell[0])
        stem = os.path.join(args.out, "figures", f"stask_N{N}_k{P.fmt(k)}_d0{P.fmt(d0)}_lam{P.fmt(lam)}")
        outs.append(figure(S, run0, T, stem, args.stat))
        print(f"wrote {outs[-1]} and {stem}.npz")
    return 0


if __name__ == "__main__":
    _rc = main()
    _dt = time.perf_counter() - _T0
    print(f"run_stask: total {_dt:.1f} s ({_dt / 60:.2f} min) from after the imports to the end", flush=True)
    sys.exit(_rc)
