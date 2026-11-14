#!/usr/bin/env python
"""Area under the current task's loss curve against task, one column per replay size, one trace per method.

    python scripts/run_loss_slope.py [--N 100] [--capacity 0.5D 4D] [--delta0 0] [--lam 4] [--seeds 4] [--tasks T]
                                     [--mode sleep|interleaved] [--gpus]

The replay counterpart of ``muP_experiment/scripts/run_loss_slope.py``.  For every task of the stream the
population loss of the task being trained is recorded before its first step and after each of its training
steps, and the area between that curve and L = 0 is taken over the task's ``n_steps`` training steps (trapezoid
rule, loss × steps; ``replay.loss_slopes``) — the sleep steps are not part of the area, so the two replay modes
and no replay are compared over the same steps.  The loss at the end of the sleep is recorded beside it.

Output, in ``results/loss_slope/figures/``: ``loss_slope_N{N}_d0{δ₀}_lam{λ}_{mode}.png`` — one column per replay
size C, x = task + 1 (log), y = the area, seed mean ± 1 s.e. band, one trace per replay method (``none`` dashed
grey, the same runs in every column).  ``loss_slope_….npz`` holds ``area_C{C}_{method}``, ``slope_…`` (the
least-squares slope of the curve per step), ``loss_start_…``, ``loss_end_…`` (after the last training step)
and ``loss_sleep_…`` (after the sleep), each (T, n_seeds), with ``task``.

Console: per method the area of task 0 and its mean over the second half of the stream, with the losses at the
start, the end of training and the end of the sleep over that half.
"""
import argparse
import os
import sys
import time
from typing import Optional

import numpy as np

import _path  # noqa: F401

import common as C
from sscl_core import plotting as P
from sscl_core import replay as R

_T0 = time.perf_counter()        # wall clock from after the imports to the end of the script


def figure(cols: dict[str, dict[str, dict[str, np.ndarray]]], run0: R.RRun, path: str) -> str:
    """``cols[capacity label][method][key]`` (T, n_seeds) → one column per capacity, one trace per method."""
    P.apply_style()
    fig, axes = P.grid(1, len(cols), sharey="row", panel=(3.4, 2.6))
    figdata, methods, n_seeds, T = {}, [], 0, 0
    for ax, (cap, by_m) in zip(axes[0], cols.items()):
        for m, d in by_m.items():
            Y = d["area"]
            T, n_seeds = max(T, Y.shape[0]), max(n_seeds, Y.shape[1])
            x = np.arange(Y.shape[0]) + 1                           # log axis: task 0 is drawn at x = 1
            mean, se = C.seed_mean(Y, 1)
            P.band(ax, x, mean - se, mean + se, C.METHOD_COLOR[m])
            C.plot_method(ax, x, mean, m)
            methods.append(m)
            for q in R.SLOPE_KEYS:
                figdata[f"{q}_C{cap}_{m}"] = d[q]
        P.ref_line(ax, 0.0)
        P.logaxis(ax, "x")
        ax.set_title(C.cap_text(cap), fontsize=P.FS_MIN + 1)
        ax.set_xlabel("task $t + 1$")
    axes[0, 0].set_ylabel("area under $L_c$ of the current task\n(loss × steps)")
    figdata["task"] = np.arange(T)
    P.cell_title(fig, d0=run0.delta0, lam=run0.lam, solver=run0.solver, eta=run0.eta,
                 extra=f"N = {run0.N}, {C.MODE_LABEL[run0.mode]}")
    P.caption(fig, f"area between the population loss $L_c = |u - w_c|^2/2D$ of the task being trained and 0 over its "
                   f"{run0.n_steps} training steps (trapezoid rule; smaller = lower loss during the task), drawn at "
                   f"t + 1 so task 0 is at 1; line: seed mean ({n_seeds} seeds), band: ±1 s.e.; no replay is the same "
                   f"runs in every column; {C.replay_text(run0)}.")
    P.fig_legend(fig, handles=C.method_handles(methods))
    return P.save(fig, path, figdata=figdata)


def main(argv: Optional[list[str]] = None) -> int:
    """Command-line entry point (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    C.add_args(ap, "loss_slope", N=[100], delta0_list=True)
    ap.add_argument("--tasks", type=int, default=None, help="tasks to run (default: the config's T_burn + T_meas)")
    args = ap.parse_args(argv)
    runs = C.build(args, "loss_slope")
    T = C.stream_T(runs, args.tasks, "--tasks")
    r0 = runs[0]
    print(f"loss area vs task: N={args.N} C={C.cap_labels(runs)} methods={C.methods_of(runs)} δ₀={args.delta0} "
          f"λ={r0.lam} {r0.solver} n_steps={r0.n_steps} T={T} seeds={args.seeds}", flush=True)
    cols, _ = C.collect("loss_slopes", runs, T, {"T": T}, lambda d, j: {q: d[q][:, j] for q in R.SLOPE_KEYS}, args,
                        out_per_task=8.0 * len(R.SLOPE_KEYS))
    cells_ = C.cells(runs)
    half = T // 2
    for N in sorted({k[0] for k in cells_}):
        for d0 in sorted({k[2] for k in cells_}):
            fig_cols, run0 = {}, None
            for (N_, cap, d0_, lam), cell in cells_.items():
                if (N_, d0_) != (N, d0):
                    continue
                by_m = C.by_method(cell, cols, finite=lambda d: all(np.isfinite(d[q]).all() for q in R.SLOPE_KEYS))
                fig_cols[cap] = {m: {q: np.stack([d[q] for d in v], 1) for q in R.SLOPE_KEYS} for m, v in by_m.items()}
                run0 = next((r for r in cell if r.method != "none"), cell[0])
                print(f"cell N={N} C={cap} δ₀={d0} λ={lam}:")
                for m, d in fig_cols[cap].items():
                    mm = {q: d[q].mean(1) for q in R.SLOPE_KEYS}
                    print(f"  {C.METHOD_LABEL[m]:16s} area: task 0 {mm['area'][0]:8.2f}, tasks {half}–{T - 1} "
                          f"{mm['area'][half:].mean():8.2f}   loss there: start {mm['loss_start'][half:].mean():.3f} → "
                          f"trained {mm['loss_end'][half:].mean():.3e} → after sleep {mm['loss_sleep'][half:].mean():.3e}",
                          flush=True)
            stem = os.path.join(args.out, "figures", f"loss_slope_N{N}_d0{P.fmt(d0)}_lam{P.fmt(run0.lam)}_{run0.mode}")
            print(f"wrote {figure(fig_cols, run0, stem)} and {stem}.npz")
    return 0


if __name__ == "__main__":
    _rc = main()
    _dt = time.perf_counter() - _T0
    print(f"run_loss_slope: total {_dt:.1f} s ({_dt / 60:.2f} min) from after the imports to the end", flush=True)
    sys.exit(_rc)
