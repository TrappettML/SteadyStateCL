#!/usr/bin/env python
"""Is every task learned under replay: the per-step loss of the task being trained at the start, middle and end.

    python scripts/run_training_verify.py [--cell N,C,delta0 ...] [--seeds 4] [--lam 4] [--tasks T]
                                          [--mode sleep|interleaved] [--gpus]

The replay counterpart of ``muP_experiment/scripts/run_training_verify.py``.  For every cell (default
``100,0.5D,0`` and ``100,4D,0``; a cell is N, replay size C, δ₀) every replay method × seed trains through the
stream (default the config's T_burn + T_meas tasks; ``--tasks`` shortens it), and the population loss
L_c = |u − w_c|²/(2D) of the task being trained is recorded before its first step and after each step for six
tasks: the first two, the middle two and the last two.  Under sleep replay the curve continues through the
sleep steps that follow the task's own (``replay.replay_task``), so the figure shows what the sleep does to the
task just learned; ``none`` has no sleep and its curve ends with the task.

Output, in ``results/training_verify/``: ``loss_N{N}_C{C}_d0{δ₀}_{mode}.png`` — three panels on a broken x axis
(first two | middle two | last two tasks), y = loss (log), one trace per replay method (``none`` dashed grey),
seed mean ± 1 s.e. band; faint dotted lines mark the start of a sleep, a solid grey line the task switch.
``loss_….npz`` holds the curves ``{method}`` (n_seeds, 6, steps + 1), with ``tasks``, ``n_steps``, ``n_sleep``.

Console: per method the loss at the start of the first and of the last traced task, at the end of its
training steps and at the end of its sleep.
"""
import argparse
import os
import sys
import time
from typing import Optional

import numpy as np

import _path  # noqa: F401

import common as C
from matplotlib.ticker import LogLocator, NullLocator
from sscl_core import plotting as P
from sscl_core import replay as R

_T0 = time.perf_counter()        # wall clock from after the imports to the end of the script


def windows(T: int) -> list[int]:
    """The six traced tasks of a ``T``-task stream: the first two, the middle two, the last two."""
    return [0, 1, T // 2 - 1, T // 2, T - 2, T - 1]


def figure(curves: dict[str, np.ndarray], run0: R.RRun, cap: str, T: int, path: str) -> str:
    """Three panels (first two | middle two | last two tasks) on a broken x axis; y = loss (log); traces = methods."""
    P.apply_style()
    n, n_sleep = run0.spec[0], run0.spec[5]
    L = n + n_sleep + 1                                             # points per traced task of a replay run
    tasks = windows(T)
    fig, axes = P.grid(1, 3, sharex=False, sharey="row", panel=(2.9, 2.6))
    figdata = {"tasks": np.array(tasks), "n_steps": np.array(n), "n_sleep": np.array(n_sleep)}
    n_seeds = max(Y.shape[0] for Y in curves.values())
    for w in range(3):
        ax = axes[0, w]
        for m, Y in curves.items():
            for half in range(2):                                   # the two tasks of the panel, side by side
                y = Y[:, 2 * w + half]
                mean, se = C.seed_mean(y, 0)
                x = half * L + np.arange(y.shape[1])
                P.band(ax, x, np.maximum(mean - se, 1e-300), mean + se, C.METHOD_COLOR[m])
                C.plot_method(ax, x, mean, m)
            figdata[m] = Y
        ax.axvline(L - 0.5, color=P.GREY_REF, lw=0.9, alpha=0.6, zorder=1)
        if n_sleep:
            for half in range(2):
                ax.axvline(half * L + n, color=P.GREY_REF, ls=":", lw=0.9, alpha=0.6, zorder=1)
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(numticks=6))
        ax.yaxis.set_minor_locator(NullLocator())
        ticks = [0, n] + ([L - 1] if n_sleep else [])
        ax.set_xticks(ticks + [L + t for t in ticks])
        ax.set_xticklabels([str(t) for t in ticks] * 2)
        ax.set_title(f"tasks {tasks[2 * w]} and {tasks[2 * w + 1]}", fontsize=P.FS_MIN + 1)
        ax.set_xlabel("step within task" + (" (training, then sleep)" if n_sleep else ""))
        if w > 0:                                                   # the broken axis: no spine between the panels
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="y", length=0)
    axes[0, 0].set_ylabel(r"$L_c = |u - w_c|^2 / 2D$ of the task being trained")
    P.cell_title(fig, d0=run0.delta0, lam=run0.lam, solver=run0.solver, eta=run0.eta,
                 extra=f"N = {run0.N}, {C.cap_text(cap)}, {C.MODE_LABEL[run0.mode]}, {n_seeds} seeds")
    P.caption(fig, f"population loss of the task being trained, before its first step and after each step, for the "
                   f"first two, the middle two and the last two of the stream's {T} tasks (broken x axis); line: "
                   f"seed mean, band: ±1 s.e.; grey line: the task switch"
                   + (f"; dotted: the start of the {n_sleep} sleep steps, which replay the buffer and no longer "
                      f"train the task (no replay has none)" if n_sleep else "")
                   + f"; {C.replay_text(run0)}.")
    P.fig_legend(fig, handles=C.method_handles(list(curves)))
    return P.save(fig, path, figdata=figdata)


def main(argv: Optional[list[str]] = None) -> int:
    """Command-line entry point (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    C.add_args(ap, "training_verify", N=[100])
    ap.add_argument("--cell", nargs="+", default=["100,0.5D,0", "100,4D,0"], help="cells N,C,delta0")
    ap.add_argument("--tasks", type=int, default=None, help="tasks of the stream (default: the config's T_burn + T_meas)")
    args = ap.parse_args(argv)
    outs = []
    for spec in args.cell:
        N, cap, d0 = spec.split(",")
        args.N, args.capacity, args.delta0 = [int(N)], [cap], float(d0)
        runs = C.build(args, "training_verify")
        T = C.stream_T(runs, args.tasks, "--tasks")
        if T < 6:
            raise SystemExit("the stream needs at least 6 tasks")
        run0 = next((r for r in runs if r.method != "none"), runs[0])
        cap_label = run0.cap_label
        print(f"training verify: N={N} C={cap_label} δ₀={d0} λ={run0.lam} {run0.solver} n_steps={run0.n_steps} "
              f"{C.MODE_LABEL[run0.mode]} T={T} seeds={args.seeds}", flush=True)
        cols, _ = C.collect("loss_curves", runs, T, {"tasks": windows(T), "T": T}, lambda Lc, j: Lc[j], args,
                            out_per_task=8.0)
        curves = {m: np.stack(v) for m, v in C.by_method(runs, cols).items()}
        n = run0.n_steps
        for m, Y in curves.items():
            mean = Y.mean(0)
            print(f"  {C.METHOD_LABEL[m]:16s} first task: start {mean[0, 0]:.3e} → trained {mean[0, n]:.3e} → after sleep "
                  f"{mean[0, -1]:.3e};  last task: start {mean[-1, 0]:.3e} → trained {mean[-1, n]:.3e} → after sleep "
                  f"{mean[-1, -1]:.3e}", flush=True)
        stem = os.path.join(args.out, f"loss_N{N}_C{cap_label}_d0{P.fmt(d0)}_{run0.mode}")
        outs.append(figure(curves, run0, cap_label, T, stem))
        print(f"wrote {outs[-1]} and {stem}.npz")
    return 0


if __name__ == "__main__":
    _rc = main()
    _dt = time.perf_counter() - _T0
    print(f"run_training_verify: total {_dt:.1f} s ({_dt / 60:.2f} min) from after the imports to the end", flush=True)
    sys.exit(_rc)
