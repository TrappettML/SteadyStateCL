#!/usr/bin/env python
"""Weight norms, end-of-task loss and buffer age against task and against width, per replay method and size.

    python scripts/run_weight_norms.py [--N 10 20 50 100] [--capacity 0.5D 4D] [--delta0 0] [--lam 4] [--seeds 4]
                                       [--mode sleep|interleaved] [--tasks T] [--gpus]

The replay counterpart of ``muP_experiment/scripts/run_weight_norms.py``.  For every (N, method, C, seed) the
stream of T = T_burn + T_meas tasks is trained (``--tasks`` sets T in units of N, burn-in = the first half) and
after every task (after its sleep) the following are recorded (``replay.NORM_KEYS``):

    ``Q_all``     hidden-layer norm, mean_i |W_i|²/D
    ``a2_all``    output-layer norm, mean_i a_i²
    ``u2``        effective weight norm |u_t|²/D, u = a W / N
    ``loss_end``  population loss |u_t − w_t|²/(2D) of the task just trained: what replay costs the new task
    ``buf_age``   mean age, in tasks, of the samples in the buffer (0 = the task just trained): what each write
                  rule keeps (FIFO: (C/m − 1)/2; random replace: C/m − 1 once stationary)

"Steady state" is the mean over the second half of the stream, t ≥ T/2.  Two figures in
``results/weight_norms/figures/``:

``norms_vs_task_C{C}_…``  rows = the five quantities, columns = N, traces = replay methods at the FIRST
    capacity listed (``none`` dashed grey); x = task; seed mean with ±1 s.e. band; burn-in shaded.
``norms_vs_N_…``          the same rows; columns = replay size C; x = N (log); y = the steady-state value
    (seed mean ± 1 s.e. bars); ``none`` repeats in every column.

``norms_….npz`` holds every per-run curve, keys ``N{N}_C{C}_{method}_{quantity}`` with value (n_seeds, T).
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

#: Rows of both figures: (key, y-axis label).
ROWS = (("Q_all", r"mean $|W_i|^2/D$"), ("a2_all", r"mean $a_i^2$"), ("u2", r"$|u_t|^2/D$"),
        ("loss_end", "loss of the task\njust trained"), ("buf_age", "mean age of\nbuffered samples"))


def collect(runs: list, T_of: dict[int, int], args) -> dict[tuple, dict[str, np.ndarray]]:
    """``{(N, C label, method): {quantity: (n_seeds, T)}}``, infeasible seeds dropped; one pool of jobs per width."""
    out = {}
    for N in sorted({r.N for r in runs}, reverse=True):
        rN = [r for r in runs if r.N == N]
        cols, _ = C.collect("norms_over_tasks", rN, T_of[N], {"T": T_of[N]},
                            lambda d, j: {q: d[q][j] for q in R.NORM_KEYS}, args, out_per_task=8.0 * len(R.NORM_KEYS))
        for key in sorted({(r.N, r.cap_label, r.method) for r in rN}):
            rs = sorted((r for r in rN if (r.N, r.cap_label, r.method) == key), key=lambda r: r.seed)
            rs = [r for r in rs if np.isfinite(cols[r.label()]["Q_all"]).all()]
            if rs:
                out[key] = {q: np.stack([cols[r.label()][q] for r in rs]) for q in R.NORM_KEYS}
    return out


def steady(Y: np.ndarray) -> np.ndarray:
    """(n_seeds,) mean of each seed's curve over the second half of the stream."""
    return Y[:, Y.shape[1] // 2:].mean(1)


def fig_vs_task(res: dict, Ns: list[int], cap: str, methods: list[str], run0: R.RRun, path: str) -> str:
    """Rows = quantities, columns = N, traces = methods at capacity ``cap``; x = task."""
    P.apply_style()
    fig, axes = P.grid(len(ROWS), len(Ns), sharex="col", sharey="row")
    for c, N in enumerate(Ns):
        for row, (q, lab) in enumerate(ROWS):
            ax = axes[row, c]
            for m in methods:
                d = res.get((N, "0" if m == "none" else cap, m))
                if d is None or not np.isfinite(d[q]).any():
                    continue
                mean, se = C.seed_mean(d[q], 0)
                x = np.arange(len(mean))
                P.band(ax, x, mean - se, mean + se, C.METHOD_COLOR[m])
                C.plot_method(ax, x, mean, m)
                T = len(mean)
            P.shade_burnin(ax, T // 2, T - T // 2)
            if c == 0:
                ax.set_ylabel(lab)
        axes[-1, c].set_xlabel("task")
    P.label_cols_N(axes, Ns)
    P.cell_title(fig, d0=run0.delta0, lam=run0.lam, solver=run0.solver, eta=run0.eta,
                 extra=f"{C.cap_text(cap)}, {C.MODE_LABEL[run0.mode]}")
    P.caption(fig, "after every task: line = seed mean, band = ±1 s.e.; grey: the burn-in half of the stream; buffer "
                   "age in tasks (no replay has no buffer).")
    P.fig_legend(fig, handles=C.method_handles(methods))
    return P.save(fig, path)


def fig_vs_N(res: dict, Ns: list[int], caps: list[str], methods: list[str], run0: R.RRun, path: str) -> str:
    """Rows = quantities, columns = capacity, traces = methods; x = N, y = steady-state value."""
    P.apply_style()
    fig, axes = P.grid(len(ROWS), len(caps), sharex="col", sharey="row", panel=(3.0, 2.0))
    figdata = {}
    for c, cap in enumerate(caps):
        for row, (q, lab) in enumerate(ROWS):
            ax = axes[row, c]
            for m in methods:
                pts = []
                for N in Ns:
                    d = res.get((N, "0" if m == "none" else cap, m))
                    if d is not None and np.isfinite(d[q]).any():
                        v = steady(d[q])
                        pts.append((N, v.mean(), v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0))
                if not pts:
                    continue
                x, mean, se = map(np.array, zip(*pts))
                C.plot_method(ax, x, mean, m, marker="o", ms=3)
                ax.errorbar(x, mean, yerr=se, fmt="none", ecolor=C.METHOD_COLOR[m], elinewidth=0.8, capsize=2, zorder=3)
                figdata[f"{q}_{m}_C{cap}"] = np.stack([x, mean, se])
            P.logaxis(ax, "x")
            P.discrete_ticks(ax, Ns, "x")
            if c == 0:
                ax.set_ylabel(lab)
        axes[0, c].set_title(C.cap_text(cap), fontsize=P.FS_MIN + 1)
        axes[-1, c].set_xlabel("$N$")
    P.cell_title(fig, d0=run0.delta0, lam=run0.lam, solver=run0.solver, eta=run0.eta, extra=C.MODE_LABEL[run0.mode])
    P.caption(fig, "steady state = mean over the second half of the stream; point = seed mean, bars = ±1 s.e.; no "
                   "replay is the same runs in every column.")
    P.fig_legend(fig, handles=C.method_handles(methods, marker="o"))
    return P.save(fig, path, figdata=figdata)


def main(argv: Optional[list[str]] = None) -> int:
    """Command-line entry point (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    C.add_args(ap, "weight_norms", N=[10, 20, 50, 100])
    ap.add_argument("--tasks", type=float, default=None, help="tasks to run, in units of N (default: the config's 14)")
    args = ap.parse_args(argv)
    runs = C.build(args, "weight_norms")
    run0 = next((r for r in runs if r.method != "none"), runs[0])
    caps, methods, Ns = C.cap_labels(runs), C.methods_of(runs), sorted(set(args.N))
    T_of = {r.N: (int(round(args.tasks * r.N)) if args.tasks else r.T_burn + r.T_meas) for r in runs}
    print(f"weight norms: N={Ns} C={caps} methods={methods} δ₀={run0.delta0} λ={run0.lam} {run0.solver} "
          f"{C.MODE_LABEL[run0.mode]} tasks={T_of} seeds={args.seeds}", flush=True)
    res = collect(runs, T_of, args)

    for (N, cap, m), d in sorted(res.items(), key=lambda kv: (kv[0][0], R.METHODS.index(kv[0][2]), kv[0][1])):
        s = {q: float(steady(d[q]).mean()) for q in R.NORM_KEYS}       # buffer age is NaN without a buffer
        print(f"  N={N:4d} C={cap:6s} {C.METHOD_LABEL[m]:16s} steady Q̄ = {s['Q_all']:.3f}  ā² = {s['a2_all']:.3f}  "
              f"|u|²/D = {s['u2']:.3f}  loss = {s['loss_end']:.3f}  buffer age = {s['buf_age']:.1f}  "
              f"({d['Q_all'].shape[0]} seeds)", flush=True)

    fdir = os.path.join(args.out, "figures")
    tag = f"d0{P.fmt(run0.delta0)}_lam{P.fmt(run0.lam)}_{run0.mode}"
    outs = []
    if caps:
        outs.append(fig_vs_task(res, Ns, caps[0], methods, run0, os.path.join(fdir, f"norms_vs_task_C{caps[0]}_{tag}")))
        outs.append(fig_vs_N(res, Ns, caps, methods, run0, os.path.join(fdir, f"norms_vs_N_{tag}")))
    os.makedirs(fdir, exist_ok=True)
    np.savez_compressed(os.path.join(fdir, f"norms_{tag}.npz"),
                        **{f"N{N}_C{cap}_{m}_{q}": d[q] for (N, cap, m), d in res.items() for q in R.NORM_KEYS})
    print("wrote " + "\n      ".join(outs + [os.path.join(fdir, f"norms_{tag}.npz")]))
    return 0


if __name__ == "__main__":
    _rc = main()
    _dt = time.perf_counter() - _T0
    print(f"run_weight_norms: total {_dt:.1f} s ({_dt / 60:.2f} min) from after the imports to the end", flush=True)
    sys.exit(_rc)
