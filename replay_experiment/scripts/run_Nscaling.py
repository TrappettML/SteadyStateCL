#!/usr/bin/env python
"""Steady-state half-life against width, per replay method and replay size: m½ from S_T(m)/S_T(0) vs N.

    python scripts/run_Nscaling.py [--N 10 20 50 100] [--capacity 0.5D 4D] [--delta0 0] [--lam 4] [--seeds 4]
                                   [--mode sleep|interleaved] [--stat mean|seed] [--T T] [--tau τ] [--gpus]

The replay counterpart of ``muP_experiment/scripts/run_Nscaling.py``.  For every (N, method, C, seed) the
network trains T = T_burn + T_meas tasks (14N under ``configs/replay.yaml``), then τ = T more while the teacher
of task T is followed, S_T(m) = u_{T+m}·w_T*/|w_T*|².  ``--T`` / ``--tau`` set both lengths in units of N
(``--T 2 --tau 1`` trains 2N tasks and follows N), so one call covers several widths.

The half-life (``--stat``, ``sscl_core.halflife``): ``mean`` (default) is the first crossing of ½ by the mean
curve Σ(m) = ⟨S_T(m)⟩/⟨S_T(0)⟩ over seeds, with a jackknife s.e.; ``seed`` the mean of the per-seed half-lives.
Two figures in ``results/Nscaling/figures/``:

``mhalf_vs_N_C{C}_…``   x = N (log), y = m½ (tasks, log), one trace per replay method at the FIRST capacity
    listed (±1 s.e. bars); ``none`` dashed grey.
``mhalf_caps_…``        one panel per replay method that has a buffer, traces = capacity in shades of the
    method's colour (lightest = smallest C), ``none`` dashed grey in every panel.

The values go to ``mhalf_….npz``: ``N{N}_C{C}_{method}`` (the statistic drawn, (n_seeds,)), ``seed_…`` (the
per-seed half-lives) and ``S0_…`` (the seed-mean S_T(0), the signal the half-life is relative to).

Placement (``sscl_core.pool``): every method and seed of one (N, C) is one ``vmap`` batch
(``replay.signal_after_tasks``), run as jobs over the GPUs (``--gpus``), CPU processes or in this process.
"""
import argparse
import os
import sys
import time
from typing import Optional

import numpy as np

import _path  # noqa: F401

import common as C
from sscl_core import halflife as H
from sscl_core import plotting as P
from sscl_core import replay as R

_T0 = time.perf_counter()        # wall clock from after the imports to the end of the script


def cond_key(r: R.RRun) -> tuple:
    """(N, capacity label, method) of a run: one point of a trace (``none``: capacity label "0")."""
    return (r.N, r.cap_label, r.method)


def lengths(r: R.RRun, T: Optional[float], tau: Optional[float]) -> tuple[int, int]:
    """(tasks before the probed one, tasks followed) of a run: ``T``·N and ``tau``·N, by default the config's
    T_burn + T_meas for both."""
    T_ = int(round(T * r.N)) if T else r.T_burn + r.T_meas
    return T_, (int(round(tau * r.N)) if tau else T_)


def collect(runs: list, T: Optional[float], tau: Optional[float], stat: str, args) -> dict[tuple, dict]:
    """``{(N, C label, method): {"mhalf", "seed", "S", "seeds"}}`` (``halflife.by_condition``), one pool of jobs
    per width (the widths differ in stream length)."""
    cols = []
    for N in sorted({r.N for r in runs}, reverse=True):
        rN = [r for r in runs if r.N == N]
        T_, tau_ = lengths(rN[0], T, tau)
        print(f"N = {N}: train {T_} tasks, follow task {T_} for {tau_}", flush=True)
        sig, _ = C.collect("signal_after_tasks", rN, T_ + tau_ + 1, {"T_pre": T_, "n_follow": tau_ + 1, "k": T_},
                           lambda S, j: S[:, j], args)
        cols += [(r, sig[r.label()]) for r in rN]
    return H.by_condition(cols, cond_key, stat)


def point(cond: dict, key: tuple) -> Optional[tuple[float, float]]:
    """(m½, s.e.) of a condition, None when it has no finite value."""
    if key not in cond or not np.isfinite(cond[key]["mhalf"]).any():
        return None
    return H.stat(cond[key]["mhalf"])


def trace(ax, cond: dict, Ns: list[int], cap: str, method: str, color, figdata: dict, ls: Optional[str] = None) -> None:
    """One m½-vs-N trace with ±1 s.e. bars."""
    pts = [(N,) + point(cond, (N, cap, method)) for N in Ns if point(cond, (N, cap, method)) is not None]
    if not pts:
        return
    x, m, se = map(np.array, zip(*pts))
    ax.plot(x, m, color=color, ls=ls or C.METHOD_STYLE[method], lw=P.LW_TRACE, marker="o", ms=3, zorder=4)
    ax.errorbar(x, m, yerr=se, fmt="none", ecolor=color, elinewidth=0.8, capsize=2, zorder=3)
    figdata[f"mhalf_{method}_C{cap}"] = np.stack([x, m, se])


def axes_style(ax, Ns: list[int]) -> None:
    P.logaxis(ax, "x"); P.logaxis(ax, "y")
    P.discrete_ticks(ax, Ns, "x")
    ax.set_xlabel("$N$")


def fig_methods(cond: dict, Ns: list[int], cap: str, methods: list[str], run0: R.RRun, path: str, stat: str) -> str:
    """m½ vs N, one trace per method at capacity ``cap``."""
    P.apply_style()
    fig, axes = P.grid(1, 1, panel=(4.0, 2.8))
    ax, figdata = axes[0, 0], {}
    for m in methods:
        trace(ax, cond, Ns, "0" if m == "none" else cap, m, C.METHOD_COLOR[m], figdata)
    axes_style(ax, Ns)
    ax.set_ylabel(r"half-life $m_{1/2}$ (tasks)")
    P.cell_title(fig, d0=run0.delta0, lam=run0.lam, solver=run0.solver, eta=run0.eta,
                 extra=f"{C.cap_text(cap)}, {C.MODE_LABEL[run0.mode]}")
    P.caption(fig, f"{H.STAT_TEXT[stat]}; S_T(m) the signal of the teacher of task T after m more tasks, at steady "
                   f"state; ungated learned shared readout, {C.replay_text(run0)} at N = {run0.N}; no replay has no "
                   f"buffer.")
    P.fig_legend(fig, handles=C.method_handles(methods, marker="o"))
    return P.save(fig, path, figdata=figdata)


def fig_caps(cond: dict, Ns: list[int], caps: list[str], methods: list[str], run0: R.RRun, path: str, stat: str) -> str:
    """One panel per buffered method; traces = capacity (shades); ``none`` dashed grey in every panel."""
    P.apply_style()
    ms = [m for m in methods if m != "none"]
    fig, axes = P.grid(1, len(ms), sharex="all", sharey="all", panel=(3.0, 2.8))
    figdata = {}
    for ax, m in zip(axes[0], ms):
        if "none" in methods:
            trace(ax, cond, Ns, "0", "none", C.METHOD_COLOR["none"], figdata)
        for i, cap in enumerate(caps):
            trace(ax, cond, Ns, cap, m, C.cap_shade(i, len(caps), m), figdata, ls="-")
        ax.set_title(C.METHOD_LABEL[m], color=C.METHOD_COLOR[m], fontsize=P.FS_MIN + 1)
        axes_style(ax, Ns)
    axes[0, 0].set_ylabel(r"half-life $m_{1/2}$ (tasks)")
    P.cell_title(fig, d0=run0.delta0, lam=run0.lam, solver=run0.solver, eta=run0.eta, extra=C.MODE_LABEL[run0.mode])
    P.caption(fig, f"{H.STAT_TEXT[stat]}; replay size C in shades of the method's colour, lightest = smallest; "
                   f"dashed grey: no replay.")
    handles = [P.Line2D([], [], color=C.cap_shade(i, len(caps)), lw=1.6, label=C.cap_text(c)) for i, c in enumerate(caps)]
    P.fig_legend(fig, handles=handles + (C.method_handles(["none"]) if "none" in methods else []))
    return P.save(fig, path, figdata=figdata)


def main(argv: Optional[list[str]] = None) -> int:
    """Command-line entry point (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    C.add_args(ap, "Nscaling", N=[10, 20, 50, 100])
    ap.add_argument("--T", type=float, default=None, help="tasks before the probed one, in units of N (default: the config's 14)")
    ap.add_argument("--tau", type=float, default=None, help="tasks followed, in units of N (default: as many as --T)")
    H.add_arg(ap)
    args = ap.parse_args(argv)
    runs = C.build(args, "Nscaling")
    run0 = next((r for r in runs if r.method != "none"), runs[0])
    caps, methods, Ns = C.cap_labels(runs), C.methods_of(runs), sorted(set(args.N))
    print(f"half-life vs N: N={Ns} C={caps} methods={methods} δ₀={run0.delta0} λ={run0.lam} {run0.solver} "
          f"{C.MODE_LABEL[run0.mode]} seeds={args.seeds} statistic={args.stat}", flush=True)
    cond = collect(runs, args.T, args.tau, args.stat, args)

    for key in sorted(cond, key=lambda k: (k[0], R.METHODS.index(k[2]), k[1])):
        d = cond[key]
        m, se = H.stat(d["mhalf"])
        print(f"  N={key[0]:4d} C={key[1]:6s} {C.METHOD_LABEL[key[2]]:16s} m½ = {m:7.2f} ± {se:5.2f}  per-seed mean "
              f"{H.stat(d['seed'])[0]:7.2f}  S_T(0) = {np.nanmean(d['S'][0]):.3f}  censored or infeasible "
              f"{int(np.isnan(d['mhalf']).sum())}/{len(d['mhalf'])}", flush=True)

    fdir = os.path.join(args.out, "figures")
    tag = f"d0{P.fmt(run0.delta0)}_lam{P.fmt(run0.lam)}_{run0.mode}"
    outs = []
    if caps:
        outs.append(fig_methods(cond, Ns, caps[0], methods, run0, os.path.join(fdir, f"mhalf_vs_N_C{caps[0]}_{tag}"), args.stat))
        outs.append(fig_caps(cond, Ns, caps, methods, run0, os.path.join(fdir, f"mhalf_caps_{tag}"), args.stat))
    data = {}
    for (N, cap, m), d in cond.items():
        data[f"N{N}_C{cap}_{m}"] = d["mhalf"]
        data[f"seed_N{N}_C{cap}_{m}"] = d["seed"]
        data[f"S0_N{N}_C{cap}_{m}"] = np.nanmean(d["S"][0])
    os.makedirs(fdir, exist_ok=True)
    np.savez_compressed(os.path.join(fdir, f"mhalf_{tag}.npz"), stat=np.array(args.stat), **data)
    print("wrote " + "\n      ".join(outs + [os.path.join(fdir, f"mhalf_{tag}.npz")]))
    return 0


if __name__ == "__main__":
    _rc = main()
    _dt = time.perf_counter() - _T0
    print(f"run_Nscaling: total {_dt:.1f} s ({_dt / 60:.2f} min) from after the imports to the end", flush=True)
    sys.exit(_rc)
