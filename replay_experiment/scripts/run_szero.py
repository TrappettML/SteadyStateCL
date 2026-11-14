#!/usr/bin/env python
"""First-teacher signal against task, one trace per replay method: S₀(t) = u_t·w₀*/|w₀*|², one figure per cell.

    python scripts/run_szero.py [--N 100] [--capacity 0.5D 4D] [--delta0 0] [--lam 4] [--seeds 4] [--tasks T]
                                [--mode sleep|interleaved] [--stat mean|seed] [--gpus]

The replay counterpart of ``muP_experiment/scripts/run_szero.py``.  From a fresh initialisation (and an empty
buffer) the ungated learned shared readout trains the task stream, and after every task t (after its sleep) the
signal of the FIRST teacher is read, S₀(t) = u_t·w₀*/|w₀*|² with u = a W / N.  One figure per cell (N, C, δ₀),
``results/szero/figures/szero_N{N}_C{C}_d0{δ₀}_lam{λ}_{mode}.png``: x = t + 1 (log), y = S₀(t)/S₀(0), one trace per
replay method (``none`` dashed grey, the same runs in every capacity's figure), ±1 s.e. band, grey lines at 0
and ½, the half-life m½ per method in the caption (``--stat``, ``sscl_core.halflife``).  The per-task arrays
are written next to the figure as ``.npz``.  No equation is drawn: the write-up has none for replay.

A second figure per cell, ``szero_area_N{N}_…png``: the area under that normalized curve from age 0 to n against
n (``area_curve``; trapezoid rule, in tasks) — the tasks' worth of the memory kept over the first n tasks.  The
half-life only sees where the curve first crosses ½, which here is within one task for every method; the area
also counts the tail a replayed memory keeps.  Its ``.npz`` holds ``area_{method}`` and ``area_se_{method}`` (n,).

Placement: every method and seed of one (N, C) is one ``vmap`` batch (``replay.signal_after_tasks`` with
``T_pre = 0, k = 0``), run as jobs over the GPUs (``--gpus``), CPU processes (``--workers``) or in this process.
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

#: What the trace is, per statistic, for captions (as the μP scripts).
LINE_TEXT = {"mean": "seed mean of the {what} divided by its seed mean {ref}",
             "seed": "seed mean of each seed's {what} divided by its own value {ref}"}
MHALF_TEXT = {"mean": "of the mean curve, ± jackknife s.e.", "seed": "of each seed, mean ± s.e. over seeds"}


def seed_signal(Y: np.ndarray, stat: str = "mean") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalized signal, its s.e. and the half-life values of a (T, n_seeds) signal (``sscl_core.halflife``)."""
    T, n = Y.shape
    if stat == "mean":
        return H.mean_curve(Y), H.mean_curve_se(Y), H.values(Y, "mean")
    with np.errstate(invalid="ignore", divide="ignore"):
        Yn = Y / Y[:1]
    return Yn.mean(1), (Yn.std(1, ddof=1) / np.sqrt(n) if n > 1 else np.zeros(T)), H.per_seed(Y)


def mhalf_text(mh: np.ndarray, T: int) -> str:
    """Mean ± s.e. of half-life values over the finite ones, and how many never reach ½ within ``T`` tasks."""
    fin = mh[np.isfinite(mh)]
    if len(fin) == 0:
        return f"> {T}"
    txt = f"{fin.mean():.2f}" + (f" ± {fin.std(ddof=1) / np.sqrt(len(fin)):.2f}" if len(fin) > 1 else "")
    if len(fin) < len(mh):
        txt += f" ({len(mh) - len(fin)}/{len(mh)} never reach ½)"
    return txt


def signals(runs: list, T_pre: int, n_follow: int, k: int, args) -> dict[tuple, dict[str, np.ndarray]]:
    """``{cell: {method: (n_follow, n_seeds)}}`` of probe ``k`` after each of ``n_follow`` tasks following ``T_pre``."""
    cols, _ = C.collect("signal_after_tasks", runs, T_pre + n_follow, {"T_pre": T_pre, "n_follow": n_follow, "k": k},
                        lambda S, j: S[:, j], args)
    return {key: {m: np.stack(v, 1) for m, v in C.by_method(cell, cols).items()} for key, cell in C.cells(runs).items()}


def figure(S: dict[str, np.ndarray], run0: R.RRun, cap: Optional[str], path: str, stat: str, steady: Optional[int]) -> str:
    """Normalized signal vs age + 1 (log), one trace per method (± 1 s.e.), half-lives in the caption.

    Parameters
    ----------
    S : dict
        ``{method: (n, n_seeds)}``.
    run0 : RRun
        A replay run of the cell (any run when the cell has none).
    cap : str or None
        The cell's capacity label.
    steady : int or None
        None: the first teacher from a fresh start (x = task); T: the teacher of task T (x = age τ).
    """
    P.apply_style()
    fig, axes = P.grid(1, 1, panel=(5.2, 2.6))
    ax = axes[0, 0]
    figdata, mh = {}, {}
    name = "S0" if steady is None else "ST"
    for m, Y in S.items():
        x = np.arange(Y.shape[0]) + 1
        mean, se, mh[m] = seed_signal(Y, stat)
        P.band(ax, x, mean - se, mean + se, C.METHOD_COLOR[m])
        C.plot_method(ax, x, mean, m)
        figdata[f"{name}_{m}"] = Y
        figdata[f"{name}norm_{m}"] = mean
        figdata[f"mhalf_{m}"] = mh[m]
        figdata[f"mhalf_seed_{m}"] = H.per_seed(Y)
    n = max(Y.shape[0] for Y in S.values())
    n_seeds = max(Y.shape[1] for Y in S.values())
    figdata["t" if steady is None else "tau"] = np.arange(n)
    P.ref_line(ax, 0.0)
    P.ref_line(ax, 0.5)
    P.logaxis(ax, "x")
    if steady is None:
        ax.set_xlabel("task $t + 1$")
        ax.set_ylabel(r"$S_0(t)\,/\,S_0(0)$,  $S_0(t) = u_t\cdot w_0^*/|w_0^*|^2$")
        what, ref = "signal of the first teacher after task t", "at t = 0"
    else:
        ax.set_xlabel(r"$\tau + 1$ (tasks after task $T$)")
        ax.set_ylabel(r"$S_T(\tau)\,/\,S_T(0)$,  $S_T(\tau) = u_{T+\tau}\cdot w_T^*/|w_T^*|^2$")
        what, ref = f"signal of teacher T = {steady} after task T + τ", "at τ = 0"
    extra = f"N = {run0.N}" + (f", {C.cap_text(cap)}, {C.MODE_LABEL[run0.mode]}" if cap else "")
    P.cell_title(fig, d0=run0.delta0, lam=run0.lam, solver=run0.solver, eta=run0.eta, extra=extra)
    mtxt = ", ".join(f"{C.METHOD_LABEL[m]} {mhalf_text(mh[m], n - 1)}" for m in S)
    P.caption(fig, f"line: {LINE_TEXT[stat].format(what=what, ref=ref)}, drawn at +1 so age 0 is at 1 ({n_seeds} "
                   f"seeds); band: ±1 s.e.; ungated learned shared readout"
                   + (f", {C.replay_text(run0)}" if run0.method != "none" else "")
                   + f"; half-life m½ (tasks), first crossing of ½ (grey line), {MHALF_TEXT[stat]}: {mtxt}.")
    P.fig_legend(fig, handles=C.method_handles(list(S)))
    return P.save(fig, path, figdata=figdata)


def area_curve(Y: np.ndarray, stat: str = "mean") -> tuple[np.ndarray, np.ndarray]:
    """Area under the normalized signal up to age n, for n = 0 … T − 1, and its s.e. over seeds: (T,), (T,).

    A(n) = ∫₀ⁿ S(m)/S(0) dm by the trapezoid rule on the integer ages, in tasks: the number of tasks' worth of
    the memory retained over the first n tasks (A(n) = n if nothing is forgotten; A(∞) is the mean lifetime of
    an exponential decay, m½/ln 2).  ``stat = "mean"``: the area under the mean curve Σ = ⟨S⟩/⟨S(0)⟩, s.e. by
    jackknife over seeds; ``"seed"``: the seed mean and s.e. of the area under each seed's own S/S(0).
    """
    cum = lambda c: np.concatenate([[0.0], np.cumsum(0.5 * (c[1:] + c[:-1]), 0)])
    T, n = Y.shape
    if stat == "mean":
        A = cum(H.mean_curve(Y))
        if n < 2:
            return A, np.zeros(T)
        tot = Y.sum(1, keepdims=True)
        loo = (tot - Y) / (n - 1)                                   # (T, n): the mean curve without seed j
        Aj = np.stack([cum(loo[:, j] / loo[0, j]) for j in range(n)], 1)
        return A, np.sqrt((n - 1) / n * ((Aj - Aj.mean(1, keepdims=True)) ** 2).sum(1))
    with np.errstate(invalid="ignore", divide="ignore"):
        Aj = np.stack([cum(Y[:, j] / Y[0, j]) for j in range(n)], 1)
    return Aj.mean(1), (Aj.std(1, ddof=1) / np.sqrt(n) if n > 1 else np.zeros(T))


def figure_area(S: dict[str, np.ndarray], run0: R.RRun, cap: Optional[str], path: str, stat: str, steady: Optional[int]) -> str:
    """Area under the normalized signal up to n tasks against n (log x), one trace per method (± 1 s.e.);
    arguments as ``figure``."""
    P.apply_style()
    fig, axes = P.grid(1, 1, panel=(5.2, 2.6))
    ax = axes[0, 0]
    figdata, last = {}, {}
    for m, Y in S.items():
        A, se = area_curve(Y, stat)
        x = np.arange(1, len(A))                                    # A(0) = 0 is left off the log axis
        P.band(ax, x, (A - se)[1:], (A + se)[1:], C.METHOD_COLOR[m])
        C.plot_method(ax, x, A[1:], m)
        figdata[f"area_{m}"], figdata[f"area_se_{m}"] = A, se
        last[m] = (A[-1], se[-1])
    n = max(Y.shape[0] for Y in S.values())
    n_seeds = max(Y.shape[1] for Y in S.values())
    figdata["n"] = np.arange(n)
    P.ref_line(ax, 0.0)
    P.logaxis(ax, "x")
    sym = "S_0" if steady is None else "S_T"
    ax.set_xlabel("$n$ (tasks after the first)" if steady is None else "$n$ (tasks after task $T$)")
    ax.set_ylabel(rf"area under ${sym}(m)/{sym}(0)$, $m \leq n$ (tasks)")
    extra = f"N = {run0.N}" + (f", {C.cap_text(cap)}, {C.MODE_LABEL[run0.mode]}" if cap else "")
    P.cell_title(fig, d0=run0.delta0, lam=run0.lam, solver=run0.solver, eta=run0.eta, extra=extra)
    what = ("the mean curve (seed mean of the signal over its seed mean at age 0), band: jackknife ±1 s.e."
            if stat == "mean" else "each seed's own signal over its value at age 0, seed mean, band: ±1 s.e.")
    atxt = ", ".join(f"{C.METHOD_LABEL[m]} {a:.2f} ± {e:.2f}" for m, (a, e) in last.items())
    P.caption(fig, f"area under {what} ({n_seeds} seeds), from age 0 to n by the trapezoid rule: the tasks' worth of "
                   f"the memory kept over the first n tasks (n if nothing were forgotten; ½ if it were gone after "
                   f"one task); at n = {n - 1}: {atxt}.")
    P.fig_legend(fig, handles=C.method_handles(list(S)))
    return P.save(fig, path, figdata=figdata)


def report_and_draw(res: dict, runs: list, out: str, stem: str, stat: str, steady: Optional[int]) -> list[str]:
    """Print each cell's summary and write its figure; returns the PNG paths."""
    cells_, outs = C.cells(runs), []
    for (N, cap, d0, lam), S in res.items():
        print(f"cell N={N} C={cap} δ₀={d0} λ={lam}:")
        for m, Y in S.items():
            mean = Y.mean(1)
            A, A_se = area_curve(Y, stat)
            print(f"  {C.METHOD_LABEL[m]:16s} S(0) = {mean[0]:.4f}  m½ = {mhalf_text(seed_signal(Y, stat)[2], Y.shape[0] - 1)}"
                  f"  S(1)/S(0) = {mean[min(1, len(mean) - 1)] / mean[0]:+.3f}  S(end)/S(0) = {mean[-1] / mean[0]:+.3f}"
                  f"  area to n = {len(A) - 1}: {A[-1]:.2f} ± {A_se[-1]:.2f}", flush=True)
        if not S:
            continue
        cell = cells_[(N, cap, d0, lam)]
        run0 = next((r for r in cell if r.method != "none"), cell[0])
        path = os.path.join(out, "figures", f"{stem}_N{N}_C{cap or 0}_d0{P.fmt(d0)}_lam{P.fmt(lam)}_{run0.mode}")
        outs.append(figure(S, run0, cap, path, stat, steady))
        print(f"wrote {outs[-1]} and {path}.npz")
        apath = path.replace(f"{stem}_N", f"{stem}_area_N")
        outs.append(figure_area(S, run0, cap, apath, stat, steady))
        print(f"wrote {outs[-1]} and {apath}.npz")
    return outs


def main(argv: Optional[list[str]] = None) -> int:
    """Command-line entry point (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    C.add_args(ap, "szero", N=[100], delta0_list=True)
    ap.add_argument("--tasks", type=int, default=None, help="tasks to run (default: the config's T_burn + T_meas)")
    H.add_arg(ap)
    args = ap.parse_args(argv)
    runs = C.build(args, "szero")
    T = C.stream_T(runs, args.tasks, "--tasks")
    r0 = runs[0]
    print(f"S₀(t) vs task: N={args.N} C={C.cap_labels(runs)} methods={C.methods_of(runs)} δ₀={args.delta0} λ={r0.lam} "
          f"{r0.solver} n_steps={r0.n_steps} T={T} seeds={args.seeds} statistic={args.stat}", flush=True)
    res = signals(runs, 0, T, 0, args)
    report_and_draw(res, runs, args.out, "szero", args.stat, None)
    return 0


if __name__ == "__main__":
    _rc = main()
    _dt = time.perf_counter() - _T0
    print(f"run_szero: total {_dt:.1f} s ({_dt / 60:.2f} min) from after the imports to the end", flush=True)
    sys.exit(_rc)
