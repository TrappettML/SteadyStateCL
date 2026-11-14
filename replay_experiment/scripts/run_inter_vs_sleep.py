#!/usr/bin/env python
"""Interleaved replay against sleep replay: the steady-state half-life per replay method, as box plots.

    python scripts/run_inter_vs_sleep.py [--N 100] [--capacity 0.5D 4D] [--delta0 0] [--lam 4] [--seeds 12]
                                         [--T T] [--tau τ] [--gpus]

Every replay method is run twice from the same seeds, once with the buffer replayed after each task
(``sleep``) and once within every step of the task (``interleaved``), at the same number of replayed samples
per task (ρ·n·B).  The half-life is that of ``run_stask.py``: train T tasks (default the config's
T_burn + T_meas), follow the teacher of task T for ``--tau`` more (default 200), and take each seed's first
crossing of half its own S_T(0) (``halflife.per_seed``) — a box needs one value per seed, so this is the
per-seed statistic, not the root of the mean curve; that one is drawn on top as a diamond.

One figure per (N, δ₀), ``results/inter_vs_sleep/figures/inter_vs_sleep_N{N}_d0{δ₀}_lam{λ}.png``: one column per
replay size C; x = replay method (no replay, FIFO, random replace, priority replace), two boxes per method —
sleep (filled) and interleaved (hollow) — and one for no replay; y = m½ in tasks.  Box: quartiles and median
over seeds, whiskers to the farthest seed within 1.5 IQR, dots: the seeds; diamond: m½ of the seed-mean curve.
Seeds that never fall to half within ``--tau`` tasks have no value and are counted in the caption.  The values
go to the ``.npz`` as ``seed_{mode}_{method}_C{C}`` (n_seeds,), ``mean_{mode}_{method}_C{C}`` and
``S0_{mode}_{method}_C{C}``.

Placement: every method and seed of one (N, C, mode) is one ``vmap`` batch (``replay.signal_after_tasks``), run
as jobs over the GPUs (``--gpus``), CPU processes (``--workers``) or in this process.
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
    """(N, δ₀, mode, method, capacity label) of a run."""
    return (r.N, r.delta0, r.mode, r.method, r.cap_label)


def boxes(ax, x: float, v: np.ndarray, mean: float, color, filled: bool, width: float) -> None:
    """One box of the finite entries of ``v`` at ``x`` with its seeds as dots and the mean-curve value as a diamond."""
    v = v[np.isfinite(v)]
    if len(v):
        bp = ax.boxplot([v], positions=[x], widths=width, patch_artist=True, showfliers=False, zorder=3,
                        medianprops={"color": "k", "lw": 1.2}, whiskerprops={"color": color, "lw": 1.0},
                        capprops={"color": color, "lw": 1.0}, boxprops={"edgecolor": color, "lw": 1.2})
        bp["boxes"][0].set_facecolor(color if filled else "white")
        bp["boxes"][0].set_alpha(0.55 if filled else 1.0)
        jit = (np.arange(len(v)) / max(len(v) - 1, 1) - 0.5) * 0.5 * width
        ax.plot(x + jit, v, ls="none", marker="o", ms=2.5, mfc=color, mec="none", alpha=0.7, zorder=4)
    if np.isfinite(mean):
        ax.plot([x], [mean], ls="none", marker="D", ms=5, mfc="white", mec="k", mew=1.0, zorder=5)


def figure(cond: dict, N: int, d0: float, caps: list[str], methods: list[str], run0: R.RRun, tau: int, path: str) -> str:
    """Columns = replay size; x = method; two boxes (sleep, interleaved) per buffered method."""
    P.apply_style()
    fig, axes = P.grid(1, len(caps), sharex="all", sharey="row", panel=(3.4, 2.8))
    figdata, n_seeds, n_cens = {}, 0, 0
    w = 0.32
    for ax, cap in zip(axes[0], caps):
        for i, m in enumerate(methods):
            slots = [("none", 0.0)] if m == "none" else [("sleep", -0.2), ("interleaved", 0.2)]
            for mode, dx in slots:
                d = cond.get((N, d0, mode, m, "0" if m == "none" else cap))
                if d is None:
                    continue
                mean = H.stat(H.values(d["S"], "mean"))[0]
                boxes(ax, i + dx, d["seed"], mean, C.METHOD_COLOR[m], mode != "interleaved", w)
                n_seeds = max(n_seeds, len(d["seed"]))
                n_cens += int(np.isnan(d["seed"]).sum())
                figdata[f"seed_{mode}_{m}_C{cap}"] = d["seed"]
                figdata[f"mean_{mode}_{m}_C{cap}"] = mean
                figdata[f"S0_{mode}_{m}_C{cap}"] = np.nanmean(d["S"][0])
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([C.METHOD_LABEL[m].replace(" ", "\n") for m in methods])
        ax.set_xlim(-0.6, len(methods) - 0.4)
        ax.set_title(C.cap_text(cap), fontsize=P.FS_MIN + 1)
        ax.grid(False, axis="x")
    axes[0, 0].set_ylabel(r"half-life $m_{1/2}$ (tasks)")
    for ax in axes[0]:
        ax.set_xlabel("replay method")
    P.cell_title(fig, d0=d0, lam=run0.lam, solver=run0.solver, eta=run0.eta, extra=f"N = {N}, {n_seeds} seeds")
    P.caption(fig, f"each seed's first crossing of half its own S_T(0), the signal of teacher T followed for {tau} "
                   f"tasks at steady state; box: quartiles and median over seeds, whiskers to 1.5 IQR, dots: seeds, "
                   f"diamond: m½ of the seed-mean curve; filled: sleep replay, hollow: interleaved replay, same "
                   f"replayed samples per task; {n_cens} seed values never reach ½ and are left out.")
    handles = [P.Rectangle((0, 0), 1, 1, fc="0.6", ec="0.3", label=C.MODE_LABEL["sleep"]),
               P.Rectangle((0, 0), 1, 1, fc="white", ec="0.3", label=C.MODE_LABEL["interleaved"]),
               P.Line2D([], [], ls="none", marker="D", mfc="white", mec="k", label="m½ of the mean curve")]
    P.fig_legend(fig, handles=handles)
    return P.save(fig, path, figdata=figdata)


def main(argv: Optional[list[str]] = None) -> int:
    """Command-line entry point (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    C.add_args(ap, "inter_vs_sleep", N=[100], delta0_list=True, seeds=12, mode=False)
    ap.add_argument("--T", type=int, default=None, help="tasks before the probed one (default: the config's T_burn + T_meas)")
    ap.add_argument("--tau", type=int, default=200, help="tasks followed after task T")
    args = ap.parse_args(argv)
    runs, seen = [], set()
    for mode in R.MODES:                                            # ``none`` is the same run under both: kept once
        for r in C.build(args, "inter_vs_sleep", mode=mode):
            if r.label() not in seen:
                seen.add(r.label())
                runs.append(r)
    T = C.stream_T(runs, args.T, "--T")
    r0 = runs[0]
    caps, methods = C.cap_labels(runs), C.methods_of(runs)
    print(f"interleaved vs sleep: N={args.N} C={caps} methods={methods} δ₀={args.delta0} λ={r0.lam} {r0.solver} "
          f"n_steps={r0.n_steps} T={T} τ≤{args.tau} seeds={args.seeds}", flush=True)
    if args.seeds < P.MIN_BOX_N:
        print(f"note: {args.seeds} seeds per box; a box summarises a distribution from about {P.MIN_BOX_N} values", flush=True)
    sig, _ = C.collect("signal_after_tasks", runs, T + args.tau + 1, {"T_pre": T, "n_follow": args.tau + 1, "k": T},
                       lambda S, j: S[:, j], args)
    cond = H.by_condition([(r, sig[r.label()]) for r in runs], cond_key, "seed")

    for key in sorted(cond, key=lambda k: (k[0], k[1], k[4], R.METHODS.index(k[3]), k[2])):
        d = cond[key]
        m, se = H.stat(d["seed"])
        print(f"  N={key[0]} δ₀={key[1]} C={key[4]:6s} {C.METHOD_LABEL[key[3]]:16s} {key[2]:11s} per-seed m½ = {m:6.2f} ± "
              f"{se:4.2f}  mean-curve m½ = {H.stat(H.values(d['S'], 'mean'))[0]:6.2f}  S_T(0) = {np.nanmean(d['S'][0]):.3f}"
              f"  never ½: {int(np.isnan(d['seed']).sum())}/{len(d['seed'])}", flush=True)
    for N in sorted(set(args.N)):
        for d0 in sorted(set(args.delta0)):
            run0 = next(r for r in runs if r.N == N and r.delta0 == d0)
            stem = os.path.join(args.out, "figures", f"inter_vs_sleep_N{N}_d0{P.fmt(d0)}_lam{P.fmt(run0.lam)}")
            print(f"wrote {figure(cond, N, d0, caps, methods, run0, args.tau, stem)} and {stem}.npz")
    return 0


if __name__ == "__main__":
    _rc = main()
    _dt = time.perf_counter() - _T0
    print(f"run_inter_vs_sleep: total {_dt:.1f} s ({_dt / 60:.2f} min) from after the imports to the end", flush=True)
    sys.exit(_rc)
