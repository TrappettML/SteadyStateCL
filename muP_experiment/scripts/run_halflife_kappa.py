#!/usr/bin/env python
"""Signal half-life against the gate scale κ: m½ of the first memory and of a steady-state memory, one figure.

    python scripts/run_halflife_kappa.py [--N 1000] [--kappa 1 2 4 … sqrtN] [--delta0 0] [--lam 4] [--seeds 4] [--T T] [--tau τ] [--gpus]

The companion of ``run_halflife_lam.py`` with κ on the x axis.  One figure for the cell (N, δ₀, λ),
``results/halflife_kappa/figures/mhalf_vs_kappa_N{N}_d0{δ₀}_lam{λ}.png``, two rows that share both axes,
x = κ (log) from the smallest gate scale the model allows, κ = 1 (a task then uses √N units on average;
``config.kappa_ok``), to κ = √N (gate density 1: every unit active), y = half-life m½ in tasks (log), one trace
per readout:

row 1, first memory      from a fresh initialisation the signal of the FIRST teacher is read after every task,
    S₀(t) = u_t·w₀*/|w₀*|², for t = 0 … T − 1; m½ is the first t with S₀(t) ≤ S₀(0)/2.
row 2, steady state      the same network trains on; the teacher of task T is followed for τ = 0 … ``--tau``
    more tasks, S_T(τ) = u_{T+τ}·w_T*/|w_T*|²; m½ is the first τ with S_T(τ) ≤ S_T(0)/2.

The gated readouts are traces over κ.  The ungated ones (learned shared, frozen) have no κ — every unit is
active, the κ = √N point of their gated counterpart — so each is one value, drawn as a dashed horizontal line
with a ±1 s.e. band.  Half-lives are linearly interpolated and computed per seed; a point is the mean ± 1 s.e.
over the seeds that reach half, and a seed that never does within its window (T tasks, row 1; τ tasks, row 2)
is censored: left out and counted in the caption and on the console.  Dotted grey: N ln 2.

N = 1000, δ₀ = 0 by default and λ the config's (4); the default κ grid is the powers of two below √N and
``sqrtN`` (1, 2, 4, 8, 16, √N at N = 1000).  T defaults to the Stage-1 stream T_burn + T_meas (14 N tasks) and
``--tau`` to T; at N = 1000 that is 28 001 tasks of 40 steps per run, so use ``--T`` / ``--tau`` to shorten it.
The per-seed half-lives are written next to the figure as ``mhalf_vs_kappa_….npz``.

Placement (``mup.pool``): a gated batch steps on the block of its active rows, about κ√N of the N, so every κ
is its own ``vmap`` batch (``experiments.group_key``), all seeds together, and the cost grows with κ.  Each
batch is ONE stream of T + τ + 1 tasks (``experiments.first_and_steady_signal``).  The batches run as jobs,
costliest first, over the GPUs (``--gpus``: every card, several processes each, a batch split over cards as far
as its share of the cost asks — ``pool.plan``), over ``--workers`` CPU processes, or in this process.
"""
import argparse
import math
import os
import sys
import time
from typing import Callable, Optional

import numpy as np

import _path  # noqa: F401

from mup import experiments as X
from mup import plotting as P
from mup import pool
from mup.config import Config, READOUT_ORDER, Run, is_gated, make_runs
from run_Nscaling import _stat, mhalf_from_signal
from run_halflife_lam import ROWS, cost, make_jobs

_T0 = time.perf_counter()        # wall clock from after the imports to the end of the script


def default_kappas(N: int) -> list[str]:
    """The κ grid when ``--kappa`` is not given: 1, 2, 4, … below √N, then ``sqrtN``."""
    out, k = [], 1
    while k < math.sqrt(N) * (1 - 1e-9):
        out.append(str(k))
        k *= 2
    return out + ["sqrtN"]


def groups(runs: list[Run], n_tasks: int) -> dict[tuple, list[Run]]:
    """``{experiments.group_key: runs}`` (one group per κ, the ungated and the closed-form readouts apart),
    costliest job first (``run_halflife_lam.cost`` over ``n_tasks`` tasks)."""
    g = {}
    for r in runs:
        g.setdefault(X.group_key(r), []).append(r)
    return {k: g[k] for k in sorted(g, key=lambda k: -cost(g[k], n_tasks))}


def cond_key(r: Run) -> tuple:
    """(readout, κ label) of a run: one point of a trace (the ungated readouts have the one label ``sqrtN``)."""
    return (r.readout, r.kappa_label)


def _report(key: tuple, d: dict, T: int, tau: int) -> None:
    """Print one condition's line: m½ (seed mean ± s.e.) and the censored count of both rows."""
    txt = []
    for (name, label, _), window in zip(ROWS, (T, tau)):
        mh = d[name]
        m, se = _stat(mh)
        txt.append(f"{label} m½ = {m:8.2f} ± {se:6.2f} (censored or infeasible {int(np.isnan(mh).sum())}/{len(mh)}, "
                   f"window {window})")
    kappa = key[1] if is_gated(key[0]) else "none"
    print(f"  κ={kappa:6s} {key[0]:15s} " + "   ".join(txt), flush=True)


def collect(runs: list[Run], T: int, tau: int, plan: pool.Plan,
            log: Callable[[str], None] = print) -> dict[tuple, dict[str, np.ndarray]]:
    """``{(readout, κ label): {"S0": (n_seeds,) m½ of the first memory, "ST": (n_seeds,) m½ at steady state}}``.

    Parameters
    ----------
    runs : list of config.Run
        One cell (N, δ₀, λ) at every κ.
    T : int
        Tasks the first memory is followed for, and trained before the steady-state probe.
    tau : int
        Tasks the steady-state probe is followed for after task T.
    plan : pool.Plan
        Placement and pieces per group (``pool.plan`` over ``groups(runs, T + tau + 1)``).
    log : callable
        Progress sink.

    Returns
    -------
    dict
        In (readout, κ) order; NaN for a censored or infeasible seed.
    """
    jobs = make_jobs(groups(runs, T + tau + 1), T, tau, plan.pieces)
    res = pool.run_jobs("mup.experiments:first_and_steady_signal", jobs, log=log, **plan.place)
    per_run, kappa_of = {}, {}
    for (_, kw, _), d in zip(jobs, res):
        mh = {name: mhalf_from_signal(d[name]) for name, _, _ in ROWS}
        for j, r in enumerate(kw["runs"]):
            per_run[(cond_key(r), r.seed)] = {name: mh[name][j] for name in mh}
            kappa_of[cond_key(r)] = r.kappa
    out = {}
    for key in sorted(kappa_of, key=lambda k: (READOUT_ORDER.index(k[0]), kappa_of[k])):
        seeds = sorted(s for k, s in per_run if k == key)
        out[key] = {name: np.array([per_run[(key, s)][name] for s in seeds]) for name, _, _ in ROWS}
        _report(key, out[key], T, tau)
    return out


def figure(res: dict[tuple, dict[str, np.ndarray]], kappas: list[tuple[float, str]], run0: Run, T: int, tau: int,
           path: str) -> str:
    """m½ vs κ: row 1 the first memory, row 2 a steady-state memory; one trace per readout (seed mean ± 1 s.e.).

    Parameters
    ----------
    res : dict
        ``collect`` output.
    kappas : list of (κ, label)
        The gated κ grid in increasing order (the x axis).
    run0 : config.Run
        Any run of the cell.
    T, tau : int
        The two follow windows, printed in the caption.
    path : str
        Figure stem.

    Returns
    -------
    str
        PNG path.
    """
    P.apply_style()
    fig, axes = P.grid(len(ROWS), 1, sharex="all", sharey="all", panel=(5.2, 2.3))
    kv = np.array([k for k, _ in kappas], float)
    figdata = {"kappa": kv}
    readouts = [ro for ro in READOUT_ORDER if any(k[0] == ro for k in res)]
    xlim = (kv.min() / 1.12, kv.max() * 1.12) if len(kv) else (0.9, math.sqrt(run0.N) * 1.12)
    n_cens = []
    for row, (name, label, tag) in enumerate(ROWS):
        ax = axes[row, 0]
        cens = 0
        for ro in readouts:
            col = P.READOUT_COLOR[ro]
            if not is_gated(ro):                                        # no κ: one value, a horizontal line
                mh = res[(ro, "sqrtN")][name]
                figdata[f"mhalf_{tag}_{ro}"] = mh
                cens += int(np.isnan(mh).sum())
                m, se = _stat(mh)
                if np.isfinite(m) and m > 0:
                    P.band(ax, np.array(xlim), np.full(2, m - se), np.full(2, m + se), col)
                    P.plot_readout(ax, np.array(xlim), np.full(2, m), ro)
                    figdata[f"{tag}_{ro}"] = np.array([m, se])
                continue
            pts = []
            for k, klabel in kappas:
                if (ro, klabel) not in res:
                    continue
                mh = res[(ro, klabel)][name]
                figdata[f"mhalf_{tag}_{ro}_k{klabel}"] = mh
                cens += int(np.isnan(mh).sum())
                m, se = _stat(mh)
                if np.isfinite(m) and m > 0:                            # log axis: a point needs a positive mean
                    pts.append((k, m, se))
            if not pts:
                continue
            x, m, se = map(np.array, zip(*pts))
            P.plot_readout(ax, x, m, ro, marker="o", ms=3)
            ax.errorbar(x, m, yerr=se, fmt="none", ecolor=col, elinewidth=0.8, capsize=2, zorder=3)
            figdata[f"{tag}_{ro}"] = np.stack([x, m, se])
        n_cens.append(cens)
        ax.axhline(run0.N * math.log(2), color=P.GREY_REF, ls=":", lw=P.LW_REF, zorder=1)
        P.logaxis(ax, "x"); P.logaxis(ax, "y")
        ax.set_ylabel(f"{label}\n" + r"half-life $m_{1/2}$ (tasks)")
    ax = axes[-1, 0]
    ax.set_xlim(*xlim)
    if 0 < len(kv) <= 8:                                                # ticks at the κ run, the last one named √N
        P.discrete_ticks(ax, kv, "x")
        ax.set_xticklabels([r"$\sqrt{N}$" if lab == "sqrtN" else P.fmt(lab) for _, lab in kappas])
    ax.set_xlabel(r"gate scale $\kappa$  (gate density $\kappa/\sqrt{N}$)")
    P.cell_title(fig, d0=run0.delta0, lam=run0.lam, solver=run0.solver, eta=run0.eta, extra=f"N = {run0.N}")
    n_seeds = max(len(d[ROWS[0][0]]) for d in res.values())
    P.caption(fig, f"m½: each seed's first crossing of half its initial signal, mean ± 1 s.e. over the seeds that "
                   f"reach it ({n_seeds} seeds per point). Top: first teacher from a fresh start, $S_0(t)/S_0(0)$ ≤ ½ "
                   f"within T = {T} tasks ({n_cens[0]} censored seeds dropped). Bottom: the teacher of task T, "
                   rf"$S_T(\tau)/S_T(0)$ ≤ ½ within τ ≤ {tau} more tasks ({n_cens[1]} dropped). Dashed horizontal "
                   f"lines: the ungated readouts, which have no κ (every unit active, as at κ = √N). Dotted grey: N ln 2.")
    P.fig_legend(fig, readouts, handles=[P.Line2D([], [], color=P.GREY_REF, ls=":", lw=1.2, label="N ln 2")])
    return P.save(fig, path, figdata=figdata)


def main(argv: Optional[list[str]] = None) -> int:
    """Command-line entry point (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(_path.ROOT, "configs", "stage1.yaml"),
                    help="config supplying the solver, λ, step-size and stream-length rules (default: Stage 1)")
    ap.add_argument("--N", type=int, default=1000)
    ap.add_argument("--kappa", nargs="+", default=None,
                    help="the κ grid of the gated readouts, the x axis: numbers in [1, √N] or 'sqrtN' "
                         "(default: 1, 2, 4, … below √N, then sqrtN)")
    ap.add_argument("--delta0", type=float, default=0.0)
    ap.add_argument("--lam", type=float, default=None, help="default: the config's first λ")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--T", type=int, default=None,
                    help="tasks the first memory is followed for and trained before the steady-state probe "
                         "(default: the config's T_burn + T_meas)")
    ap.add_argument("--tau", type=int, default=None, help="tasks the steady-state probe is followed for (default: T)")
    ap.add_argument("--readout", nargs="*", default=list(READOUT_ORDER))
    ap.add_argument("--out", default=os.path.join(_path.ROOT, "results", "halflife_kappa"))
    pool.add_args(ap)
    args = ap.parse_args(argv)

    cfg = Config.from_yaml(args.config)
    over = {"name": "halflife_kappa", "N": [args.N], "kappa": list(args.kappa or default_kappas(args.N)),
            "readout": list(args.readout), "delta0": [args.delta0],
            "lam": [args.lam if args.lam is not None else cfg.lam[0]], "seeds": args.seeds, "extra": (), "trace": False}
    cfg = Config(**{**cfg.__dict__, **over})
    runs = make_runs(cfg)                             # κ outside [1, √N] is skipped with a note
    if not runs:
        print("no runs left after skipping κ outside [1, √N]; nothing to do", flush=True)
        return 0
    kappas = sorted({(r.kappa, r.kappa_label) for r in runs if r.gated == 1.0})
    T = args.T or (runs[0].T_burn + runs[0].T_meas)
    tau = args.tau or T
    plan = pool.plan(args, [(g, T + tau + 1) for g in groups(runs, T + tau + 1).values()])
    r0 = runs[0]
    print(f"m½ vs κ: N={args.N} κ={[lab for _, lab in kappas]} δ₀={r0.delta0} λ={r0.lam} {r0.solver} "
          f"n_steps={r0.n_steps} T={T} τ≤{tau} seeds={args.seeds}  [{plan.describe()}]", flush=True)
    res = collect(runs, T, tau, plan)

    stem = os.path.join(args.out, "figures", f"mhalf_vs_kappa_N{args.N}_d0{P.fmt(r0.delta0)}_lam{P.fmt(r0.lam)}")
    out = figure(res, kappas, r0, T, tau, stem)
    print(f"wrote {out} and {stem}.npz")
    return 0


if __name__ == "__main__":
    _rc = main()
    _dt = time.perf_counter() - _T0
    print(f"run_halflife_kappa: total {_dt:.1f} s ({_dt / 60:.2f} min) from after the imports to the end", flush=True)
    sys.exit(_rc)
