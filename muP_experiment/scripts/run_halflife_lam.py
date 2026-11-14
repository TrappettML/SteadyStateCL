#!/usr/bin/env python
"""Signal half-life against the hidden-layer learning per task λ: m½ of the first memory and of a steady-state memory, one figure.

    python scripts/run_halflife_lam.py [--N 100] [--kappa 1] [--delta0 0] [--lam 0.5 1 2 … 20] [--seeds 4] [--T T] [--tau τ]
                                       [--stat mean|seed] [--clock realized|nominal] [--gpus]

λ is the exponent of the hidden channel (the frozen-readout exponent, eq. lam): a task removes θ = 1 − e^{−λ} of
its residual through the hidden layer.  The readout's own exponent λᵃ_c = Λ‖M_c‖ is not an axis; it follows
from (λ, δ₀) and the state (write-up, "Readout exponent").

One figure for the cell (N, κ, δ₀), ``results/halflife_lam/figures/mhalf_vs_lam_N{N}_k{κ}_d0{δ₀}.png``, two rows
that share both axes, x = λ on a linear axis from 0 to the largest λ (20 by default), y = half-life m½ in
tasks (log), one trace per readout (the gated ones at κ, the ungated ones have no κ and are drawn dashed):

row 1, first memory      from a fresh initialisation the signal of the FIRST teacher is read after every task,
    S₀(t) = u_t·w₀*/|w₀*|² (``run_szero.py``), for t = 0 … T − 1; m½ is the first t with S₀(t) ≤ S₀(0)/2.
row 2, steady state      the same network trains on; the teacher of task T is followed for τ = 0 … ``--tau``
    more tasks, S_T(τ) = u_{T+τ}·w_T*/|w_T*|² (``run_stask.py``, ``run_Nscaling.py``); m½ is the first τ with
    S_T(τ) ≤ S_T(0)/2.

Both half-lives are linearly interpolated.  ``--stat mean`` (default) is the write-up's m½, the first crossing
of ½ by the mean curve over seeds (eq. Sigma) with a jackknife s.e.; ``--stat seed`` is the mean ± 1 s.e. of
the per-seed half-lives over the seeds that reach half (``sscl_core.halflife``).  A curve that never reaches half
within its window (T tasks, row 1; τ tasks, row 2) is censored: left out and counted in the caption and on
the console.  Dotted, in the readout's colour: the half-life of the readout's equation at each λ
(``theory.mhalf``), which scales as 1/θ — eq. (S_heads), eq. (S_frozen) and, in the steady-state row,
eq. (S_learned) with the key drift of eqs. (delta_est), (abarss); the first-memory row draws only the two that
hold from a fresh start (heads, frozen).  Dotted grey: N ln 2, their common scale at θ = 1.

N = 100, κ = 1, δ₀ = 0 by default; the solver and the step-size rule come from ``configs/stage1.yaml``, so a
task takes n_steps = 10 λ population steps at δ₀ = 0 (h ā² ≤ 0.1) and the λ = 4 points are the cell of
``run_szero.py`` / ``run_stask.py``.  T defaults to the Stage-1 stream T_burn + T_meas (14 N tasks) and
``--tau`` to T.  λ = 0 takes no step (nothing is learned, so there is no signal to halve) and is skipped; the
axis still starts there.  The half-lives (the statistic drawn, and the per-seed ones under ``…_seed``) are
written next to the figure as ``mhalf_vs_lam_….npz``.

Placement (``sscl_core.pool``): n_steps is static in the kernels, so every λ is its own set of ``vmap`` batches —
per λ one batch per ``experiments.group_key`` (the stepped gated readouts, the closed-form frozen ones, the
ungated ones), every seed together.  Each batch is ONE stream of T + τ + 1 tasks
(``experiments.first_and_steady_signal``): its first T tasks are the life of the first memory and the burn-in
of the steady-state one.  The batches run as jobs, costliest (largest λ) first, over the GPUs (``--gpus``: the
four cards, several processes each — ``pool.plan``), over ``--workers`` CPU processes, or in this process.
"""
import argparse
import math
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
from run_Nscaling import _stat

_T0 = time.perf_counter()        # wall clock from after the imports to the end of the script

#: λ grid of the figure when ``--lam`` is not given: 0 < λ ≤ 20, denser where a task is only partly learned.
LAM_DEFAULT = (0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0)

#: The two rows: (key of ``first_and_steady_signal``, row label, npz tag).
ROWS = (("S0", "first memory", "first"), ("ST", "steady state", "steady"))


def cost(runs: Sequence[Run], n_tasks: int) -> float:
    """Device time of a job, for ``pool.run_jobs`` to balance the cards by: ``experiments.job_cost``, which grows
    with n_steps = 10 λ, except for a closed-form batch (the frozen readouts under the population solver), whose
    task is one update whatever λ is.  Counting those at the stepped rate would fill some cards with jobs that
    finish in seconds and stack the long large-λ jobs on the others."""
    c = X.job_cost(runs, n_tasks)
    return c / runs[0].n_steps if runs[0].closed else c


def groups(runs: list[Run], n_tasks: int) -> dict[tuple, list[Run]]:
    """``{(λ,) + group_key: runs}``, costliest job first (``cost`` over ``n_tasks`` tasks)."""
    g = {}
    for r in runs:
        g.setdefault((r.lam,) + X.group_key(r), []).append(r)
    return {k: g[k] for k in sorted(g, key=lambda k: -cost(g[k], n_tasks))}


def make_jobs(groups_: dict[tuple, list[Run]], T: int, tau: int, pieces: Sequence[int]) -> list[tuple]:
    """One job per piece of a group: ``(key, kwargs of experiments.first_and_steady_signal, cost)``."""
    jobs = []
    for (key, runs), k in zip(groups_.items(), pieces):
        for i, part in enumerate(X.split(runs, k)):
            jobs.append(((key, i), {"runs": part, "T": T, "n_follow": tau + 1}, cost(part, T + tau + 1)))
    return jobs


def cond_key(r: Run) -> tuple:
    """(readout, λ) of a run: one point of a trace."""
    return (r.readout, r.lam)


def reduce_rows(jobs: list[tuple], res: list[dict], key: Callable, stat: str) -> dict[tuple, dict[str, np.ndarray]]:
    """Half-lives of both rows by condition, from the jobs' ``first_and_steady_signal`` results.

    Returns ``{key(run): {"S0", "ST": (n_seeds,) values of ``stat`` (``halflife.values``), "S0_seed", "ST_seed":
    (n_seeds,) per-seed half-lives}}``; a condition's seeds may be spread over several jobs.
    """
    out: dict = {}
    for name, _, _ in ROWS:
        cols = [(r, d[name][:, j]) for (_, kw, _), d in zip(jobs, res) for j, r in enumerate(kw["runs"])]
        for k, c in H.by_condition(cols, key, stat).items():
            out.setdefault(k, {})[name] = c["mhalf"]
            out[k][name + "_seed"] = c["seed"]
    return out


def theory_mhalf(readout: str, row: str, N: int, kappa: float, lam: float, delta0: float, sigma0: float = 1.0) -> float:
    """Half-life the write-up's equations give a readout in a row (``theory.mhalf``), NaN where they give none.

    Steady state (``ST``): eq. (S_heads), (S_frozen) or (S_learned).  First memory (``S0``): the heads and
    frozen curves, which hold from a fresh start as at steady state; the learned shared readout has no
    fresh-start curve (its key drift and leak-removal factor depend on norms that are still growing).
    """
    if row == "S0" and TH.READOUT_KIND[readout][0] == "learned":
        return float("nan")
    return TH.mhalf(readout, N, kappa, lam, delta0, sigma0)


def _report(key: tuple, d: dict, T: int, tau: int) -> None:
    """Print one condition's line: m½ ± s.e. of the chosen statistic and the censored count of both rows."""
    txt = []
    for (name, label, _), window in zip(ROWS, (T, tau)):
        mh = d[name]
        m, se = _stat(mh)
        txt.append(f"{label} m½ = {m:8.2f} ± {se:6.2f} (censored or infeasible {int(np.isnan(mh).sum())}/{len(mh)}, "
                   f"window {window})")
    print(f"  λ={key[1]:5g} {key[0]:15s} " + "   ".join(txt), flush=True)


def collect(runs: list[Run], T: int, tau: int, plan: pool.Plan, stat: str = "mean",
            log: Callable[[str], None] = print) -> dict[tuple, dict[str, np.ndarray]]:
    """``{(readout, λ): {"S0": (n_seeds,) m½ values of the first memory, "ST": those at steady state, "S0_seed",
    "ST_seed": the per-seed half-lives}}`` (``reduce_rows``; ``stat`` as ``sscl_core.halflife``).

    Parameters
    ----------
    runs : list of config.Run
        One cell (N, κ, δ₀) at every λ.
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
        In (readout, λ) order; NaN for a censored or infeasible seed.
    """
    jobs = make_jobs(groups(runs, T + tau + 1), T, tau, plan.pieces)
    res = pool.run_jobs("sscl_core.experiments:first_and_steady_signal", jobs, log=log, **plan.place)
    red = reduce_rows(jobs, res, cond_key, stat)
    out = {}
    for key in sorted(red, key=lambda k: (READOUT_ORDER.index(k[0]), k[1])):
        out[key] = red[key]
        _report(key, out[key], T, tau)
    return out


def figure(res: dict[tuple, dict[str, np.ndarray]], lams: list[float], run0: Run, T: int, tau: int, path: str,
           stat: str = "mean", kappa_of: Optional[dict] = None) -> str:
    """m½ vs λ: row 1 the first memory, row 2 a steady-state memory; one trace per readout (± 1 s.e.), dotted
    the half-life of its equation (``theory_mhalf``; ``kappa_of[readout]`` its κ).

    Parameters
    ----------
    res : dict
        ``collect`` output.
    lams : list of float
        The λ grid (sets the x range, 0 … max λ).
    run0 : config.Run
        A gated run of the cell when there is one (its κ goes in the title), else any run.
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
    figdata = {"lam": np.array(lams, float)}
    readouts = [ro for ro in READOUT_ORDER if any(k[0] == ro for k in res)]
    n_cens = []
    for row, (name, label, tag) in enumerate(ROWS):
        ax = axes[row, 0]
        cens = 0
        for ro in readouts:
            pts = []
            for lam in lams:
                if (ro, lam) not in res:
                    continue
                mh = res[(ro, lam)][name]
                figdata[f"mhalf_{tag}_{ro}_lam{P.fmt(lam)}"] = mh
                figdata[f"mhalf_{tag}_{ro}_lam{P.fmt(lam)}_seed"] = res[(ro, lam)][name + "_seed"]
                cens += int(np.isnan(mh).sum())
                m, se = _stat(mh)
                if np.isfinite(m) and m > 0:                            # log axis: a point needs a positive mean
                    pts.append((lam, m, se))
            if not pts:
                continue
            x, m, se = map(np.array, zip(*pts))
            P.plot_readout(ax, x, m, ro, marker="o", ms=3)
            ax.errorbar(x, m, yerr=se, fmt="none", ecolor=P.READOUT_COLOR[ro], elinewidth=0.8, capsize=2, zorder=3)
            figdata[f"{tag}_{ro}"] = np.stack([x, m, se])
            xt = np.linspace(min(lams), max(lams), 80)                  # the equation's half-life over the λ axis
            yt = np.array([theory_mhalf(ro, name, run0.N, (kappa_of or {}).get(ro, run0.kappa), lam, run0.delta0,
                                        run0.sigma0) for lam in xt])
            if np.isfinite(yt).any():
                ax.plot(xt, yt, color=P.READOUT_COLOR[ro], ls=":", lw=P.LW_REF, zorder=2)
                figdata[f"theory_{tag}_{ro}"] = np.stack([xt, yt])
        n_cens.append(cens)
        ax.axhline(run0.N * math.log(2), color=P.GREY_REF, ls=":", lw=P.LW_REF, zorder=1)
        P.logaxis(ax, "y")
        ax.set_ylabel(f"{label}\n" + r"half-life $m_{1/2}$ (tasks)")
    lam_max = max(lams)
    axes[0, 0].set_xlim(0.0, lam_max * 1.03)
    if lam_max >= 4:                                                    # ticks at 0, 5, …, 20 rather than MaxNLocator's 6s
        step = max(1, int(round(lam_max / 4)))
        P.discrete_ticks(axes[-1, 0], range(0, int(lam_max) + 1, step), "x")
    axes[-1, 0].set_xlabel(r"hidden-layer learning per task $\lambda$")
    kappa = run0.kappa_label if run0.gated == 1.0 else None
    P.cell_title(fig, kappa=kappa, d0=run0.delta0, solver=run0.solver, eta=run0.eta, extra=f"N = {run0.N}")
    n_seeds = max(len(d[ROWS[0][0]]) for d in res.values())
    P.caption(fig, f"{H.STAT_TEXT[stat]} ({n_seeds} seeds per point). Top: first teacher from a fresh start, "
                   f"$S_0(t)/S_0(0)$ ≤ ½ within T = {T} tasks ({n_cens[0]} values missing: censored). Bottom: the teacher "
                   rf"of task T, $S_T(\tau)/S_T(0)$ ≤ ½ within τ ≤ {tau} more tasks ({n_cens[1]} missing). Dotted in a "
                   f"readout's colour: the half-life of its equation, (S_heads), (S_frozen) or, at steady state, "
                   f"(S_learned). Dotted grey: N ln 2; the ungated readouts (dashed) have no κ.")
    P.fig_legend(fig, readouts, styles=[(":", "equation / N ln 2 (grey)")])
    return P.save(fig, path, figdata=figdata)


def main(argv: Optional[list[str]] = None) -> int:
    """Command-line entry point (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(_path.ROOT, "configs", "stage1.yaml"),
                    help="config supplying the solver, step-size and stream-length rules (default: Stage 1)")
    ap.add_argument("--N", type=int, default=100)
    ap.add_argument("--kappa", default="1", help="κ of the gated readouts, as typed (the ungated ones have none)")
    ap.add_argument("--delta0", type=float, default=0.0)
    ap.add_argument("--lam", type=float, nargs="+", default=list(LAM_DEFAULT),
                    help="the λ grid, the x axis (default: 0.5 … 20; λ = 0 is skipped)")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--T", type=int, default=None,
                    help="tasks the first memory is followed for and trained before the steady-state probe "
                         "(default: the config's T_burn + T_meas)")
    ap.add_argument("--tau", type=int, default=None, help="tasks the steady-state probe is followed for (default: T)")
    ap.add_argument("--readout", nargs="*", default=list(READOUT_ORDER))
    ap.add_argument("--out", default=os.path.join(_path.ROOT, "results", "halflife_lam"))
    H.add_arg(ap)
    add_clock_arg(ap)
    pool.add_args(ap)
    args = ap.parse_args(argv)

    lams = sorted({float(l) for l in args.lam if l > 0})
    if any(l < 0 for l in args.lam):
        raise SystemExit("λ must be ≥ 0")
    if len(lams) < len(set(args.lam)):
        print("note: λ = 0 takes no step (nothing is learned, no signal to halve); skipped", flush=True)
    if not lams:
        print("no λ > 0; nothing to do", flush=True)
        return 0
    cfg = Config.from_yaml(args.config)
    over = {"name": "halflife_lam", "N": [args.N], "kappa": [args.kappa], "readout": list(args.readout),
            "delta0": [args.delta0], "lam": lams, "seeds": args.seeds, "extra": (), "trace": False, **clock_override(args)}
    cfg = Config(**{**cfg.__dict__, **over})
    runs = make_runs(cfg)
    if not runs:
        print("no runs left after skipping κ outside [1, √N]; nothing to do", flush=True)
        return 0
    T = args.T or (runs[0].T_burn + runs[0].T_meas)
    tau = args.tau or T
    plan = pool.plan(args, [(g, T + tau + 1) for g in groups(runs, T + tau + 1).values()])
    r0 = runs[0]
    print(f"m½ vs λ: N={args.N} κ={args.kappa} δ₀={args.delta0} λ={lams} {r0.solver} "
          f"n_steps={min(r.n_steps for r in runs)}…{max(r.n_steps for r in runs)} {r0.clock} clock T={T} τ≤{tau} "
          f"seeds={args.seeds} statistic={args.stat}  [{plan.describe()}]", flush=True)
    res = collect(runs, T, tau, plan, args.stat)

    run0 = next((r for r in runs if r.gated == 1.0), r0)
    stem = os.path.join(args.out, "figures",
                        f"mhalf_vs_lam_N{args.N}_k{P.fmt(run0.kappa_label)}_d0{P.fmt(run0.delta0)}")
    out = figure(res, lams, run0, T, tau, stem, args.stat, {r.readout: r.kappa for r in runs})
    print(f"wrote {out} and {stem}.npz")
    return 0


if __name__ == "__main__":
    _rc = main()
    _dt = time.perf_counter() - _T0
    print(f"run_halflife_lam: total {_dt:.1f} s ({_dt / 60:.2f} min) from after the imports to the end", flush=True)
    sys.exit(_rc)
