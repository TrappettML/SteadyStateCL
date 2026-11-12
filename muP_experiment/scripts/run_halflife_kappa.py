#!/usr/bin/env python
"""Signal half-life against the gate scale κ: m½ of the first memory and of a steady-state memory, one figure.

    python scripts/run_halflife_kappa.py [--synaptic] [--logx] [--N 1000] [--n-kappa 6] [--kappa …] [--delta0 0] [--lam 4] [--seeds 4]
                                         [--T T] [--tau τ] [--stat mean|seed] [--clock realized|nominal] [--gpus]

The companion of ``run_halflife_lam.py`` with κ on the x axis.  One figure for the cell (N, δ₀, λ),
``results/halflife_kappa/figures/mhalf_vs_kappa_N{N}_d0{δ₀}_lam{λ}.png``, two rows that share both axes,
x = κ from the smallest gate scale the model allows, κ = 1 (a task then uses √N units on average;
``config.kappa_ok``), to κ = √N (gate density 1: every unit active), y = half-life m½ in tasks (log), one trace
per readout.  The κ grid is ``--n-kappa`` values evenly spaced on a linear x axis, or with ``--logx`` evenly
spaced in log κ on a log x axis; ``--kappa`` lists the values instead (``--logx`` then only sets the axis).

row 1, first memory      from a fresh initialisation the signal of the FIRST teacher is read after every task,
    S₀(t) = u_t·w₀*/|w₀*|², for t = 0 … T − 1; m½ is the first t with S₀(t) ≤ S₀(0)/2.
row 2, steady state      the same network trains on; the teacher of task T is followed for τ = 0 … ``--tau``
    more tasks, S_T(τ) = u_{T+τ}·w_T*/|w_T*|²; m½ is the first τ with S_T(τ) ≤ S_T(0)/2.

The gated readouts are traces over κ.  The ungated ones (learned shared, frozen) have no κ — every unit is
active, the κ = √N point of their gated counterpart — so each is one value, drawn as a dashed horizontal line
with a ±1 s.e. band.  Half-lives are linearly interpolated.  ``--stat mean`` (default) is the write-up's m½, the
first crossing of ½ by the mean curve over seeds (eq. Sigma) with a jackknife s.e.; ``--stat seed`` is the mean
± 1 s.e. of the per-seed half-lives (``mup.halflife``).  A curve that never reaches half within its window
(T tasks, row 1; τ tasks, row 2) is censored: left out and counted in the caption and on the console.
Dotted, in a gated readout's colour: the half-life of its equation over κ (``theory.mhalf``: eqs. S_heads,
S_frozen and, in the steady-state row, S_learned; ``run_halflife_lam.theory_mhalf``).  Dotted grey: N ln 2.

``--synaptic`` adds, for every gated readout, the same runs with the gate on the synapse (write-up Section
"Synaptic gating", ``mup.synapse``): task c has a mask Γ_c ∈ {0, 1}^{N×D} of i.i.d. indicators with
Pr = p = κ/√N applied entrywise to W, u_{c,j} = Σ_i γ_cij a_i W_ij / pN; the step is eq. (syn_pop) and the
projection runs on all N units after every step.  A synaptic run has the W⁽⁰⁾, readout, teachers and fresh heads
of its unit-gated twin.  Under ``--clock nominal`` (clock (i)) it also has its task duration, so the pair
differs in the mask alone — the condition under which the write-up shows equal mean signals, eq. (syn_Sigma).
Under the default ``--clock realized`` (clock (ii), the write-up's experiment) each gating runs its own
realized duration λ/ā²_c, the synaptic one at the mean rate over the input coordinates, and the half-lives
differ by the factor θν/θ̄ of eq. (syn_heads), which tends to 1 as κ√N grows.  The synaptic traces are
dash-dotted with open squares in the colour of their unit-gated trace.
A synaptic step touches all N·D synapses whatever κ is (a factor √N/κ over unit gating), so at N = 1000 these
runs dominate the cost.

N = 1000, δ₀ = 0 by default and λ the config's (4).  T defaults to the Stage-1 stream T_burn + T_meas (14 N tasks) and
``--tau`` to T; at N = 1000 that is 28 001 tasks of 40 steps per run, so use ``--T`` / ``--tau`` to shorten it.
The half-lives (the statistic drawn, and the per-seed ones under ``…_seed``) are written next to the figure as
``mhalf_vs_kappa_….npz``.

Placement (``mup.pool``): a gated batch steps on the block of its active rows, about κ√N of the N, so every κ
is its own ``vmap`` batch (``experiments.group_key``), all seeds together, and the cost grows with κ.  Each
batch is ONE stream of T + τ + 1 tasks (``experiments.first_and_steady_signal``).  The batches run as jobs,
costliest first, over the GPUs (``--gpus``: every card, several processes each, a batch split over cards as far
as its share of the cost asks — ``pool.plan``), over ``--workers`` CPU processes, or in this process.  The
synaptic runs have no block of active rows, so every κ shares one batch per ``synapse.group_key``; they run as a
second set of jobs after the unit-gated ones.
"""
import argparse
import math
import os
import sys
import time
from typing import Callable, Optional, Sequence

import numpy as np

import _path  # noqa: F401

from mup import experiments as X
from mup import halflife as H
from mup import plotting as P
from mup import pool
from mup import synapse as S
from mup.config import Config, READOUT_ORDER, Run, add_clock_arg, clock_override, is_gated, make_runs
from run_Nscaling import _stat
from run_halflife_lam import ROWS, cost, make_jobs, reduce_rows, theory_mhalf
from run_synapse_comp import CLOCK_TEXT

_T0 = time.perf_counter()        # wall clock from after the imports to the end of the script


#: Line style and marker of a synaptic trace (as ``run_synapse_comp.py``).
SYN_LS, SYN_MARKER = "-.", "s"


def default_kappas(N: int, n: int = 6, logx: bool = False) -> list[str]:
    """The κ grid when ``--kappa`` is not given: ``n`` values from 1 to √N, evenly spaced (in log κ with
    ``logx``), as the labels the runs carry: 4 significant figures, the last one ``sqrtN``."""
    top = math.sqrt(N)
    if n < 2 or top <= 1.0:
        return ["sqrtN"]
    ks = np.geomspace(1.0, top, n) if logx else np.linspace(1.0, top, n)
    return [f"{k:.4g}" for k in ks[:-1]] + ["sqrtN"]


def groups(runs: list[Run], n_tasks: int) -> dict[tuple, list[Run]]:
    """``{experiments.group_key: runs}`` (one group per κ, the ungated and the closed-form readouts apart),
    costliest job first (``run_halflife_lam.cost`` over ``n_tasks`` tasks)."""
    g = {}
    for r in runs:
        g.setdefault(X.group_key(r), []).append(r)
    return {k: g[k] for k in sorted(g, key=lambda k: -cost(g[k], n_tasks))}


def syn_groups(runs: list[Run], n_tasks: int) -> dict[tuple, list[Run]]:
    """``{synapse.group_key: gated runs}`` (every κ together; the closed-form frozen readout apart), costliest
    first.  The runs at gate density 1 (κ = √N) are left out: there the two gatings are the same network and the
    unit-gated result is reused (``synapse.all_open``)."""
    g = {}
    for r in runs:
        if r.gated == 1.0 and not S.all_open(r):
            g.setdefault(S.group_key(r), []).append(r)
    return {k: g[k] for k in sorted(g, key=lambda k: -S.job_cost(g[k], n_tasks))}


def syn_jobs(groups_: dict[tuple, list[Run]], T: int, tau: int, pieces: Sequence[int]) -> list[tuple]:
    """One job per piece of a synaptic group: ``(key, kwargs of synapse.first_and_steady_signal, cost)``."""
    jobs = []
    for (key, runs), k in zip(groups_.items(), pieces):
        for i, part in enumerate(X.split(runs, k)):
            jobs.append(((("syn",) + key, i), {"runs": part, "T": T, "n_follow": tau + 1}, S.job_cost(part, T + tau + 1)))
    return jobs


def cond_key(r: Run) -> tuple:
    """(readout, κ label) of a run: one point of a trace (the ungated readouts have the one label ``sqrtN``)."""
    return (r.readout, r.kappa_label)


def _report(key: tuple, d: dict, T: int, tau: int) -> None:
    """Print one condition's line: m½ ± s.e. of the chosen statistic and the censored count of both rows."""
    txt = []
    for (name, label, _), window in zip(ROWS, (T, tau)):
        mh = d[name]
        m, se = _stat(mh)
        txt.append(f"{label} m½ = {m:8.2f} ± {se:6.2f} (censored or infeasible {int(np.isnan(mh).sum())}/{len(mh)}, "
                   f"window {window})")
    kappa = key[1] if is_gated(key[0]) else "none"
    print(f"  κ={kappa:6s} {key[0]:15s} " + "   ".join(txt), flush=True)


def collect(runs: list[Run], T: int, tau: int, plan: pool.Plan, synaptic: bool = False, stat: str = "mean",
            log: Callable[[str], None] = print) -> dict[tuple, dict[str, np.ndarray]]:
    """``{(readout, κ label): {"S0": (n_seeds,) m½ values of the first memory, "ST": those at steady state,
    "S0_seed", "ST_seed": the per-seed half-lives}}`` (``run_halflife_lam.reduce_rows``; ``stat`` as ``mup.halflife``).

    Parameters
    ----------
    runs : list of config.Run
        One cell (N, δ₀, λ) at every κ.
    T : int
        Tasks the first memory is followed for, and trained before the steady-state probe.
    tau : int
        Tasks the steady-state probe is followed for after task T.
    plan : pool.Plan
        Placement and pieces per group (``pool.plan`` over ``groups(runs, T + tau + 1)``, or over
        ``syn_groups`` when ``synaptic``).
    synaptic : bool
        Run the gated runs of ``runs`` with the gate on the synapse (``mup.synapse``) instead.
    log : callable
        Progress sink.

    Returns
    -------
    dict
        In (readout, κ) order; NaN for a censored or infeasible seed.
    """
    if synaptic:
        jobs = syn_jobs(syn_groups(runs, T + tau + 1), T, tau, plan.pieces)
        res = pool.run_jobs("mup.synapse:first_and_steady_signal", jobs, log=log, **plan.place)
    else:
        jobs = make_jobs(groups(runs, T + tau + 1), T, tau, plan.pieces)
        res = pool.run_jobs("mup.experiments:first_and_steady_signal", jobs, log=log, **plan.place)
    red = reduce_rows(jobs, res, cond_key, stat)
    kappa_of = {cond_key(r): r.kappa for _, kw, _ in jobs for r in kw["runs"]}
    out = {}
    for key in sorted(red, key=lambda k: (READOUT_ORDER.index(k[0]), kappa_of[k])):
        out[key] = red[key]
        _report(key, out[key], T, tau)
    return out


def figure(res: dict[tuple, dict[str, np.ndarray]], syn: Optional[dict], kappas: list[tuple[float, str]], run0: Run,
           T: int, tau: int, logx: bool, path: str, stat: str = "mean") -> str:
    """m½ vs κ: row 1 the first memory, row 2 a steady-state memory; one trace per readout (± 1 s.e.), dotted the
    half-life of a gated readout's equation over κ.

    Parameters
    ----------
    res : dict
        ``collect`` output.
    syn : dict or None
        ``collect(..., synaptic=True)`` output: the gated readouts again, drawn dash-dotted with open squares.
    kappas : list of (κ, label)
        The gated κ grid in increasing order (the x axis).
    run0 : config.Run
        Any run of the cell.
    T, tau : int
        The two follow windows, printed in the caption.
    logx : bool
        Log x axis (else linear).
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
    lo, hi = (kv.min(), kv.max()) if len(kv) else (1.0, math.sqrt(run0.N))
    pad = 0.04 * (hi - lo) or 0.5
    xlim = (lo / 1.12, hi * 1.12) if logx else (lo - pad, hi + pad)
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
                figdata[f"mhalf_{tag}_{ro}_k{klabel}_seed"] = res[(ro, klabel)][name + "_seed"]
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
            xt = np.geomspace(lo, hi, 60) if logx else np.linspace(lo, hi, 60)   # the equation's half-life over κ
            yt = np.array([theory_mhalf(ro, name, run0.N, k, run0.lam, run0.delta0, run0.sigma0) for k in xt])
            if np.isfinite(yt).any():
                ax.plot(xt, yt, color=col, ls=":", lw=P.LW_REF, zorder=2)
                figdata[f"theory_{tag}_{ro}"] = np.stack([xt, yt])
            pts = []
            for k, klabel in (kappas if syn else []):
                if (ro, klabel) not in syn:
                    continue
                mh = syn[(ro, klabel)][name]
                figdata[f"mhalf_{tag}_syn_{ro}_k{klabel}"] = mh
                figdata[f"mhalf_{tag}_syn_{ro}_k{klabel}_seed"] = syn[(ro, klabel)][name + "_seed"]
                cens += int(np.isnan(mh).sum())
                m, se = _stat(mh)
                if np.isfinite(m) and m > 0:
                    pts.append((k, m, se))
            if pts:
                x, m, se = map(np.array, zip(*pts))
                P.plot_readout(ax, x, m, ro, ls=SYN_LS, marker=SYN_MARKER, ms=4, mfc="white", zorder=5)
                ax.errorbar(x, m, yerr=se, fmt="none", ecolor=col, elinewidth=0.8, capsize=2, zorder=3)
                figdata[f"{tag}_syn_{ro}"] = np.stack([x, m, se])
        n_cens.append(cens)
        ax.axhline(run0.N * math.log(2), color=P.GREY_REF, ls=":", lw=P.LW_REF, zorder=1)
        if logx:
            P.logaxis(ax, "x")
        P.logaxis(ax, "y")
        ax.set_ylabel(f"{label}\n" + r"half-life $m_{1/2}$ (tasks)")
    ax = axes[-1, 0]
    ax.set_xlim(*xlim)
    if 0 < len(kv) <= 8:                                                # ticks at the κ run, the last one named √N
        P.discrete_ticks(ax, kv, "x")
        ax.set_xticklabels([r"$\sqrt{N}$" if lab == "sqrtN" else P.fmt(lab) for _, lab in kappas])
    ax.set_xlabel(r"gate scale $\kappa$  (gate density $\kappa/\sqrt{N}$)")
    P.cell_title(fig, d0=run0.delta0, lam=run0.lam, solver=run0.solver, eta=run0.eta, extra=f"N = {run0.N}")
    n_seeds = max(len(d[ROWS[0][0]]) for d in res.values())
    P.caption(fig, f"{H.STAT_TEXT[stat]} ({n_seeds} seeds per point). Top: first teacher from a fresh start, "
                   f"$S_0(t)/S_0(0)$ ≤ ½ within T = {T} tasks ({n_cens[0]} values missing: censored). Bottom: the teacher "
                   rf"of task T, $S_T(\tau)/S_T(0)$ ≤ ½ within τ ≤ {tau} more tasks ({n_cens[1]} missing). Dashed horizontal "
                   f"lines: the ungated readouts, which have no κ (every unit active, as at κ = √N). Dotted in a readout's "
                   f"colour: the half-life of its equation, (S_heads), (S_frozen) or, at steady state, (S_learned). "
                   f"Dotted grey: N ln 2."
                   + (" Dash-dotted, open squares: the gated readouts with the gate on the synapse, same density, "
                      f"initial weights and teachers as the unit-gated trace of their colour, {CLOCK_TEXT[run0.clock]}."
                      if syn else ""))
    handles = [P.Line2D([], [], color="k", ls=SYN_LS, lw=1.2, marker=SYN_MARKER, ms=4, mfc="white",
                        label="synaptic gating")] if syn else []
    P.fig_legend(fig, readouts, handles=handles + [P.Line2D([], [], color=P.GREY_REF, ls=":", lw=1.2, label="N ln 2")])
    return P.save(fig, path, figdata=figdata)


def main(argv: Optional[list[str]] = None) -> int:
    """Command-line entry point (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(_path.ROOT, "configs", "stage1.yaml"),
                    help="config supplying the solver, λ, step-size and stream-length rules (default: Stage 1)")
    ap.add_argument("--synaptic", action="store_true",
                    help="also run every gated readout with the gate on the synapse and add it to the figure as a trace")
    ap.add_argument("--logx", action="store_true",
                    help="log x axis and a κ grid evenly spaced in log κ (default: linear axis, evenly spaced κ)")
    ap.add_argument("--N", type=int, default=1000)
    ap.add_argument("--n-kappa", type=int, default=6, help="number of κ values from 1 to √N (default 6)")
    ap.add_argument("--kappa", nargs="+", default=None,
                    help="the κ values of the gated readouts instead of the --n-kappa grid: numbers in [1, √N] or 'sqrtN'")
    ap.add_argument("--delta0", type=float, default=0.0)
    ap.add_argument("--lam", type=float, default=None, help="default: the config's first λ")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--T", type=int, default=None,
                    help="tasks the first memory is followed for and trained before the steady-state probe "
                         "(default: the config's T_burn + T_meas)")
    ap.add_argument("--tau", type=int, default=None, help="tasks the steady-state probe is followed for (default: T)")
    ap.add_argument("--readout", nargs="*", default=list(READOUT_ORDER))
    ap.add_argument("--out", default=os.path.join(_path.ROOT, "results", "halflife_kappa"))
    H.add_arg(ap)
    add_clock_arg(ap)
    pool.add_args(ap)
    args = ap.parse_args(argv)

    cfg = Config.from_yaml(args.config)
    over = {"name": "halflife_kappa", "N": [args.N], "kappa": list(args.kappa or default_kappas(args.N, args.n_kappa, args.logx)),
            "readout": list(args.readout), "delta0": [args.delta0],
            "lam": [args.lam if args.lam is not None else cfg.lam[0]], "seeds": args.seeds, "extra": (), "trace": False,
            **clock_override(args)}
    cfg = Config(**{**cfg.__dict__, **over})
    runs = make_runs(cfg)                             # κ outside [1, √N] is skipped with a note
    if not runs:
        print("no runs left after skipping κ outside [1, √N]; nothing to do", flush=True)
        return 0
    if args.synaptic and cfg.solver != "pop":
        raise SystemExit("--synaptic: synaptic gating is implemented for the population solver only")
    kappas = sorted({(r.kappa, r.kappa_label) for r in runs if r.gated == 1.0})
    T = args.T or (runs[0].T_burn + runs[0].T_meas)
    tau = args.tau or T
    plan = pool.plan(args, [(g, T + tau + 1) for g in groups(runs, T + tau + 1).values()])
    r0 = runs[0]
    print(f"m½ vs κ: N={args.N} κ={[lab for _, lab in kappas]} δ₀={r0.delta0} λ={r0.lam} {r0.solver} "
          f"n_steps={r0.n_steps} {r0.clock} clock T={T} τ≤{tau} seeds={args.seeds} statistic={args.stat}  "
          f"[{plan.describe()}]", flush=True)
    if args.synaptic:
        print("unit (hidden-layer) gating:", flush=True)
    res = collect(runs, T, tau, plan, stat=args.stat)
    syn = None
    if args.synaptic and syn_groups(runs, T + tau + 1):
        plan_syn = pool.plan(args, [(g, T + tau + 1) for g in syn_groups(runs, T + tau + 1).values()])
        print(f"synaptic gating:  [{plan_syn.describe()}]", flush=True)
        syn = collect(runs, T, tau, plan_syn, synaptic=True, stat=args.stat)
        same = sorted({cond_key(r) for r in runs if r.gated == 1.0 and S.all_open(r)})
        syn.update({key: res[key] for key in same})        # density 1: the unit-gated run is the synaptic run
        if same:
            print(f"  κ=sqrtN: every synapse open, the same network under either gating; unit-gated values reused "
                  f"for {', '.join(ro for ro, _ in same)}", flush=True)

    stem = os.path.join(args.out, "figures", f"mhalf_vs_kappa_N{args.N}_d0{P.fmt(r0.delta0)}_lam{P.fmt(r0.lam)}")
    out = figure(res, syn, kappas, r0, T, tau, args.logx, stem, args.stat)
    print(f"wrote {out} and {stem}.npz")
    return 0


if __name__ == "__main__":
    _rc = main()
    _dt = time.perf_counter() - _T0
    print(f"run_halflife_kappa: total {_dt:.1f} s ({_dt / 60:.2f} min) from after the imports to the end", flush=True)
    sys.exit(_rc)
