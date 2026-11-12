#!/usr/bin/env python
"""Hidden-layer (unit) gating against synaptic gating: steady-state half-life m½ vs width N (write-up Section "Synaptic gating").

    python scripts/run_synapse_comp.py [--synapse] [--N 5 10 20 50 100] [--kappa 1 2 4] [--delta0 0] [--lam 4] [--seeds 4]
                                       [--stat mean|seed] [--theory eq|nln2|nk2] [--clock realized|nominal] [--gpus]

The experiment of ``run_Nscaling.py`` — for every (N, κ, readout, seed) train T = T_burn + T_meas tasks, follow
the teacher of task T for T more, m½ = the first crossing of ½ by the mean curve Σ(m) = ⟨S_T(m)⟩/⟨S_T(0)⟩
(``--stat mean``, the write-up's m½, jackknife s.e.; ``--stat seed`` averages per-seed half-lives) — with one
more axis.  ``--synapse`` adds, for every gated readout, the same run with the gate on the synapse: a mask
Γ_c ∈ {0, 1}^{N×D} of i.i.d. indicators at the density p = κ/√N of the unit gate, drawn per task and applied
entrywise to W (``mup.synapse``).  A synaptic run starts from the W⁽⁰⁾ and readout of its unit-gated twin and
sees the same teachers.  Without ``--synapse`` the script is ``run_Nscaling.py`` (unit gating only).  The
ungated readouts have no gate and no synaptic form.

The task clock (``--clock``) decides what the write-up predicts:

``realized``, clock (ii), the default and the clock of the write-up's experiment: a unit-gated task lasts
    Λ_c = λ/ā²_c with its own key norm, a synaptic task Λ_c = λ/ā²_c at the mean rate over the input
    coordinates.  The durations are then the only difference between the gatings: S(0) = θ̄ < θ under synaptic
    gating (eq. syn_theta) and the heads' half-life is N ln 2/θ̄ against N ln 2/(θν) (eq. syn_heads), a ratio
    θν/θ̄ → 1 as κ√N grows (1.017 at N = 100, κ = 1, λ = 4).
``nominal``, clock (i): both gatings run the fixed duration Λ = λ/(1 + δ₀), so the pair differs in the mask
    alone and the mean signals are equal at every age, eq. (syn_Sigma): exactly for a held key (frozen shared
    readout; task heads under its assumptions), at leading order for the learned shared readout.

Two figures in ``results/synapse_comp/figures/`` (the layout of ``run_Nscaling.py``, the synaptic runs added as
traces: dash-dotted, open squares, in the colour of the unit-gated trace they belong to):

``mhalf_vs_N_k{κ}_…``   x = N (log), y = m½ (tasks, log), one trace per readout at the first κ listed (±1 s.e.
    bars); dotted grey N ln 2.
``mhalf_theory_…``      one panel per gated readout, traces = κ in shades of the readout's colour (lightest =
    smallest κ); solid = unit gating, dash-dotted = synaptic gating, dotted = the equation (``--theory``: ``eq``
    the half-life of the readout's own curve, eqs. S_heads, S_frozen, S_learned; ``nln2``; ``nk2``).

The values are written to ``mhalf_….npz`` beside the figures, keys ``N{N}_k{κ}_{readout}`` (unit) and
``syn_N{N}_k{κ}_{readout}``, the per-seed half-lives under ``seed_…``.  The console prints every condition and,
with ``--synapse``, the two gatings side by side with their ratio and the predicted θν/θ̄ for the heads.

Placement (``mup.pool``): the unit-gated runs are the jobs of ``run_Nscaling.py``.  A synaptic step touches all
N·D synapses (no block of active rows), so every κ of a width shares one ``vmap`` batch per
``synapse.group_key`` (the stepped readouts, the closed-form frozen one), advanced ``--chunk`` tasks per device
call with S_T evaluated inside the scan (``synapse.signal_after_tasks``).  The two sets of jobs run one after
the other over the GPUs (``--gpus``), ``--workers`` CPU processes, or in this process.
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
from mup import theory as TH
from mup.config import Config, READOUT_ORDER, Run, add_clock_arg, clock_override, is_gated, make_runs
from run_Nscaling import GATED, THEORY_LABEL, _report, _stat, collect, cond_key, groups, reduce, theory

_T0 = time.perf_counter()        # wall clock from after the imports to the end of the script

#: Line style and marker of a synaptic trace (solid = gated, dashed = ungated, dotted = equation are taken).
SYN_LS, SYN_MARKER = "-.", "s"


#: How the task durations of the two gatings relate under each clock, for captions.
CLOCK_TEXT = {"nominal": "equal task duration λ/(1 + δ₀) (clock (i))",
              "realized": "each at its own realized duration λ/ā²_c (clock (ii); the synaptic one at the mean rate)"}


def syn_groups(runs: list[Run]) -> dict[tuple, list[Run]]:
    """``{synapse.group_key + (T,): gated runs}``, costliest job first.  Runs at gate density 1 (κ = √N) are left
    out: the two gatings are then the same network and ``collect_syn`` reuses the unit-gated result."""
    g = {}
    for r in runs:
        if r.gated == 1.0 and not S.all_open(r):
            g.setdefault(S.group_key(r) + (r.T_burn + r.T_meas,), []).append(r)
    return {k: g[k] for k in sorted(g, key=lambda k: -S.job_cost(g[k], 2 * k[-1] + 1))}


def syn_jobs(groups_: dict[tuple, list[Run]], pieces: Sequence[int]) -> list[tuple]:
    """One job per piece of a group: train T tasks, follow probe T for T + 1 more (as ``run_Nscaling.make_jobs``)."""
    jobs = []
    for (key, runs), k in zip(groups_.items(), pieces):
        T = key[-1]
        for i, part in enumerate(X.split(runs, k)):
            jobs.append((("syn",) + key + (i,), {"runs": part, "T_pre": T, "n_follow": T + 1, "k": T},
                         S.job_cost(part, 2 * T + 1)))
    return jobs


def collect_syn(runs: list[Run], plan: pool.Plan, unit: dict, unit_seed: dict, stat: str = "mean",
                log: Callable[[str], None] = print) -> tuple[dict[tuple, np.ndarray], dict[tuple, np.ndarray]]:
    """``(res, seed)`` of the gated runs under synaptic gating, as ``run_Nscaling.collect``; conditions at gate
    density 1 take their values from ``unit`` / ``unit_seed``, the unit-gated results."""
    jobs = syn_jobs(syn_groups(runs), plan.pieces)
    res = pool.run_jobs("mup.synapse:signal_after_tasks", jobs, log=log, **plan.place)
    cond = reduce(jobs, res, stat)
    run_of = {cond_key(r): r for r in runs if r.gated == 1.0}
    out, seed = {}, {}
    for key in sorted(run_of, key=lambda k: (k[0], READOUT_ORDER.index(k[2]), k[1])):
        r = run_of[key]
        if S.all_open(r):                                   # the same network under either gating
            if key in unit:
                out[key], seed[key] = unit[key], unit_seed[key]
            continue
        v = cond[key]["mhalf"]
        ok = np.isfinite(v).any()
        _report(key, v if ok else None, r.T_burn + r.T_meas, cond[key]["seed"])
        if ok:
            out[key], seed[key] = v, cond[key]["seed"]
    return out, seed


def compare(res: dict, syn: dict, runs: list[Run]) -> None:
    """Print the two gatings side by side: m½ ± s.e. and synaptic / unit, with the ratio the write-up predicts
    for the task heads: 1 under clock (i) (eq. syn_Sigma), θν/θ̄ under clock (ii) (eq. syn_heads)."""
    print("unit against synaptic gating, m½ (tasks):", flush=True)
    run_of = {cond_key(r): r for r in runs}
    for key in syn:
        if key not in res:
            continue
        (mu, su), (ms, ss) = _stat(res[key]), _stat(syn[key])
        r = run_of[key]
        pred = TH.theta(r.lam) * TH.nu(r.N, r.p) / TH.syn_theta_bar(r.lam, r.N, r.p) if r.clock == "realized" else 1.0
        print(f"  N={key[0]:4d} κ={key[1]:5s} {key[2]:15s} unit {mu:7.2f} ± {su:5.2f}   synaptic {ms:7.2f} ± {ss:5.2f}"
              f"   synaptic/unit {ms / mu:5.3f}" + (f"   (heads, predicted {pred:5.3f})" if key[2] == "task_heads" else ""),
              flush=True)


def _trace(res: dict, Ns: list[int], k: str, ro: str) -> Optional[tuple]:
    """(x, mean, se, censored count) of a condition over the N that have it; None when none does."""
    pts = [(N,) + _stat(res[(N, k, ro)]) for N in Ns if (N, k, ro) in res]
    if not pts:
        return None
    x, m, se = map(np.array, zip(*pts))
    return x, m, se, sum(int(np.isnan(res[(N, k, ro)]).sum()) for N in Ns if (N, k, ro) in res)


def _syn_handle(color="k"):
    """Legend handle of a synaptic trace."""
    return P.Line2D([], [], color=color, ls=SYN_LS, lw=1.2, marker=SYN_MARKER, ms=4, mfc="white", label="synaptic gating")


def fig_readouts(res: dict, syn: Optional[dict], Ns: list[int], klabel: str, run0: Run, path: str, stat: str = "mean") -> str:
    """m½ vs N, one trace per readout at κ = ``klabel``; with ``syn`` the gated readouts again under synaptic gating."""
    P.apply_style()
    fig, axes = P.grid(1, 1, panel=(4.4, 3.0))
    ax = axes[0, 0]
    figdata = {}
    n_cens = 0
    readouts = []
    for ro in READOUT_ORDER:
        k = klabel if is_gated(ro) else "sqrtN"
        tr = _trace(res, Ns, k, ro)
        if tr is None:
            continue
        readouts.append(ro)
        x, m, se, cens = tr
        n_cens += cens
        P.plot_readout(ax, x, m, ro, marker="o", ms=3)
        ax.errorbar(x, m, yerr=se, fmt="none", ecolor=P.READOUT_COLOR[ro], elinewidth=0.8, capsize=2, zorder=3)
        figdata[f"mhalf_{ro}"] = np.stack([x, m, se])
        tr = _trace(syn, Ns, k, ro) if syn else None
        if tr is not None:
            x, m, se, cens = tr
            n_cens += cens
            P.plot_readout(ax, x, m, ro, ls=SYN_LS, marker=SYN_MARKER, ms=4, mfc="white", zorder=5)
            ax.errorbar(x, m, yerr=se, fmt="none", ecolor=P.READOUT_COLOR[ro], elinewidth=0.8, capsize=2, zorder=3)
            figdata[f"mhalf_syn_{ro}"] = np.stack([x, m, se])
    Nr = np.array(Ns, float)
    ax.plot(Nr, Nr * math.log(2), color=P.GREY_REF, ls=":", lw=P.LW_REF, zorder=1)
    P.logaxis(ax, "x"); P.logaxis(ax, "y")
    P.discrete_ticks(ax, Ns, "x")
    ax.set_xlabel("$N$")
    ax.set_ylabel(r"half-life $m_{1/2}$ (tasks)")
    P.cell_title(fig, kappa=klabel, d0=run0.delta0, lam=run0.lam, solver=run0.solver, eta=run0.eta)
    P.caption(fig, f"{H.STAT_TEXT[stat]}; S_T(m) the signal of the teacher of task T = T_burn + T_meas after m more "
                   f"tasks, followed for T ({n_cens} seeds without a value dropped); dotted grey: N ln 2; the ungated "
                   f"readouts (dashed) have no κ and no gate"
                   + ("; dash-dotted, open squares: the gated readouts with the gate on the synapse, same density, "
                      f"initial weights and teachers as the unit-gated trace of their colour, {CLOCK_TEXT[run0.clock]}."
                      if syn else "."))
    P.fig_legend(fig, readouts, handles=([_syn_handle()] if syn else [])
                 + [P.Line2D([], [], color=P.GREY_REF, ls=":", lw=1.2, label="N ln 2")])
    return P.save(fig, path, figdata=figdata)


def fig_theory(res: dict, syn: Optional[dict], Ns: list[int], klabels: list[str], run0: Run, kind: str, path: str,
               stat: str = "mean") -> str:
    """One panel per gated readout; traces = κ (shades); solid = unit gating, dash-dotted = synaptic, dotted = equation."""
    P.apply_style()
    ros = [ro for ro in GATED if any(k[2] == ro for k in res)]
    fig, axes = P.grid(1, len(ros), sharex="all", sharey="all", panel=(3.0, 2.8))
    figdata = {}
    Nr = np.array(Ns, float)
    kvals = [float(P.fmt(k)) for k in klabels]
    for c, ro in enumerate(ros):
        ax = axes[0, c]
        for i, (k, kv) in enumerate(zip(klabels, kvals)):
            col = P.kappa_shades(i, len(klabels), ro)
            tr = _trace(res, Ns, k, ro)
            if tr is not None:
                x, m, se, _ = tr
                ax.plot(x, m, color=col, ls="-", lw=P.LW_TRACE, marker="o", ms=3, zorder=4)
                ax.errorbar(x, m, yerr=se, fmt="none", ecolor=col, elinewidth=0.8, capsize=2, zorder=3)
                figdata[f"sim_{ro}_k{P.fmt(k)}"] = np.stack([x, m, se])
            tr = _trace(syn, Ns, k, ro) if syn else None
            if tr is not None:
                x, m, se, _ = tr
                ax.plot(x, m, color=col, ls=SYN_LS, lw=P.LW_TRACE, marker=SYN_MARKER, ms=4, mfc="white", zorder=5)
                ax.errorbar(x, m, yerr=se, fmt="none", ecolor=col, elinewidth=0.8, capsize=2, zorder=3)
                figdata[f"syn_{ro}_k{P.fmt(k)}"] = np.stack([x, m, se])
            ok = [N for N in Nr if kv <= math.sqrt(N) * 1.0001]         # the equation needs p = κ/√N ≤ 1
            th = np.array([theory(N, kv, kind, ro, run0.lam, run0.delta0, run0.sigma0) for N in ok])
            ax.plot(ok, th, color=col, ls=":", lw=P.LW_TRACE, zorder=2)
            figdata[f"theory_{ro}_k{P.fmt(k)}"] = np.stack([np.array(ok), th])
        ax.set_title(P.READOUT_LABEL[ro], color=P.READOUT_COLOR[ro], fontsize=P.FS_MIN + 1)
        P.logaxis(ax, "x"); P.logaxis(ax, "y")
        P.discrete_ticks(ax, Ns, "x")
        ax.set_xlabel("$N$")
    axes[0, 0].set_ylabel(r"half-life $m_{1/2}$ (tasks)")
    eq = THEORY_LABEL[kind]
    P.cell_title(fig, d0=run0.delta0, lam=run0.lam, solver=run0.solver, eta=run0.eta)
    P.caption(fig, f"simulation, S_T(m) followed for T tasks after T of training; {H.STAT_TEXT[stat]}: solid = unit gating"
                   + (", dash-dotted with open squares = synaptic gating at the same density, initial weights and "
                      f"teachers, {CLOCK_TEXT[run0.clock]}" if syn else "")
                   + "; dotted: " + ("the root of Σ(m) = ½ for the readout's own curve under unit gating, " if kind == "eq"
                                     else "the equation m½ = ")
                   + f"{eq}; κ in shades of the readout colour, lightest = smallest κ.")
    handles = [P.Line2D([], [], color=P.kappa_shades(i, len(klabels)), lw=1.6, label=f"κ = {P.fmt(k)}")
               for i, k in enumerate(klabels)]
    handles.append(P.Line2D([], [], color="k", ls="-", lw=1.2, marker="o", ms=3, label="unit gating"))
    if syn:
        handles.append(_syn_handle())
    handles.append(P.Line2D([], [], color="k", ls=":", lw=1.2, label="equation" if kind == "eq" else f"equation {eq}"))
    P.fig_legend(fig, None, handles=handles)
    return P.save(fig, path, figdata=figdata)


def main(argv: Optional[list[str]] = None) -> int:
    """Command-line entry point (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(_path.ROOT, "configs", "stage1.yaml"))
    ap.add_argument("--synapse", action="store_true",
                    help="also run every gated readout with the gate on the synapse and add it to the figures as a trace")
    ap.add_argument("--N", type=int, nargs="+", default=[5, 10, 20, 50, 100])
    ap.add_argument("--kappa", nargs="+", default=["1", "2", "4"])
    ap.add_argument("--delta0", type=float, default=0.0)
    ap.add_argument("--lam", type=float, default=None, help="default: the config's first λ")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--readout", nargs="*", default=list(READOUT_ORDER))
    ap.add_argument("--theory", choices=list(THEORY_LABEL), default="eq")
    ap.add_argument("--out", default=os.path.join(_path.ROOT, "results", "synapse_comp"))
    H.add_arg(ap)
    add_clock_arg(ap)
    pool.add_args(ap)
    args = ap.parse_args(argv)

    cfg = Config.from_yaml(args.config)
    over = {"name": "synapse_comp", "N": list(args.N), "kappa": list(args.kappa), "readout": list(args.readout),
            "delta0": [args.delta0], "lam": [args.lam if args.lam is not None else cfg.lam[0]],
            "seeds": args.seeds, "extra": (), "trace": False, **clock_override(args)}
    cfg = Config(**{**cfg.__dict__, **over})
    runs = make_runs(cfg)
    if not runs:
        print("no runs left after skipping κ outside [1, √N]; nothing to do", flush=True)
        return 0
    if args.synapse and cfg.solver != "pop":
        raise SystemExit("--synapse: synaptic gating is implemented for the population solver only")
    run0 = runs[0]
    plan = pool.plan(args, [(g, 2 * key[-1] + 1) for key, g in groups(runs).items()])
    print(f"half-life vs N, unit{' and synaptic' if args.synapse else ''} gating: N={args.N} κ={args.kappa} "
          f"δ₀={run0.delta0} λ={run0.lam} {run0.solver} {run0.clock} clock seeds={args.seeds} statistic={args.stat}  "
          f"[{plan.describe()}]", flush=True)
    print("unit (hidden-layer) gating:", flush=True)
    res, seed = collect(runs, plan, args.stat)
    syn, syn_seed = None, {}
    if args.synapse:
        sg = syn_groups(runs)
        if sg:
            plan_syn = pool.plan(args, [(g, 2 * key[-1] + 1) for key, g in sg.items()])
            print(f"synaptic gating:  [{plan_syn.describe()}]", flush=True)
            syn, syn_seed = collect_syn(runs, plan_syn, res, seed, args.stat)
            compare(res, syn, runs)
        else:
            print("--synapse: no gated readout among the runs; nothing to add", flush=True)

    klabels = [str(k) for k in args.kappa]              # the runs' kappa_label is the argument as typed
    fdir = os.path.join(args.out, "figures")
    tag = f"d0{P.fmt(run0.delta0)}_lam{P.fmt(run0.lam)}"
    f1 = fig_readouts(res, syn, list(args.N), klabels[0], run0, os.path.join(fdir, f"mhalf_vs_N_k{P.fmt(klabels[0])}_{tag}"),
                      args.stat)
    f2 = fig_theory(res, syn, list(args.N), klabels, run0, args.theory, os.path.join(fdir, f"mhalf_theory_{args.theory}_{tag}"),
                    args.stat)
    data = {f"N{N}_k{k}_{ro}": v for (N, k, ro), v in res.items()}
    data.update({f"syn_N{N}_k{k}_{ro}": v for (N, k, ro), v in (syn or {}).items()})
    data.update({f"seed_N{N}_k{k}_{ro}": v for (N, k, ro), v in seed.items()})
    data.update({f"seed_syn_N{N}_k{k}_{ro}": v for (N, k, ro), v in syn_seed.items()})
    np.savez_compressed(os.path.join(fdir, f"mhalf_{tag}.npz"), stat=np.array(args.stat), clock=np.array(run0.clock), **data)
    print(f"wrote {f1}\n      {f2}")
    return 0


if __name__ == "__main__":
    _rc = main()
    _dt = time.perf_counter() - _T0
    print(f"run_synapse_comp: total {_dt:.1f} s ({_dt / 60:.2f} min) from after the imports to the end", flush=True)
    sys.exit(_rc)
