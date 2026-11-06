#!/usr/bin/env python
"""Steady-state half-life against width (PLAN.md §12.13): m½ from S_T(m)/S_T(0) vs N.

    python scripts/run_Nscaling.py [--N 5 10 20 50 100] [--kappa 1 2 4] [--delta0 0] [--lam 4] [--seeds 4] [--gpus auto]

For every (N, κ, readout, seed) the network trains the Stage-1 stream of
T = T_burn + T_meas tasks (14N under ``configs/stage1.yaml``), then T
more while the teacher of task T is followed: S_T(m) = u_{T+m}·w_T*/|w_T*|²
for m = 0 … T, with the gate of task T and its stored head (task heads)
or the current shared readout.  The half-life m½ is the first m at
which S_T(m)/S_T(0) ≤ ½ (linear interpolation), i.e. the number of tasks
a memory survives at half alignment or better; a seed that never
crosses in T tasks is censored (NaN, counted in the caption).  Two
figures in ``results/Nscaling/figures/``:

``mhalf_vs_N_k{κ}_…``   x = N (log), y = m½ (tasks, log), one trace per
    readout at the first κ listed (seed mean, ±1 s.e. bars over seeds);
    the ungated readouts have no κ and appear on this figure only.
    Dotted grey: the plan's estimate m½ = N ln 2 (PLAN.md §12.3 M4).
``mhalf_theory_…``      one panel per gated readout, traces = κ in shades
    of the readout's colour (lightest = smallest κ); solid = simulation
    (seed mean ± 1 s.e.), dotted = the equation.  ``--theory nln2``
    (default) draws m½ = N ln 2 for every κ; ``--theory nk2`` draws
    m½ = N ln 2 / κ² (the N/κ² claim of PLAN.md §12.3).

The per-condition values are written to ``mhalf_….npz`` beside the figures.

Placement (``mup.pool``): the runs of one width — every κ, readout and
seed, which share (N, D, n_steps, T) — form one ``vmap`` batch advanced
``--chunk`` tasks per device call with S_T evaluated inside the scan
(``experiments.signal_after_tasks``) and runs as a job, largest N first,
over the GPUs (``--gpus``: the four cards, several processes each, a width
split only as far as needed to fill them — ``pool.plan``), over
``--workers`` CPU processes, or in this process.
"""
import argparse
import math
import os
import sys
import time

import numpy as np

import _path  # noqa: F401

from mup import experiments as X
from mup import plotting as P
from mup import pool
from mup.config import Config, READOUT_ORDER, is_gated, make_runs

_T0 = time.perf_counter()        # wall clock from after the imports to the end of the script

GATED = tuple(ro for ro in READOUT_ORDER if is_gated(ro))

def half_life(m: np.ndarray) -> float:
    """First task at which ``m`` falls to ``m[0]/2``, linearly interpolated; NaN if never (as ``run_szero.half_life``)."""
    half = m[0] / 2
    below = np.nonzero(m <= half)[0]
    if len(below) == 0:
        return float("nan")
    j = int(below[0])
    if j == 0:
        return 0.0
    return j - 1 + (m[j - 1] - half) / (m[j - 1] - m[j])


def theory(N: float, kappa: float, kind: str) -> float:
    """Equation half-life: ``nln2`` → N ln 2; ``nk2`` → N ln 2 / κ²."""
    return N * math.log(2) / (kappa ** 2 if kind == "nk2" else 1.0)


def groups(runs: list) -> dict:
    """``{(N, D, solver, n_steps, T): runs}``, largest N first."""
    g = {}
    for r in runs:
        g.setdefault(X.group_key(r) + (r.T_burn + r.T_meas,), []).append(r)
    return {k: g[k] for k in sorted(g, key=lambda k: -k[0])}


def make_jobs(groups_: dict, pieces) -> list:
    """One job per piece of a (N, D, n_steps, T) group: train T tasks, follow probe T for T + 1 more."""
    jobs = []
    for (key, runs), k in zip(groups_.items(), pieces):
        T = key[-1]
        for i, part in enumerate(X.split(runs, k)):
            jobs.append(((key, i), {"runs": part, "T_pre": T, "n_follow": T + 1, "k": T},
                         X.run_cost(part[0], 2 * T + 1) * len(part)))
    return jobs


def mhalf_from_signal(S: np.ndarray) -> np.ndarray:
    """m½ of every column of ``S`` (T + 1, B); NaN when censored or infeasible (NaN column)."""
    return np.array([half_life(S[:, j]) if np.isfinite(S[:, j]).all() else float("nan") for j in range(S.shape[1])])


def _report(key, mh, T):
    if mh is None:
        print(f"  N={key[0]} κ={key[1]} {key[2]}: every seed infeasible, skipped", flush=True)
        return
    n_ok = int(np.isfinite(mh).sum())
    se = np.nanstd(mh, ddof=1) / math.sqrt(n_ok) if n_ok > 1 else 0.0
    print(f"  N={key[0]:4d} κ={key[1]:5s} {key[2]:15s} T={T:5d}  m½ = {np.nanmean(mh):7.2f} ± {se:5.2f}"
          f"  censored or infeasible {int(np.isnan(mh).sum())}/{len(mh)}", flush=True)


def collect(runs: list, plan: pool.Plan, log=print) -> dict:
    """``{(N, kappa_label, readout): (n_seeds,) m½ array}`` for every condition of ``runs``.

    Parameters
    ----------
    runs : list of config.Run
    plan : pool.Plan
        Placement and pieces per width (``pool.plan`` over ``groups(runs)``).
    """
    jobs = make_jobs(groups(runs), plan.pieces)
    res = pool.run_jobs("mup.experiments:signal_after_tasks", jobs, log=log, **plan.place)
    mh = {}
    for (_, kw, _), S in zip(jobs, res):
        for r, v in zip(kw["runs"], mhalf_from_signal(S)):
            mh[(r.N, r.kappa_label, r.readout, r.seed)] = v
    cond = {}
    for r in runs:
        cond.setdefault((r.N, r.kappa_label, r.readout), []).append(r)
    out = {}
    for key in sorted(cond, key=lambda k: (k[0], READOUT_ORDER.index(k[2]), k[1])):
        v = np.array([mh[key + (r.seed,)] for r in cond[key]])
        T = cond[key][0].T_burn + cond[key][0].T_meas
        if not np.isfinite(v).any():
            _report(key, None, T)
            continue
        out[key] = v
        _report(key, v, T)
    return out


def _stat(mh: np.ndarray) -> tuple:
    """(mean, s.e.) over the uncensored seeds."""
    v = mh[np.isfinite(mh)]
    if len(v) == 0:
        return np.nan, np.nan
    return float(v.mean()), float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else 0.0


def fig_readouts(res: dict, Ns: list, klabel: str, run0, path: str) -> str:
    """m½ vs N, one trace per readout at κ = ``klabel`` (ungated readouts at their κ-free value)."""
    P.apply_style()
    fig, axes = P.grid(1, 1, panel=(4.0, 2.8))
    ax = axes[0, 0]
    figdata = {}
    n_cens = 0
    readouts = []
    for ro in READOUT_ORDER:
        k = klabel if is_gated(ro) else "sqrtN"
        pts = [(N,) + _stat(res[(N, k, ro)]) for N in Ns if (N, k, ro) in res]
        if not pts:
            continue
        readouts.append(ro)
        x, m, se = map(np.array, zip(*pts))
        n_cens += sum(int(np.isnan(res[(N, k, ro)]).sum()) for N in Ns if (N, k, ro) in res)
        P.plot_readout(ax, x, m, ro, marker="o", ms=3)
        ax.errorbar(x, m, yerr=se, fmt="none", ecolor=P.READOUT_COLOR[ro], elinewidth=0.8, capsize=2, zorder=3)
        figdata[f"mhalf_{ro}"] = np.stack([x, m, se])
    Nr = np.array(Ns, float)
    ax.plot(Nr, Nr * math.log(2), color=P.GREY_REF, ls=":", lw=P.LW_REF, zorder=1)
    P.logaxis(ax, "x"); P.logaxis(ax, "y")
    P.discrete_ticks(ax, Ns, "x")
    ax.set_xlabel("$N$")
    ax.set_ylabel(r"half-life $m_{1/2}$ (tasks)")
    P.cell_title(fig, kappa=klabel, d0=run0.delta0, lam=run0.lam, solver=run0.solver, eta=run0.eta)
    P.caption(fig, f"m½: first m with S_T(m)/S_T(0) ≤ ½ after T = T_burn + T_meas tasks of training, followed for T more "
                   f"(seed mean, ±1 s.e. bars; {n_cens} censored seeds dropped); dotted grey: N ln 2; the ungated "
                   f"readouts have no κ (every unit active) and sit at < 1 task.")
    P.fig_legend(fig, readouts, styles=[(":", "N ln 2")])
    return P.save(fig, path, figdata=figdata)


def fig_theory(res: dict, Ns: list, klabels: list, run0, kind: str, path: str) -> str:
    """One panel per gated readout; traces = κ (shades); solid = simulation, dotted = equation."""
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
            pts = [(N,) + _stat(res[(N, k, ro)]) for N in Ns if (N, k, ro) in res]
            if pts:
                x, m, se = map(np.array, zip(*pts))
                ax.plot(x, m, color=col, ls="-", lw=P.LW_TRACE, marker="o", ms=3, zorder=4)
                ax.errorbar(x, m, yerr=se, fmt="none", ecolor=col, elinewidth=0.8, capsize=2, zorder=3)
                figdata[f"sim_{ro}_k{P.fmt(k)}"] = np.stack([x, m, se])
            th = np.array([theory(N, kv, kind) for N in Nr])
            ax.plot(Nr, th, color=col, ls=":", lw=P.LW_TRACE, zorder=2)
            figdata[f"theory_{ro}_k{P.fmt(k)}"] = np.stack([Nr, th])
        ax.set_title(P.READOUT_LABEL[ro], color=P.READOUT_COLOR[ro], fontsize=P.FS_MIN + 1)
        P.logaxis(ax, "x"); P.logaxis(ax, "y")
        P.discrete_ticks(ax, Ns, "x")
        ax.set_xlabel("$N$")
    axes[0, 0].set_ylabel(r"half-life $m_{1/2}$ (tasks)")
    eq = "N ln 2" if kind == "nln2" else "N ln 2 / κ²"
    P.cell_title(fig, d0=run0.delta0, lam=run0.lam, solver=run0.solver, eta=run0.eta)
    P.caption(fig, f"solid: simulated m½ (first m with S_T(m)/S_T(0) ≤ ½ after T tasks, followed for T more; seed mean, "
                   f"±1 s.e. bars); dotted: the equation m½ = {eq}; κ in shades of the readout colour, lightest = smallest κ.")
    handles = [P.Line2D([], [], color=P.kappa_shades(i, len(klabels)), lw=1.6, label=f"κ = {P.fmt(k)}")
               for i, k in enumerate(klabels)]
    P.fig_legend(fig, handles=handles, styles=[("-", "simulation"), (":", f"equation {eq}")])
    return P.save(fig, path, figdata=figdata)


def main(argv=None):
    """Command-line entry point (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(_path.ROOT, "configs", "stage1.yaml"))
    ap.add_argument("--N", type=int, nargs="+", default=[5, 10, 20, 50, 100])
    ap.add_argument("--kappa", nargs="+", default=["1", "2", "4"])
    ap.add_argument("--delta0", type=float, default=0.0)
    ap.add_argument("--lam", type=float, default=None, help="default: the config's first λ")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--readout", nargs="*", default=list(READOUT_ORDER))
    ap.add_argument("--theory", choices=["nln2", "nk2"], default="nln2")
    ap.add_argument("--out", default=os.path.join(_path.ROOT, "results", "Nscaling"))
    pool.add_args(ap)
    args = ap.parse_args(argv)

    cfg = Config.from_yaml(args.config)
    over = {"name": "Nscaling", "N": list(args.N), "kappa": list(args.kappa), "readout": list(args.readout),
            "delta0": [args.delta0], "lam": [args.lam if args.lam is not None else cfg.lam[0]],
            "seeds": args.seeds, "extra": (), "trace": False}
    cfg = Config(**{**cfg.__dict__, **over})
    runs = make_runs(cfg)
    run0 = runs[0]
    plan = pool.plan(args, [(g, 2 * key[-1] + 1) for key, g in groups(runs).items()])
    print(f"half-life vs N: N={args.N} κ={args.kappa} δ₀={run0.delta0} λ={run0.lam} {run0.solver} seeds={args.seeds}  "
          f"[{plan.describe()}]", flush=True)
    res = collect(runs, plan)

    klabels = [P.fmt(k) for k in args.kappa]
    fdir = os.path.join(args.out, "figures")
    tag = f"d0{P.fmt(run0.delta0)}_lam{P.fmt(run0.lam)}"
    f1 = fig_readouts(res, list(args.N), klabels[0], run0, os.path.join(fdir, f"mhalf_vs_N_k{klabels[0]}_{tag}"))
    f2 = fig_theory(res, list(args.N), klabels, run0, args.theory, os.path.join(fdir, f"mhalf_theory_{args.theory}_{tag}"))
    np.savez_compressed(os.path.join(fdir, f"mhalf_{tag}.npz"),
                        **{f"N{N}_k{k}_{ro}": v for (N, k, ro), v in res.items()})
    print(f"wrote {f1}\n      {f2}")
    return 0


if __name__ == "__main__":
    _rc = main()
    _dt = time.perf_counter() - _T0
    print(f"run_Nscaling: total {_dt:.1f} s ({_dt / 60:.2f} min) from after the imports to the end", flush=True)
    sys.exit(_rc)
