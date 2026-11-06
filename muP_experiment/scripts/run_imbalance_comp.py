#!/usr/bin/env python
"""Imbalance comparison: steady-state weight norms and half-life against N and κ across several δ₀.

    python scripts/run_imbalance_comp.py [--N 5 10 20 50 100] [--kappa 1 2 4] [--delta0 0 1 2 4] [--lam 4] [--seeds 4] [--gpus]

The grid of ``run_weight_norms.py`` and ``run_Nscaling.py`` (every readout ×
κ × seed at each N) is run at every ``--delta0`` and the two observables are
recorded per (N, κ, readout, δ₀, seed): the norms after each task
(``experiments.norms_over_tasks``; steady state = the mean over the
measurement window t ≥ T_burn) and the steady-state half-life m½ of the
probe stored after T tasks (``experiments.signal_after_tasks`` with
``T_pre = T, k = T``; ``run_Nscaling.mhalf_from_signal``).

Output, under ``--out`` (default ``results/imbalance_results``):

``weight_norm_by_delta/norms_vs_N_{readout}.png``
    one file per readout (method); rows = the four rows of
    ``run_weight_norms.fig_vs_N`` (mean Q_ii, mean a_i², |u_t|²/D and the
    collapse Q̄/pN or Q̄²/pN), columns = κ, x = N (log), y = steady-state
    value (seed mean ± 1 s.e. bars), traces = δ₀ in shades of the readout's
    colour (lightest = smallest δ₀).  An ungated readout has no κ and
    repeats in every column.
``weight_norm_by_model/norms_vs_N_d0{δ₀}.png``
    the roles flipped: one file per δ₀, traces = readouts — exactly
    ``run_weight_norms.fig_vs_N`` at that δ₀.
``halflife_by_delta0.png``
    rows = readouts, columns = κ, x = N (log), y = m½ (seed mean ± 1 s.e.
    over the uncensored seeds), traces = δ₀ in shades of the row's readout
    colour; dotted grey N ln 2.
``halflife_by_model.png``
    rows = δ₀, columns = κ, traces = readouts.
``imbalance_comp.npz``
    every per-run curve and half-life: ``N{N}_k{κ}_{readout}_d0{δ₀}_{key}``
    with value (n_seeds, T) for the norm keys and ``_mhalf`` (n_seeds,).

Placement (``mup.pool``): the runs of one (width, δ₀-independent group)
form one ``vmap`` batch per observable, split over the GPUs (``--gpus``),
``--workers`` CPU processes or this process as ``run_weight_norms.py``;
the half-life jobs follow the norm jobs in a second pool.
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
from mup.config import Config, READOUT_ORDER, Run, is_gated, make_runs
from run_Nscaling import _stat, mhalf_from_signal
from run_weight_norms import ALL_KEYS, QUANT, Q_SCALED, fig_vs_N, q_scaled

_T0 = time.perf_counter()        # wall clock from after the imports to the end of the script


# ----------------------------------------------------------------------------
# jobs: one batch per (group, T) for the norms, another for the half-life signal
# ----------------------------------------------------------------------------
def groups(runs: list[Run]) -> dict[tuple, list[Run]]:
    """``{(N, D, solver, n_steps, closed, T): runs}`` over every readout, κ, δ₀ and seed, largest N first."""
    g = {}
    for r in runs:
        g.setdefault(X.group_key(r) + (r.T_burn + r.T_meas,), []).append(r)
    return {k: g[k] for k in sorted(g, key=lambda k: -k[0])}


def make_jobs(groups_: dict, pieces, kind: str) -> list:
    """One job per piece of a group: ``kind = "norms"`` (``norms_over_tasks``) or ``"mhalf"`` (``signal_after_tasks``)."""
    jobs = []
    for (key, runs), k in zip(groups_.items(), pieces):
        T = key[-1]
        for i, part in enumerate(X.split(runs, k)):
            if kind == "norms":
                jobs.append(((key, i), {"runs": part, "T": T}, X.run_cost(part[0], T) * len(part)))
            else:
                jobs.append(((key, i), {"runs": part, "T_pre": T, "n_follow": T + 1, "k": T},
                             X.run_cost(part[0], 2 * T + 1) * len(part)))
    return jobs


def cond_key(r: Run) -> tuple:
    """(N, κ label, readout, δ₀) of a run."""
    return (r.N, r.kappa_label, r.readout, r.delta0)


def collect(runs: list[Run], plan_norms: pool.Plan, plan_mhalf: pool.Plan, log=print) -> dict[tuple, dict]:
    """``{(N, κ label, readout, δ₀): {key: (n_seeds, T) for key in ALL_KEYS, "mhalf": (n_seeds,)}}``."""
    g = groups(runs)
    out = {}
    per_run: dict[tuple, dict] = {}
    log("weight norms:")
    jobs = make_jobs(g, plan_norms.pieces, "norms")
    res = pool.run_jobs("mup.experiments:norms_over_tasks", jobs, log=log, **plan_norms.place)
    for (_, kw, _), d in zip(jobs, res):
        for j, r in enumerate(kw["runs"]):
            per_run[(cond_key(r), r.seed)] = {q: d[q][j] for q in ALL_KEYS}
    log("half-life:")
    jobs = make_jobs(g, plan_mhalf.pieces, "mhalf")
    res = pool.run_jobs("mup.experiments:signal_after_tasks", jobs, log=log, **plan_mhalf.place)
    for (_, kw, _), S in zip(jobs, res):
        mh = mhalf_from_signal(S)
        for j, r in enumerate(kw["runs"]):
            per_run[(cond_key(r), r.seed)]["mhalf"] = mh[j]
    for key in sorted({k for k, _ in per_run}, key=lambda k: tuple(str(x) for x in k)):
        seeds = sorted(s for k, s in per_run if k == key)
        out[key] = {q: np.stack([per_run[(key, s)][q] for s in seeds]) for q in ALL_KEYS}
        out[key]["mhalf"] = np.array([per_run[(key, s)]["mhalf"] for s in seeds])
    for key, d in out.items():
        r = next(rr for rr in runs if cond_key(rr) == key)
        ss = {q: np.nanmean(d[q][:, r.T_burn:], 1) for q, _ in QUANT}
        m, se = _stat(d["mhalf"])
        print(f"  N={key[0]:4d} κ={key[1]:5s} {key[2]:15s} δ₀={key[3]:5g}  steady mean Q = {np.nanmean(ss['Q_all']):7.3f}  "
              f"mean a² = {np.nanmean(ss['a2_all']):7.3f}  |u|²/D = {np.nanmean(ss['u2']):7.3f}  "
              f"m½ = {m:7.2f} ± {se:5.2f}  censored or infeasible {int(np.isnan(d['mhalf']).sum())}/{len(d['mhalf'])}",
              flush=True)
    return out


# ----------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------
def _ss(res: dict, key: tuple, q: str, run: Run) -> np.ndarray:
    """Per-seed steady-state value of quantity ``q`` (the collapse variable for ``Q_SCALED``) of a condition."""
    if q == Q_SCALED[0]:
        return q_scaled(np.nanmean(res[key]["Q_all"][:, run.T_burn:], 1), run)
    return np.nanmean(res[key][q][:, run.T_burn:], 1)


def _points(values_of, Ns: list[int]) -> tuple:
    """(x, mean, se) over the N with finite per-seed values from ``values_of(N) -> (n_seeds,) or None``."""
    pts = []
    for N in Ns:
        v = values_of(N)
        if v is None:
            continue
        v = v[np.isfinite(v)]
        if len(v):
            pts.append((N, v.mean(), v.std(ddof=1) / math.sqrt(len(v)) if len(v) > 1 else 0.0))
    return tuple(map(np.array, zip(*pts))) if pts else (np.array([]), np.array([]), np.array([]))


def d0_handles(d0s: list[float]) -> list:
    """Legend handles for δ₀ traces (grey shades, lightest = smallest δ₀)."""
    return [P.Line2D([], [], color=P.kappa_shades(i, len(d0s)), lw=P.LW_TRACE, label=f"δ₀ = {P.fmt(d)}")
            for i, d in enumerate(d0s)]


def fig_norms_by_delta(res: dict, Ns: list[int], klabels: list[str], d0s: list[float], ro: str, runs: list[Run],
                       path: str) -> str:
    """``norms_vs_N`` of one readout: rows = quantities, columns = κ, x = N, traces = δ₀ (shades)."""
    P.apply_style()
    rows = QUANT + (Q_SCALED,)
    fig, axes = P.grid(len(rows), len(klabels), sharex="all", sharey="row", panel=(3.0, 2.4))
    figdata = {}
    for c, klabel in enumerate(klabels):
        k = klabel if is_gated(ro) else "sqrtN"
        for r, (q, lab) in enumerate(rows):
            ax = axes[r, c]
            for i, d0 in enumerate(d0s):
                def values(N, k=k, d0=d0, q=q):
                    key = (N, k, ro, d0)
                    if key not in res:
                        return None
                    run = next(rr for rr in runs if cond_key(rr) == key)
                    return _ss(res, key, q, run)
                x, m, se = _points(values, Ns)
                if not len(x):
                    continue
                col = P.kappa_shades(i, len(d0s), ro)
                ax.plot(x, m, color=col, marker="o", ms=3, lw=P.LW_TRACE, ls="-" if is_gated(ro) else "--")
                ax.errorbar(x, m, yerr=se, fmt="none", ecolor=col, elinewidth=0.8, capsize=2, zorder=3)
                figdata[f"{q}_k{klabel}_d0{P.fmt(d0)}"] = np.stack([x, m, se])
            P.logaxis(ax, "x")
            P.discrete_ticks(ax, Ns, "x")
            if r == len(rows) - 1:
                ax.set_xlabel("$N$")
        axes[0, c].set_title(f"κ = {klabel}" if is_gated(ro) else f"(κ = {klabel}: no κ for this readout)")
    for r, (q, lab) in enumerate(rows):
        axes[r, 0].set_ylabel(lab)
    r0 = runs[0]
    P.cell_title(fig, lam=r0.lam, solver=r0.solver, eta=r0.eta, extra=P.READOUT_LABEL[ro])
    P.caption(fig, f"{P.READOUT_LABEL[ro]}: steady-state norms (mean over the measurement window t ≥ T_burn) of "
                   "mean_i Q_ii (top), mean_i a_i² (second) and |u_t|²/D (third), seed mean with ±1 s.e. bars; "
                   "bottom: the collapse Q̄/pN (frozen) or Q̄²/pN (learned), pN = κ√N or N when ungated; "
                   "traces = δ₀, lightest = smallest.")
    P.fig_legend(fig, None, handles=d0_handles(d0s))
    return P.save(fig, path, figdata=figdata)


def fig_halflife(res: dict, Ns: list[int], klabels: list[str], d0s: list[float], readouts: list[str],
                 runs: list[Run], by: str, path: str) -> str:
    """m½ vs N: columns = κ; ``by = "delta0"``: rows = readouts, traces = δ₀; ``by = "model"``: rows = δ₀, traces = readouts."""
    P.apply_style()
    row_keys = readouts if by == "delta0" else d0s
    fig, axes = P.grid(len(row_keys), len(klabels), sharex="all", sharey="all", panel=(3.0, 2.3))
    figdata = {}
    for c, klabel in enumerate(klabels):
        for r, row in enumerate(row_keys):
            ax = axes[r, c]
            traces = d0s if by == "delta0" else readouts
            for i, tr in enumerate(traces):
                ro, d0 = (row, tr) if by == "delta0" else (tr, row)
                k = klabel if is_gated(ro) else "sqrtN"
                x, m, se = _points(lambda N: res[(N, k, ro, d0)]["mhalf"] if (N, k, ro, d0) in res else None, Ns)
                if not len(x):
                    continue
                if by == "delta0":
                    col = P.kappa_shades(i, len(d0s), ro)
                    ax.plot(x, m, color=col, marker="o", ms=3, lw=P.LW_TRACE, ls="-" if is_gated(ro) else "--")
                else:
                    col = P.READOUT_COLOR[ro]
                    P.plot_readout(ax, x, m, ro, marker="o", ms=3)
                ax.errorbar(x, m, yerr=se, fmt="none", ecolor=col, elinewidth=0.8, capsize=2, zorder=3)
                figdata[f"mhalf_k{klabel}_{ro}_d0{P.fmt(d0)}"] = np.stack([x, m, se])
            xs = np.array(Ns, float)
            ax.plot(xs, xs * math.log(2), color=P.GREY_REF, ls=":", lw=1.0, zorder=1)
            P.logaxis(ax, "x"); P.logaxis(ax, "y")
            P.discrete_ticks(ax, Ns, "x")
            if r == len(row_keys) - 1:
                ax.set_xlabel("$N$")
            if c == 0:
                ax.set_ylabel((P.READOUT_LABEL[row] if by == "delta0" else f"δ₀ = {P.fmt(row)}") + "\n$m_{1/2}$ (tasks)")
        axes[0, c].set_title(f"κ = {klabel}")
    r0 = runs[0]
    P.cell_title(fig, lam=r0.lam, solver=r0.solver, eta=r0.eta,
                 extra="half-life by " + ("δ₀" if by == "delta0" else "readout"))
    P.caption(fig, "steady-state half-life m½ (first age with S_T(m) ≤ S_T(0)/2 after T = T_burn + T_meas tasks, "
                   "per seed, then mean ± 1 s.e. over the uncensored seeds) against N; "
                   + ("rows = readouts, traces = δ₀ (lightest = smallest)" if by == "delta0"
                      else "rows = δ₀, traces = readouts")
                   + "; the ungated readouts have no κ and repeat in every column; dotted grey: N ln 2.")
    if by == "delta0":
        P.fig_legend(fig, None, handles=d0_handles(d0s))
    else:
        P.fig_legend(fig, readouts)
    return P.save(fig, path, figdata=figdata)


def main(argv=None) -> int:
    """Command-line entry point (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(_path.ROOT, "configs", "stage1.yaml"))
    ap.add_argument("--N", type=int, nargs="+", default=[5, 10, 20, 50, 100])
    ap.add_argument("--kappa", nargs="+", default=["1", "2", "4"])
    ap.add_argument("--delta0", type=float, nargs="+", default=[0.0, 1.0, 2.0, 4.0])
    ap.add_argument("--lam", type=float, default=None, help="default: the config's first λ")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--readout", nargs="*", default=list(READOUT_ORDER))
    ap.add_argument("--out", default=os.path.join(_path.ROOT, "results", "imbalance_results"))
    pool.add_args(ap)
    args = ap.parse_args(argv)

    cfg = Config.from_yaml(args.config)
    over = {"name": "imbalance_comp", "N": list(args.N), "kappa": list(args.kappa), "readout": list(args.readout),
            "delta0": [float(d) for d in args.delta0], "lam": [args.lam if args.lam is not None else cfg.lam[0]],
            "seeds": args.seeds, "extra": (), "trace": False}
    cfg = Config(**{**cfg.__dict__, **over})
    runs = make_runs(cfg)
    run0 = runs[0]
    g = groups(runs)
    plan_norms = pool.plan(args, [(v, k[-1]) for k, v in g.items()], out_per_task=8 * len(ALL_KEYS))
    plan_mhalf = pool.plan(args, [(v, 2 * k[-1] + 1) for k, v in g.items()])
    d0s = sorted(set(cfg.delta0))
    klabels = [str(k) for k in args.kappa]
    readouts = [ro for ro in READOUT_ORDER if ro in args.readout]
    print(f"imbalance comparison: N={args.N} κ={args.kappa} δ₀={d0s} λ={run0.lam} {run0.solver} seeds={args.seeds}  "
          f"[{plan_norms.describe()}]", flush=True)
    res = collect(runs, plan_norms, plan_mhalf)

    outs = []
    d_by = os.path.join(args.out, "weight_norm_by_delta")
    for ro in readouts:
        outs.append(fig_norms_by_delta(res, list(args.N), klabels, d0s, ro, runs, os.path.join(d_by, f"norms_vs_N_{ro}")))
    d_model = os.path.join(args.out, "weight_norm_by_model")
    for d0 in d0s:
        res_d0 = {(N, k, ro): d for (N, k, ro, dd), d in res.items() if dd == d0}
        runs_d0 = [r for r in runs if r.delta0 == d0]
        outs.append(fig_vs_N(res_d0, list(args.N), klabels, runs_d0, os.path.join(d_model, f"norms_vs_N_d0{P.fmt(d0)}")))
    outs.append(fig_halflife(res, list(args.N), klabels, d0s, readouts, runs, "delta0", os.path.join(args.out, "halflife_by_delta0")))
    outs.append(fig_halflife(res, list(args.N), klabels, d0s, readouts, runs, "model", os.path.join(args.out, "halflife_by_model")))
    np.savez_compressed(os.path.join(args.out, "imbalance_comp.npz"),
                        **{f"N{N}_k{k}_{ro}_d0{P.fmt(d0)}_{q}": v for (N, k, ro, d0), d in res.items() for q, v in d.items()})
    print("wrote\n  " + "\n  ".join(outs) + f"\n  {os.path.join(args.out, 'imbalance_comp.npz')}")
    return 0


if __name__ == "__main__":
    _rc = main()
    _dt = time.perf_counter() - _T0
    print(f"run_imbalance_comp: total {_dt:.1f} s ({_dt / 60:.2f} min) from after the imports to the end", flush=True)
    sys.exit(_rc)
