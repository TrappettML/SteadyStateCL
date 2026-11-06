#!/usr/bin/env python
"""Weight norms against tasks and against width (PLAN.md §12.1, §12.3).

    python scripts/run_weight_norms.py [--N 5 10 20 50 100] [--kappa 1 2 4] [--delta0 0] [--lam 4] [--seeds 4] [--gpus auto]

For every (N, κ, readout, seed) the network trains the Stage-1 stream of
T = T_burn + T_meas tasks (14N under ``configs/stage1.yaml``) and, after
every task, records the hidden-layer norm mean_i Q_ii = mean_i |W_i|²/D
(= tr(WWᵀ)/(N D)) and the output-layer norm mean_i a_i², both over all
N units (``_all``) and over the task's active units (``_act``), and the
effective weight norm |u_t|²/D of the task just trained, u_t = (g ⊙ a) W / pN
(the gate g of that task; for task heads ``a`` is its head, as in
``run_Nscaling.mhalf_run``).  Figures in ``results/weight_norms/figures/``:

``norms_vs_task_k{κ}_…``   rows = (mean Q_ii, mean a_i², |u_t|²/D), columns = N,
    traces = readouts at the first κ listed (ungated readouts have no κ),
    x = task; seed mean with a ±1 s.e. band, burn-in shaded.
``norms_vs_N_…``           rows = the same three quantities plus the scaling
    collapse Q̄/pN for the frozen readouts and Q̄²/pN for the learned ones
    (Q̄ the steady-state mean_i Q_ii of the top row, pN = κ√N, or N when
    ungated; a constant in N confirms the mechanism, and a frozen constant
    with a drifting learned one points at extra dynamics in a), columns = κ,
    x = N (log), y = the steady-state value (mean over the measurement
    window t ≥ T_burn, then seed mean ± 1 s.e. bars), traces = readouts;
    the ungated readouts repeat in every column.

The per-run curves are written to ``norms_….npz`` beside the figures.

Placement (``mup.pool``): the runs of one width — every κ, readout and
seed, which share (N, D, n_steps, T) — form one ``vmap`` batch advanced
``--chunk`` tasks per device call with the norms evaluated inside the scan
(``experiments.norms_over_tasks``, the JAX kernels of the sweep) and runs
as a job, largest N first, over the GPUs (``--gpus``: the four cards,
several processes each, a width split only as far as needed to fill them
— ``pool.plan``), over ``--workers`` CPU processes, or in this process.
"""
import argparse
import math
import warnings
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

#: Recorded per task: (key, y-axis label)
QUANT = (("Q_all", r"mean $Q_{ii} = |W_i|^2/D$"), ("a2_all", r"mean $a_i^2$"), ("u2", r"$|u_t|^2/D$"))
ALL_KEYS = X.NORM_KEYS                      # ("Q_all", "Q_act", "a2_all", "a2_act", "u2")
#: Fourth row of ``norms_vs_N``: the scaling collapse of Q̄ = mean Q_ii (derived from ``Q_all`` per seed)
Q_SCALED = ("Q_scaled", r"$\bar Q/pN$ (frozen),  $\bar Q^2/pN$ (learned)")


def q_scaled(ss_Q: np.ndarray, run) -> np.ndarray:
    """Collapse variable of one condition: Q̄/pN when the readout is frozen, Q̄²/pN when it learns.

    Parameters
    ----------
    ss_Q : (n_seeds,) per-seed steady-state mean Q_ii
    run : config.Run of the condition (supplies ``pN`` and ``learn_a``)
    """
    pN = float(run.pN)
    return ss_Q / pN if run.learn_a == 0.0 else ss_Q * ss_Q / pN


def groups(runs: list) -> dict:
    """``{(N, D, solver, n_steps, T): runs}``, largest N first."""
    g = {}
    for r in runs:
        g.setdefault(X.group_key(r) + (r.T_burn + r.T_meas,), []).append(r)
    return {k: g[k] for k in sorted(g, key=lambda k: -k[0])}


def make_jobs(groups_: dict, pieces) -> list:
    """One job per piece of a (N, D, n_steps, T) group: the norms after each of its T tasks."""
    jobs = []
    for (key, runs), k in zip(groups_.items(), pieces):
        T = key[-1]
        for i, part in enumerate(X.split(runs, k)):
            jobs.append(((key, i), {"runs": part, "T": T}, X.run_cost(part[0], T) * len(part)))
    return jobs


def collect(runs: list, plan: pool.Plan, log=print) -> dict:
    """``{(N, kappa_label, readout): {key: (n_seeds, T) array}}`` for every condition of ``runs``."""
    jobs = make_jobs(groups(runs), plan.pieces)
    res = pool.run_jobs("mup.experiments:norms_over_tasks", jobs, log=log, **plan.place)
    per_run = {}
    for (_, kw, _), d in zip(jobs, res):
        for j, r in enumerate(kw["runs"]):
            per_run[(r.N, r.kappa_label, r.readout, r.seed)] = {q: d[q][j] for q in ALL_KEYS}
    cond = {}
    for r in runs:
        cond.setdefault((r.N, r.kappa_label, r.readout), []).append(r)
    out = {}
    for key in sorted(cond, key=lambda k: (k[0], READOUT_ORDER.index(k[2]), k[1])):
        b = cond[key]
        out[key] = {q: np.stack([per_run[key + (r.seed,)][q] for r in b]) for q in ALL_KEYS}
        T_burn = b[0].T_burn
        ss = {q: np.nanmean(out[key][q][:, T_burn:], 1) for q, _ in QUANT}
        n_ok = int(np.isfinite(ss["Q_all"]).sum())
        print(f"  N={key[0]:4d} κ={key[1]:5s} {key[2]:15s} T={out[key]['Q_all'].shape[1]:5d}  "
              f"steady mean Q = {np.nanmean(ss['Q_all']):7.3f}  mean a² = {np.nanmean(ss['a2_all']):7.3f}  "
              f"|u|²/D = {np.nanmean(ss['u2']):7.3f}  "
              f"infeasible {len(b) - n_ok}/{len(b)}", flush=True)
    return out


def _mean_se(Y: np.ndarray) -> tuple:
    """Seed mean and s.e. along axis 0 over the finite entries (NaN where none, s.e. 0 where one)."""
    n = np.isfinite(Y).sum(0)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = np.nanmean(Y, 0)
        sd = np.nanstd(Y, 0, ddof=1)
    se = np.where(n > 1, sd / np.sqrt(np.maximum(n, 1)), 0.0)
    return m, se


def fig_vs_task(res: dict, Ns: list, klabel: str, run0, path: str) -> str:
    """Rows = quantities, columns = N, traces = readouts at κ = ``klabel``; seed mean ± 1 s.e. band."""
    P.apply_style()
    Ns = [N for N in Ns if any(k[0] == N for k in res)]
    fig, axes = P.grid(len(QUANT), len(Ns), sharex="col", sharey="row", panel=(2.6, 2.0))
    figdata = {}
    readouts = []
    n_inf = 0
    for c, N in enumerate(Ns):
        for r, (q, lab) in enumerate(QUANT):
            ax = axes[r, c]
            for ro in READOUT_ORDER:
                k = klabel if is_gated(ro) else "sqrtN"
                if (N, k, ro) not in res:
                    continue
                if ro not in readouts:
                    readouts.append(ro)
                Y = res[(N, k, ro)][q]
                if r == 0:
                    n_inf += int((~np.isfinite(Y).any(1)).sum())
                m, se = _mean_se(Y)
                x = np.arange(1, Y.shape[1] + 1)
                P.band(ax, x, m - se, m + se, P.READOUT_COLOR[ro])
                P.plot_readout(ax, x, m, ro, zorder=4)
                figdata[f"{q}_N{N}_{ro}"] = np.stack([x, m, se])
            run = next(rr for rr in run0 if rr.N == N)
            P.shade_burnin(ax, run.T_burn, run.T_meas)
            if r == len(QUANT) - 1:
                ax.set_xlabel("task")
        axes[0, c].set_title(f"N = {N}")
    for r, (q, lab) in enumerate(QUANT):
        axes[r, 0].set_ylabel(lab)
    r0 = run0[0]
    P.cell_title(fig, kappa=klabel, d0=r0.delta0, lam=r0.lam, solver=r0.solver, eta=r0.eta)
    P.caption(fig, f"after each task: mean over all N units of Q_ii = |W_i|²/D (top) and of a_i² (middle; the head "
                   f"just trained for task heads), and the effective weight norm |u_t|²/D of the task just trained, "
                   f"u_t = (g ⊙ a) W / pN (bottom); seed mean, band ±1 s.e. over seeds; grey: burn-in, dotted: half of "
                   f"the measurement window; the ungated readouts have no κ; {n_inf} infeasible seeds dropped.")
    P.fig_legend(fig, readouts)
    return P.save(fig, path, figdata=figdata)


def fig_vs_N(res: dict, Ns: list, klabels: list, run0, path: str) -> str:
    """Rows = quantities, columns = κ, x = N (log), y = steady-state value (seed mean ± 1 s.e.), traces = readouts."""
    P.apply_style()
    rows = QUANT + (Q_SCALED,)
    fig, axes = P.grid(len(rows), len(klabels), sharex="all", sharey="row", panel=(3.0, 2.4))
    figdata = {}
    readouts = []
    for c, klabel in enumerate(klabels):
        for r, (q, lab) in enumerate(rows):
            ax = axes[r, c]
            for ro in READOUT_ORDER:
                k = klabel if is_gated(ro) else "sqrtN"
                pts = []
                for N in Ns:
                    if (N, k, ro) not in res:
                        continue
                    run = next(rr for rr in run0 if rr.N == N and rr.readout == ro and rr.kappa_label == k)
                    if q == Q_SCALED[0]:
                        ss = q_scaled(np.nanmean(res[(N, k, ro)]["Q_all"][:, run.T_burn:], 1), run)
                    else:
                        ss = np.nanmean(res[(N, k, ro)][q][:, run.T_burn:], 1)     # per-seed steady-state value
                    v = ss[np.isfinite(ss)]
                    if len(v):
                        pts.append((N, v.mean(), v.std(ddof=1) / math.sqrt(len(v)) if len(v) > 1 else 0.0))
                if not pts:
                    continue
                if ro not in readouts:
                    readouts.append(ro)
                x, m, se = map(np.array, zip(*pts))
                P.plot_readout(ax, x, m, ro, marker="o", ms=3)
                ax.errorbar(x, m, yerr=se, fmt="none", ecolor=P.READOUT_COLOR[ro], elinewidth=0.8, capsize=2, zorder=3)
                figdata[f"{q}_k{klabel}_{ro}"] = np.stack([x, m, se])
            P.logaxis(ax, "x")
            P.discrete_ticks(ax, Ns, "x")
            if r == len(rows) - 1:
                ax.set_xlabel("$N$")
        axes[0, c].set_title(f"κ = {klabel}")
    for r, (q, lab) in enumerate(rows):
        axes[r, 0].set_ylabel(lab)
    r0 = run0[0]
    P.cell_title(fig, d0=r0.delta0, lam=r0.lam, solver=r0.solver, eta=r0.eta)
    P.caption(fig, "steady-state norms: mean over the measurement window (t ≥ T_burn) of mean_i Q_ii (top), "
                   "mean_i a_i² (second) and the effective weight norm |u_t|²/D, u_t = (g ⊙ a) W / pN (third), "
                   "then seed mean with ±1 s.e. bars; bottom: the scaling collapse of the top row, Q̄/pN for the "
                   "frozen readouts and Q̄²/pN for the learned ones (pN = κ√N, or N when ungated) — flat in N "
                   "confirms the mechanism; the ungated readouts have no κ and repeat in every column.")
    P.fig_legend(fig, readouts, marker=True)
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
    ap.add_argument("--out", default=os.path.join(_path.ROOT, "results", "weight_norms"))
    pool.add_args(ap)
    args = ap.parse_args(argv)

    cfg = Config.from_yaml(args.config)
    over = {"name": "weight_norms", "N": list(args.N), "kappa": list(args.kappa), "readout": list(args.readout),
            "delta0": [args.delta0], "lam": [args.lam if args.lam is not None else cfg.lam[0]],
            "seeds": args.seeds, "extra": (), "trace": False}
    cfg = Config(**{**cfg.__dict__, **over})
    runs = make_runs(cfg)
    run0 = runs[0]
    plan = pool.plan(args, [(g, key[-1]) for key, g in groups(runs).items()], out_per_task=8 * len(ALL_KEYS))
    print(f"weight norms: N={args.N} κ={args.kappa} δ₀={run0.delta0} λ={run0.lam} {run0.solver} seeds={args.seeds}  "
          f"[{plan.describe()}]", flush=True)
    res = collect(runs, plan)

    klabels = [P.fmt(k) for k in args.kappa]
    fdir = os.path.join(args.out, "figures")
    tag = f"d0{P.fmt(run0.delta0)}_lam{P.fmt(run0.lam)}"
    f1 = fig_vs_task(res, list(args.N), klabels[0], runs, os.path.join(fdir, f"norms_vs_task_k{klabels[0]}_{tag}"))
    f2 = fig_vs_N(res, list(args.N), klabels, runs, os.path.join(fdir, f"norms_vs_N_{tag}"))
    np.savez_compressed(os.path.join(fdir, f"norms_{tag}.npz"),
                        **{f"N{N}_k{k}_{ro}_{q}": v for (N, k, ro), d in res.items() for q, v in d.items()})
    print(f"wrote {f1}\n      {f2}")
    return 0


if __name__ == "__main__":
    _rc = main()
    _dt = time.perf_counter() - _T0
    print(f"run_weight_norms: total {_dt:.1f} s ({_dt / 60:.2f} min) from after the imports to the end", flush=True)
    sys.exit(_rc)
