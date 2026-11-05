#!/usr/bin/env python
"""Weight norms against tasks and against width (PLAN.md §12.1, §12.3), NumPy, one run per process.

    python scripts/run_weight_norms.py [--N 5 10 20 50 100] [--kappa 1 2 4] [--delta0 0] [--lam 4] [--seeds 4] [--workers W]

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
``norms_vs_N_…``           rows = the same three quantities, columns = κ,
    x = N (log), y = the steady-state value (mean over the measurement
    window t ≥ T_burn, then seed mean ± 1 s.e. bars), traces = readouts;
    the ungated readouts repeat in every column.

The per-run curves are written to ``norms_….npz`` beside the figures.
The dynamics run through ``mup.npmodel`` (no JAX); every run is one job
of a ``ProcessPoolExecutor`` of ``--workers`` processes (default:
``psutil.cpu_count(logical=False)``; 1 runs everything in this process),
largest N first.
"""
import argparse
import math
import warnings
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

import _path  # noqa: F401

from mup import plotting as P
from mup import npmodel
from mup.config import Config, READOUT_ORDER, is_gated, make_runs

_T0 = time.perf_counter()        # wall clock from after the imports to the end of the script

#: Recorded per task: (key, y-axis label)
QUANT = (("Q_all", r"mean $Q_{ii} = |W_i|^2/D$"), ("a2_all", r"mean $a_i^2$"), ("u2", r"$|u_t|^2/D$"))
ALL_KEYS = ("Q_all", "Q_act", "a2_all", "a2_act", "u2")


def norms_run(run) -> dict:
    """Per-task norms of one run: ``{key: (T,) array}`` for ``ALL_KEYS``; NaN arrays when infeasible."""
    T = run.T_burn + run.T_meas
    out = {k: np.full(T, np.nan) for k in ALL_KEYS}
    rng, W, a = npmodel.init_run(run)
    if not np.isfinite(a).all():
        return out                                            # protocol.feasible: δ₀ ≤ −min_i Q0_i
    for t, W, a, g, w, fcum, Q, u in npmodel.tasks(run, rng, W, a, T):
        # Q and u come tracked from the active block: nothing here reads W (PLAN: SCALING_N10k.md step A)
        a2 = a * a
        n_act = g.sum()
        out["Q_all"][t] = Q.mean()
        out["a2_all"][t] = a2.mean()
        out["Q_act"][t] = (g * Q).sum() / n_act if n_act > 0 else np.nan
        out["a2_act"][t] = (g * a2).sum() / n_act if n_act > 0 else np.nan
        out["u2"][t] = (u @ u) / W.shape[1]                   # effective weight of the task just trained
    return out


def physical_cores() -> int:
    """Physical (not logical) core count: ``psutil.cpu_count(logical=False)``, else ``sysctl`` on macOS, else ``os.cpu_count()``."""
    try:
        import psutil
        n = psutil.cpu_count(logical=False)
        if n:
            return int(n)
    except ImportError:
        pass
    if sys.platform == "darwin":
        try:
            return int(subprocess.check_output(["sysctl", "-n", "hw.physicalcpu"]).decode().strip())
        except (OSError, ValueError, subprocess.CalledProcessError):
            pass
    return os.cpu_count() or 1


def _job(key: tuple, run) -> tuple:
    """One run in a worker process: ``(key, norms)``."""
    return key, norms_run(run)


def collect(runs: list, workers: int = 1) -> dict:
    """``{(N, kappa_label, readout): {key: (n_seeds, T) array}}`` for every condition of ``runs``."""
    cond = {}
    for r in runs:
        cond.setdefault((r.N, r.kappa_label, r.readout), []).append(r)
    jobs = [((r.N, r.kappa_label, r.readout, r.seed), r) for r in runs]
    jobs.sort(key=lambda j: (-j[0][0], READOUT_ORDER.index(j[0][2]), j[0][1], j[0][3]))   # largest N first
    per_seed = {}
    if workers <= 1:
        for k, r in jobs:
            _, v = _job(k, r)
            per_seed.setdefault(k[:3], {})[k[3]] = v
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_job, k, r) for k, r in jobs]
            for f in as_completed(futs):
                (N, k, ro, seed), v = f.result()
                per_seed.setdefault((N, k, ro), {})[seed] = v
    out = {}
    for key in sorted(cond, key=lambda k: (k[0], READOUT_ORDER.index(k[2]), k[1])):
        b = cond[key]
        out[key] = {q: np.stack([per_seed[key][r.seed][q] for r in b]) for q in ALL_KEYS}
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
    fig, axes = P.grid(len(QUANT), len(klabels), sharex="all", sharey="row", panel=(3.0, 2.4))
    figdata = {}
    readouts = []
    for c, klabel in enumerate(klabels):
        for r, (q, lab) in enumerate(QUANT):
            ax = axes[r, c]
            for ro in READOUT_ORDER:
                k = klabel if is_gated(ro) else "sqrtN"
                pts = []
                for N in Ns:
                    if (N, k, ro) not in res:
                        continue
                    run = next(rr for rr in run0 if rr.N == N)
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
            if r == len(QUANT) - 1:
                ax.set_xlabel("$N$")
        axes[0, c].set_title(f"κ = {klabel}")
    for r, (q, lab) in enumerate(QUANT):
        axes[r, 0].set_ylabel(lab)
    r0 = run0[0]
    P.cell_title(fig, d0=r0.delta0, lam=r0.lam, solver=r0.solver, eta=r0.eta)
    P.caption(fig, "steady-state norms: mean over the measurement window (t ≥ T_burn) of mean_i Q_ii (top), "
                   "mean_i a_i² (middle) and the effective weight norm |u_t|²/D, u_t = (g ⊙ a) W / pN (bottom), "
                   "then seed mean with ±1 s.e. bars; the ungated readouts have no κ and "
                   "repeat in every column.")
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
    ap.add_argument("--workers", type=int, default=None,
                    help="processes in the pool, one run each (default: psutil physical cores; 1: in-process)")
    ap.add_argument("--out", default=os.path.join(_path.ROOT, "results", "weight_norms"))
    args = ap.parse_args(argv)

    cfg = Config.from_yaml(args.config)
    over = {"name": "weight_norms", "N": list(args.N), "kappa": list(args.kappa), "readout": list(args.readout),
            "delta0": [args.delta0], "lam": [args.lam if args.lam is not None else cfg.lam[0]],
            "seeds": args.seeds, "extra": (), "trace": False}
    cfg = Config(**{**cfg.__dict__, **over})
    runs = make_runs(cfg)
    run0 = runs[0]
    workers = args.workers or physical_cores()
    print(f"weight norms: N={args.N} κ={args.kappa} δ₀={run0.delta0} λ={run0.lam} {run0.solver} seeds={args.seeds} "
          f"workers={workers}", flush=True)
    res = collect(runs, workers)

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
