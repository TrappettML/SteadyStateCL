#!/usr/bin/env python
"""Steady-state half-life against width (PLAN.md §12.13): m½ from S_T(m)/S_T(0) vs N.

    python scripts/run_Nscaling.py [--N 5 10 20 50 100] [--kappa 1 2 4] [--delta0 0] [--lam 4] [--seeds 4] [--workers W]

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

Every run (N, κ, readout, seed) is independent, so each is one job of
a ``ProcessPoolExecutor`` of ``--workers`` processes (default: the
number of physical cores; 1 runs everything in this process), each
pinned to one XLA thread as ``mup.driver`` does for the sweep; the
per-seed half-lives are gathered per condition afterwards.  Largest N
first, so the long runs do not trail at the end.  The workers share a
JAX persistent compilation cache in ``<out>/jax_cache`` (safe to delete).
"""
import argparse
import math
import multiprocessing as mp
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import jax
import jax.numpy as jnp
import numpy as np

import _path  # noqa: F401

from mup import plotting as P
from mup.config import Config, READOUT_ORDER, is_gated, make_runs
from mup.observe import age_schedule
from mup.driver import PINNED_ENV
from mup.protocol import batched_init, batched_run_task, feasible, make_params
from run_stask import _signal_k
from run_szero import half_life

GATED = tuple(ro for ro in READOUT_ORDER if is_gated(ro))


def theory(N: float, kappa: float, kind: str) -> float:
    """Equation half-life: ``nln2`` → N ln 2; ``nk2`` → N ln 2 / κ²."""
    return N * math.log(2) / (kappa ** 2 if kind == "nk2" else 1.0)


def mhalf_batch(batch: list) -> np.ndarray:
    """Per-seed half-lives of one homogeneous batch: train T tasks, follow teacher T for T more.

    Parameters
    ----------
    batch : list of config.Run
        Same (N, κ, readout); one entry per seed.

    Returns
    -------
    (n_seeds,) array
        m½ in tasks, NaN where censored (no crossing within T tasks).
    """
    r0 = batch[0]
    T = r0.T_burn + r0.T_meas
    p = make_params(batch)
    keys = jnp.stack([jax.random.PRNGKey(r.rng_seed()) for r in batch])
    state = batched_init(r0.N, r0.D, 2 * T + 1)(keys, p)
    ages = jnp.asarray(age_schedule(r0.N, r0.kappa_s, T, r0.n_ages_geom, r0.m_max_factor))
    step = batched_run_task(r0.n_steps, r0.solver)
    for _ in range(T):
        state, _ = step(state, p, ages)
    S = np.zeros((T + 1, len(batch)))
    for m in range(T + 1):
        state, _ = step(state, p, ages)
        S[m] = np.asarray(_signal_k(state, p, T))
    return np.array([half_life(S[:, j]) for j in range(len(batch))])


def physical_cores() -> int:
    """Physical (not logical) core count: psutil, else ``sysctl`` on macOS, else ``os.cpu_count()``."""
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
    """One run (one seed) in a worker process: ``(key, m½)``; NaN when the run is infeasible (δ₀ ≤ −min_i Q0_i)."""
    if not feasible([run])[0]:
        return key, float("nan")
    return key, float(mhalf_batch([run])[0])


def _report(key, mh, T):
    if mh is None:
        print(f"  N={key[0]} κ={key[1]} {key[2]}: every seed infeasible, skipped", flush=True)
        return
    n_ok = int(np.isfinite(mh).sum())
    se = np.nanstd(mh, ddof=1) / math.sqrt(n_ok) if n_ok > 1 else 0.0
    print(f"  N={key[0]:4d} κ={key[1]:5s} {key[2]:15s} T={T:5d}  m½ = {np.nanmean(mh):7.2f} ± {se:5.2f}"
          f"  censored or infeasible {int(np.isnan(mh).sum())}/{len(mh)}", flush=True)


def collect(runs: list, workers: int = 1, cache_dir: str = "") -> dict:
    """``{(N, kappa_label, readout): (n_seeds,) m½ array}`` for every condition of ``runs``.

    Parameters
    ----------
    runs : list of config.Run
    workers : int
        Processes in the pool, one run (seed) per job; 1 runs the jobs
        here, in order.  Each worker is pinned to one XLA thread
        (``driver.PINNED_ENV``), so ``workers`` should not exceed the
        physical core count.
    cache_dir : str
        JAX persistent compilation cache shared by the workers (empty: none).
        Every worker compiles the same few programs (one per N); with the
        cache the first to finish writes them and the rest load them, which
        also silences XLA's "constant folding is taking > 1s" alarm on a
        many-core machine where dozens of single-threaded compiles collide.
    """
    cond = {}                                             # condition → runs, in seed order
    for r in runs:
        cond.setdefault((r.N, r.kappa_label, r.readout), []).append(r)
    T_of = {k: b[0].T_burn + b[0].T_meas for k, b in cond.items()}
    jobs = [((r.N, r.kappa_label, r.readout, r.seed), r) for r in runs]
    jobs.sort(key=lambda j: (-j[0][0], READOUT_ORDER.index(j[0][2]), j[0][1], j[0][3]))   # largest N first
    per_seed = {}
    if workers <= 1:
        results = (_job(k, r) for k, r in jobs)
    else:
        # spawn (never fork: the parent has JAX initialised) with single-threaded XLA in every child
        os.environ.update(PINNED_ENV)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            os.environ.update(JAX_COMPILATION_CACHE_DIR=os.path.abspath(cache_dir),
                              JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS="0")
        ctx = mp.get_context("spawn")
        ex = ProcessPoolExecutor(max_workers=workers, mp_context=ctx)
        results = (f.result() for f in as_completed([ex.submit(_job, k, r) for k, r in jobs]))
    try:
        for (N, k, ro, seed), mh in results:
            per_seed[(N, k, ro)] = per_seed.get((N, k, ro), {})
            per_seed[(N, k, ro)][seed] = mh
    finally:
        if workers > 1:
            ex.shutdown()
    out = {}
    for key in sorted(cond, key=lambda k: (k[0], READOUT_ORDER.index(k[2]), k[1])):
        mh = np.array([per_seed[key][r.seed] for r in cond[key]])
        if np.isnan(mh).all() and not np.isfinite(mh).any():
            _report(key, None, T_of[key])
            continue
        out[key] = mh
        _report(key, mh, T_of[key])
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
    ap.add_argument("--workers", type=int, default=None,
                    help="processes in the pool, one condition each (default: physical cores; 1: in-process)")
    ap.add_argument("--out", default=os.path.join(_path.ROOT, "results", "Nscaling"))
    args = ap.parse_args(argv)

    cfg = Config.from_yaml(args.config)
    over = {"name": "Nscaling", "N": list(args.N), "kappa": list(args.kappa), "readout": list(args.readout),
            "delta0": [args.delta0], "lam": [args.lam if args.lam is not None else cfg.lam[0]],
            "seeds": args.seeds, "extra": (), "trace": False}
    cfg = Config(**{**cfg.__dict__, **over})
    runs = make_runs(cfg)
    run0 = runs[0]
    workers = args.workers or physical_cores()
    print(f"half-life vs N: N={args.N} κ={args.kappa} δ₀={run0.delta0} λ={run0.lam} {run0.solver} seeds={args.seeds} "
          f"workers={workers}", flush=True)
    res = collect(runs, workers, cache_dir=os.path.join(args.out, "jax_cache"))

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
    sys.exit(main())
