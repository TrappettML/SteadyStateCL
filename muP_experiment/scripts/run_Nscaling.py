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

This experiment does not use JAX: the dynamics of ``mup.model`` (teacher
and gate draws, population / SGD step, row projection, stored-head
rescaling) are re-implemented here for one run in float64 NumPy, so a
run is a plain Python loop with no compilation.  A run's random stream is
``numpy.random.default_rng(run.rng_seed())`` (so the seeds do not
reproduce the JAX sweep's draws bit for bit, only their distribution).

Every run (N, κ, readout, seed) is independent, so each is one job of
a ``ProcessPoolExecutor`` of ``--workers`` processes (default:
``psutil.cpu_count(logical=False)``, the physical cores; 1 runs
everything in this process), every worker pinned to one BLAS thread.
Largest N first, so the long runs do not trail at the end.
"""
import argparse
import math
import multiprocessing as mp
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

import _path  # noqa: F401

from mup import plotting as P
from mup.config import Config, READOUT_ORDER, is_gated, make_runs

GATED = tuple(ro for ro in READOUT_ORDER if is_gated(ro))

#: One BLAS/OpenMP thread per worker process; set before the pool spawns.
PINNED_ENV = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
              "VECLIB_MAXIMUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}


def half_life(m: np.ndarray) -> float:
    """First task at which ``m`` falls to ``m[0]/2``, linearly interpolated; NaN if never (as ``run_szero.half_life``, which imports JAX)."""
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


# ----------------------------------------------------------------------------
# float64 NumPy port of the single-run kernels of ``mup.model`` (PLAN.md §2–§3)
# ----------------------------------------------------------------------------
def _Q_diag(W: np.ndarray) -> np.ndarray:
    """Q_ii = |W_i|² / D for every row."""
    return (W * W).sum(-1) / W.shape[1]


def _draw_readout(rng: np.random.Generator, Q: np.ndarray, eta: float, Delta0: float) -> np.ndarray:
    """a_i = ε_i √(η (Δ₀ + Q_ii/η)), ε_i = ±1; NaN where Δ₀ + Q_ii/η < 0 (infeasible δ₀)."""
    eps = rng.choice(np.array([-1.0, 1.0]), size=Q.shape)
    with np.errstate(invalid="ignore"):
        return eps * np.sqrt(eta * (Delta0 + Q / eta))


def _sgd_step(W, a, g, w, rng, eta, pN, learn_a):
    """One online-SGD step, eqs. (sgdW)–(sgda); returns (W, a)."""
    D = W.shape[1]
    sqD = math.sqrt(D)
    x = rng.standard_normal(D)
    hx = W @ x / sqD
    ga = g * a
    e = ga @ hx / pN - w @ x / sqD
    W_n = W - (eta * e / sqD) * np.outer(ga, x)
    a_n = a - learn_a * eta / D * g * e * hx
    return W_n, a_n


def _pop_step(W, a, g, w, h, eta, pN, learn_a):
    """One population-gradient step, eq. (pop), simultaneous in W and a; returns (W, a)."""
    D = W.shape[1]
    ga = g * a
    r = ga @ W / pN - w
    a_n = a - learn_a * h * eta * g * (W @ r) / D
    W_n = W - (h * eta) * np.outer(ga, r)
    return W_n, a_n


def _project_rows(W, a, g, eta, Delta0, do_proj):
    """Rescale every active unit onto Δ_i = Δ₀, eq. (proj), in the cancellation-free product form.

    Returns ``(W, a, f)`` with ``f`` the (N,) factor applied to the rows of W
    (1 for untouched units) so the stored heads can be rescaled by 1/f.
    """
    N = W.shape[0]
    if not do_proj:
        return W, a, np.ones(N)
    Q = _Q_diag(W)
    A = a * a / eta
    B = Q / eta
    AB = A * B
    big = 0.5 * (abs(Delta0) + np.hypot(Delta0, 2.0 * np.sqrt(AB)))
    small = AB / big
    Ap, Bp = (big, small) if Delta0 >= 0.0 else (small, big)
    act = g == 1.0
    a_new = np.where(act, np.sign(a) * np.sqrt(eta * Ap), a)
    f = np.where(act, np.sqrt(eta * Bp / Q), 1.0)
    W_new = W * f[:, None]
    return W_new, a_new, f


def mhalf_run(run) -> float:
    """Half-life m½ of one run: train T tasks, follow the teacher of task T for T more.

    Parameters
    ----------
    run : config.Run

    Returns
    -------
    float
        m½ in tasks; NaN when censored (no crossing within T tasks) or
        infeasible (δ₀ ≤ −min_i Q0_i, so the initial readout is not real).
    """
    N, D = run.N, run.D
    T = run.T_burn + run.T_meas
    eta, h, pN, Delta0 = float(run.eta), float(run.h), float(run.pN), float(run.Delta0)
    gated, learn_a = run.gated == 1.0, float(run.learn_a)
    do_proj, use_heads = run.do_proj == 1.0, run.use_heads == 1.0
    n_steps, solver = int(run.n_steps), run.solver
    assert n_steps >= 1
    if solver not in ("pop", "sgd"):
        raise ValueError(f"unknown solver {solver!r}")
    rng = np.random.default_rng(run.rng_seed())

    W = rng.standard_normal((N, D))
    a = _draw_readout(rng, _Q_diag(W), eta, Delta0)
    if not np.isfinite(a).all():
        return float("nan")                                   # protocol.feasible
    w_T = g_T = head_T = None                                 # probe of task T

    def task(t):
        """Train task ``t``; stores the probe when t == T and rescales the stored head afterwards."""
        nonlocal W, a, w_T, g_T, head_T
        w = rng.standard_normal(D)
        g = (rng.uniform(size=N) < pN / N).astype(np.float64) if gated else np.ones(N)
        head = _draw_readout(rng, _Q_diag(W), eta, Delta0)
        if use_heads:
            a = head
        fcum = np.ones(N)
        for i in range(n_steps):
            if solver == "sgd":
                W, a = _sgd_step(W, a, g, w, rng, eta, pN, learn_a)
            else:
                W, a = _pop_step(W, a, g, w, h, eta, pN, learn_a)
            W, a, f = _project_rows(W, a, g, eta, Delta0, do_proj)
            fcum *= f
        if use_heads and head_T is not None:
            head_T = head_T / fcum                            # model.rescale_heads for the stored probe
        if t == T:
            w_T, g_T, head_T = w, g, (a.copy() if use_heads else None)

    for t in range(T):
        task(t)
    S = np.empty(T + 1)
    for m in range(T + 1):
        task(T + m)
        a_T = head_T if use_heads else a
        u = (g_T * a_T) @ W / pN                              # run_stask._signal_k
        S[m] = (u * w_T).sum() / (w_T * w_T).sum()
    return float(half_life(S))


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
    """One run (one seed) in a worker process: ``(key, m½)``; NaN when censored or infeasible."""
    return key, mhalf_run(run)


def _report(key, mh, T):
    if mh is None:
        print(f"  N={key[0]} κ={key[1]} {key[2]}: every seed infeasible, skipped", flush=True)
        return
    n_ok = int(np.isfinite(mh).sum())
    se = np.nanstd(mh, ddof=1) / math.sqrt(n_ok) if n_ok > 1 else 0.0
    print(f"  N={key[0]:4d} κ={key[1]:5s} {key[2]:15s} T={T:5d}  m½ = {np.nanmean(mh):7.2f} ± {se:5.2f}"
          f"  censored or infeasible {int(np.isnan(mh).sum())}/{len(mh)}", flush=True)


def collect(runs: list, workers: int = 1) -> dict:
    """``{(N, kappa_label, readout): (n_seeds,) m½ array}`` for every condition of ``runs``.

    Parameters
    ----------
    runs : list of config.Run
    workers : int
        Processes in the pool, one run (seed) per job; 1 runs the jobs
        here, in order.  Each worker is pinned to one BLAS thread
        (``PINNED_ENV``), so ``workers`` should not exceed the physical
        core count.
    """
    cond = {}                                             # condition → runs, in seed order
    for r in runs:
        cond.setdefault((r.N, r.kappa_label, r.readout), []).append(r)
    T_of = {k: b[0].T_burn + b[0].T_meas for k, b in cond.items()}
    jobs = [((r.N, r.kappa_label, r.readout, r.seed), r) for r in runs]
    jobs.sort(key=lambda j: (-j[0][0], READOUT_ORDER.index(j[0][2]), j[0][1], j[0][3]))   # largest N first
    per_seed = {}
    if workers <= 1:
        for k, r in jobs:
            _, mh = _job(k, r)
            per_seed.setdefault(k[:3], {})[k[3]] = mh
    else:
        os.environ.update(PINNED_ENV)                     # inherited by the spawned workers before they import numpy
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            futs = [ex.submit(_job, k, r) for k, r in jobs]
            for f in as_completed(futs):
                (N, k, ro, seed), mh = f.result()
                per_seed.setdefault((N, k, ro), {})[seed] = mh
    out = {}
    for key in sorted(cond, key=lambda k: (k[0], READOUT_ORDER.index(k[2]), k[1])):
        mh = np.array([per_seed[key][r.seed] for r in cond[key]])
        if not np.isfinite(mh).any():
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
                    help="processes in the pool, one run each (default: psutil physical cores; 1: in-process)")
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
    res = collect(runs, workers)

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
