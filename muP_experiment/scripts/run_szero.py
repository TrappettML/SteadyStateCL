#!/usr/bin/env python
"""First-teacher signal against task (PLAN.md §12.11): S₀(t) = u_t·w₀*/|w₀*|², one trace per readout, one cell.

    python scripts/run_szero.py [--N 100] [--kappa 1] [--delta0 0] [--lam 4] [--seeds 4] [--tasks T]

One figure, ``results/szero/figures/szero_N{N}_k{κ}_d0{δ₀}_lam{λ}.png``:
x = t + 1 on a log axis (task 0 at x = 1), y = S₀(t)/S₀(0) (every trace starts at 1), with S₀(t) how well the student after task t
still aligns with the FIRST teacher w₀*: u_t = (g₀ ⊙ a) W_t / pN with the
gate g₀ of task 0 and its stored head (task heads) or the current shared
readout, projected on w₀* (PLAN.md §5 with the probe fixed at k = 0 and
age m = t).  S₀(0) is the signal just after learning task 0 and the
decay of S₀(t) from there is the memory of that task; the half-life m½
is where the seed mean crosses S₀(0)/2 and is given per readout in the caption.  One
trace per readout of §4 as the seed mean with a ±1 s.e. band over seeds.
N = 100, κ = 1, δ₀ = 0 by default; solver, λ and step-size rule come from
``configs/stage1.yaml``; the stream is the
Stage-1 one, T_burn + T_meas tasks (1400 at N = 100), unless ``--tasks``.  The
per-task arrays are written next to the figure as ``szero_….npz``.
"""
import argparse
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

import _path  # noqa: F401

from mup import plotting as P
from mup.config import Config, READOUT_ORDER, make_runs
from mup.observe import age_schedule
from mup.protocol import batched_run_task, feasible, init_batch

_T0 = time.perf_counter()        # wall clock from after the imports to the end of the script


@jax.jit
def _signal0(state, p):
    """S₀ of every run of a batch: probe 0 (teacher, gate, head) against the current W."""
    def one(s, q):
        w0, g0 = s.probes_w[0], s.probes_g[0]
        a0 = jnp.where(q.use_heads == 1.0, s.heads[0], s.a)
        u = (g0 * a0) @ s.W / q.pN
        return (u * w0).sum() / (w0 * w0).sum()
    return jax.vmap(one)(state, p)


def szero_by_readout(runs: list, T: int) -> dict:
    """Run every readout's seeds as one batch for ``T`` tasks and collect S₀(t) after each task.

    Parameters
    ----------
    runs : list of config.Run
        One cell (N, κ, δ₀, λ), every readout, every seed.
    T : int
        Tasks to run (task 0 is the probed one).

    Returns
    -------
    dict
        ``{readout: (T, n_seeds) array}``; row t is S₀ after task t.
    """
    out = {}
    for ro in READOUT_ORDER:
        batch = [r for r in runs if r.readout == ro]
        if not batch:
            continue
        ok = feasible(batch)
        batch = [r for r, f in zip(batch, ok) if f]
        if not batch:
            print(f"  {ro}: every seed infeasible (δ₀ ≤ −min_i Q0_i), skipped", flush=True)
            continue
        r0 = batch[0]
        state, p = init_batch(batch)
        ages = jnp.asarray(age_schedule(r0.N, r0.kappa_s, r0.T_burn + r0.T_meas, r0.n_ages_geom, r0.m_max_factor))
        step = batched_run_task(r0.n_steps, r0.solver)
        S = np.zeros((T, len(batch)))
        for t in range(T):
            state, _ = step(state, p, ages)
            S[t] = np.asarray(_signal0(state, p))
        out[ro] = S
        print(f"  {ro}: {len(batch)} seeds × {T} tasks done", flush=True)
    return out


def figure(S0: dict, run0, path: str) -> str:
    """S₀(t) vs task, one trace per readout (seed mean ± 1 s.e.), half-life in the caption.

    Parameters
    ----------
    S0 : dict
        From ``szero_by_readout``.
    run0 : config.Run
        Any run of the cell.
    path : str
        Figure stem.

    Returns
    -------
    str
        PNG path.
    """
    P.apply_style()
    fig, axes = P.grid(1, 1, panel=(5.2, 2.6))
    ax = axes[0, 0]
    figdata = {}
    mh = {}
    for ro, Y in S0.items():
        T, n = Y.shape
        t = np.arange(T)
        m = Y.mean(1)
        se = Y.std(1, ddof=1) / np.sqrt(n) if n > 1 else np.zeros(T)
        x = t + 1                                                       # log axis: task 0 is drawn at x = 1
        s0 = m[0]                                                       # y is S₀(t)/S₀(0): every trace starts at 1
        P.band(ax, x, (m - se) / s0, (m + se) / s0, P.READOUT_COLOR[ro])
        P.plot_readout(ax, x, m / s0, ro)
        mh[ro] = half_life(m)
        figdata[f"S0_{ro}"] = Y
        figdata[f"S0norm_{ro}"] = m / s0
    figdata["task"] = np.arange(max(Y.shape[0] for Y in S0.values()))
    P.ref_line(ax, 0.0)
    P.ref_line(ax, 0.5)
    P.logaxis(ax, "x")
    ax.set_xlabel("task $t + 1$")
    ax.set_ylabel(r"$S_0(t)\,/\,S_0(0)$,  $S_0(t) = u_t\cdot w_0^*/|w_0^*|^2$")
    n_seeds = max(Y.shape[1] for Y in S0.values())
    P.cell_title(fig, kappa=run0.kappa_label, d0=run0.delta0, lam=run0.lam, solver=run0.solver, eta=run0.eta,
                 extra=f"N = {run0.N}")
    def _m(ro):
        v = mh[ro]
        return "> T" if not np.isfinite(v) else f"{v:.1f}"
    mtxt = ", ".join(f"{P.READOUT_LABEL[ro]} {_m(ro)}" for ro in S0)
    P.caption(fig, f"line: seed mean of the first teacher's signal after task t divided by its value after task 0, "
                   f"drawn at t + 1 so task 0 is at 1 ({n_seeds} seeds); band: ±1 s.e. over seeds; half-life m½ "
                   f"(tasks) where the mean crosses ½ (grey line): {mtxt}.")
    P.fig_legend(fig, list(S0))
    return P.save(fig, path, figdata=figdata)


def half_life(m: np.ndarray) -> float:
    """First task at which ``m`` falls to ``m[0]/2``, linearly interpolated; NaN if never."""
    half = m[0] / 2
    below = np.nonzero(m <= half)[0]
    if len(below) == 0:
        return float("nan")
    j = int(below[0])
    if j == 0:
        return 0.0
    return j - 1 + (m[j - 1] - half) / (m[j - 1] - m[j])


def main(argv=None):
    """Command-line entry point (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(_path.ROOT, "configs", "stage1.yaml"),
                    help="config supplying the solver, λ, step-size and stream-length rules (default: Stage 1)")
    ap.add_argument("--N", type=int, default=100)
    ap.add_argument("--kappa", default="1.0")
    ap.add_argument("--delta0", type=float, default=0.0)
    ap.add_argument("--lam", type=float, default=None, help="default: the config's first λ")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--tasks", type=int, default=None, help="tasks to run (default: the config's T_burn + T_meas)")
    ap.add_argument("--readout", nargs="*", default=list(READOUT_ORDER))
    ap.add_argument("--out", default=os.path.join(_path.ROOT, "results", "szero"))
    args = ap.parse_args(argv)

    cfg = Config.from_yaml(args.config)
    over = {"name": "szero", "N": [args.N], "kappa": [args.kappa], "readout": list(args.readout),
            "delta0": [args.delta0], "lam": [args.lam if args.lam is not None else cfg.lam[0]],
            "seeds": args.seeds, "extra": (), "trace": False}
    cfg = Config(**{**cfg.__dict__, **over})
    runs = make_runs(cfg)
    run0 = runs[0]
    T = args.tasks or (run0.T_burn + run0.T_meas)
    print(f"S₀(t) vs task: N={run0.N} κ={run0.kappa_label} δ₀={run0.delta0} λ={run0.lam} {run0.solver} "
          f"n_steps={run0.n_steps} T={T} seeds={args.seeds}", flush=True)
    S0 = szero_by_readout(runs, T)
    for ro, Y in S0.items():
        m = Y.mean(1)
        print(f"  {ro:15s} S₀(0) = {m[0]:.4f}  m½ = {half_life(m):.1f}  S₀(T−1) = {m[-1]:+.4f}", flush=True)

    stem = os.path.join(args.out, "figures", f"szero_N{run0.N}_k{P.fmt(run0.kappa_label)}_d0{P.fmt(run0.delta0)}_lam{P.fmt(run0.lam)}")
    out = figure(S0, run0, stem)
    print(f"wrote {out} and {stem}.npz")
    return 0


if __name__ == "__main__":
    _rc = main()
    _dt = time.perf_counter() - _T0
    print(f"run_szero: total {_dt:.1f} s ({_dt / 60:.2f} min) from after the imports to the end", flush=True)
    sys.exit(_rc)
