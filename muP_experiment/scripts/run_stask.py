#!/usr/bin/env python
"""Steady-state teacher signal against age (PLAN.md §12.12): S_T(τ) = u_{T+τ}·w_T*/|w_T*|², one trace per readout, one cell.

    python scripts/run_stask.py [--N 100] [--kappa 1] [--delta0 0] [--lam 4] [--seeds 4] [--T 1400] [--tau 1000]

The companion of ``run_szero.py`` with the probe taken after the network
has reached its steady state: train T tasks (default the Stage-1 stream
length, T_burn + T_meas = 1400 at N = 100), then follow the teacher of
task T for τ = 0 … ``--tau`` more tasks.  S_T(τ) = u_{T+τ}·w_T*/|w_T*|²
with u = (g_T ⊙ a) W / pN, the gate of task T and its stored head (task
heads) or the current shared readout, evaluated after every task T + τ.
One figure, ``results/stask/figures/stask_N{N}_k{κ}_d0{δ₀}_lam{λ}.png``:
x = τ + 1 on a log axis (τ = 0, the task just learned, at x = 1), y =
S_T(τ)/S_T(0) so every trace starts at 1, one trace per readout of §4 as the seed mean with a ±1 s.e.
band over seeds; the half-life m½ (first crossing of S_T(0)/2) is in
the caption.  N = 100, κ = 1, δ₀ = 0 by default; solver, λ and step-size
rule come from ``configs/stage1.yaml``.  The per-τ arrays are written
next to the figure as ``stask_….npz``.
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
from mup.protocol import batched_init, batched_run_task, feasible, make_params
from run_szero import half_life

_T0 = time.perf_counter()        # wall clock from after the imports to the end of the script


@jax.jit
def _signal_k(state, p, k):
    """S of probe ``k`` (teacher, gate, head of task k) against the current W, for every run of a batch."""
    def one(s, q):
        wk, gk = s.probes_w[k], s.probes_g[k]
        ak = jnp.where(q.use_heads == 1.0, s.heads[k], s.a)
        u = (gk * ak) @ s.W / q.pN
        return (u * wk).sum() / (wk * wk).sum()
    return jax.vmap(one)(state, p)


def stask_by_readout(runs: list, T: int, tau: int) -> dict:
    """Train ``T`` tasks, then collect S_T(τ) for τ = 0 … ``tau`` after each further task.

    Parameters
    ----------
    runs : list of config.Run
        One cell (N, κ, δ₀, λ), every readout, every seed.
    T : int
        Tasks before the probed one (task T is the probe).
    tau : int
        Further tasks after task T.

    Returns
    -------
    dict
        ``{readout: (tau + 1, n_seeds) array}``; row τ is S_T after task T + τ.
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
        p = make_params(batch)
        keys = jnp.stack([jax.random.PRNGKey(r.rng_seed()) for r in batch])
        state = batched_init(r0.N, r0.D, T + tau + 1)(keys, p)        # probe slots for every task run
        ages = jnp.asarray(age_schedule(r0.N, r0.kappa_s, r0.T_burn + r0.T_meas, r0.n_ages_geom, r0.m_max_factor))
        step = batched_run_task(r0.n_steps, r0.solver)
        for _ in range(T):
            state, _ = step(state, p, ages)
        S = np.zeros((tau + 1, len(batch)))
        for i in range(tau + 1):
            state, _ = step(state, p, ages)
            S[i] = np.asarray(_signal_k(state, p, T))
        out[ro] = S
        print(f"  {ro}: {len(batch)} seeds × ({T} + {tau + 1}) tasks done", flush=True)
    return out


def figure(S: dict, run0, T: int, path: str) -> str:
    """S_T(τ) vs τ + 1 (log), one trace per readout (seed mean ± 1 s.e.), half-life in the caption.

    Parameters
    ----------
    S : dict
        From ``stask_by_readout``.
    run0 : config.Run
        Any run of the cell.
    T : int
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
    for ro, Y in S.items():
        n_tau, n = Y.shape
        x = np.arange(n_tau) + 1                                        # log axis: τ = 0 at x = 1
        m = Y.mean(1)
        se = Y.std(1, ddof=1) / np.sqrt(n) if n > 1 else np.zeros(n_tau)
        s0 = m[0]                                                       # y is S_T(τ)/S_T(0): every trace starts at 1
        P.band(ax, x, (m - se) / s0, (m + se) / s0, P.READOUT_COLOR[ro])
        P.plot_readout(ax, x, m / s0, ro)
        mh[ro] = half_life(m)
        figdata[f"ST_{ro}"] = Y
        figdata[f"STnorm_{ro}"] = m / s0
    figdata["tau"] = np.arange(max(Y.shape[0] for Y in S.values()))
    P.ref_line(ax, 0.0)
    P.ref_line(ax, 0.5)
    P.logaxis(ax, "x")
    ax.set_xlabel(r"$\tau + 1$ (tasks after task $T$)")
    ax.set_ylabel(r"$S_T(\tau)\,/\,S_T(0)$,  $S_T(\tau) = u_{T+\tau}\cdot w_T^*/|w_T^*|^2$")
    n_seeds = max(Y.shape[1] for Y in S.values())
    mtxt = ", ".join(f"{P.READOUT_LABEL[ro]} {mh[ro]:.1f}" if np.isfinite(mh[ro]) else f"{P.READOUT_LABEL[ro]} > τ"
                     for ro in S)
    P.cell_title(fig, kappa=run0.kappa_label, d0=run0.delta0, lam=run0.lam, solver=run0.solver, eta=run0.eta,
                 extra=f"N = {run0.N}, T = {T}")
    P.caption(fig, f"line: seed mean of the signal of teacher T = {T} after task T + τ divided by its value at τ = 0, "
                   f"drawn at τ + 1 so τ = 0 is at 1 ({n_seeds} seeds); band: ±1 s.e. over seeds; half-life m½ "
                   f"(tasks) where the mean crosses ½ (grey line): {mtxt}.")
    P.fig_legend(fig, list(S))
    return P.save(fig, path, figdata=figdata)


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
    ap.add_argument("--T", type=int, default=None, help="tasks before the probed one (default: the config's T_burn + T_meas)")
    ap.add_argument("--tau", type=int, default=1000, help="tasks followed after task T")
    ap.add_argument("--readout", nargs="*", default=list(READOUT_ORDER))
    ap.add_argument("--out", default=os.path.join(_path.ROOT, "results", "stask"))
    args = ap.parse_args(argv)

    cfg = Config.from_yaml(args.config)
    over = {"name": "stask", "N": [args.N], "kappa": [args.kappa], "readout": list(args.readout),
            "delta0": [args.delta0], "lam": [args.lam if args.lam is not None else cfg.lam[0]],
            "seeds": args.seeds, "extra": (), "trace": False}
    cfg = Config(**{**cfg.__dict__, **over})
    runs = make_runs(cfg)
    run0 = runs[0]
    T = args.T or (run0.T_burn + run0.T_meas)
    print(f"S_T(τ) vs τ: N={run0.N} κ={run0.kappa_label} δ₀={run0.delta0} λ={run0.lam} {run0.solver} "
          f"n_steps={run0.n_steps} T={T} τ≤{args.tau} seeds={args.seeds}", flush=True)
    S = stask_by_readout(runs, T, args.tau)
    for ro, Y in S.items():
        m = Y.mean(1)
        print(f"  {ro:15s} S_T(0) = {m[0]:.4f}  m½ = {half_life(m):.1f}  S_T(τ_max) = {m[-1]:+.4f}", flush=True)

    stem = os.path.join(args.out, "figures", f"stask_N{run0.N}_k{P.fmt(run0.kappa_label)}_d0{P.fmt(run0.delta0)}_lam{P.fmt(run0.lam)}")
    out = figure(S, run0, T, stem)
    print(f"wrote {out} and {stem}.npz")
    return 0


if __name__ == "__main__":
    _rc = main()
    _dt = time.perf_counter() - _T0
    print(f"run_stask: total {_dt:.1f} s ({_dt / 60:.2f} min) from after the imports to the end", flush=True)
    sys.exit(_rc)
