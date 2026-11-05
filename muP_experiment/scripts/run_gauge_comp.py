#!/usr/bin/env python
"""Gauge comparison (PLAN.md §12.10): Δ_i per step with and without the projection, one cell.

    python scripts/run_gauge_comp.py [--N 100] [--kappa 1] [--delta0 0] [--lam 4] [--seed 0] [--tasks T]

One figure, ``results/gauge_comp/figures/gauge_comp_N{N}_k{κ}_d0{δ₀}_lam{λ}.png``:
two rows × one column, x = global training step (log), y = the per-unit
imbalance Δ_i = (a_i² − Q_ii)/η averaged over the active units of the
current task (min–max over active units as a band).  Row 1 runs every
readout with the projection switched off (``do_proj = 0``), row 2 with it
switched on (``do_proj = 1``); every other setting of the readout (gating,
learned/frozen a, task heads) is what PLAN.md §4 defines.  One trace per
readout, one seed, N = 100 and κ = 1 by default; the task stream is the
Stage-1 one (``configs/stage1.yaml``: λ = 4, population solver, T_burn =
T_meas = 7N) unless ``--tasks`` shortens it.  The per-step arrays are
written next to the figure as ``gauge_comp_….npz``.
"""
import argparse
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

import _path  # noqa: F401

from mup import model
from mup import plotting as P
from mup.config import Config, READOUT_ORDER, make_runs
from mup.protocol import _step_fn, make_params

_T0 = time.perf_counter()        # wall clock from after the imports to the end of the script

#: the two rows: (do_proj value, row title)
ROWS = ((0.0, "no gauge projection"), (1.0, "gauge projection after every step"))


def trace_imbalance(run, do_proj: float, n_tasks: int) -> dict:
    """Train one run for ``n_tasks`` tasks and return per-step statistics of Δ_i over the active units.

    The loop is the one of ``protocol.run_task`` (task draw, head draw,
    ``n_steps`` of step + ``project_rows``) without the probes, and with
    ``Params.do_proj`` forced to ``do_proj`` so that the same readout can be
    run both ways.

    Parameters
    ----------
    run : config.Run
    do_proj : float
        0 or 1.
    n_tasks : int

    Returns
    -------
    dict
        ``mean``, ``min``, ``max`` of Δ_i over the active units after every
        step, each (n_tasks · n_steps,); ``abs_max`` the max |Δ_i − Δ₀|.
    """
    p = make_params([run])
    p = p._replace(do_proj=jnp.full_like(p.do_proj, do_proj))
    p = jax.tree_util.tree_map(lambda x: x[0], p)                      # one run: drop the batch axis
    N, D = run.N, run.D
    state = model.init_state(jax.random.PRNGKey(run.rng_seed()), p, N, D, 1)
    n_steps = run.n_steps

    def stats(s):
        g = s.g
        d = model.imbalance(s, p)
        n = jnp.maximum(g.sum(), 1.0)
        big = jnp.where(g == 1.0, d, -jnp.inf)
        small = jnp.where(g == 1.0, d, jnp.inf)
        return jnp.stack([(g * d).sum() / n, small.min(), big.max(), jnp.abs(jnp.where(g == 1.0, d - p.Delta0, 0.0)).max()])

    def task(s, _):
        kt = jax.random.fold_in(s.key, s.t)
        k_teach, k_head, k_stream = jax.random.split(kt, 3)
        w, g = model.draw_task(k_teach, N, D, p.pN, p.gated)
        head = model.draw_readout(k_head, model.Q_diag(s.W), p)
        a = jnp.where(p.use_heads == 1.0, head, s.a)
        s = s._replace(w=w, g=g, a=a)
        step = _step_fn(p, run.solver, k_stream)

        def body(s, i):
            s = step(s, i)
            s, _ = model.project_rows(s, p)
            return s, stats(s)

        s, rows = jax.lax.scan(body, s, jnp.arange(n_steps, dtype=jnp.int32))
        return s._replace(t=s.t + 1), rows

    _, rows = jax.jit(lambda s: jax.lax.scan(task, s, None, length=n_tasks))(state)
    rows = np.asarray(rows).reshape(-1, 4)
    return {"mean": rows[:, 0], "min": rows[:, 1], "max": rows[:, 2], "abs_max": rows[:, 3]}


def figure(traces: dict, run0, n_tasks: int, path: str) -> str:
    """Two rows (no projection / projection), one trace per readout, x = step (log).

    Parameters
    ----------
    traces : dict
        ``traces[(do_proj, readout)]`` from ``trace_imbalance``.
    run0 : config.Run
        Any run of the cell (N, κ, δ₀, λ, n_steps).
    n_tasks : int
    path : str
        Figure stem.

    Returns
    -------
    str
        PNG path.
    """
    P.apply_style()
    fig, axes = P.grid(len(ROWS), 1, sharex="col", sharey=False, panel=(5.2, 2.2))
    readouts = [ro for ro in READOUT_ORDER if any(k[1] == ro for k in traces)]
    figdata = {}
    for r, (do_proj, title) in enumerate(ROWS):
        ax = axes[r, 0]
        for ro in readouts:
            tr = traces[(do_proj, ro)]
            x = np.arange(1, len(tr["mean"]) + 1)
            P.band(ax, x, tr["min"], tr["max"], P.READOUT_COLOR[ro])
            P.plot_readout(ax, x, tr["mean"], ro)
            tag = f"{'proj' if do_proj else 'noproj'}_{ro}"
            figdata.update({f"{tag}_mean": tr["mean"], f"{tag}_min": tr["min"], f"{tag}_max": tr["max"]})
        P.ref_line(ax, run0.Delta0)
        P.logaxis(ax, "x")
        ax.set_title(title, fontsize=P.FS_MIN + 1, loc="left")
        ax.set_ylabel(r"$\Delta_i$ (active-unit mean)")
    axes[-1, 0].set_xlabel("training step")
    figdata["step"] = np.arange(1, n_tasks * run0.n_steps + 1)
    P.cell_title(fig, kappa=run0.kappa_label, d0=run0.delta0, lam=run0.lam, solver=run0.solver, eta=run0.eta,
                 extra=f"N = {run0.N}, seed {run0.seed}")
    P.caption(fig, f"line: mean of Δ_i = (a_i² − Q_ii)/η over the active units after each step; band: min–max over "
                   f"active units; grey line: Δ₀ = {P.fmt(run0.Delta0)}.  {n_tasks} tasks × {run0.n_steps} steps.")
    P.fig_legend(fig, readouts)
    return P.save(fig, path, figdata=figdata)


def main(argv=None):
    """Command-line entry point (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(_path.ROOT, "configs", "stage1.yaml"),
                    help="config supplying the solver, λ and step-size rules (default: Stage 1)")
    ap.add_argument("--N", type=int, default=100)
    ap.add_argument("--kappa", default="1.0")
    ap.add_argument("--delta0", type=float, default=0.0)
    ap.add_argument("--lam", type=float, default=None, help="default: the config's first λ")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tasks", type=int, default=None, help="tasks to run (default: the config's T_burn + T_meas)")
    ap.add_argument("--readout", nargs="*", default=list(READOUT_ORDER))
    ap.add_argument("--out", default=os.path.join(_path.ROOT, "results", "gauge_comp"))
    args = ap.parse_args(argv)

    cfg = Config.from_yaml(args.config)
    cfg = Config(**{**cfg.__dict__, "name": "gauge_comp", "N": [args.N], "kappa": [args.kappa],
                    "readout": list(args.readout), "delta0": [args.delta0],
                    "lam": [args.lam if args.lam is not None else cfg.lam[0]], "seeds": [args.seed],
                    "extra": (), "trace": False})
    runs = {r.readout: r for r in make_runs(cfg)}
    run0 = next(iter(runs.values()))
    n_tasks = args.tasks or (run0.T_burn + run0.T_meas)
    print(f"gauge comparison: N={run0.N} κ={run0.kappa_label} δ₀={run0.delta0} λ={run0.lam} {run0.solver} "
          f"n_steps={run0.n_steps} tasks={n_tasks} ({n_tasks * run0.n_steps} steps)", flush=True)

    traces = {}
    for do_proj, title in ROWS:
        for ro, run in runs.items():
            traces[(do_proj, ro)] = trace_imbalance(run, do_proj, n_tasks)
            tr = traces[(do_proj, ro)]
            print(f"  {title:36s} {ro:15s} final mean Δ = {tr['mean'][-1]:+.3e}  max|Δ−Δ₀| = {tr['abs_max'].max():.3e}", flush=True)

    stem = os.path.join(args.out, "figures", f"gauge_comp_N{run0.N}_k{P.fmt(run0.kappa_label)}_d0{P.fmt(run0.delta0)}_lam{P.fmt(run0.lam)}")
    out = figure(traces, run0, n_tasks, stem)
    print(f"wrote {out} and {stem}.npz")
    return 0


if __name__ == "__main__":
    _rc = main()
    _dt = time.perf_counter() - _T0
    print(f"run_gauge_comp: total {_dt:.1f} s ({_dt / 60:.2f} min) from after the imports to the end", flush=True)
    sys.exit(_rc)
