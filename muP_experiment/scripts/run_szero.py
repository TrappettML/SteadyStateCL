#!/usr/bin/env python
"""First-teacher signal against task (PLAN.md §12.11): S₀(t) = u_t·w₀*/|w₀*|², one trace per readout, one figure per cell.

    python scripts/run_szero.py [--N 100 200] [--kappa 1 2] [--delta0 0] [--lam 4] [--seeds 4] [--tasks T]
                                [--stat mean|seed] [--clock realized|nominal] [--gpus auto]

One figure per cell (N, κ, δ₀), ``results/szero/figures/szero_N{N}_k{κ}_d0{δ₀}_lam{λ}.png``:
x = t + 1 on a log axis (task 0 at x = 1), y = S₀(t)/S₀(0) (every trace starts at 1), with S₀(t) how well the student after task t
still aligns with the FIRST teacher w₀*: u_t = (g₀ ⊙ a) W_t / pN with the
gate g₀ of task 0 and its stored head (task heads) or the current shared
readout, projected on w₀* (PLAN.md §5 with the probe fixed at k = 0 and
age m = t).  S₀(0) is the signal just after learning task 0 and the
decay of S₀(t) from there is the memory of that task.  One trace per readout of §4.  With ``--stat mean``
(default) the trace is the write-up's mean normalized signal Σ(t) = ⟨S₀(t)⟩/⟨S₀(0)⟩, the seed mean of the
signal divided by its seed mean after task 0 (eq. Sigma; band: jackknife ±1 s.e.), and the half-life m½ in
the caption is its first crossing of ½, the root of Σ = ½ that the theory predicts.  With ``--stat seed`` the
trace is the seed mean of each seed's own S₀(t)/S₀(0) and m½ the mean of the per-seed half-lives, a different
statistic (``sscl_core.halflife``).  Dotted, in the readout's colour: the write-up's curve for the readouts whose
equation holds from a fresh start, eq. (S_heads) for task heads and eq. (S_frozen) for the frozen ones
(``theory.sigma``); the learned shared readout has no fresh-start curve (its key drift and leak-removal factor
depend on norms that are still growing; ``run_stask.py`` draws eq. S_learned at steady state).
N = 100, κ = 1, δ₀ = 0 by default; solver, λ and step-size rule come from
``configs/stage1.yaml``; the stream is the
Stage-1 one, T_burn + T_meas tasks (1400 at N = 100), unless ``--tasks``.  The
per-task arrays are written next to the figure as ``szero_….npz``.

Placement (``sscl_core.pool``): every readout and seed of a cell is one ``vmap``
batch advanced ``--chunk`` tasks per device call with S₀ evaluated inside
the scan (``experiments.signal_after_tasks``) and runs as a job over the GPUs
(``--gpus``: the four cards, several processes each, the cells split only
as far as needed to fill them — ``pool.plan``), over ``--workers`` CPU
processes, or in this process.
"""
import argparse
import os
import sys
import time
from typing import Callable, Optional, Sequence

import numpy as np

import _path  # noqa: F401

from sscl_core import experiments as X
from sscl_core import halflife as H
from sscl_core import plotting as P
from sscl_core import pool
from sscl_core import theory as TH
from sscl_core.config import Config, READOUT_ORDER, Run, add_clock_arg, clock_override, make_runs

_T0 = time.perf_counter()        # wall clock from after the imports to the end of the script


def cell_key(r: Run) -> tuple:
    """(N, κ label, δ₀, λ) of a run: one figure per cell."""
    return (r.N, r.kappa_label if r.gated == 1.0 else None, r.delta0, r.lam)


def cells(runs: list[Run]) -> dict[tuple, list[Run]]:
    """``{(N, κ label, δ₀, λ): runs}``; the ungated readouts (no κ) are listed in every κ cell."""
    out = {}
    kappas = sorted({r.kappa_label for r in runs if r.gated == 1.0}, key=float) or [None]
    for N in sorted({r.N for r in runs}):
        for k in kappas:
            for d0 in sorted({r.delta0 for r in runs}):
                for lam in sorted({r.lam for r in runs}):
                    cell = [r for r in runs if r.N == N and r.delta0 == d0 and r.lam == lam
                            and (r.kappa_label == k if r.gated == 1.0 else True)]
                    if cell:
                        out[(N, k, d0, lam)] = cell
    return out


def groups(cells_: dict[tuple, list[Run]]) -> dict[tuple, list[Run]]:
    """``{group_key: distinct runs}`` over every cell: a run listed in several cells (the ungated readouts, which
    join every κ cell) is trained once and its column is read by each of them."""
    seen, out = set(), {}
    for cell in cells_.values():
        for r in cell:
            if r.label() not in seen:
                seen.add(r.label())
                out.setdefault(X.group_key(r), []).append(r)
    return out


def make_jobs(groups_: dict[tuple, list[Run]], T: int, pieces: Sequence[int]) -> list[tuple]:
    """One job per piece of a group: ``(key, kwargs of experiments.signal_after_tasks, cost)``; ``pieces`` per group."""
    jobs = []
    for (key, runs), k in zip(groups_.items(), pieces):
        for i, part in enumerate(X.split(runs, k)):
            jobs.append(((key, i), {"runs": part, "T_pre": 0, "n_follow": T, "k": 0}, X.job_cost(part, T)))
    return jobs


def columns(jobs: list[tuple], res: list[np.ndarray]) -> dict[str, np.ndarray]:
    """``{run label: (n_follow,) signal column}`` from the job results."""
    out = {}
    for (_, kw, _), S in zip(jobs, res):
        for j, r in enumerate(kw["runs"]):
            out[r.label()] = S[:, j]
    return out


def per_cell(cells_: dict[tuple, list[Run]], cols: dict[str, np.ndarray]) -> dict[tuple, dict[str, np.ndarray]]:
    """``{cell: {readout: (n, n_seeds)}}``: each cell's runs' columns, by readout."""
    return {key: by_readout(cell, np.stack([cols[r.label()] for r in cell], 1)) for key, cell in cells_.items()}


def by_readout(cell: list[Run], S: np.ndarray) -> dict[str, np.ndarray]:
    """``{readout: (T, n_seeds) array}`` of a cell from its stacked job results, infeasible seeds dropped."""
    out = {}
    for ro in READOUT_ORDER:
        cols = [j for j, r in enumerate(cell) if r.readout == ro]
        if not cols:
            continue
        Y = S[:, cols]
        Y = Y[:, np.isfinite(Y).all(0)]
        if Y.shape[1] == 0:
            print(f"  {ro}: every seed infeasible (δ₀ ≤ −min_i Q0_i), skipped", flush=True)
            continue
        out[ro] = Y
    return out


def collect(cells_: dict[tuple, list[Run]], T: int, plan: pool.Plan,
            log: Callable[[str], None] = print) -> dict[tuple, dict[str, np.ndarray]]:
    """Run the distinct runs of every cell through ``pool.run_jobs`` and return ``{cell: {readout: (T, n_seeds)}}``."""
    jobs = make_jobs(groups(cells_), T, plan.pieces)
    res = pool.run_jobs("sscl_core.experiments:signal_after_tasks", jobs, log=log, **plan.place)
    return per_cell(cells_, columns(jobs, res))


def theory_curve(ax, ro: str, run0: Run, n: int, steady: bool, figdata: dict) -> None:
    """Draw the write-up's Σ(m), m = 0 … n − 1, for readout ``ro`` at x = m + 1, dotted in the readout's colour.

    ``theory.sigma``: eq. (S_heads), eq. (S_frozen) or — only when ``steady`` — eq. (S_learned) with the key
    drift of eqs. (delta_est), (abarss).  The heads and frozen curves hold from a fresh start as at steady state.
    """
    if not steady and TH.READOUT_KIND[ro][0] == "learned":
        return
    m = np.arange(n)
    y = TH.sigma(ro, m, run0.N, run0.kappa, run0.lam, run0.delta0, run0.sigma0)
    ax.plot(m + 1, y, color=P.READOUT_COLOR[ro], ls=":", lw=P.LW_REF, zorder=5)
    figdata[f"theory_{ro}"] = y


def figure(S0: dict[str, np.ndarray], run0: Run, path: str, stat: str = "mean") -> str:
    """S₀(t) vs task, one trace per readout (± 1 s.e.), its equation dotted, half-life in the caption.

    Parameters
    ----------
    S0 : dict
        ``{readout: (T, n_seeds)}`` from ``by_readout``.
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
        x = np.arange(T) + 1                                            # log axis: task 0 is drawn at x = 1
        m, se, mh[ro] = seed_signal(Y, stat)                            # normalized signal and its half-life
        P.band(ax, x, m - se, m + se, P.READOUT_COLOR[ro])
        P.plot_readout(ax, x, m, ro)
        theory_curve(ax, ro, run0, T, False, figdata)
        figdata[f"S0_{ro}"] = Y
        figdata[f"S0norm_{ro}"] = m
        figdata[f"mhalf_{ro}"] = mh[ro]
        figdata[f"mhalf_seed_{ro}"] = H.per_seed(Y)
    figdata["task"] = np.arange(max(Y.shape[0] for Y in S0.values()))
    P.ref_line(ax, 0.0)
    P.ref_line(ax, 0.5)
    P.logaxis(ax, "x")
    ax.set_xlabel("task $t + 1$")
    ax.set_ylabel(r"$S_0(t)\,/\,S_0(0)$,  $S_0(t) = u_t\cdot w_0^*/|w_0^*|^2$")
    n_seeds = max(Y.shape[1] for Y in S0.values())
    P.cell_title(fig, kappa=run0.kappa_label, d0=run0.delta0, lam=run0.lam, solver=run0.solver, eta=run0.eta,
                 extra=f"N = {run0.N}")
    T_all = max(Y.shape[0] for Y in S0.values())
    mtxt = ", ".join(f"{P.READOUT_LABEL[ro]} {mhalf_text(mh[ro], T_all)}" for ro in S0)
    P.caption(fig, f"line: {LINE_TEXT[stat].format(what='first-teacher signal after task t', ref='after task 0')}, "
                   f"drawn at t + 1 so task 0 is at 1 ({n_seeds} seeds); band: ±1 s.e.; dotted in the readout's "
                   f"colour: eq. (S_heads) for task heads, eq. (S_frozen) for the frozen readouts; half-life m½ "
                   f"(tasks), first crossing of ½ (grey line), {MHALF_TEXT[stat]}: {mtxt}.")
    P.fig_legend(fig, list(S0), styles=[(":", "equation")])
    return P.save(fig, path, figdata=figdata)


#: First crossing of half the initial value, linearly interpolated (``sscl_core.halflife``).
half_life = H.half_life

#: What the trace is, per statistic (``what`` the signal, ``ref`` its reference point), for captions.
LINE_TEXT = {"mean": "seed mean of the {what} divided by its seed mean {ref} (the write-up's Σ)",
             "seed": "seed mean of each seed's {what} divided by its own value {ref}"}
#: What the half-life in the caption is, per statistic.
MHALF_TEXT = {"mean": "of the mean curve, ± jackknife s.e.", "seed": "of each seed, mean ± s.e. over seeds"}


def seed_signal(Y: np.ndarray, stat: str = "mean") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalized signal and half-life of a (T, n_seeds) signal under a half-life statistic (``sscl_core.halflife``).

    Returns
    -------
    (m, se, mh)
        ``stat = "mean"``: m = Σ(t) = ⟨Y(t)⟩/⟨Y(0)⟩ over seeds (eq. Sigma), its jackknife s.e., and the
        jackknife values of its half-life (their mean is the half-life of Σ, their s.e. its error).
        ``stat = "seed"``: the seed mean and s.e. of each seed's own Y(t)/Y(0), and each seed's own half-life
        (NaN when it never reaches half).
    """
    T, n = Y.shape
    if stat == "mean":
        return H.mean_curve(Y), H.mean_curve_se(Y), H.values(Y, "mean")
    with np.errstate(invalid="ignore", divide="ignore"):
        Yn = Y / Y[:1]
    m = Yn.mean(1)
    se = Yn.std(1, ddof=1) / np.sqrt(n) if n > 1 else np.zeros(T)
    return m, se, H.per_seed(Y)


def mhalf_text(mh: np.ndarray, T: int) -> str:
    """Mean ± s.e. of a half-life values array (``seed_signal``) over its finite entries, and how many are not:
    seeds (or, for the mean curve, leave-one-out curves) that never reach half within the ``T`` tasks followed."""
    fin = mh[np.isfinite(mh)]
    if len(fin) == 0:
        return f"> {T}"
    txt = f"{fin.mean():.1f}" + (f" ± {fin.std(ddof=1) / np.sqrt(len(fin)):.1f}" if len(fin) > 1 else "")
    if len(fin) < len(mh):
        txt += f" ({len(mh) - len(fin)}/{len(mh)} never reach ½, left out)"
    return txt


def main(argv: Optional[list[str]] = None) -> int:
    """Command-line entry point (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(_path.ROOT, "configs", "stage1.yaml"),
                    help="config supplying the solver, λ, step-size and stream-length rules (default: Stage 1)")
    ap.add_argument("--N", type=int, nargs="+", default=[100])
    ap.add_argument("--kappa", nargs="+", default=["1.0"])
    ap.add_argument("--delta0", type=float, nargs="+", default=[0.0])
    ap.add_argument("--lam", type=float, default=None, help="default: the config's first λ")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--tasks", type=int, default=None, help="tasks to run (default: the config's T_burn + T_meas)")
    ap.add_argument("--readout", nargs="*", default=list(READOUT_ORDER))
    ap.add_argument("--out", default=os.path.join(_path.ROOT, "results", "szero"))
    H.add_arg(ap)
    add_clock_arg(ap)
    pool.add_args(ap)
    args = ap.parse_args(argv)

    cfg = Config.from_yaml(args.config)
    over = {"name": "szero", "N": list(args.N), "kappa": list(args.kappa), "readout": list(args.readout),
            "delta0": list(args.delta0), "lam": [args.lam if args.lam is not None else cfg.lam[0]],
            "seeds": args.seeds, "extra": (), "trace": False, **clock_override(args)}
    cfg = Config(**{**cfg.__dict__, **over})
    runs = make_runs(cfg)
    if not runs:
        print("no runs left after skipping κ outside [1, √N]; nothing to do", flush=True)
        return 0
    cells_ = cells(runs)
    T_of = {key: args.tasks or (cell[0].T_burn + cell[0].T_meas) for key, cell in cells_.items()}
    if len(set(T_of.values())) != 1:
        raise SystemExit("cells differ in T; pass --tasks")
    T = next(iter(T_of.values()))
    plan = pool.plan(args, [(g, T) for g in groups(cells_).values()])
    r0 = runs[0]
    print(f"S₀(t) vs task: N={args.N} κ={args.kappa} δ₀={args.delta0} λ={r0.lam} {r0.solver} n_steps={r0.n_steps} "
          f"{r0.clock} clock T={T} seeds={args.seeds} statistic={args.stat}  [{plan.describe()}]", flush=True)
    res = collect(cells_, T, plan)

    outs = []
    for (N, k, d0, lam), S0 in res.items():
        print(f"cell N={N} κ={k} δ₀={d0} λ={lam}:")
        for ro, Y in S0.items():
            m = Y.mean(1)
            th = "" if TH.READOUT_KIND[ro][0] == "learned" else (
                f"  equation m½ = {TH.mhalf(ro, N, cells_[(N, k, d0, lam)][0].kappa if k else 1.0, lam, d0):.1f}")
            print(f"  {ro:15s} S₀(0) = {m[0]:.4f} (θ = {TH.theta(lam):.4f})  m½ = "
                  f"{mhalf_text(seed_signal(Y, args.stat)[2], Y.shape[0])}  S₀(T−1) = {m[-1]:+.4f}{th}", flush=True)
        if not S0:
            continue
        cell = cells_[(N, k, d0, lam)]
        run0 = next((r for r in cell if r.gated == 1.0), cell[0])
        stem = os.path.join(args.out, "figures", f"szero_N{N}_k{P.fmt(k)}_d0{P.fmt(d0)}_lam{P.fmt(lam)}")
        outs.append(figure(S0, run0, stem, args.stat))
        print(f"wrote {outs[-1]} and {stem}.npz")
    return 0


if __name__ == "__main__":
    _rc = main()
    _dt = time.perf_counter() - _T0
    print(f"run_szero: total {_dt:.1f} s ({_dt / 60:.2f} min) from after the imports to the end", flush=True)
    sys.exit(_rc)
