#!/usr/bin/env python
"""Half-life over the two rate parameters, the hidden-layer learning per task λ and the imbalance δ₀: heat maps per readout.

    python scripts/run_lr_comp.py [--N 100] [--kappa 1] [--lam 0.5 1 2 4 8 16] [--delta0 -0.75 -0.5 0 1 4 16]
                                  [--seeds 4] [--T T] [--tau τ] [--force] [--stat mean|seed] [--clock realized|nominal] [--gpus]

The two rates η_a, η_W of the model are the pair (η, δ₀), with η entering through λ (write-up, "The rate ratio is
an imbalance"): λ is the hidden-layer learning per task (the frozen-readout exponent) and δ₀ = a_i² − Q_ii the
rate ratio η_a/η_W in equal-rate variables.  Neither sets the learning through the readout directly: the readout
exponent λᵃ_c = Λ‖M_c‖ follows from them and the state, ∝ 1/(1 + δ₀) at fixed λ ("Readout exponent").  One
figure for the cell (N, κ), ``results/lr_comp/figures/mhalf_lr_N{N}_k{κ}.png``: a column per readout, two rows,
every panel a heat map with x = δ₀ and y = λ (both as categories, in increasing order), colour and the number
in the cell = the half-life m½ in tasks:

row 1, first memory      S₀(t) = u_t·w₀*/|w₀*|² from a fresh start, m½ the first t with S₀(t) ≤ S₀(0)/2.
row 2, steady state      the teacher of task T followed for τ more tasks, m½ the first τ with S_T(τ) ≤ S_T(0)/2.

(the two rows of ``run_halflife_lam.py``; half-lives linearly interpolated; ``--stat mean``, the default, is the
write-up's m½, the first crossing of ½ by the mean curve over seeds, ``--stat seed`` the mean of the per-seed
half-lives — ``sscl_core.halflife``.)  Each row has its own colour bar, shared by its panels, with limits taken from
that row's data: logarithmic when the row spans more than a factor ``LOG_RATIO``, linear otherwise.

Safe (λ, δ₀).  A cell is run only when the write-up's conditions hold for its readout (``safety``):

1. λ > 0 (a task takes a step) and δ₀ > −σ₀² = −1 (the initial readout a_i² = δ₀ + Q_ii is real; "Initialization").
2. the population step contracts the residual in every direction, eq. (popstab) of "Population step h":
   h η (n_c/pN)(δ₀ + 2 Q̄_c) < 2 for the projected readouts (gated shared, learned shared, task heads), with
   n_c/pN at its mean 1 and Q̄_c the larger of its initial value σ₀² and the steady-state estimate
   Q̄_act = ā_act − δ₀ of eq. (abarss), θ = 1 − e^{−λ}, pN = κ√N (N ungated) — ``config.Run.popstab``,
   ``theory.popstab_steady``.  Under clock (ii), the default, hη = λ/(n ā²_c) follows the realized key norm
   and the left side is (λ/n)(δ₀ + 2Q̄)/(δ₀ + Q̄), which the step rule λ/n ≤ 0.1 keeps far below 2; under
   ``--clock nominal`` h is fixed from ā²_nom = 1 + δ₀ and the condition fails where the settled norm has
   outgrown it (small or negative δ₀, large pN).  The frozen readouts have the hidden channel alone,
   which the step-size rule always resolves.

The bound is sufficient, not necessary, and rests on a leading-order norm estimate, so it is conservative
(strongly so for the ungated learned readout, whose norm the estimate was not derived for).  A cell that fails
is skipped and left BLANK; ``--force`` runs the cells that fail condition 2 anyway (never condition 1).  A cell
that ran but has no half-life (censored within its window, or infeasible, δ₀ < −min_i Q0_i) is
hatched grey.  Skipped and hatched cells are listed on the console.

N = 100, κ = 1 by default; solver and step rule from ``configs/stage1.yaml``.  T defaults to T_burn + T_meas
(14 N tasks) and ``--tau`` to T.  The half-lives (the statistic drawn, the per-seed ones under ``…_seed``) and
the safety table go to ``mhalf_lr_….npz``.

Placement (``sscl_core.pool``): n_steps is static in the kernels, so every λ is its own set of ``vmap`` batches, one
per ``experiments.group_key`` with every δ₀ and seed together (δ₀ < −0.9 has its own n_steps, hence its own
batch).  Each batch is ONE stream of T + τ + 1 tasks (``experiments.first_and_steady_signal``).  The batches
run as jobs, costliest first, over the GPUs (``--gpus``), over ``--workers`` CPU processes, or in this process.
"""
import argparse
import math
import os
import sys
import time
from typing import Callable, Optional

import numpy as np

import _path  # noqa: F401

from sscl_core import experiments as X
from sscl_core import halflife as H
from sscl_core import plotting as P
from sscl_core import pool
from sscl_core.config import Config, READOUT_ORDER, Run, add_clock_arg, clock_override, make_runs
from run_Nscaling import _stat
from run_halflife_lam import ROWS, cost, make_jobs, reduce_rows

_T0 = time.perf_counter()        # wall clock from after the imports to the end of the script

#: Default grids: the λ axis in octaves, the δ₀ of Stage 1 (Table of the write-up).
LAM_DEFAULT = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
DELTA0_DEFAULT = (-0.75, -0.5, 0.0, 1.0, 4.0, 16.0)

#: Sequential colour map of the half-life (perceptually uniform, monotone in lightness, colour-blind safe).
CMAP = "viridis"
#: A row whose half-lives span more than this factor gets a logarithmic colour scale.
LOG_RATIO = 8.0


def stab_bound(r: Run) -> float:
    """Left side of the contraction condition of the population step for run ``r`` (safe iff < 2): eq. (popstab)
    at the estimated steady state under the run's clock, ``config.Run.popstab`` (``theory.popstab_steady``).

    Projected readouts: h η (δ₀ + 2 Q̄) at n_c = pN, with Q̄ = max(σ₀², ā_act − δ₀) and ā_act of eq. (abarss);
    hη = λ/(n(δ₀ + Q̄)) under clock (ii).  Frozen readouts: the hidden channel alone.
    """
    return r.popstab()


def safety(r: Run) -> tuple[bool, float]:
    """(safe, bound) of a run: ``stab_bound`` < 2 (λ > 0 and δ₀ > −1 are filtered before the runs are made)."""
    b = stab_bound(r)
    return b < 2.0, b


def cond_key(r: Run) -> tuple:
    """(readout, λ, δ₀) of a run: one cell of a heat map."""
    return (r.readout, r.lam, r.delta0)


def groups(runs: list[Run], n_tasks: int) -> dict[tuple, list[Run]]:
    """``{(λ,) + group_key: runs}`` over every δ₀ and seed, costliest job first."""
    g = {}
    for r in runs:
        g.setdefault((r.lam,) + X.group_key(r), []).append(r)
    return {k: g[k] for k in sorted(g, key=lambda k: -cost(g[k], n_tasks))}


def collect(runs: list[Run], T: int, tau: int, plan: pool.Plan, stat: str = "mean",
            log: Callable[[str], None] = print) -> dict[tuple, dict[str, np.ndarray]]:
    """``{(readout, λ, δ₀): {"S0": (n_seeds,) m½ values of the first memory, "ST": those at steady state, "S0_seed",
    "ST_seed": the per-seed half-lives}}`` (``run_halflife_lam.reduce_rows``; ``stat`` as ``sscl_core.halflife``).

    Parameters
    ----------
    runs : list of config.Run
        The safe runs of one cell (N, κ).
    T, tau : int
        Tasks the first memory is followed for (and trained before the steady-state probe); tasks that probe is followed for.
    plan : pool.Plan
        Placement and pieces per group (``pool.plan`` over ``groups(runs, T + tau + 1)``).
    log : callable
        Progress sink.

    Returns
    -------
    dict
        NaN for a censored or infeasible seed.
    """
    jobs = make_jobs(groups(runs, T + tau + 1), T, tau, plan.pieces)
    res = pool.run_jobs("sscl_core.experiments:first_and_steady_signal", jobs, log=log, **plan.place)
    red = reduce_rows(jobs, res, cond_key, stat)
    out = {}
    for key in sorted(red, key=lambda k: (READOUT_ORDER.index(k[0]), k[1], k[2])):
        out[key] = red[key]
        txt = []
        for (name, label, _), window in zip(ROWS, (T, tau)):
            m, se = _stat(out[key][name])
            n_nan = int(np.isnan(out[key][name]).sum())
            txt.append(f"{label} m½ = {m:8.2f} ± {se:6.2f} (censored or infeasible {n_nan}/{len(out[key][name])}, window {window})")
        log(f"  {key[0]:15s} λ={key[1]:5g} δ₀={key[2]:6g} " + "   ".join(txt))
    return out


def colour_scale(values: np.ndarray) -> tuple[bool, float, float]:
    """Colour limits of one row from its data: (log?, lo, hi) over the finite positive cells.

    Logarithmic when max/min > ``LOG_RATIO`` (half-lives of the gated and ungated readouts differ by ~N);
    a linear scale is padded by 2 % so the extreme cells are not at the very ends of the map.
    """
    v = values[np.isfinite(values) & (values > 0)]
    if not len(v):
        return False, 0.0, 1.0
    lo, hi = float(v.min()), float(v.max())
    if hi / lo > LOG_RATIO:
        return True, lo, hi
    pad = 0.02 * (hi - lo) if hi > lo else 0.05 * hi
    return False, lo - pad, hi + pad


def log_ticks(lo: float, hi: float) -> list[float]:
    """1–2–5 ticks inside [lo, hi] (decades alone once there are more than eight)."""
    ks = range(int(math.floor(math.log10(lo))) - 1, int(math.ceil(math.log10(hi))) + 2)
    t = [m * 10.0 ** k for k in ks for m in (1, 2, 5) if lo * 0.999 <= m * 10.0 ** k <= hi * 1.001]
    if len(t) > 8:
        t = [x for x in t if abs(math.log10(x) - round(math.log10(x))) < 1e-9]
    return t or [lo, hi]


def cell_fmt(v: float) -> str:
    """Half-life printed in a cell: two significant digits below 10, an integer above."""
    return f"{v:.2g}" if v < 9.95 else f"{v:.0f}"


def figure(res: dict, status: dict, lams: list[float], d0s: list[float], readouts: list[str], run0: Run,
           T: int, tau: int, path: str, stat: str = "mean") -> str:
    """Heat maps of m½ on (λ rows, δ₀ columns): a column per readout, row 1 first memory, row 2 steady state.

    Parameters
    ----------
    res : dict
        ``collect`` output.
    status : dict
        ``{(readout, λ, δ₀): (ran, bound)}`` — whether the cell was run, and its ``stab_bound`` (NaN: δ₀ ≤ −1).
    lams, d0s : list of float
        The grids, increasing (rows upward, columns rightward).
    readouts : list of str
        The columns.
    run0 : config.Run
        A gated run when there is one (its κ goes in the title), else any run.
    T, tau : int
        The two follow windows, printed in the caption.
    path : str
        Figure stem.

    Returns
    -------
    str
        PNG path.
    """
    from matplotlib.patches import Rectangle
    from matplotlib.ticker import FuncFormatter

    P.apply_style()
    fig, axes = P.grid(len(ROWS), len(readouts), sharex="all", sharey="all", panel=(2.25, 2.1), extra_w=0.9)
    skipped = np.array([[[not status[(ro, lam, d0)][0] for d0 in d0s] for lam in lams] for ro in readouts])
    figdata = {"lam": np.array(lams, float), "delta0": np.array(d0s, float),
               "stab_bound": np.array([[[status[(ro, lam, d0)][1] for d0 in d0s] for lam in lams] for ro in readouts]),
               "skipped": skipped}
    n_nohalf = []
    for row, (name, label, tag) in enumerate(ROWS):
        M = np.full((len(readouts), len(lams), len(d0s)), np.nan)
        for c, ro in enumerate(readouts):
            for i, lam in enumerate(lams):
                for j, d0 in enumerate(d0s):
                    if (ro, lam, d0) in res:
                        mh = res[(ro, lam, d0)][name]
                        figdata[f"mhalf_{tag}_{ro}_lam{P.fmt(lam)}_d0{P.fmt(d0)}"] = mh
                        figdata[f"mhalf_{tag}_{ro}_lam{P.fmt(lam)}_d0{P.fmt(d0)}_seed"] = res[(ro, lam, d0)][name + "_seed"]
                        M[c, i, j] = _stat(mh)[0]
        M[~(M > 0)] = np.nan
        figdata[f"{tag}_mean"] = M
        n_nohalf.append(int((np.isnan(M) & ~skipped).sum()))
        is_log, lo, hi = colour_scale(M)
        tr = np.log10 if is_log else (lambda v: v)
        with np.errstate(invalid="ignore"):
            C = tr(M)
        im = None
        for c, ro in enumerate(readouts):
            ax = axes[row, c]
            im = P.heat_map(ax, C[c], M[c], lams, d0s, CMAP, float(tr(lo)), float(tr(hi)), cell_fmt)
            for (i, j), s in np.ndenumerate(skipped[c]):            # an unsafe cell is blank, not hatched
                if s:
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fc="white", ec="none", zorder=2.5))
            ax.tick_params(axis="x", labelrotation=0, length=0)
            ax.tick_params(axis="y", length=0)
            if row == 0:
                ax.set_title(P.READOUT_LABEL[ro], fontsize=9, color="k")
            if row == len(ROWS) - 1:
                ax.set_xlabel(r"imbalance $\delta_0$")
        axes[row, 0].set_ylabel(f"{label}\n" + r"hidden-layer learning per task $\lambda$")
        cb = fig.colorbar(im, ax=list(axes[row, :]), pad=0.015, fraction=0.03, aspect=18)
        if is_log:
            cb.set_ticks([math.log10(t) for t in log_ticks(lo, hi)])
            cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{10.0 ** v:g}"))
            cb.ax.minorticks_off()
        cb.set_label(r"$m_{1/2}$ (tasks)" + (", log scale" if is_log else ""))
        cb.outline.set_linewidth(0.5)
    kappa = run0.kappa_label if run0.gated == 1.0 else None
    P.cell_title(fig, kappa=kappa, solver=run0.solver, eta=run0.eta, extra=f"N = {run0.N}")
    n_seeds = max((len(d[ROWS[0][0]]) for d in res.values()), default=0)
    P.caption(fig, f"Colour and number: half-life in tasks; {H.STAT_TEXT[stat]} ({n_seeds} seeds per "
                   f"cell); each row has its own colour scale. Top: first teacher from a fresh start, within T = {T} "
                   f"tasks. Bottom: the teacher of task T, within τ ≤ {tau} more. Blank: (λ, δ₀) not safe for that "
                   f"readout, not run ({int(skipped.sum())} cells per row). Hatched: run, never below half "
                   f"({n_nohalf[0]} top, {n_nohalf[1]} bottom).")
    return P.save(fig, path, figdata=figdata)


def main(argv: Optional[list[str]] = None) -> int:
    """Command-line entry point (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(_path.ROOT, "configs", "stage1.yaml"),
                    help="config supplying the solver, step-size and stream-length rules (default: Stage 1)")
    ap.add_argument("--N", type=int, default=100)
    ap.add_argument("--kappa", default="1", help="κ of the gated readouts, as typed (the ungated ones have none)")
    ap.add_argument("--lam", type=float, nargs="+", default=list(LAM_DEFAULT), help="the λ grid, the y axis")
    ap.add_argument("--delta0", type=float, nargs="+", default=list(DELTA0_DEFAULT), help="the δ₀ grid, the x axis")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--T", type=int, default=None,
                    help="tasks the first memory is followed for and trained before the steady-state probe "
                         "(default: the config's T_burn + T_meas)")
    ap.add_argument("--tau", type=int, default=None, help="tasks the steady-state probe is followed for (default: T)")
    ap.add_argument("--readout", nargs="*", default=list(READOUT_ORDER))
    ap.add_argument("--force", action="store_true",
                    help="also run the cells that fail the step-contraction bound (never λ ≤ 0 or δ₀ ≤ −1)")
    ap.add_argument("--out", default=os.path.join(_path.ROOT, "results", "lr_comp"))
    H.add_arg(ap)
    add_clock_arg(ap)
    pool.add_args(ap)
    args = ap.parse_args(argv)

    lams, d0s = sorted({float(l) for l in args.lam}), sorted({float(d) for d in args.delta0})
    lam_ok, d0_ok = [l for l in lams if l > 0], [d for d in d0s if d > -1.0]
    for bad, why in ((sorted(set(lams) - set(lam_ok)), "λ ≤ 0 takes no step"),
                     (sorted(set(d0s) - set(d0_ok)), "δ₀ ≤ −1 makes the initial readout imaginary")):
        if bad:
            print(f"unsafe, left blank for every readout: {bad} ({why})", flush=True)
    cfg = Config.from_yaml(args.config)
    over = {"name": "lr_comp", "N": [args.N], "kappa": [args.kappa], "readout": list(args.readout),
            "delta0": d0_ok, "lam": lam_ok, "seeds": args.seeds, "extra": (), "trace": False, **clock_override(args)}
    all_runs = make_runs(Config(**{**cfg.__dict__, **over})) if lam_ok and d0_ok else []
    if not all_runs:
        print("no runs (no safe λ and δ₀, or κ outside [1, √N]); nothing to do", flush=True)
        return 0
    readouts = [ro for ro in READOUT_ORDER if any(r.readout == ro for r in all_runs)]

    status = {(ro, lam, d0): (False, float("nan")) for ro in readouts for lam in lams for d0 in d0s}
    runs = []
    for r in all_runs:
        ok, b = safety(r)
        status[cond_key(r)] = (ok or args.force, b)
        if ok or args.force:
            runs.append(r)
    unsafe = sorted(((k, b) for k, (_, b) in status.items() if np.isfinite(b) and b >= 2.0),
                    key=lambda kb: (READOUT_ORDER.index(kb[0][0]), kb[0][1], kb[0][2]))
    if unsafe:
        print(f"step-contraction bound h η (δ₀ + 2 Q̄) ≥ 2 (eq. popstab, {all_runs[0].clock} clock) in {len(unsafe)} cells "
              f"({'run anyway, --force' if args.force else 'skipped, left blank'}):", flush=True)
        for (ro, lam, d0), b in unsafe:
            print(f"  {ro:15s} λ={lam:5g} δ₀={d0:6g}  bound = {b:.2f}", flush=True)
    if not runs:
        print("no safe cell; nothing to do", flush=True)
        return 0

    T = args.T or (runs[0].T_burn + runs[0].T_meas)
    tau = args.tau or T
    plan = pool.plan(args, [(g, T + tau + 1) for g in groups(runs, T + tau + 1).values()])
    r0 = runs[0]
    n_cells = sum(1 for ran, _ in status.values() if ran)
    print(f"m½ over (λ, δ₀): N={args.N} κ={args.kappa} λ={lam_ok} δ₀={d0_ok} {r0.solver} "
          f"n_steps={min(r.n_steps for r in runs)}…{max(r.n_steps for r in runs)} {r0.clock} clock T={T} τ≤{tau} "
          f"seeds={args.seeds} statistic={args.stat} cells={n_cells}/{len(status)}  [{plan.describe()}]", flush=True)
    res = collect(runs, T, tau, plan, args.stat)

    run0 = next((r for r in runs if r.gated == 1.0), r0)
    stem = os.path.join(args.out, "figures", f"mhalf_lr_N{args.N}_k{P.fmt(run0.kappa_label)}")
    out = figure(res, status, lams, d0s, readouts, run0, T, tau, stem, args.stat)
    print(f"wrote {out} and {stem}.npz")
    return 0


if __name__ == "__main__":
    _rc = main()
    _dt = time.perf_counter() - _T0
    print(f"run_lr_comp: total {_dt:.1f} s ({_dt / 60:.2f} min) from after the imports to the end", flush=True)
    sys.exit(_rc)
