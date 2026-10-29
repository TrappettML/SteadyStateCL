#!/usr/bin/env python
"""Weight-norm figures (PLAN.md §12.3): norms vs task from batch files, norms vs step from trace files.

    python scripts/plot_norms.py results/<sweep> [--fig-dir DIR] [--save-figdata]

Family A (``figures/norms/by_task/``): one figure per (κ, δ₀, λ, solver, η);
rows = quantity, columns = N, traces = readouts, seeds thin with the seed
mean on top; a companion ``norms_prepost_*`` shows the size of the
rescaling.  Family B (``figures/norms/by_step/``): from ``traces/``, one
figure per (readout, κ, δ₀, λ, solver, η) with every measurement task
overlaid (coloured by task index), the across-task median and the 10–90 %
band, plus one ``steps_summary`` per solver.  Called by ``analyze.py``.
"""
import argparse
import os
import sys
import warnings

import numpy as np

import _path  # noqa: F401

from mup import io
from mup import plotting as P
from mup.config import is_gated
from mup.results import by_kappa, group_by, half_stats, load_runs, seed_stack

#: family-A rows: (scalar key, axis label); task_heads adds heads_fro2_post
ROWS_A = (("Q_act_pre", r"$Q_{act}$ (pre)"), ("a2_act_post", r"$a^2_{act}$"), ("aQ_act_post", r"$|a_i|\sqrt{Q_{ii}}$ act"), ("logf_sum", "Σ log f"))
ROWS_PREPOST = (("Q_act_pre-Q_act_post", r"$Q_{act}$ pre − post"), ("logf_sum", "Σ log f"), ("logf_max", r"max $|\log f|$"))
#: §12.1 keys printed in the half-window comparison
CHECK_KEYS = ("Q_act_pre", "Q_all_pre", "W_fro2_pre", "a2_act_post", "a2_all_post", "aQ_act_post", "aQ_max_post",
              "heads_fro2_post", "logf_sum", "Q_q05", "Q_q95", "aQ_q95")
#: family-B rows: (trace quantity, relative to step 0?, axis label)
ROWS_B = (("Q_act_pre", True, r"$Q_{act} - Q_{act}(0)$"), ("a2_act", True, r"$a^2_{act} - a^2_{act}(0)$"),
          ("aQ_act", True, r"$|a|\sqrt{Q} - (0)$"), ("logf", False, r"$\log f$"))


def _series(rec, key):
    """
    A per-task series by key; ``"A-B"`` gives the difference of two scalars.

    Parameters
    ----------
    rec : RunRecord
    key : str

    Returns
    -------
    (T,) array
    """
    if "-" in key:
        a, b = key.split("-")
        return rec.scalars[a] - rec.scalars[b]
    return rec.scalars[key]


def _draw_task_rows(axes, members, Ns, rows, band_keys=None):
    """Draw family-A style panels: rows × N columns, readouts as traces, seeds thin, mean on top.

    Parameters
    ----------
    axes : (n_rows, n_N) array of Axes
    members : list of RunRecord
    Ns : sorted widths
    rows : sequence of (key, label)
    band_keys : dict, optional
        ``{row key: (lo key, hi key)}`` for a seed-mean 5–95 % band.

    Returns
    -------
    dict
        Figure data (seed means) keyed ``{readout}_N{N}_{key}``.
    """
    data = {}
    for j, N in enumerate(Ns):
        for ro in P.READOUT_COLOR:
            recs = [r for r in members if r.run.N == N and r.run.readout == ro]
            if not recs:
                continue
            x = np.arange(recs[0].T)
            for i, (key, _) in enumerate(rows):
                if key.split("-")[0] not in recs[0].scalars:
                    continue
                ys = np.stack([_series(r, key) for r in recs])
                P.seeds_and_mean(axes[i, j], x, ys, ro)
                if band_keys and key in band_keys:
                    lo, hi = (np.nanmean(seed_stack(recs, k), 0) for k in band_keys[key])
                    P.band(axes[i, j], x, lo, hi, P.READOUT_COLOR[ro])
                data[f"{ro}_N{N}_{key}"] = np.nanmean(ys, 0)
        for i in range(len(rows)):
            P.shade_burnin(axes[i, j], recs[0].T_burn, recs[0].T_meas) if recs else None
    return data


def family_A(records, fig_dir, index, figdata):
    """Norms vs task (§12.3 A) and the pre/post companion, one figure each per (κ, δ₀, λ, solver, η)."""
    for kappa, kmembers in by_kappa(records).items():
        for cell, members in group_by(kmembers, ("delta0", "lam", "solver", "eta")).items():
            d0, lam, solver, eta = cell
            Ns = sorted({r.run.N for r in members})
            rows = list(ROWS_A) + ([("heads_fro2_post", r"mean $|a_k|^2$ (heads)")] if any(r.run.readout == "task_heads" for r in members) else [])
            fig, axes = P.grid(len(rows), len(Ns), sharex="col", sharey="row")
            data = _draw_task_rows(axes, members, Ns, rows, {"Q_act_pre": ("Q_q05", "Q_q95")})
            for i, (_, lab) in enumerate(rows):
                axes[i, 0].set_ylabel(lab)
            for j in range(len(Ns)):
                axes[-1, j].set_xlabel("task")
            P.ref_line(axes[rows.index(("logf_sum", "Σ log f")), 0], 0.0)
            P.label_cols_N(axes, Ns)
            P.readout_legend(axes[0, 0], {r.run.readout for r in members})
            P.cell_title(fig, kappa, d0, lam, solver, eta)
            P.caption(fig, "thin: seeds; thick: seed mean; grey: burn-in; dotted: half boundary; band: 5–95 % of $Q_i$ over active units")
            stem = P.cell_stem(kappa, d0, lam, solver, eta)
            P.save(fig, os.path.join(fig_dir, "norms", "by_task", "norms_" + stem), data if figdata else None, index, "norms/by_task", stem)

            fig, axes = P.grid(len(ROWS_PREPOST), len(Ns), sharex="col", sharey="row")
            data = _draw_task_rows(axes, members, Ns, ROWS_PREPOST)
            for i, (_, lab) in enumerate(ROWS_PREPOST):
                axes[i, 0].set_ylabel(lab); P.ref_line(axes[i, 0], 0.0)
            for j in range(len(Ns)):
                axes[-1, j].set_xlabel("task")
            P.label_cols_N(axes, Ns)
            P.readout_legend(axes[0, 0], {r.run.readout for r in members})
            P.cell_title(fig, kappa, d0, lam, solver, eta, "rescaling per task")
            P.caption(fig, "thin: seeds; thick: seed mean; grey: burn-in")
            P.save(fig, os.path.join(fig_dir, "norms", "by_task", "norms_prepost_" + stem), data if figdata else None, index, "norms/by_task", stem)


def print_half_windows(records):
    """Print the check-2 half-window comparison (half1/half0 − 1, seed mean) for every §12.1 key, per condition."""
    print("half-window ratios (half 1 / half 0 − 1, mean over seeds; * = |z| > 3 in every seed):")
    keys = [k for k in CHECK_KEYS if k in records[0].scalars]
    print("  " + " " * 44 + " ".join(f"{k[:11]:>11s}" for k in keys))
    for cond, recs in sorted(group_by(records, ("N", "kappa_label", "readout", "delta0", "lam")).items(), key=str):
        cells = []
        for k in keys:
            hs = [half_stats(r.scalars[k], r.T_burn, r.T_meas) for r in recs]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ratio = np.nanmean([h["ratio"] for h in hs])
            flag = all(h["z"] > 3 for h in hs)
            cells.append(f"{ratio:+10.3f}{'*' if flag else ' '}")
        N, kl, ro, d0, lam = cond
        print(f"  N={N:4d} κ={kl:5s} {ro:14s} δ₀={d0:5.2f} λ={lam:3.1f} " + " ".join(cells))


def family_B(out_dir, fig_dir, index, figdata):
    """Norms vs step within a task (§12.3 B) from ``traces/``; one figure per (readout, κ, δ₀, λ, solver, η).

    Columns are (N, seed) pairs (seeds are never pooled into one median),
    split into several files when more than ``plotting.MAX_COLS``.  A
    ``steps_summary`` per solver shows the medians only at δ₀ = 0, λ = 1.
    """
    traces = io.load_traces(out_dir)
    if not traces:
        return
    quant = traces[0][3]["quantities"]
    groups = {}
    for tr, tidx, run, meta in traces:
        groups.setdefault((run.readout, run.kappa_label, run.delta0, run.lam, run.solver, run.eta), []).append((tr, tidx, run, meta))
    medians = {}
    for cell, items in groups.items():
        ro, kappa, d0, lam, solver, eta = cell
        all_cols = sorted({(it[2].N, it[2].seed) for it in items})
        for part, cols in enumerate(P.col_chunks(all_cols)):
            fig, axes = P.grid(len(ROWS_B), len(cols), sharex="col", sharey="row")
            data = {}
            for j, (N, seed) in enumerate(cols):
                tr, tidx, run, meta = next(it for it in items if (it[2].N, it[2].seed) == (N, seed))
                T_burn, T_meas = meta["T_burn"], meta["T_meas"]
                x = tidx.astype(float)
                for i, (q, rel, lab) in enumerate(ROWS_B):
                    ax = axes[i, j]
                    y = tr[T_burn:T_burn + T_meas, :, quant.index(q)].astype(float)
                    if rel:
                        y = y - y[:, :1]
                    med = P.overlay(ax, x, y, P.READOUT_COLOR[ro], cmap="viridis")
                    if solver == "sgd":
                        ax.set_xscale("symlog", linthresh=10)
                    data[f"N{N}_seed{seed}_{q}_median"] = med
                    medians[(cell, N, seed, q)] = (x, med)
                axes[0, j].set_title(f"N = {N}, seed {seed}")
                axes[-1, j].set_xlabel("step within task")
            for i, (_, _, lab) in enumerate(ROWS_B):
                axes[i, 0].set_ylabel(lab); P.ref_line(axes[i, 0], 0.0)
            P.cell_title(fig, kappa if is_gated(ro) else None, d0, lam, solver, eta, P.READOUT_LABEL[ro])
            P.caption(fig, "thin: individual measurement tasks (viridis = task index); black: median over tasks; band: 10–90 %")
            stem = P.cell_stem(kappa if is_gated(ro) else None, d0, lam, solver, eta, readout=ro) + (f"_part{part + 1}" if part else "")
            P.save(fig, os.path.join(fig_dir, "norms", "by_step", "steps_" + stem), data if figdata else None, index, "norms/by_step", stem)

    # summary per solver: readouts × N grid of medians at δ₀ = 0, λ = 1 (§12.5 item 8), lowest seed
    for (solver, eta) in sorted({(c[4], c[5]) for c in groups}):
        sel = {k: v for k, v in medians.items() if k[0][4] == solver and k[0][5] == eta and k[0][2] == 0.0 and k[0][3] == 1.0}
        if not sel:
            continue
        ros = [r for r in P.READOUT_COLOR if any(k[0][0] == r for k in sel)]
        Ns = sorted({k[1] for k in sel})
        fig, axes = P.grid(len(ros), len(Ns), sharex="col", sharey=False)
        data = {}
        shade = {"Q_act_pre": None, "a2_act": "0.3", "aQ_act": "0.6"}
        for i, ro in enumerate(ros):
            for j, N in enumerate(Ns):
                for q, rel, lab in ROWS_B[:3]:
                    seeds = [k[2] for k in sel if k[0][0] == ro and k[1] == N and k[3] == q]
                    if not seeds:
                        continue
                    x, med = sel[next(k for k in sel if k[0][0] == ro and k[1] == N and k[3] == q and k[2] == min(seeds))]
                    axes[i, j].plot(x, med, lw=P.LW_TRACE, label=lab, color=shade[q] or P.READOUT_COLOR[ro], zorder=4)
                    data[f"{ro}_N{N}_{q}"] = med
                if solver == "sgd":
                    axes[i, j].set_xscale("symlog", linthresh=10)
                P.ref_line(axes[i, j], 0.0)
            axes[i, 0].set_ylabel(P.READOUT_LABEL[ro], color=P.READOUT_COLOR[ro])
        P.label_cols_N(axes, Ns)
        axes[0, 0].legend(loc="best")
        for j in range(len(Ns)):
            axes[-1, j].set_xlabel("step within task")
        P.cell_title(fig, None, 0.0, 1.0, solver, eta, "step summary")
        P.caption(fig, "medians over measurement tasks, relative to step 0; readout colour: $Q_{act}$, dark grey: $a^2_{act}$, light grey: $|a|\\sqrt{Q}$")
        stem = P.cell_stem(None, None, None, solver, eta)
        P.save(fig, os.path.join(fig_dir, "norms", "by_step", "steps_summary_" + stem), data if figdata else None, index, "norms/by_step", stem)


def run(out_dir, fig_dir, records=None, figdata=False):
    """Make every norm figure of a sweep.

    Parameters
    ----------
    out_dir : str
        ``results/<sweep>``.
    fig_dir : str
        Figure root (``figures/``); ``norms/by_task`` and ``norms/by_step`` are created under it.
    records : list of RunRecord, optional
        Loaded records (loaded here when omitted).
    figdata : bool
        Write the plotted arrays as ``.npz`` next to each figure.

    Returns
    -------
    list of (family, cell, file)
        INDEX.md rows.
    """
    if records is None:
        records, _ = load_runs(out_dir)
    index = []
    if records:
        family_A(records, fig_dir, index, figdata)
        print_half_windows(records)
    family_B(out_dir, fig_dir, index, figdata)
    return index


def main(argv=None):
    """
    Command-line entry point (see the module docstring).

    Parameters
    ----------
    argv : list of str, optional

    Returns
    -------
    int
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir")
    ap.add_argument("--fig-dir", default=None)
    ap.add_argument("--save-figdata", action="store_true")
    args = ap.parse_args(argv)
    fig_dir = args.fig_dir or os.path.join(args.out_dir, "figures")
    index = run(args.out_dir, fig_dir, None, args.save_figdata)
    print(f"{len(index)} figures → {fig_dir}/norms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
