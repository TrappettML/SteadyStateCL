#!/usr/bin/env python
"""Weight-norm figures (PLAN.md §12.3): norms vs task from batch files; per-step traces only with ``--diag``.

    python scripts/plot_norms.py results/<sweep> [--fig-dir DIR] [--save-figdata] [--diag]

S1, the default (``slices/norms_{readout}_…``): one file per readout and
κ; rows = quantity (rows constant by construction for that readout
dropped), columns = δ₀, traces = N, x = t κ_s²/N (lifetimes).
M8 (``slices/norms_lt2_{readout}_…``, PLAN.md §12.3): S1 on the measured
time scale, x = t/m½ with the condition's own half-life and y divided by
its second-half mean, so 1 is the steady level; burn-in end marked per N.
Per-cell, only for the records ``analyze.py`` selects with ``--cells``:
F1 (``cells/norms/norms_…``): rows = quantity, columns = N, traces = readouts.
F2 (``cells/norms/normsN_…`` and ``normsN_lt_…``): rows = quantity, columns =
readouts, traces = N (x = task, or lifetimes in the ``_lt`` file).
Every trace is a block mean over half a lifetime (§12.7 G10) with the raw
per-task seed mean faint behind it; nothing is shaded, the half boundary
is one dotted line.  The by-step family of §12.2 (``diag/by_step/``) is
drawn only with ``--diag``.  Called by ``analyze.py``.
"""
import argparse
import os
import sys

import numpy as np

import _path  # noqa: F401

from mup import io
from mup import plotting as P
from mup.config import is_gated
from mup.results import by_kappa, group_by, kappa_min, load_runs, mhalf_by_condition, seed_stack, slice_groups

#: F1/F2 rows: (scalar key, quantile keys or None); heads_fro2_post is appended when task heads are present
ROWS = (("Q_act_pre", ("Q_q05", "Q_q95")), ("a2_act_post", None), ("aQ_act_post", ("aQ_q05", "aQ_q95")),
        ("aQ_max_post", None), ("logf_sum", None))
#: family-B rows (diag): (trace quantity, relative to step 0?, axis label)
ROWS_B = (("Q_act_pre", True, P.qlabel("Q_act_pre") + " − step 0"), ("a2_act", True, P.qlabel("a2_act") + " − step 0"),
          ("aQ_act", True, P.qlabel("aQ_act") + " − step 0"), ("logf", False, P.qlabel("logf")))


def _na():
    from plot_check2 import NA_BY_CONSTRUCTION
    return NA_BY_CONSTRUCTION


def _rows(members):
    rows = list(ROWS)
    if any(r.run.readout == "task_heads" for r in members):
        rows.append(("heads_fro2_post", None))
    return rows


def _block_trace(ax, x, Y, w, cuts, color, ls, quantiles=None):
    """Block mean (lw 1.6, ±1 s.e. band) over the faint per-task seed mean; quantile block means as thin lines.

    Returns
    -------
    (xb, mb, sb)
    """
    with np.errstate(invalid="ignore"):
        raw = np.nanmean(np.asarray(Y, float), 0)
    ax.plot(x, raw, color=color, ls=ls, lw=0.5, alpha=0.25, zorder=2)
    xb, mb, sb = P.block_mean(x, Y, w, cuts)
    ax.plot(xb, mb, color=color, ls=ls, lw=P.LW_TRACE, zorder=4)
    P.band(ax, xb, mb - sb, mb + sb, color)
    for Q in (quantiles or []):
        _, qb, _ = P.block_mean(x, Q, w, cuts)
        ax.plot(xb, qb, color=color, ls=ls, lw=P.LW_OVERLAY, zorder=3)
    return xb, mb, sb


def _omitted(members, rows):
    """Caption text naming the (quantity, readout) pairs left out as constant by construction."""
    na = _na()
    ros = {r.run.readout for r in members}
    parts = [f"{P.qlabel(k)} for {', '.join(P.READOUT_LABEL[ro] for ro in P.READOUT_COLOR if ro in ros and (k, ro) in na)}"
             for k, _ in rows if any((k, ro) in na for ro in ros)]
    return ("; omitted (constant by construction): " + "; ".join(parts)) if parts else ""


def _panel_missing(key, ros, N, na, missing):
    """Whether every readout of an F1 panel (quantity ``key``, width ``N``) lacks the scalar (older batch files)."""
    live = [ro for ro in ros if (key, ro) not in na]
    return bool(live) and all((key, ro, N) in missing for ro in live)


def _missing_text(missing):
    """Caption text naming (quantity, readout, N) traces absent because their batch files predate the scalar."""
    if not missing:
        return ""
    by_N = {}
    for key, ro, N in missing:
        by_N.setdefault(N, set()).add(key)
    parts = [f"N = {P.fmt(N)}: " + ", ".join(P.qlabel(k) for k in sorted(keys)) for N, keys in sorted(by_N.items())]
    return "; not recorded (older batch files, rerun the sweep): " + "; ".join(parts)


def fig_S1(records, fig_dir, index, figdata):
    """S1: norms slice, one file per (readout, κ, λ, solver, η); rows = quantities, columns = δ₀, traces = N, x in lifetimes (PLAN.md §12.3)."""
    na, kmin = _na(), kappa_min(records)
    for (ro, kappa, lam, solver, eta), members in slice_groups(records).items():
        rows = [(k, qk) for k, qk in _rows(members) if (k, ro) not in na]
        d0s_all = sorted({r.run.delta0 for r in members})
        Ns = sorted({r.run.N for r in members})
        for part, d0s in enumerate(P.col_chunks(d0s_all)):
            fig, axes = P.grid(len(rows), len(d0s), sharex="all", sharey="row")
            data, missing = {}, set()
            for j, d0 in enumerate(d0s):
                for i_N, N in enumerate(Ns):
                    recs = [r for r in members if r.run.N == N and r.run.delta0 == d0]
                    if not recs:
                        continue
                    r0 = recs[0]
                    ks = r0.kappa_sched(kmin)
                    scale = ks ** 2 / N
                    x = np.arange(r0.T, dtype=float) * scale
                    w = max(1, int(round(N / (2 * ks ** 2))))
                    cuts = (r0.T_burn + r0.T_meas // 2,)
                    col = P.n_shade(i_N, len(Ns))
                    for i, (key, qk) in enumerate(rows):
                        if key not in r0.scalars:
                            missing.add((key, ro, N))
                            continue
                        xb, mb, sb = _block_trace(axes[i, j], x, seed_stack(recs, key), w, cuts, col, P.READOUT_STYLE[ro],
                                                  [seed_stack(recs, k) for k in qk] if qk else None)
                        data[f"d{d0}_N{N}_{key}"] = np.stack([xb, mb, sb])
                    half = (r0.T_burn + r0.T_meas / 2) * scale
                    for i in range(len(rows)):
                        axes[i, j].axvline(half, color=col, ls=":", lw=P.LW_EVENT, zorder=1)
                for i, (key, _) in enumerate(rows):
                    if all((key, ro, N) in missing for N in Ns if any(r.run.N == N and r.run.delta0 == d0 for r in members)):
                        P.na_slot(axes[i, j], "not recorded\n(older batch file)")
                axes[0, j].set_title(rf"$\delta_0$ = {P.fmt(d0)}")
                axes[-1, j].set_xlabel(r"$t\kappa_s^2/N$ (lifetimes)")
            for i, (key, _) in enumerate(rows):
                axes[i, 0].set_ylabel(P.qlabel(key))
            keys = [k for k, _ in rows]
            if "logf_sum" in keys:
                P.ref_line(axes[keys.index("logf_sum"), 0], 0.0)
            P.fig_legend(fig, None, handles=P.n_handles(Ns))
            P.cell_title(fig, kappa, None, lam, solver, eta, P.READOUT_LABEL[ro] + ", traces = N")
            P.caption(fig, "thick: mean over seeds and blocks of half a lifetime, band ±1 s.e.; faint: per-task seed mean; "
                           "thin: 5 % and 95 % over active units; dotted: half boundary of each N"
                      + ("; dashed: ungated" if not is_gated(ro) else "") + _omitted(members, _rows(members)) + _missing_text(missing))
            stem = P.cell_stem(kappa, None, lam, solver, eta, readout=ro) + (f"_part{part + 1}" if part else "")
            P.save(fig, os.path.join(fig_dir, "slices", "norms_" + stem), data if figdata else None, index, "slices", stem)


def fig_M8(records, fig_dir, index, figdata):
    """M8: S1 on the measured time scale: x = t/m½ (the condition's own seed-pooled half-life), y relative to the second-half mean (PLAN.md §12.3).

    Block width = half a measured half-life; the end of burn-in is marked per N
    in its shade (dashed) and the half boundary dotted; grey reference at 1.
    Conditions whose half-life is censored have no time scale and are left out
    (named in the caption).
    """
    na, mh = _na(), mhalf_by_condition(records)
    for (ro, kappa, lam, solver, eta), members in slice_groups(records).items():
        rows = [(k, qk) for k, qk in _rows(members) if (k, ro) not in na]
        d0s_all = sorted({r.run.delta0 for r in members})
        Ns = sorted({r.run.N for r in members})
        for part, d0s in enumerate(P.col_chunks(d0s_all)):
            fig, axes = P.grid(len(rows), len(d0s), sharex="all", sharey="row")
            data, missing, left_out = {}, set(), []
            for j, d0 in enumerate(d0s):
                drawn = set()
                for i_N, N in enumerate(Ns):
                    recs = [r for r in members if r.run.N == N and r.run.delta0 == d0]
                    if not recs:
                        continue
                    r0 = recs[0]
                    m = mh[r0.key()]["m_half"]
                    if not (np.isfinite(m) and m > 0):
                        left_out.append(f"δ₀ = {P.fmt(d0)}, N = {P.fmt(N)}")
                        continue
                    x = np.arange(r0.T, dtype=float) / m
                    w = max(1, int(round(m / 2)))
                    mid = r0.T_burn + r0.T_meas // 2
                    cuts = (r0.T_burn, mid)
                    col = P.n_shade(i_N, len(Ns))
                    for i, (key, qk) in enumerate(rows):
                        if key not in r0.scalars:
                            missing.add((key, ro, N))
                            continue
                        Y = seed_stack(recs, key)
                        with np.errstate(invalid="ignore"):
                            level = np.nanmean(Y[:, mid:r0.T])
                        if not np.isfinite(level) or level == 0:
                            continue
                        xb, mb, sb = _block_trace(axes[i, j], x, Y / level, w, cuts, col, P.READOUT_STYLE[ro],
                                                  [seed_stack(recs, k) / level for k in qk] if qk else None)
                        drawn.add(key)
                        data[f"d{d0}_N{N}_{key}"] = np.stack([xb, mb, sb])
                        data[f"d{d0}_N{N}_{key}_level"] = np.array([level, m])
                    for i in range(len(rows)):
                        axes[i, j].axvline(r0.T_burn / m, color=col, ls="--", lw=P.LW_EVENT, zorder=1)
                        axes[i, j].axvline((r0.T_burn + r0.T_meas / 2) / m, color=col, ls=":", lw=P.LW_EVENT, zorder=1)
                for i, (key, _) in enumerate(rows):
                    if key not in drawn and all((key, ro, N) in missing for N in Ns if any(r.run.N == N and r.run.delta0 == d0 for r in members)):
                        P.na_slot(axes[i, j], "not recorded\n(older batch file)")
                    elif key not in drawn:
                        P.na_slot(axes[i, j], "no time scale\n(censored m½)")
                axes[0, j].set_title(rf"$\delta_0$ = {P.fmt(d0)}")
                axes[-1, j].set_xlabel(r"$t / m_{1/2}$ (half-lives)")
            for i, (key, _) in enumerate(rows):
                axes[i, 0].set_ylabel(P.qlabel(key) + "\n/ level")
                for j in range(len(d0s)):
                    P.ref_line(axes[i, j], 1.0)
            P.fig_legend(fig, None, handles=P.n_handles(Ns))
            P.cell_title(fig, kappa, None, lam, solver, eta, P.READOUT_LABEL[ro] + ", in measured half-lives")
            cap = ("x: task / the condition's own m½; y: quantity / its mean over the second half of the measurement window (grey: 1); "
                   "thick: mean over seeds and blocks of half a half-life, band ±1 s.e.; faint: per-task seed mean; thin: 5 % and 95 % over active units; "
                   "dashed: end of burn-in of each N; dotted: its half boundary; pass: every trace at 1 before burn-in ends")
            if not is_gated(ro):
                cap += "; dashed traces: ungated"
            if left_out:
                cap += "; left out (censored m½): " + ", ".join(left_out[:4]) + (", …" if len(left_out) > 4 else "")
            P.caption(fig, cap + _omitted(members, _rows(members)) + _missing_text(missing))
            stem = P.cell_stem(kappa, None, lam, solver, eta, readout=ro) + (f"_part{part + 1}" if part else "")
            P.save(fig, os.path.join(fig_dir, "slices", "norms_lt2_" + stem), data if figdata else None, index, "slices", stem)


def fig_F1(records, fig_dir, index, figdata):
    """F1: norms vs task, columns = N, traces = readouts; one figure per (κ, δ₀, λ, solver, η)."""
    na, kmin = _na(), kappa_min(records)
    for kappa, kmembers in by_kappa(records).items():
        for cell, members in group_by(kmembers, ("delta0", "lam", "solver", "eta")).items():
            d0, lam, solver, eta = cell
            Ns = sorted({r.run.N for r in members})
            rows = _rows(members)
            fig, axes = P.grid(len(rows), len(Ns), sharex="col", sharey="row")
            data, missing = {}, set()
            for j, N in enumerate(Ns):
                col = [r for r in members if r.run.N == N]
                gated = [r for r in col if is_gated(r.run.readout)]
                ref = gated[0] if gated else col[0]
                scale = ref.kappa_sched(kmin) ** 2 / N
                for ro in P.READOUT_COLOR:
                    recs = [r for r in col if r.run.readout == ro]
                    if not recs:
                        continue
                    r0 = recs[0]
                    ks = r0.kappa_sched(kmin)
                    x = np.arange(r0.T, dtype=float)
                    w = max(1, int(round(N / (2 * ks ** 2))))
                    cuts = (r0.T_burn + r0.T_meas // 2,)
                    for i, (key, qk) in enumerate(rows):
                        if (key, ro) in na:
                            continue
                        if key not in r0.scalars:
                            missing.add((key, ro, N))
                            continue
                        xb, mb, sb = _block_trace(axes[i, j], x, seed_stack(recs, key), w, cuts, P.READOUT_COLOR[ro], P.READOUT_STYLE[ro],
                                                  [seed_stack(recs, k) for k in qk] if qk else None)
                        data[f"{ro}_N{N}_{key}"] = np.stack([xb, mb, sb])
                for i, (key, _) in enumerate(rows):
                    if _panel_missing(key, [r.run.readout for r in col], N, na, missing):
                        P.na_slot(axes[i, j], "not recorded\n(older batch file)")
                for half in sorted({r.T_burn + r.T_meas / 2 for r in col}):
                    for i in range(len(rows)):
                        P.event_line(axes[i, j], half)
                top = axes[0, j].secondary_xaxis("top", functions=(lambda u, s=scale: u * s, lambda v, s=scale: v / s))
                top.set_xlabel(r"$t\kappa_s^2/N$ (lifetimes)"); top.xaxis.set_major_locator(P.MaxNLocator(4))
                axes[-1, j].set_xlabel("task t")
            for i, (key, _) in enumerate(rows):
                axes[i, 0].set_ylabel(P.qlabel(key))
            P.ref_line(axes[[k for k, _ in rows].index("logf_sum"), 0], 0.0)
            P.label_cols_N(axes, Ns)
            P.fig_legend(fig, {r.run.readout for r in members})
            P.cell_title(fig, kappa, d0, lam, solver, eta)
            P.caption(fig, "thick: mean over seeds and blocks of half a lifetime, band ±1 s.e.; faint: per-task seed mean; "
                           "thin: 5 % and 95 % over active units; dotted: half boundary" + _omitted(members, rows) + _missing_text(missing))
            stem = P.cell_stem(kappa, d0, lam, solver, eta)
            P.save(fig, os.path.join(fig_dir, "cells", "norms", "norms_" + stem), data if figdata else None, index, "cells/norms", stem)


def fig_F2(records, fig_dir, index, figdata):
    """F2: norms vs task, columns = readouts, traces = N; ``normsN_`` (x = task) and ``normsN_lt_`` (x = lifetimes)."""
    na, kmin = _na(), kappa_min(records)
    for kappa, kmembers in by_kappa(records).items():
        for cell, members in group_by(kmembers, ("delta0", "lam", "solver", "eta")).items():
            d0, lam, solver, eta = cell
            Ns = sorted({r.run.N for r in members})
            ros = [ro for ro in P.READOUT_COLOR if any(r.run.readout == ro for r in members)]
            rows = _rows(members)
            for lt in (False, True):
                fig, axes = P.grid(len(rows), len(ros), sharex="all", sharey="row")
                data, missing = {}, set()
                for j, ro in enumerate(ros):
                    for i_N, N in enumerate(Ns):
                        recs = [r for r in members if r.run.N == N and r.run.readout == ro]
                        if not recs:
                            continue
                        r0 = recs[0]
                        ks = r0.kappa_sched(kmin)
                        scale = ks ** 2 / N if lt else 1.0
                        x = np.arange(r0.T, dtype=float) * scale
                        w = max(1, int(round(N / (2 * ks ** 2))))
                        cuts = (r0.T_burn + r0.T_meas // 2,)
                        col = P.n_shade(i_N, len(Ns))
                        for i, (key, qk) in enumerate(rows):
                            if (key, ro) in na:
                                continue
                            if key not in r0.scalars:
                                missing.add((key, ro, N))
                                continue
                            xb, mb, sb = _block_trace(axes[i, j], x, seed_stack(recs, key), w, cuts, col, P.READOUT_STYLE[ro],
                                                      [seed_stack(recs, k) for k in qk] if qk else None)
                            data[f"{ro}_N{N}_{key}"] = np.stack([xb, mb, sb])
                        half = (r0.T_burn + r0.T_meas / 2) * scale
                        for i in range(len(rows)):
                            axes[i, j].axvline(half, color=col, ls=":", lw=P.LW_EVENT, zorder=1)
                    for i, (key, _) in enumerate(rows):
                        if (key, ro) not in na and all((key, ro, N) in missing for N in Ns if any(r.run.N == N and r.run.readout == ro for r in members)):
                            P.na_slot(axes[i, j], "not recorded\n(older batch file)")
                    axes[0, j].set_title(P.READOUT_LABEL[ro], color=P.READOUT_COLOR[ro])
                    axes[-1, j].set_xlabel(r"$t\kappa_s^2/N$ (lifetimes)" if lt else "task t")
                for i, (key, _) in enumerate(rows):
                    axes[i, 0].set_ylabel(P.qlabel(key))
                P.ref_line(axes[[k for k, _ in rows].index("logf_sum"), 0], 0.0)
                P.fig_legend(fig, None, handles=P.n_handles(Ns))
                P.cell_title(fig, kappa, d0, lam, solver, eta, "traces = N" + (", x in lifetimes" if lt else ""))
                P.caption(fig, "thick: mean over seeds and blocks of half a lifetime, band ±1 s.e.; faint: per-task seed mean; "
                               "thin: 5 % and 95 % over active units; dotted: half boundary of each N; dashed columns: ungated"
                          + _omitted(members, rows) + _missing_text(missing))
                stem = P.cell_stem(kappa, d0, lam, solver, eta)
                P.save(fig, os.path.join(fig_dir, "cells", "norms", ("normsN_lt_" if lt else "normsN_") + stem), data if figdata else None, index, "cells/norms", stem)


# ---------------------------------------------------------------------------
# diagnostics: per-step traces (PLAN.md §12.2), only with --diag
# ---------------------------------------------------------------------------
def family_B(out_dir, fig_dir, index, figdata):
    """Norms vs step within a task from ``traces/``; one figure per (readout, κ, δ₀, λ, solver, η) → ``diag/by_step/``."""
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
                    if q == "logf" and "frozen" in ro:
                        P.na_slot(ax, "not projected, f = 1"); continue
                    if q == "a2_act" and "frozen" in ro:
                        P.na_slot(ax, "a fixed within a task"); continue
                    y = tr[T_burn:T_burn + T_meas, :, quant.index(q)].astype(float)
                    if rel:
                        y = y - y[:, :1]
                    med = P.overlay(ax, x, y, P.READOUT_COLOR[ro], cmap="viridis")
                    if solver == "sgd":
                        ax.set_xscale("symlog", linthresh=10)
                    data[f"N{N}_seed{seed}_{q}_median"] = med
                    medians[(cell, N, seed, q)] = (x, med)
                axes[0, j].set_title(f"N = {P.fmt(N)}, seed {seed}")
                axes[-1, j].set_xlabel("step within task")
            for i, (_, _, lab) in enumerate(ROWS_B):
                axes[i, 0].set_ylabel(lab); P.ref_line(axes[i, 0], 0.0)
            P.cell_title(fig, kappa if is_gated(ro) else None, d0, lam, solver, eta, P.READOUT_LABEL[ro])
            P.caption(fig, "thin: individual measurement tasks (viridis = task index); black: median over tasks; band: 10–90 %")
            stem = P.cell_stem(kappa if is_gated(ro) else None, d0, lam, solver, eta, readout=ro) + (f"_part{part + 1}" if part else "")
            P.save(fig, os.path.join(fig_dir, "diag", "by_step", "steps_" + stem), data if figdata else None, index, "diag/by_step", stem)


def run(out_dir, fig_dir, records=None, figdata=False, diag=False, cells=None):
    """Make the norms slices S1 and M8 (and, for ``cells``, F1/F2; with ``diag``, the by-step family).

    Parameters
    ----------
    out_dir : str
        ``results/<sweep>``.
    fig_dir : str
        Figure root (``figures/``).
    records : list of RunRecord, optional
        Loaded records (loaded here when omitted).
    figdata : bool
        Write the plotted arrays as ``.npz`` next to each figure.
    diag : bool
        Also draw ``diag/by_step/`` from the trace files.
    cells : list of RunRecord, optional
        Records of the cells selected with ``--cells``; F1 and F2 are drawn
        for these into ``cells/norms/`` (nothing per cell when ``None``).

    Returns
    -------
    list of (family, cell, file)
        INDEX.md rows.
    """
    if records is None:
        records, _ = load_runs(out_dir)
    index = []
    if records:
        fig_S1(records, fig_dir, index, figdata)
        fig_M8(records, fig_dir, index, figdata)
    if cells:
        fig_F1(cells, fig_dir, index, figdata)
        fig_F2(cells, fig_dir, index, figdata)
    if diag:
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
    ap.add_argument("--diag", action="store_true", help="also draw the per-step family from traces/ (diag/by_step/)")
    ap.add_argument("--cells", action="store_true", help="also draw the per-cell families F1/F2 for every cell (cells/norms/)")
    args = ap.parse_args(argv)
    fig_dir = args.fig_dir or os.path.join(args.out_dir, "figures")
    records, _ = load_runs(args.out_dir)
    index = run(args.out_dir, fig_dir, records, args.save_figdata, args.diag, records if args.cells else None)
    print(f"{len(index)} figures → {fig_dir}/slices")
    return 0


if __name__ == "__main__":
    sys.exit(main())
