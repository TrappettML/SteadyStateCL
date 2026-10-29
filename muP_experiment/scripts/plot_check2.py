#!/usr/bin/env python
"""Steady-state (check 2) figures and ``check2_cells.csv`` (PLAN.md §12.6).

    python scripts/plot_check2.py results/<sweep> [--fig-dir DIR] [--save-figdata]

C2-A trajectories vs task in lifetimes; C2-B half-window dumbbells and
ratios over the (δ₀, λ) grid; C2-C drift-rate maps; C2-D S(m), E(m) from
the two halves overlaid; C2-E box-plot summaries of the drift statistics
(z, ρ, β) over the (κ, δ₀, λ) grid and flag-fraction bars, with the
per-cell statistics in ``check2_cells.csv``.  Everything goes to
``figures/check2/``.  Called by ``analyze.py``.
"""
import argparse
import csv
import os
import sys

import numpy as np

import _path  # noqa: F401

from mup import plotting as P
from mup.config import is_gated
from mup.observe import half_life
from mup.results import by_kappa, drift_rate, group_by, half_stats, load_runs, seed_stack

#: C2-A rows and the C2-B/C quantities: (scalar key, label, band keys or None)
ROWS = (("Q_act_pre", r"$Q_{act}$ (pre)", ("Q_q05", "Q_q95")), ("a2_act_post", r"$a^2_{act}$", None),
        ("aQ_act_post", r"mean $|a_i|\sqrt{Q_{ii}}$", ("aQ_q05", "aQ_q95")), ("aQ_max_post", r"max $|a_i|\sqrt{Q_{ii}}$", None),
        ("logf_sum", "Σ log f", None), ("S0", "S(0)", None))
QUANTS = tuple(r[0] for r in ROWS)
#: C2-C maps: quantities fitted as log q(t) (β = fractional change per lifetime) or raw (logf_sum)
DRIFT_QUANTS = ("aQ_act_post", "Q_act_pre", "logf_sum")
#: C2-E rows (top → bottom) and their short labels; ``dm_half`` is Δm_half / m_half_0 (PLAN.md §12.6)
SUMMARY_ROWS = (("aQ_act_post", r"mean $|a|\sqrt{Q}$"), ("aQ_max_post", r"max $|a|\sqrt{Q}$"), ("Q_act_pre", r"$Q_{act}$ (pre)"),
                ("a2_act_post", r"$a^2_{act}$"), ("logf_sum", "Σ log f"), ("S0", "S(0)"), ("dm_half", r"$\Delta m_{1/2}/m_{1/2}$"))
#: columns of check2_cells.csv
CELL_COLS = ("readout", "N", "kappa", "delta0", "lam", "solver", "eta", "quantity", "n_seeds",
             "half0", "half1", "se", "z", "rho", "flag", "beta", "se_beta", "m_half_0", "m_half_1", "censored")


def _cond_stats(recs, key):
    """Seed-averaged half-window and drift statistics of one condition for one quantity.

    Returns
    -------
    dict
        ``half0, half1, se`` (seed means; ``se`` combined over seeds),
        ``z`` = (half1 − half0)/se (the check-2 statistic, signed), ``rho`` =
        half1/half0 − 1 (``ratio`` is kept as an alias), ``flag`` = |z| > 3,
        ``beta, se_beta`` (seed means of the per-run fit).
    """
    hs = [half_stats(r.scalars[key], r.T_burn, r.T_meas) for r in recs]
    h0, h1 = np.nanmean([h["half0"] for h in hs]), np.nanmean([h["half1"] for h in hs])
    se = float(np.sqrt(np.nansum([h["se"] ** 2 for h in hs])) / len(hs))
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = float(h1 / h0 - 1.0)
        z = float((h1 - h0) / se)
        flag = bool(abs(z) > 3.0)
    fits = []
    for r in recs:
        x = r.scalars[key]
        if key == "logf_sum":          # may be ≤ 0: raw slope instead of log slope
            x = np.exp(x)
        fits.append(drift_rate(x, r.lifetimes(), r.T_burn, r.T_meas))
    beta, se_beta = np.nanmean([f[0] for f in fits]), np.nanmean([f[1] for f in fits]) / np.sqrt(len(fits))
    return {"half0": float(h0), "half1": float(h1), "se": se, "ratio": ratio, "rho": ratio, "z": z, "flag": flag,
            "beta": float(beta), "se_beta": float(se_beta)}


def fig_traj(records, fig_dir, index, figdata):
    """C2-A: trajectories vs task in lifetimes (t κ²/N), one figure per (κ, δ₀, λ, solver, η)."""
    for kappa, kmembers in by_kappa(records).items():
        for cell, members in group_by(kmembers, ("delta0", "lam", "solver", "eta")).items():
            d0, lam, solver, eta = cell
            Ns = sorted({r.run.N for r in members})
            fig, axes = P.grid(len(ROWS), len(Ns), sharex="col", sharey="row")
            data = {}
            for j, N in enumerate(Ns):
                gated = [r for r in members if r.run.N == N and is_gated(r.run.readout)]
                ref = gated[0] if gated else next(r for r in members if r.run.N == N)
                scale = ref.run.kappa ** 2 / N
                for i in range(len(ROWS)):
                    P.shade_burnin(axes[i, j], ref.T_burn, ref.T_meas, scale)
                for ro in P.READOUT_COLOR:
                    recs = [r for r in members if r.run.N == N and r.run.readout == ro]
                    if not recs:
                        continue
                    x = recs[0].lifetimes()
                    for i, (key, _, bk) in enumerate(ROWS):
                        ys = seed_stack(recs, key)
                        m = P.seeds_and_mean(axes[i, j], x, ys, ro)
                        if bk:
                            P.band(axes[i, j], x, np.nanmean(seed_stack(recs, bk[0]), 0), np.nanmean(seed_stack(recs, bk[1]), 0), P.READOUT_COLOR[ro])
                        data[f"{ro}_N{N}_{key}"] = m
                top = axes[0, j].secondary_xaxis("top", functions=(lambda u, s=scale: u / s, lambda t, s=scale: t * s))
                top.set_xlabel("task (gated κ)"); top.xaxis.set_major_locator(P.MaxNLocator(4))
                axes[-1, j].set_xlabel(r"$t\kappa^2/N$ (lifetimes)")
            for i, (_, lab, _) in enumerate(ROWS):
                axes[i, 0].set_ylabel(lab)
            P.ref_line(axes[QUANTS.index("logf_sum"), 0], 0.0)
            P.label_cols_N(axes, Ns)
            P.readout_legend(axes[0, 0], {r.run.readout for r in members})
            P.cell_title(fig, kappa, d0, lam, solver, eta)
            P.caption(fig, "thin: seeds; thick: seed mean; grey: burn-in; dotted: half boundary; bands: 5–95 % over active units")
            stem = P.cell_stem(kappa, d0, lam, solver, eta)
            P.save(fig, os.path.join(fig_dir, "check2", "traj_" + stem), data if figdata else None, index, "check2", stem)


def _grid_stats(records):
    """
    Half-window and drift statistics of every (readout, N, κ, δ₀, λ, solver, η, quantity).

    Parameters
    ----------
    records : list of RunRecord

    Returns
    -------
    list of dict
        One row per condition and quantity with the keys ``CELL_COLS`` (written
        to ``check2_cells.csv``).  The pseudo-quantity ``dm_half`` carries
        ``rho`` = (m_half(1) − m_half(0)) / m_half(0) of the seed-pooled S(m),
        ``se`` = s.e.m. over seeds of the per-seed ratio, ``z`` = rho/se and
        ``censored`` when either half never reaches S(0)/2.
    """
    rows = []
    for cond, recs in group_by(records, ("readout", "N", "kappa_label", "delta0", "lam", "solver", "eta")).items():
        ro, N, kappa, d0, lam, solver, eta = cond
        base = dict(readout=ro, N=N, kappa=kappa, delta0=d0, lam=lam, solver=solver, eta=eta, n_seeds=len(recs))
        mh0, mh1 = (half_life(recs[0].ages, _pooled_S(recs, h))[0] for h in ((0,), (1,)))
        censored = not (np.isfinite(mh0) and np.isfinite(mh1))
        for q in QUANTS:
            if q not in recs[0].scalars:
                continue
            rows.append(dict(base, quantity=q, m_half_0=mh0, m_half_1=mh1, censored=censored, **_cond_stats(recs, q)))
        # half-life shift, with the seed scatter as its error
        per_seed = []
        for r in recs:
            a, b = (half_life(r.ages, _pooled_S([r], h))[0] for h in ((0,), (1,)))
            per_seed.append((b - a) / a if np.isfinite(a) and np.isfinite(b) and a > 0 else np.nan)
        per_seed = np.asarray(per_seed, float)
        ok = np.isfinite(per_seed)
        se = float(np.std(per_seed[ok], ddof=1) / np.sqrt(ok.sum())) if ok.sum() > 1 else np.nan
        with np.errstate(invalid="ignore", divide="ignore"):
            rho = float((mh1 - mh0) / mh0) if not censored else np.nan
            z = float(rho / se)
        rows.append(dict(base, quantity="dm_half", half0=mh0, half1=mh1, se=se, z=z, rho=rho, ratio=rho,
                         flag=bool(np.isfinite(z) and abs(z) > 3.0), beta=np.nan, se_beta=np.nan,
                         m_half_0=mh0, m_half_1=mh1, censored=censored))
    return rows


def _pooled_S(recs, halves):
    """
    Seed-pooled S(m) of a condition from the given measurement halves.

    Parameters
    ----------
    recs : list of RunRecord
    halves : tuple

    Returns
    -------
    (n_ages,) array
    """
    hs = list(halves)
    with np.errstate(invalid="ignore", divide="ignore"):
        return sum(r.moments["sum_s"][:, hs].sum(-1) for r in recs) / sum(r.moments["count"][:, hs].sum(-1) for r in recs)


def fig_halves(rows, fig_dir, index, figdata):
    """C2-B: half-window dumbbells (and the ratio companion) across the (δ₀, λ) grid, one file per quantity and (κ, solver, η)."""
    for (solver, eta, q), sub in _group_rows(rows, ("solver", "eta", "quantity")).items():
        if q == "dm_half":
            continue
        kappas = sorted({r["kappa"] for r in sub if is_gated(r["readout"])}) or ["sqrtN"]
        for kappa in kappas:
            rs = [r for r in sub if (r["kappa"] == kappa) or not is_gated(r["readout"])]
            ros = [r for r in P.READOUT_COLOR if any(x["readout"] == r for x in rs)]
            Ns = sorted({r["N"] for r in rs})
            cells = sorted({(r["delta0"], r["lam"]) for r in rs})
            labels = [rf"$\delta_0$={d}, $\lambda$={l}" for d, l in cells]
            for kind in ("halves", "halves_ratio"):
                fig, axes = P.grid(len(ros), len(Ns), sharex="col", sharey="row" if kind == "halves" else False)
                data = {}
                for i, ro in enumerate(ros):
                    for j, N in enumerate(Ns):
                        ax = axes[i, j]
                        for c, (d0, lam) in enumerate(cells):
                            r = next((x for x in rs if x["readout"] == ro and x["N"] == N and (x["delta0"], x["lam"]) == (d0, lam)), None)
                            if r is None:
                                continue
                            if kind == "halves":
                                P.dumbbell(ax, c, r["half0"], r["half1"], r["se"], P.READOUT_COLOR[ro], r["flag"])
                                data[f"{ro}_N{N}_{c}"] = [r["half0"], r["half1"], r["se"]]
                            else:
                                ax.plot(c, r["ratio"], "o", color=P.UP if r["ratio"] > 0 else P.DOWN, ms=5,
                                        mec="k" if r["flag"] else "none", mew=1.0, zorder=4)
                                data[f"{ro}_N{N}_{c}"] = [r["ratio"]]
                        ax.set_xticks(range(len(cells)))
                        ax.set_xticklabels(labels if i == len(ros) - 1 else [], rotation=90, fontsize=P.FS_MIN)
                        if kind == "halves_ratio":
                            P.ref_line(ax, 0.0)
                            vals = [abs(r["ratio"]) for r in rs if r["readout"] == ro and r["N"] == N and np.isfinite(r["ratio"])]
                            lim = 1.1 * max(vals) if vals and max(vals) > 0 else 0.01
                            ax.set_ylim(-lim, lim)
                    axes[i, 0].set_ylabel(P.READOUT_LABEL[ro] + ("" if kind == "halves" else "\nhalf 1 / half 0 − 1"), color=P.READOUT_COLOR[ro])
                P.label_cols_N(axes, Ns)
                P.cell_title(fig, kappa, None, None, solver, eta, q)
                P.caption(fig, "open: half 0, filled: half 1; bars: ±3 s.e.; black + red ×: flagged" if kind == "halves"
                          else "orange: up, blue: down; black edge: flagged")
                stem = f"{q}_" + P.cell_stem(kappa, None, None, solver, eta)
                P.save(fig, os.path.join(fig_dir, "check2", f"{kind}_" + stem), data if figdata else None, index, "check2", stem)


def fig_drift(rows, fig_dir, index, figdata):
    """C2-C: drift-rate map β over (δ₀ rows, λ columns); rows = readouts, columns = N; one file per quantity and (κ, solver, η)."""
    for (solver, eta, q), sub in _group_rows(rows, ("solver", "eta", "quantity")).items():
        if q not in DRIFT_QUANTS:
            continue
        kappas = sorted({r["kappa"] for r in sub if is_gated(r["readout"])}) or ["sqrtN"]
        for kappa in kappas:
            rs = [r for r in sub if (r["kappa"] == kappa) or not is_gated(r["readout"])]
            ros = [r for r in P.READOUT_COLOR if any(x["readout"] == r for x in rs)]
            Ns = sorted({r["N"] for r in rs})
            d0s, lams = sorted({r["delta0"] for r in rs}), sorted({r["lam"] for r in rs})
            grids = {}
            for ro in ros:
                for N in Ns:
                    B = np.full((len(d0s), len(lams)), np.nan); SE = np.full_like(B, np.nan)
                    for r in rs:
                        if r["readout"] == ro and r["N"] == N:
                            B[d0s.index(r["delta0"]), lams.index(r["lam"])] = r["beta"]
                            SE[d0s.index(r["delta0"]), lams.index(r["lam"])] = r["se_beta"]
                    grids[(ro, N)] = (B, SE)
            vmax = np.nanmax([np.nanmax(np.abs(g[0])) for g in grids.values()] or [1.0])
            vmax = float(vmax) if np.isfinite(vmax) and vmax > 0 else 1.0
            fig, axes = P.grid(len(ros), len(Ns), sharex=False, sharey=False, extra_w=1.0)
            im = None
            for i, ro in enumerate(ros):
                for j, N in enumerate(Ns):
                    B, SE = grids[(ro, N)]
                    im = P.drift_map(axes[i, j], B, d0s, lams, vmax, SE)
                    if i < len(ros) - 1:
                        axes[i, j].set_xticklabels([])
                    if j > 0:
                        axes[i, j].set_yticklabels([])
                    if i == len(ros) - 1:
                        axes[i, j].set_xlabel(r"$\lambda$")
                axes[i, 0].set_ylabel(P.READOUT_LABEL[ro] + "\n" + r"$\delta_0$", color=P.READOUT_COLOR[ro])
            P.label_cols_N(axes, Ns)
            cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, pad=0.02)
            cbar.set_label("β (per lifetime)"); cbar.locator = P.MaxNLocator(3); cbar.update_ticks()
            P.cell_title(fig, kappa, None, None, solver, eta, f"{q}, vmax = {vmax:.3g}")
            P.caption(fig, ("slope of log q" if q != "logf_sum" else "slope of q") + " vs $t\\kappa^2/N$ over the measurement window; black: |β| > 3 s.e.; hatched: missing")
            stem = f"{q}_" + P.cell_stem(kappa, None, None, solver, eta)
            P.save(fig, os.path.join(fig_dir, "check2", "drift_" + stem), {f"{ro}_N{N}": g[0] for (ro, N), g in grids.items()} if figdata else None, index, "check2", stem)


def fig_Sm_halves(records, fig_dir, index, figdata):
    """C2-D: S(m) and E(m) from the two halves overlaid (half 0 solid, half 1 dotted), m_half ticks and Δm_half."""
    for kappa, kmembers in by_kappa(records).items():
        for cell, members in group_by(kmembers, ("delta0", "lam", "solver", "eta")).items():
            d0, lam, solver, eta = cell
            Ns = sorted({r.run.N for r in members})
            fig, axes = P.grid(2, len(Ns), sharex="col", sharey="row")
            data = {}
            for j, N in enumerate(Ns):
                notes = []
                for ro in P.READOUT_COLOR:
                    recs = [r for r in members if r.run.N == N and r.run.readout == ro]
                    if not recs:
                        continue
                    ages = recs[0].ages
                    mh = []
                    for h, ls in ((0, "-"), (1, ":")):
                        with np.errstate(invalid="ignore", divide="ignore"):
                            n = sum(r.moments["count"][:, h] for r in recs)
                            S = sum(r.moments["sum_s"][:, h] for r in recs) / n
                            E = sum(r.moments["sum_e"][:, h] for r in recs) / n
                        P.plot_readout(axes[0, j], ages[1:], S[1:], ro, ls=ls, lw=1.0)
                        P.plot_readout(axes[1, j], ages[1:], E[1:], ro, ls=ls, lw=1.0)
                        m, _ = half_life(ages, S)
                        mh.append(m)
                        if np.isfinite(m) and m > 0:
                            axes[0, j].plot([m], [axes[0, j].get_ylim()[0]], marker="|", color=P.READOUT_COLOR[ro], ms=5, clip_on=False, zorder=6)
                        data[f"{ro}_N{N}_S_half{h}"], data[f"{ro}_N{N}_E_half{h}"] = S, E
                    notes.append(f"{P.READOUT_LABEL[ro]}: " + rf"$\Delta m_{{1/2}}$ = {mh[1] - mh[0]:+.2f}")
                P.logaxis(axes[1, j], "x"); axes[1, j].set_xlabel("age m (tasks)")
                axes[1, j].text(0.02, 0.98, "\n".join(notes), transform=axes[1, j].transAxes, fontsize=P.FS_MIN, va="top")
            axes[0, 0].set_ylabel("S(m)"); axes[1, 0].set_ylabel("E(m)")
            P.label_cols_N(axes, Ns)
            P.readout_legend(axes[0, 0], {r.run.readout for r in members})
            P.cell_title(fig, kappa, d0, lam, solver, eta)
            P.caption(fig, "solid: half 0, dotted: half 1; ticks: $m_{1/2}$ of each half")
            stem = P.cell_stem(kappa, d0, lam, solver, eta)
            P.save(fig, os.path.join(fig_dir, "check2", "Sm_halves_" + stem), data if figdata else None, index, "check2", stem)


def _group_rows(rows, fields):
    """Group plain dicts by the given fields → ``{tuple: [rows]}``."""
    out = {}
    for r in rows:
        out.setdefault(tuple(r[f] for f in fields), []).append(r)
    return out


def fig_summary(rows, fig_dir, index, figdata):
    """C2-E summary: box plots per readout of z (``summary_*``), ρ (``summary_rho_*``) and β/se_β (``summary_beta_*``).

    Rows = quantity (``SUMMARY_ROWS``), columns = N; each box pools one
    readout's (κ, δ₀, λ) cells (seeds already averaged).  The ``dm_half``
    row shows ρ with a ±2 s.e. band from the seed scatter of m_half and the
    number of censored cells in the corner.
    """
    for (solver, eta), sub in _group_rows(rows, ("solver", "eta")).items():
        Ns = sorted({r["N"] for r in sub})
        for kind, stat, ylab, band in (("summary", "z", "z = (half 1 − half 0) / s.e.", 3.0),
                                       ("summary_rho", "rho", "half 1 / half 0 − 1", None),
                                       ("summary_beta", "zbeta", r"$\beta / \mathrm{se}_\beta$", 3.0)):
            qrows = [qr for qr in SUMMARY_ROWS if not (kind == "summary_beta" and qr[0] == "dm_half")]
            fig, axes = P.grid(len(qrows), len(Ns), sharex="col", sharey="row")
            data = {}
            for i, (q, lab) in enumerate(qrows):
                for j, N in enumerate(Ns):
                    ax = axes[i, j]
                    cells = [r for r in sub if r["quantity"] == q and r["N"] == N and not (q == "dm_half" and r["censored"])]
                    if q == "dm_half" and kind == "summary":
                        vals = {ro: np.array([r["rho"] for r in cells if r["readout"] == ro]) for ro in P.READOUT_COLOR}
                        ses = np.array([r["se"] for r in cells], float)
                        half = 2.0 * float(np.nanmedian(ses)) if np.isfinite(ses).any() else 0.0
                        P.zero_band(ax, -half, half)
                        n_cens = sum(r["censored"] for r in sub if r["quantity"] == q and r["N"] == N)
                        ax.text(0.98, 0.02, f"{n_cens} censored", transform=ax.transAxes, ha="right", va="bottom", fontsize=P.FS_MIN, color=P.GREY_CAPTION)
                    else:
                        if stat == "zbeta":
                            with np.errstate(invalid="ignore", divide="ignore"):
                                vals = {ro: np.array([r["beta"] for r in cells if r["readout"] == ro], float) / np.array([r["se_beta"] for r in cells if r["readout"] == ro], float) for ro in P.READOUT_COLOR}
                        else:
                            vals = {ro: np.array([r[stat] for r in cells if r["readout"] == ro]) for ro in P.READOUT_COLOR}
                        if band is not None:
                            P.zero_band(ax, -band, band)
                        else:
                            P.ref_line(ax, 0.0)
                    vals = {ro: v for ro, v in vals.items() if len(v)}
                    P.boxes_by_readout(ax, vals, labels=(i == len(qrows) - 1))
                    if kind == "summary_rho":
                        ax.set_yscale("symlog", linthresh=0.01)
                    for ro, v in vals.items():
                        data[f"{q}_{ro}_N{N}"] = v
                axes[i, 0].set_ylabel(lab)
            if kind == "summary":          # the per-cell statistics, indexed by the first CELL_COLS fields (§12.6)
                data.update({"cells_" + c: np.array([r[c] for r in sub]) for c in CELL_COLS})
            P.label_cols_N(axes, Ns)
            P.cell_title(fig, None, None, None, solver, eta, {"summary": "check 2: z", "summary_rho": "check 2: ρ", "summary_beta": "check 2: β"}[kind])
            P.caption(fig, "box: IQR over (κ, δ₀, λ) cells; whiskers: 5–95 %; hatched: ungated; "
                      + ("band: |z| ≤ 3 (last row: ±2 s.e.)" if kind == "summary" else
                         "band: |β/se| ≤ 3" if kind == "summary_beta" else "symlog, linthresh 0.01"))
            axes[0, 0].set_ylabel(qrows[0][1] + "\n" + ylab)
            stem = P.cell_stem(None, None, None, solver, eta)
            P.save(fig, os.path.join(fig_dir, "check2", f"{kind}_" + stem), data if figdata else None, index, "check2", stem)


def fig_flags(rows, fig_dir, index, figdata):
    """C2-E flag bars (``flags_*``): fraction of (κ, δ₀, λ) cells flagged per (quantity, readout), one panel per N."""
    quantities = [q for q, _ in SUMMARY_ROWS]
    for (solver, eta), sub in _group_rows(rows, ("solver", "eta")).items():
        Ns = sorted({r["N"] for r in sub})
        fig, axes = P.grid(1, len(Ns), sharex=True, sharey=False, panel=(P.PANEL[0], 3.4))
        data = {}
        for j, N in enumerate(Ns):
            fr, ct = {}, {}
            for q in quantities:
                for ro in P.READOUT_COLOR:
                    cells = [r for r in sub if r["quantity"] == q and r["N"] == N and r["readout"] == ro and not r["censored"]]
                    if not cells:
                        continue
                    k = sum(bool(r["flag"]) for r in cells)
                    fr[(q, ro)], ct[(q, ro)] = k / len(cells), f"{k}/{len(cells)}"
                    data[f"{q}_{ro}_N{N}"] = [k, len(cells)]
            P.flag_bars(axes[0, j], fr, ct, quantities, labels=(j == 0))
            axes[0, j].set_xlabel("fraction of cells flagged")
        P.label_cols_N(axes, Ns)
        P.cell_title(fig, None, None, None, solver, eta, "check 2: flags")
        P.caption(fig, "bar: fraction of (κ, δ₀, λ) cells with |z| > 3; grey line: chance (0.3 %); hatched: ungated")
        stem = P.cell_stem(None, None, None, solver, eta)
        P.save(fig, os.path.join(fig_dir, "check2", "flags_" + stem), data if figdata else None, index, "check2", stem)


def write_cells(rows, fig_dir):
    """Write ``check2/check2_cells.csv`` (every per-cell statistic; the same arrays go into ``summary_*.npz`` with ``--save-figdata``).

    Prints the flag counts per (readout, quantity) and the three worst cells by |z|.

    Parameters
    ----------
    rows : list of dict
        Output of ``_grid_stats``.
    fig_dir : str
    """
    os.makedirs(os.path.join(fig_dir, "check2"), exist_ok=True)
    path = os.path.join(fig_dir, "check2", "check2_cells.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(CELL_COLS))
        w.writeheader()
        for r in rows:
            w.writerow({c: r[c] for c in CELL_COLS})
    live = [r for r in rows if not r["censored"] or r["quantity"] != "dm_half"]
    print(f"check 2  flagged cells (|z| > 3) per readout and quantity → {path}")
    for (ro, q), rs in sorted(_group_rows(live, ("readout", "quantity")).items(), key=lambda kv: (list(P.READOUT_COLOR).index(kv[0][0]), kv[0][1])):
        k = sum(bool(r["flag"]) for r in rs)
        if k:
            print(f"   {ro:14s} {q:12s} {k}/{len(rs)}")
    worst = sorted((r for r in live if np.isfinite(r["z"])), key=lambda r: -abs(r["z"]))[:3]
    for r in worst:
        print(f"   worst: {r['readout']:14s} N={r['N']:4d} κ={r['kappa']:5s} δ₀={r['delta0']:5.2f} λ={r['lam']:3.1f} {r['quantity']:12s} "
              f"z={r['z']:+.1f} ρ={r['rho']:+.3f} β={r['beta']:+.3g}")


def run(out_dir, fig_dir, records=None, figdata=False):
    """Make every check-2 figure (C2-A … C2-E) and ``check2_cells.csv``.

    Parameters
    ----------
    out_dir : str
    fig_dir : str
    records : list of RunRecord, optional
    figdata : bool

    Returns
    -------
    list of (family, cell, file)
        INDEX.md rows.
    """
    if records is None:
        records, _ = load_runs(out_dir)
    index = []
    if not records:
        return index
    rows = _grid_stats(records)
    write_cells(rows, fig_dir)
    fig_traj(records, fig_dir, index, figdata)
    fig_halves(rows, fig_dir, index, figdata)
    fig_drift(rows, fig_dir, index, figdata)
    fig_Sm_halves(records, fig_dir, index, figdata)
    fig_summary(rows, fig_dir, index, figdata)
    fig_flags(rows, fig_dir, index, figdata)
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
    print(f"{len(index)} figures → {fig_dir}/check2")
    return 0


if __name__ == "__main__":
    sys.exit(main())
