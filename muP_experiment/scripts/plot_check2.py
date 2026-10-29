#!/usr/bin/env python
"""Check statistics → the ✓/✗ checks table (PLAN.md §12.6); the C2 diagnostic figures only with ``--diag``.

    python scripts/plot_check2.py results/<sweep> [--fig-dir DIR] [--save-figdata] [--diag] [--levels]

Default output: ``checks/checks_{solver}_eta{η}.png`` (one table per
solver and η; rows = the checks of §6 and the stationarity statistics,
columns = readout × N), ``checks_{solver}_eta{η}.csv`` (every number behind
a mark) and ``checks/check2_cells.csv`` (per-cell statistics).  With
``--diag`` the figures of the previous revision (C2-A … C2-E) go to
``diag/check2/`` for locating a ✗.  Called by ``analyze.py``.
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
from mup.results import by_kappa, drift_rate, group_by, half_stats, kappa_min, load_runs, seed_stack

#: C2-A rows: (scalar key, quantile keys or None); labels come from plotting.QUANTITY_LABEL (§12.7 G2)
ROWS = (("Q_act_pre", ("Q_q05", "Q_q95")), ("a2_act_post", None), ("aQ_act_post", ("aQ_q05", "aQ_q95")),
        ("aQ_max_post", None), ("logf_sum", None), ("S0", None))
QUANTS = tuple(r[0] for r in ROWS)
#: C2-C maps: quantities fitted as log q(t) (β = fractional change per lifetime) or raw (logf_sum)
DRIFT_QUANTS = ("aQ_act_post", "Q_act_pre", "logf_sum")
#: C2-E rows (top → bottom); ``dm_half`` is Δm_half / m_half_0 (PLAN.md §12.6)
SUMMARY_ROWS = ("aQ_act_post", "aQ_max_post", "Q_act_pre", "a2_act_post", "logf_sum", "S0", "dm_half")
#: (quantity, readout) pairs that are constant by construction (§12.7 G4): no statistics, an n/a slot in the figures
NA_BY_CONSTRUCTION = {
    ("logf_sum", "frozen"): "not projected, f = 1",
    ("logf_sum", "gated_frozen"): "not projected, f = 1",
    ("a2_act_post", "frozen"): "a fixed, all units active",
    **{("Q_act_post", ro): "pinned by the projection" for ro in ("gated_shared", "learned_shared", "task_heads")},
    **{("heads_fro2_post", ro): "no stored heads" for ro in ("gated_shared", "learned_shared", "gated_frozen", "frozen")},
}
#: columns of check2_cells.csv
CELL_COLS = ("readout", "N", "kappa", "delta0", "lam", "solver", "eta", "quantity", "n_seeds",
             "half0", "half1", "se", "z", "rho", "flag", "beta", "se_beta", "m_half_0", "m_half_1", "censored")
Z_FLOOR, Z_CLIP, RHO_FLOOR_PCT = 4.0, 8.0, 2.0


def is_na(q: str, ro: str) -> bool:
    """Whether (quantity, readout) is constant by construction."""
    return (q, ro) in NA_BY_CONSTRUCTION


def _cond_stats(recs, key):
    """Seed-averaged half-window and drift statistics of one condition for one quantity.

    Returns
    -------
    dict
        ``half0, half1, se`` (seed means; ``se`` combined over seeds),
        ``z`` = (half1 − half0)/se (the check-2 statistic, signed), ``rho`` =
        half1/half0 − 1, ``flag`` = |z| > 3, ``beta, se_beta`` (seed means of
        the per-run fit).
    """
    hs = [half_stats(r.scalars[key], r.T_burn, r.T_meas) for r in recs]
    h0, h1 = np.nanmean([h["half0"] for h in hs]), np.nanmean([h["half1"] for h in hs])
    se = float(np.sqrt(np.nansum([h["se"] ** 2 for h in hs])) / len(hs))
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = float(h1 / h0 - 1.0)
        z = float((h1 - h0) / se)
        flag = bool(np.isfinite(z) and abs(z) > 3.0)
    fits = []
    for r in recs:
        x = r.scalars[key]
        if key == "logf_sum":          # may be ≤ 0: raw slope instead of log slope
            x = np.exp(x)
        fits.append(drift_rate(x, r.lifetimes(kappa_min(recs)), r.T_burn, r.T_meas))
    beta, se_beta = np.nanmean([f[0] for f in fits]), np.nanmean([f[1] for f in fits]) / np.sqrt(len(fits))
    return {"half0": float(h0), "half1": float(h1), "se": se, "rho": rho, "z": z, "flag": flag,
            "beta": float(beta), "se_beta": float(se_beta)}


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


def _grid_stats(records):
    """
    Half-window and drift statistics of every (readout, N, κ, δ₀, λ, solver, η, quantity).

    Pairs in ``NA_BY_CONSTRUCTION`` get no row (§12.7 G4).

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
    kmin = kappa_min(records)
    for cond, recs in group_by(records, ("readout", "N", "kappa_label", "delta0", "lam", "solver", "eta")).items():
        ro, N, kappa, d0, lam, solver, eta = cond
        base = dict(readout=ro, N=N, kappa=kappa, delta0=d0, lam=lam, solver=solver, eta=eta, n_seeds=len(recs))
        mh0, mh1 = (half_life(recs[0].ages, _pooled_S(recs, h))[0] for h in ((0,), (1,)))
        censored = not (np.isfinite(mh0) and np.isfinite(mh1))
        for q in QUANTS:
            if q not in recs[0].scalars or is_na(q, ro):
                continue
            st = _cond_stats(recs, q)
            rows.append(dict(base, quantity=q, m_half_0=mh0, m_half_1=mh1, censored=censored, **st))
        per_seed = []
        for r in recs:
            a, b = (half_life(r.ages, _pooled_S([r], h))[0] for h in ((0,), (1,)))
            per_seed.append((b - a) / a if np.isfinite(a) and np.isfinite(b) and a > 0 else np.nan)
        per_seed = np.asarray(per_seed, float)
        ok = np.isfinite(per_seed)
        se = float(np.std(per_seed[ok], ddof=1) / np.sqrt(ok.sum())) if ok.sum() > 1 else np.nan
        with np.errstate(invalid="ignore", divide="ignore"):
            rho = float((mh1 - mh0) / mh0) if not censored else np.nan
            z = float(np.float64(rho) / np.float64(se))
        rows.append(dict(base, quantity="dm_half", half0=mh0, half1=mh1, se=se, z=z, rho=rho,
                         flag=bool(np.isfinite(z) and abs(z) > 3.0), beta=np.nan, se_beta=np.nan,
                         m_half_0=mh0, m_half_1=mh1, censored=censored))
    del kmin
    return rows


def _group_rows(rows, fields):
    """Group plain dicts by the given fields → ``{tuple: [rows]}``."""
    out = {}
    for r in rows:
        out.setdefault(tuple(r[f] for f in fields), []).append(r)
    return out


# ---------------------------------------------------------------------------
# C2-A
# ---------------------------------------------------------------------------
def fig_traj(records, fig_dir, index, figdata):
    """C2-A: block-mean trajectories vs task in lifetimes (t κ_s²/N), one figure per (κ, δ₀, λ, solver, η)."""
    kmin = kappa_min(records)
    for kappa, kmembers in by_kappa(records).items():
        for cell, members in group_by(kmembers, ("delta0", "lam", "solver", "eta")).items():
            d0, lam, solver, eta = cell
            Ns = sorted({r.run.N for r in members})
            fig, axes = P.grid(len(ROWS), len(Ns), sharex="col", sharey="row")
            data = {}
            for j, N in enumerate(Ns):
                col = [r for r in members if r.run.N == N]
                gated = [r for r in col if is_gated(r.run.readout)]
                ref = gated[0] if gated else col[0]
                ks_ref = ref.run.kappa if gated else kmin
                scale = ks_ref ** 2 / N
                for i in range(len(ROWS)):
                    P.shade_burnin(axes[i, j], ref.T_burn, ref.T_meas, scale)
                for ro in P.READOUT_COLOR:
                    recs = [r for r in col if r.run.readout == ro]
                    if not recs:
                        continue
                    ks = recs[0].run.kappa if is_gated(ro) else kmin
                    x = recs[0].lifetimes(kmin)
                    w = max(1, int(round(N / (2 * ks ** 2))))
                    cuts = (recs[0].T_burn, recs[0].T_burn + recs[0].T_meas // 2)
                    for i, (key, qk) in enumerate(ROWS):
                        if is_na(key, ro):
                            continue
                        ax = axes[i, j]
                        Y = seed_stack(recs, key)
                        with np.errstate(invalid="ignore"):
                            raw = np.nanmean(Y, 0)
                        P.plot_readout(ax, x, raw, ro, lw=0.5, alpha=0.25, zorder=2)
                        xb, mb, sb = P.block_mean(x, Y, w, cuts)
                        P.plot_readout(ax, xb, mb, ro, lw=P.LW_TRACE, zorder=5)
                        P.band(ax, xb, mb - sb, mb + sb, P.READOUT_COLOR[ro])
                        if qk:
                            for k in qk:
                                _, qb, _ = P.block_mean(x, seed_stack(recs, k), w, cuts)
                                P.plot_readout(ax, xb, qb, ro, lw=P.LW_OVERLAY, zorder=3)
                        data[f"{ro}_N{N}_{key}"] = np.stack([xb, mb, sb])
                top = axes[0, j].secondary_xaxis("top", functions=(lambda u, s=scale: u / s, lambda t, s=scale: t * s))
                top.set_xlabel("task"); top.xaxis.set_major_locator(P.MaxNLocator(4))
                axes[-1, j].set_xlabel(r"$t\kappa_s^2/N$ (lifetimes)")
            for i, (key, _) in enumerate(ROWS):
                axes[i, 0].set_ylabel(P.qlabel(key))
            P.ref_line(axes[QUANTS.index("logf_sum"), 0], 0.0)
            P.label_cols_N(axes, Ns)
            P.fig_legend(fig, {r.run.readout for r in members})
            P.cell_title(fig, kappa, d0, lam, solver, eta)
            P.caption(fig, "thick: mean over seeds and blocks of half a lifetime, band ±1 s.e.; faint: per-task seed mean; "
                           "thin: 5 % and 95 % over active units; grey: burn-in; dotted: half boundary; "
                           "Σ log f omits the frozen readouts (f = 1)")
            stem = P.cell_stem(kappa, d0, lam, solver, eta)
            P.save(fig, os.path.join(fig_dir, "diag", "check2", "traj_" + stem), data if figdata else None, index, "diag/check2", stem)


# ---------------------------------------------------------------------------
# C2-B
# ---------------------------------------------------------------------------
def _ratio_cells(rs):
    """Sorted (κ_label, δ₀, λ) cells of a set of rows (gated κ numerically, ungated 'sqrtN' last)."""
    def key(c):
        k = c[0]
        return (1, 0.0) if k == "sqrtN" else (0, float(k)), c[1], c[2]
    return sorted({(r["kappa"], r["delta0"], r["lam"]) for r in rs}, key=key)


def fig_halves(rows, fig_dir, index, figdata, levels=False):
    """C2-B: half-to-half change ρ (in %) across the (κ, δ₀, λ) cells, one file per quantity and (solver, η); levels only with ``--levels``."""
    for (solver, eta, q), sub in _group_rows(rows, ("solver", "eta", "quantity")).items():
        if q == "dm_half":
            continue
        ros = [r for r in P.READOUT_COLOR if any(x["readout"] == r for x in sub)]
        Ns = sorted({r["N"] for r in sub})
        n_cells = max(len(_ratio_cells([r for r in sub if r["readout"] == ro])) for ro in ros)
        panel = (max(P.PANEL[0], 0.12 * n_cells + 0.6), P.PANEL[1])
        kinds = ["ratio"] + (["levels"] if levels else [])
        for kind in kinds:
            fig, axes = P.grid(len(ros), len(Ns), sharex=False, sharey=False if kind == "ratio" else "row", panel=panel)
            data, allv = {}, []
            for i, ro in enumerate(ros):
                rs_ro = [r for r in sub if r["readout"] == ro]
                cells = _ratio_cells(rs_ro)
                for j, N in enumerate(Ns):
                    ax = axes[i, j]
                    for c, cell in enumerate(cells):
                        r = next((x for x in rs_ro if x["N"] == N and (x["kappa"], x["delta0"], x["lam"]) == cell), None)
                        if r is None:
                            continue
                        if kind == "ratio":
                            with np.errstate(invalid="ignore", divide="ignore"):
                                y = 100.0 * r["rho"]
                                e = 100.0 * 3.0 * r["se"] / abs(r["half0"])
                            ax.errorbar([c], [y], yerr=[e] if np.isfinite(e) else None, fmt="none", ecolor=P.GREY_REF, elinewidth=P.LW_REF, zorder=2)
                            ax.plot([c], [y], "o", ms=5, mfc=P.READOUT_COLOR[ro] if is_gated(ro) else "white",
                                    mec=P.READOUT_COLOR[ro], mew=0.9, zorder=4)
                            if r["flag"]:
                                P.flag_mark(ax, c, y)
                            allv.append(y)
                            data[f"{ro}_N{N}_{c}"] = [y, e]
                        else:
                            P.dumbbell(ax, c, r["half0"], r["half1"], r["se"], P.READOUT_COLOR[ro], r["flag"], hollow=not is_gated(ro))
                            data[f"{ro}_N{N}_{c}"] = [r["half0"], r["half1"], r["se"]]
                    P.cell_ticks(ax, cells, labels=True)
                    if kind == "ratio":
                        P.ref_line(ax, 0.0)
                    ax._mup_no_const_check = True
                axes[i, 0].set_ylabel(P.READOUT_LABEL[ro] + ("\nhalf 1 / half 0 − 1 (%)" if kind == "ratio" else ""), color=P.READOUT_COLOR[ro])
            if kind == "ratio":
                P.sym_limits(axes, allv, RHO_FLOOR_PCT)
            P.label_cols_N(axes, Ns)
            P.fig_legend(fig, ros, marker=True)
            P.cell_title(fig, None, None, None, solver, eta, "check 2: " + P.qlabel(q))
            P.caption(fig, "bars: ±3 s.e.; hollow: ungated; red strip and ring: flagged (|z| > 3); one y-range for the figure"
                      if kind == "ratio" else "open: half 0, filled: half 1; bars: ±3 s.e.; red strip and ring: flagged")
            stem = f"{q}_" + P.cell_stem(None, None, None, solver, eta)
            name = "halves_ratio_" if kind == "ratio" else "halves_"
            P.save(fig, os.path.join(fig_dir, "diag", "check2", name + stem), data if figdata else None, index, "diag/check2", stem)


# ---------------------------------------------------------------------------
# C2-C
# ---------------------------------------------------------------------------
def fig_drift(rows, fig_dir, index, figdata):
    """C2-C: drift maps, colour = β/se on fixed ±5, β printed; rows = readouts, columns = N; one file per quantity and (κ, solver, η)."""
    for (solver, eta, q), sub in _group_rows(rows, ("solver", "eta", "quantity")).items():
        if q not in DRIFT_QUANTS:
            continue
        kappas = sorted({r["kappa"] for r in sub if is_gated(r["readout"])}) or ["sqrtN"]
        for kappa in kappas:
            rs = [r for r in sub if (r["kappa"] == kappa) or not is_gated(r["readout"])]
            ros = [r for r in P.READOUT_COLOR if any(x["readout"] == r for x in rs)]
            Ns = sorted({r["N"] for r in rs})
            d0s, lams = sorted({r["delta0"] for r in rs}), sorted({r["lam"] for r in rs})
            pct = q != "logf_sum"
            bfmt = (lambda b: P.fmt2(100 * b) + "%") if pct else P.fmt2
            fig, axes = P.grid(len(ros), len(Ns), sharex=False, sharey=False, extra_w=1.0)
            im, data = None, {}
            for i, ro in enumerate(ros):
                for j, N in enumerate(Ns):
                    B = np.full((len(d0s), len(lams)), np.nan); Z = np.full_like(B, np.nan)
                    for r in rs:
                        if r["readout"] == ro and r["N"] == N:
                            a, b = d0s.index(r["delta0"]), lams.index(r["lam"])
                            B[a, b] = r["beta"]
                            with np.errstate(invalid="ignore", divide="ignore"):
                                Z[a, b] = r["beta"] / r["se_beta"]
                    im = P.drift_map(axes[i, j], Z, B, d0s, lams, bfmt)
                    if i < len(ros) - 1:
                        axes[i, j].set_xticklabels([])
                    if j > 0:
                        axes[i, j].set_yticklabels([])
                    if i == len(ros) - 1:
                        axes[i, j].set_xlabel(r"$\lambda$")
                    data[f"{ro}_N{N}_beta"], data[f"{ro}_N{N}_z"] = B, Z
                axes[i, 0].set_ylabel(P.READOUT_LABEL[ro] + "\n" + r"$\delta_0$", color=P.READOUT_COLOR[ro])
            P.label_cols_N(axes, Ns)
            cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, pad=0.02, ticks=[-5, -3, 0, 3, 5])
            cbar.set_label(r"$\beta/\mathrm{se}(\beta)$")
            na = [P.READOUT_LABEL[r] for r in P.READOUT_COLOR if is_na(q, r)]
            P.cell_title(fig, kappa, None, None, solver, eta, "drift of " + P.qlabel(q))
            P.caption(fig, ("cell text: β, % change per lifetime" if pct else "cell text: β, change per lifetime")
                      + " from a least-squares fit over the measurement window; colour: β/se; black outline: |β/se| > 3; hatched: missing"
                      + (f"; omitted (constant by construction): {', '.join(na)}" if na else ""))
            stem = f"{q}_" + P.cell_stem(kappa, None, None, solver, eta)
            P.save(fig, os.path.join(fig_dir, "diag", "check2", "drift_" + stem), data if figdata else None, index, "diag/check2", stem)


# ---------------------------------------------------------------------------
# C2-D
# ---------------------------------------------------------------------------
def _half_curve(recs, h):
    """Pooled S, E, their s.e. and the count at every age for one half."""
    n = sum(r.moments["count"][:, h] for r in recs)
    with np.errstate(invalid="ignore", divide="ignore"):
        S = sum(r.moments["sum_s"][:, h] for r in recs) / n
        E = sum(r.moments["sum_e"][:, h] for r in recs) / n
        S2 = sum(r.moments["sum_s2"][:, h] for r in recs) / n
        se = np.sqrt(np.maximum(S2 - S ** 2, 0.0) / n)
    return S, E, se, n


def fig_Sm_halves(records, fig_dir, index, figdata):
    """C2-D: S(m), E(m) from the two halves overlaid (half 0 solid, half 1 dotted), a ΔS row, m½ ticks in a strip."""
    for kappa, kmembers in by_kappa(records).items():
        for cell, members in group_by(kmembers, ("delta0", "lam", "solver", "eta")).items():
            d0, lam, solver, eta = cell
            Ns = sorted({r.run.N for r in members})
            fig, axes = P.grid(3, len(Ns), sharex="col", sharey="row")
            data, thinned, max_age = {}, [], 1.0
            for j, N in enumerate(Ns):
                mh, edge = {}, {0: ([], []), 1: ([], []), 2: ([], [])}
                for ro in P.READOUT_COLOR:
                    recs = [r for r in members if r.run.N == N and r.run.readout == ro]
                    if not recs:
                        continue
                    ages = recs[0].ages.astype(float)
                    max_age = max(max_age, float(ages.max()))
                    curves = {h: _half_curve(recs, h) for h in (0, 1)}
                    for h, ls in ((0, "-"), (1, ":")):
                        S, E, se, n = curves[h]
                        ok = n >= 0.25 * np.nanmax(n)
                        if not ok.all():
                            thinned.append(f"{P.READOUT_LABEL[ro]} N = {N} half {h}: {int(n[-1])} at age {P.fmt(ages[-1])}")
                        for i, y in ((0, S), (1, E)):
                            ax = axes[i, j]
                            P.plot_readout(ax, ages, np.where(ok, y, np.nan), ro, ls=ls, lw=1.0)
                            P.plot_readout(ax, ages, y, ro, ls=ls, lw=P.LW_OVERLAY, alpha=0.5, zorder=3, scaley=False)
                            edge[i][0].extend(ages[~ok]); edge[i][1].extend(y[~ok])
                        mh.setdefault(ro, [np.nan, np.nan])[h] = half_life(recs[0].ages, S)[0]
                        data[f"{ro}_N{N}_S_half{h}"], data[f"{ro}_N{N}_E_half{h}"] = S, E
                    S0, _, se0, n0 = curves[0]; S1, _, se1, n1 = curves[1]
                    dS = S1 - S0
                    okd = (n0 >= 0.25 * np.nanmax(n0)) & (n1 >= 0.25 * np.nanmax(n1))
                    sed = 2.0 * np.sqrt(se0 ** 2 + se1 ** 2)
                    P.plot_readout(axes[2, j], ages, np.where(okd, dS, np.nan), ro, ls="-", lw=1.0)
                    P.plot_readout(axes[2, j], ages, dS, ro, ls="-", lw=P.LW_OVERLAY, alpha=0.5, zorder=3, scaley=False)
                    P.band(axes[2, j], ages, np.where(okd, dS - sed, np.nan), np.where(okd, dS + sed, np.nan), P.READOUT_COLOR[ro])
                    edge[2][0].extend(ages[~okd]); edge[2][1].extend(dS[~okd])
                    data[f"{ro}_N{N}_dS"] = dS
                for i in range(3):
                    P.age_axis(axes[i, j], max_age)
                P.ref_line(axes[2, j], 0.0)
                axes[2, j].set_xlabel("age m (tasks)")
                P.mhalf_ticks(axes[0, j], {ro: tuple(v) for ro, v in mh.items()})
                for i in range(3):
                    P.edge_markers(axes[i, j], *edge[i])
            axes[0, 0].set_ylabel("S(m)"); axes[1, 0].set_ylabel("E(m)"); axes[2, 0].set_ylabel(r"$\Delta S(m)$ = half 1 − half 0")
            P.label_cols_N(axes, Ns)
            P.fig_legend(fig, {r.run.readout for r in members}, styles=[("-", "half 0"), (":", "half 1")])
            P.cell_title(fig, kappa, d0, lam, solver, eta)
            cap = "strip: m½ of half 0 (open) and half 1 (filled); ΔS band: ±2 s.e.; thin: ages with < 25 % of the trace's maximum count"
            if thinned:
                cap += " (count at the last age: " + "; ".join(thinned[:2]) + (", …" if len(thinned) > 2 else "") + ")"
            P.caption(fig, cap)
            stem = P.cell_stem(kappa, d0, lam, solver, eta)
            P.save(fig, os.path.join(fig_dir, "diag", "check2", "Sm_halves_" + stem), data if figdata else None, index, "diag/check2", stem)


# ---------------------------------------------------------------------------
# C2-E
# ---------------------------------------------------------------------------
def _slot_values(sub, q, N, stat):
    """``{readout: array}`` of one statistic over the (κ, δ₀, λ) cells of quantity ``q`` at width ``N``."""
    out = {}
    for ro in P.READOUT_COLOR:
        cells = [r for r in sub if r["quantity"] == q and r["N"] == N and r["readout"] == ro and not (q == "dm_half" and r["censored"])]
        if not cells:
            continue
        if stat == "zbeta":
            with np.errstate(invalid="ignore", divide="ignore"):
                v = np.array([r["beta"] for r in cells], float) / np.array([r["se_beta"] for r in cells], float)
        elif stat == "rho_pct":
            v = 100.0 * np.array([r["rho"] for r in cells], float)
        else:
            v = np.array([r[stat] for r in cells], float)
        out[ro] = v
    return out


def fig_summary(rows, fig_dir, index, figdata):
    """C2-E summary: per-cell points (box only at ≥ MIN_BOX_N) per readout of z (``summary_*``), ρ (``summary_rho_*``) and β/se (``summary_beta_*``).

    The three files have the same rows in the same order; the ``dm_half``
    row is a ratio with its own range (n/a in the β file).
    """
    for (solver, eta), sub in _group_rows(rows, ("solver", "eta")).items():
        Ns = sorted({r["N"] for r in sub})
        ros_all = [r for r in P.READOUT_COLOR if any(x["readout"] == r for x in sub)]
        for kind, stat, title in (("summary", "z", "check 2: z = (half 1 − half 0) / s.e."),
                                  ("summary_rho", "rho_pct", "check 2: ρ = half 1 / half 0 − 1 (%)"),
                                  ("summary_beta", "zbeta", "check 2: β / se(β)")):
            fig, axes = P.grid(len(SUMMARY_ROWS), len(Ns), sharex="col", sharey=False)
            data, main_vals, dm_vals, dm_se = {}, [], [], []
            for i, q in enumerate(SUMMARY_ROWS):
                for j, N in enumerate(Ns):
                    ax = axes[i, j]
                    last = i == len(SUMMARY_ROWS) - 1
                    if q == "dm_half" and kind == "summary_beta":
                        P.na_slot(ax, "no drift fit for m½")
                        ax.set_xticks([])
                        continue
                    vals = _slot_values(sub, q, N, "rho_pct" if q == "dm_half" else stat)
                    na = {ro: NA_BY_CONSTRUCTION[(q, ro)] for ro in ros_all if is_na(q, ro)}
                    if q == "dm_half":
                        ses = 100.0 * np.array([r["se"] for r in sub if r["quantity"] == q and r["N"] == N and not r["censored"]], float)
                        half = 2.0 * float(np.nanmedian(ses)) if np.isfinite(ses).any() else 0.0
                        P.dist_by_readout(ax, vals, band=half if half > 0 else None, na=na, labels=last)
                        n_cens = sum(r["censored"] for r in sub if r["quantity"] == q and r["N"] == N)
                        ax.text(0.98, 0.04, f"{n_cens} censored", transform=ax.transAxes, ha="right", va="bottom", fontsize=P.FS_MIN, color=P.GREY_CAPTION)
                        dm_vals += [v for v in vals.values()]; dm_se.append(half)
                    else:
                        P.dist_by_readout(ax, vals, band=3.0 if stat != "rho_pct" else None, clip=Z_CLIP if stat != "rho_pct" else None, na=na, labels=last)
                        main_vals += [v for v in vals.values()]
                    for ro, v in vals.items():
                        data[f"{q}_{ro}_N{N}"] = v
                axes[i, 0].set_ylabel(P.qlabel(q) + ("\n(ratio, own range)" if q == "dm_half" else ""))
            main_axes = axes[:-1].ravel()
            flat = np.concatenate([np.ravel(v) for v in main_vals]) if main_vals else np.array([])
            if stat == "rho_pct":
                lim = P.sym_limits(main_axes, flat, RHO_FLOOR_PCT)
                if lim > 50:
                    for ax in main_axes:
                        ax.set_yscale("symlog", linthresh=1.0)
                        P.discrete_ticks(ax, [-100, -10, -1, 0, 1, 10, 100], "y")
            else:
                P.sym_limits(main_axes, flat, Z_FLOOR, Z_CLIP)
                for ax in main_axes:
                    cx, cy = getattr(ax, "_mup_clipped", ([], []))
                    P.edge_markers(ax, cx, cy)
            if kind != "summary_beta":
                flat_dm = np.concatenate([np.ravel(v) for v in dm_vals]) if dm_vals else np.array([])
                P.sym_limits(axes[-1], np.concatenate([flat_dm, 1.2 * np.array(dm_se)]), RHO_FLOOR_PCT)
            P.label_cols_N(axes, Ns)
            if kind == "summary":          # the per-cell statistics, indexed by the first CELL_COLS fields (§12.6)
                data.update({"cells_" + c: np.array([r[c] for r in sub]) for c in CELL_COLS})
            P.fig_legend(fig, ros_all, marker=True)
            P.cell_title(fig, None, None, None, solver, eta, title)
            P.caption(fig, "points: (κ, δ₀, λ) cells, seeds averaged; black tick: median; box only from ≥ 10 cells; hollow: ungated; "
                      + ("band: |z| ≤ 3; ▲▼: beyond ±8" if kind == "summary" else
                         "band: |β/se| ≤ 3; ▲▼: beyond ±8" if kind == "summary_beta" else "one y-range for the figure")
                      + "; last row: Δm½/m½ in %, band ±2 s.e. over seeds")
            P.row_rule(fig, axes, len(SUMMARY_ROWS) - 1)
            stem = P.cell_stem(None, None, None, solver, eta)
            P.save(fig, os.path.join(fig_dir, "diag", "check2", f"{kind}_" + stem), data if figdata else None, index, "diag/check2", stem)


def fig_flags(rows, fig_dir, index, figdata):
    """C2-E flag table (``flags_*``): k/n flagged (κ, δ₀, λ) cells per (quantity, readout), one panel per N."""
    for (solver, eta), sub in _group_rows(rows, ("solver", "eta")).items():
        Ns = sorted({r["N"] for r in sub})
        ros = [r for r in P.READOUT_COLOR if any(x["readout"] == r for x in sub)]
        fig, axes = P.grid(1, len(Ns), sharex=False, sharey=False, panel=(max(P.PANEL[0], 0.5 * len(ros) + 0.4), 0.32 * len(SUMMARY_ROWS) + 0.4))
        data = {}
        for j, N in enumerate(Ns):
            counts, na = {}, {}
            for q in SUMMARY_ROWS:
                for ro in ros:
                    if is_na(q, ro):
                        na[(q, ro)] = NA_BY_CONSTRUCTION[(q, ro)]
                        continue
                    cells = [r for r in sub if r["quantity"] == q and r["N"] == N and r["readout"] == ro and not r["censored"]]
                    if cells:
                        counts[(q, ro)] = (sum(bool(r["flag"]) for r in cells), len(cells))
                        data[f"{q}_{ro}_N{N}"] = list(counts[(q, ro)])
            P.flag_table(axes[0, j], counts, list(SUMMARY_ROWS), ros, na, row_labels=(j == 0))
        P.label_cols_N(axes, Ns)
        P.cell_title(fig, None, None, None, solver, eta, "check 2: flagged cells")
        P.caption(fig, "k/n: (κ, δ₀, λ) cells with |z| > 3 of the cells run (censored m½ cells excluded); fill: flagged fraction; "
                       "—: constant by construction; chance at 3 s.e. ≈ 0.3 % of cells")
        stem = P.cell_stem(None, None, None, solver, eta)
        P.save(fig, os.path.join(fig_dir, "diag", "check2", "flags_" + stem), data if figdata else None, index, "diag/check2", stem)


# ---------------------------------------------------------------------------
# the checks table (PLAN.md §12.6)
# ---------------------------------------------------------------------------
#: pass criteria, set once here and printed in the table's caption: key → (label, ok, warn) thresholds
CRITERIA = {
    "gauge_delta": ("1 gauge: Δᵢ pinned", 1e-10, 1e-8),
    "gauge_uc": ("1 gauge: u_c invariant", 1e-10, 1e-8),
    "b_aQ_act_post": ("2 bounded: mean |a|√Q", 0, 1),
    "b_aQ_max_post": ("2 bounded: max |a|√Q", 0, 1),
    "b_Q_act_pre": ("2 bounded: Q_act (pre)", 0, 1),
    "b_a2_act_post": ("2 bounded: a²_act", 0, 1),
    "b_logf_sum": ("2 bounded: Σ log f", 0, 1),
    "b_S0": ("2 steady memory: S(0)", 0, 1),
    "mhalf": ("2 steady memory: m½", 0, 1),
    "drift": ("2 drift: β", 0, 1),
    "cv": ("3 self-averaging γ", -0.4, -0.2),
    "collapse": ("4 collapse spread", 0.05, 0.10),
    "feasible": ("feasibility (skipped)", 0.0, 0.10),
}
CHECK_ROWS = [(k, v[0]) for k, v in CRITERIA.items()]


def _mark_count(k: int, fail_also: bool = False) -> str:
    """Mark of a flagged-cell count: 0 ✓, 1 ~, ≥ 2 ✗."""
    return "fail" if (k >= 2 or fail_also) else ("warn" if k == 1 else "ok")


def _mark_value(v: float, ok: float, warn: float, lower_is_better: bool = True) -> str:
    """Mark of a continuous statistic against (ok, warn) thresholds."""
    if not np.isfinite(v):
        return "na"
    if lower_is_better:
        return "ok" if v < ok else ("warn" if v < warn else "fail")
    return "ok" if v <= ok else ("warn" if v <= warn else "fail")


def _parse_skipped(label: str):
    """(N, readout) of a skipped-run label ``N=10 k=1.0 gated_frozen d0=… lam=… eta=… seed=…``."""
    parts = label.split()
    try:
        return int(parts[0].split("=")[1]), parts[2]
    except (IndexError, ValueError):
        return None


def _cv_gamma(records, age_index=1):
    """Check 3 per readout: mean over (κ, δ₀, λ) cells of the slope of log CV(s, m = 1) vs log N.

    Cells whose pooled mean s is within 3 s.e. of zero at a width are dropped
    at that width (CV is undefined as the mean → 0).

    Returns
    -------
    dict
        ``{readout: (gamma, n_widths)}``; ``gamma`` NaN with fewer than two usable widths.
    """
    from mup.observe import cv_exponent
    out = {}
    for (ro,), recs_ro in group_by(records, ("readout",)).items():
        slopes, nN = [], 0
        for _, cell_recs in group_by(recs_ro, ("kappa_label", "delta0", "lam", "solver", "eta")).items():
            pts = []
            for (N,), recs in sorted(group_by(cell_recs, ("N",)).items()):
                n = sum(r.moments["count"].sum(-1) for r in recs)
                if len(recs[0].ages) <= age_index or n[age_index] <= 0:
                    continue
                with np.errstate(invalid="ignore", divide="ignore"):
                    S = sum(r.moments["sum_s"].sum(-1) for r in recs) / n
                    S2 = sum(r.moments["sum_s2"].sum(-1) for r in recs) / n
                    sd = np.sqrt(np.maximum(S2 - S ** 2, 0))
                    cv = sd[age_index] / abs(S[age_index])
                    se = sd[age_index] / np.sqrt(n[age_index])
                if abs(S[age_index]) > 3 * se and np.isfinite(cv):
                    pts.append((N, cv))
            if len(pts) >= 2:
                Ns, cvs = zip(*pts)
                slopes.append(cv_exponent(Ns, cvs)[0]); nN = max(nN, len(pts))
        out[ro] = (float(np.nanmean(slopes)) if slopes else np.nan, nN)
    return out


def _collapse_spread(records):
    """Check 4 per frozen readout: worst over cells of max_m spread across N of S(m)/S(0) at matched m κ_s²/N.

    Returns
    -------
    dict
        ``{readout: spread}``; NaN with fewer than two widths.
    """
    kmin = kappa_min(records)
    out = {}
    for ro in ("gated_frozen", "frozen"):
        worst = np.nan
        recs_ro = [r for r in records if r.run.readout == ro]
        if not recs_ro:
            continue
        for _, cell_recs in group_by(recs_ro, ("kappa_label", "delta0", "lam", "solver", "eta")).items():
            curves = []
            for (N,), recs in sorted(group_by(cell_recs, ("N",)).items()):
                n = sum(r.moments["count"].sum(-1) for r in recs)
                with np.errstate(invalid="ignore", divide="ignore"):
                    S = sum(r.moments["sum_s"].sum(-1) for r in recs) / n
                    y = S / S[0]
                x = recs[0].ages * recs[0].kappa_sched(kmin) ** 2 / N
                ok = np.isfinite(y) & (x > 0)
                if ok.sum() >= 2:
                    curves.append((x[ok], y[ok]))
            if len(curves) < 2:
                continue
            lo, hi = max(c[0].min() for c in curves), min(c[0].max() for c in curves)
            if not hi > lo:
                continue
            grid = np.geomspace(lo, hi, 20)
            Y = np.stack([np.interp(np.log(grid), np.log(c[0]), c[1]) for c in curves])
            spread = float(np.max(Y.max(0) - Y.min(0)))
            worst = spread if not np.isfinite(worst) else max(worst, spread)
        out[ro] = worst
    return out


def table_marks(records, rows, skipped=None):
    """Compute every mark of the checks table.

    Parameters
    ----------
    records : list of RunRecord
    rows : list of dict
        Output of ``_grid_stats``.
    skipped : list of str, optional
        Labels of runs refused as infeasible.

    Returns
    -------
    (marks, cols, details)
        ``marks`` ``{(row_key, (readout, N)): (mark, text)}``; ``cols`` the
        (readout, N) columns in fixed readout order; ``details`` the CSV rows
        (row, readout, N, mark, value, threshold_ok, threshold_warn, worst_cell).
    """
    skipped = skipped or []
    ros = [ro for ro in P.READOUT_COLOR if any(r.run.readout == ro for r in records)]
    Ns = sorted({r.run.N for r in records})
    cols = [(ro, N) for ro in ros for N in Ns if any(r.run.readout == ro and r.run.N == N for r in records)]
    marks, details = {}, []

    def put(rk, col, mark, value, text, worst=""):
        marks[(rk, col)] = (mark, text)
        lab, ok, warn = CRITERIA[rk]
        details.append(dict(row=lab, readout=col[0], N=col[1], mark=mark or "", value=value, threshold_ok=ok, threshold_warn=warn, worst_cell=worst))

    def cell_of(r):
        return f"κ={P.fmt(r['kappa'])} δ₀={P.fmt(r['delta0'])} λ={P.fmt(r['lam'])}"

    gamma = _cv_gamma(records)
    spread = _collapse_spread(records)
    for col in cols:
        ro, N = col
        recs = [r for r in records if r.run.readout == ro and r.run.N == N]
        cell_rows = [r for r in rows if r["readout"] == ro and r["N"] == N]
        # check 1 (projected readouts only)
        for rk in ("gauge_delta", "gauge_uc"):
            if "frozen" in ro:
                put(rk, col, "na", np.nan, "")
                continue
            v, worst = max((float(np.nanmax(r.scalars[rk])), r.run.label()) for r in recs)
            put(rk, col, _mark_value(v, *CRITERIA[rk][1:]), v, P.fmt2(v), worst)
        # check 2: bounded norms and S(0)
        for q in ("aQ_act_post", "aQ_max_post", "Q_act_pre", "a2_act_post", "logf_sum", "S0"):
            rk = "b_" + q
            if is_na(q, ro):
                put(rk, col, "na", np.nan, "")
                continue
            cr = [r for r in cell_rows if r["quantity"] == q]
            k, n = sum(bool(r["flag"]) for r in cr), len(cr)
            worst = max(cr, key=lambda r: abs(r["z"]) if np.isfinite(r["z"]) else -1, default=None)
            put(rk, col, _mark_count(k) if n else "na", k, f"{k}/{n}" if n else "",
                f"{cell_of(worst)} z={worst['z']:+.1f}" if worst else "")
        # m½: |Δm½| > 2 s.e.(m½ over seeds); censored cells counted separately
        cr = [r for r in cell_rows if r["quantity"] == "dm_half"]
        live = [r for r in cr if not r["censored"]]
        cens = len(cr) - len(live)
        k = sum(bool(np.isfinite(r["rho"]) and np.isfinite(r["se"]) and abs(r["rho"]) > 2 * r["se"]) for r in live)
        worst = max(live, key=lambda r: abs(r["z"]) if np.isfinite(r["z"]) else -1, default=None)
        put("mhalf", col, _mark_count(k, fail_also=cens > 0) if cr else "na", k,
            (f"{k}/{len(live)}" + (f", {cens} cens." if cens else "")) if cr else "",
            f"{cell_of(worst)} Δm½/m½={100 * worst['rho']:+.1f}%" if worst else "")
        # drift: |β/se| > 3 for mean |a|√Q (and Q_act pre for the frozen readouts)
        keys = ("aQ_act_post", "Q_act_pre") if "frozen" in ro else ("aQ_act_post",)
        by_cell = {}
        for r in cell_rows:
            if r["quantity"] in keys and np.isfinite(r["beta"]) and np.isfinite(r["se_beta"]) and r["se_beta"] > 0:
                c = (r["kappa"], r["delta0"], r["lam"])
                by_cell[c] = max(by_cell.get(c, 0.0), abs(r["beta"] / r["se_beta"]))
        k, n = sum(v > 3 for v in by_cell.values()), len(by_cell)
        wc = max(by_cell, key=by_cell.get, default=None)
        put("drift", col, _mark_count(k) if n else "na", k, f"{k}/{n}" if n else "",
            f"κ={P.fmt(wc[0])} δ₀={P.fmt(wc[1])} λ={P.fmt(wc[2])} |β/se|={by_cell[wc]:.1f}" if wc else "")
        # check 3 (one value per readout; a two-point slope is printed in grey with no mark)
        g, nN = gamma.get(ro, (np.nan, 0))
        if nN < 3:
            put("cv", col, None, g, f"{g:+.2f}\n(2-pt)" if np.isfinite(g) else "n/a")
        else:
            put("cv", col, _mark_value(g, *CRITERIA["cv"][1:], lower_is_better=False), g, f"{g:+.2f}")
        # check 4 (frozen readouts)
        if ro in spread:
            v = spread[ro]
            put("collapse", col, _mark_value(v, *CRITERIA["collapse"][1:]) if np.isfinite(v) else None, v,
                P.fmt2(v) if np.isfinite(v) else "1 width")
        else:
            put("collapse", col, "na", np.nan, "")
        # feasibility
        k_skip = sum(1 for lab in skipped if _parse_skipped(lab) == (N, ro))
        n_tot = k_skip + len(recs)
        frac = k_skip / n_tot if n_tot else 0.0
        put("feasible", col, "ok" if k_skip == 0 else ("warn" if frac <= CRITERIA["feasible"][2] else "fail"), frac, f"{k_skip}/{n_tot}")
    return marks, cols, details


def fig_table(records, rows, fig_dir, index, skipped=None, figdata=False):
    """Draw ``checks/checks_{solver}_eta{η}.png``, write its CSV, print the text form and the worst cells per ✗."""
    for (solver, eta), recs in group_by(records, ("solver", "eta")).items():
        sub = [r for r in rows if r["solver"] == solver and r["eta"] == eta]
        marks, cols, details = table_marks(recs, sub, skipped)
        fig, axes = P.grid(1, 1, panel=(0.62 * len(cols) + 0.4, 0.48 * (len(CHECK_ROWS) + 2)))
        P.check_table(axes[0, 0], marks, CHECK_ROWS, cols)
        P.cell_title(fig, None, None, None, solver, eta, "checks")
        c = CRITERIA
        P.caption(fig, f"✓ / ~ / ✗ thresholds — gauge: {c['gauge_delta'][1]:g}, {c['gauge_delta'][2]:g}; flagged cells (|z| > 3, |Δm½| > 2 s.e., |β/se| > 3): 0, 1, ≥ 2 "
                       f"(any censored m½ = ✗); γ: ≤ {c['cv'][1]}, ≤ {c['cv'][2]}; collapse spread: < {c['collapse'][1]}, < {c['collapse'][2]}; "
                       f"skipped: 0, ≤ {100 * c['feasible'][2]:g} %; —: not applicable by construction; number under a mark: what it was judged on")
        stem = P.cell_stem(None, None, None, solver, eta)
        os.makedirs(os.path.join(fig_dir, "checks"), exist_ok=True)
        with open(os.path.join(fig_dir, "checks", f"checks_{stem}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["row", "readout", "N", "mark", "value", "threshold_ok", "threshold_warn", "worst_cell"])
            w.writeheader(); w.writerows(details)
        P.save(fig, os.path.join(fig_dir, "checks", "checks_" + stem),
               {"cells_" + k: np.array([r[k] for r in sub]) for k in CELL_COLS} if figdata and sub else None, index, "checks", stem)
        glyph = {"ok": "✓", "warn": "~", "fail": "✗", "na": "—", None: " "}
        print(f"checks  {solver} η={P.fmt(eta)}")
        print(" " * 28 + " ".join(f"{P.READOUT_LABEL[ro][:7]:>7s}/{P.fmt(N):<3s}" for ro, N in cols))
        for rk, lab in CHECK_ROWS:
            print(f"  {lab:26s}" + " ".join(f"{glyph[marks.get((rk, col), (None, ''))[0]]:^11s}" for col in cols))
        for d in [d for d in details if d["mark"] == "fail" and d["worst_cell"]][:12]:
            print(f"  ✗ {d['row']:24s} {d['readout']:14s} N={d['N']:4d}  worst: {d['worst_cell']}")


def write_cells(rows, fig_dir):
    """Write ``checks/check2_cells.csv`` (every per-cell statistic, the machine-readable export)."""
    os.makedirs(os.path.join(fig_dir, "checks"), exist_ok=True)
    path = os.path.join(fig_dir, "checks", "check2_cells.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(CELL_COLS))
        w.writeheader()
        for r in rows:
            w.writerow({c: r[c] for c in CELL_COLS})
    return path


def run(out_dir, fig_dir, records=None, figdata=False, levels=False, diag=False, skipped=None):
    """Compute the check statistics, draw the checks table, and with ``diag`` the C2 figures.

    Parameters
    ----------
    out_dir : str
    fig_dir : str
    records : list of RunRecord, optional
    figdata : bool
    levels : bool
        With ``diag``, also write the absolute half-window levels (``halves_*``).
    diag : bool
        Draw the diagnostic figures C2-A … C2-E into ``diag/check2/``.
    skipped : list of str, optional
        Labels of infeasible runs (feasibility row); loaded here when ``records`` is None.

    Returns
    -------
    list of (family, cell, file)
        INDEX.md rows.
    """
    if records is None:
        records, skipped = load_runs(out_dir)
    index = []
    if not records:
        return index
    rows = _grid_stats(records)
    write_cells(rows, fig_dir)
    fig_table(records, rows, fig_dir, index, skipped, figdata)
    if diag:
        fig_Sm_halves(records, fig_dir, index, figdata)
        fig_summary(rows, fig_dir, index, figdata)
        fig_flags(rows, fig_dir, index, figdata)
        fig_halves(rows, fig_dir, index, figdata, levels)
        fig_drift(rows, fig_dir, index, figdata)
        fig_traj(records, fig_dir, index, figdata)
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
    ap.add_argument("--diag", action="store_true", help="draw the C2-A … C2-E diagnostic figures (diag/check2/)")
    ap.add_argument("--levels", action="store_true", help="with --diag, also the absolute half-window levels (halves_*)")
    args = ap.parse_args(argv)
    fig_dir = args.fig_dir or os.path.join(args.out_dir, "figures")
    index = run(args.out_dir, fig_dir, None, args.save_figdata, args.levels, args.diag)
    print(f"{len(index)} figures → {fig_dir}/checks" + (" and diag/check2" if args.diag else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
