#!/usr/bin/env python
"""Merge the batch files of a sweep: summary.csv, curves.json, the four figure families F1–F4 and the checks table.

    python scripts/analyze.py results/pilot [--fig-dir DIR] [--no-figures] [--save-figdata]
                                            [--cells d0=0,lam=1] [--all | --diag mhalf kaczmarz by_step check2]

Default figures (PLAN.md §12.3, §12.6): ``norms/`` (F1, F2 via
``plot_norms.py``), ``curves/`` (F3, F4) and ``checks/`` (the ✓/✗ table via
``plot_check2.py``).  The diagnostic families of the previous revision
(``diag/{mhalf,kaczmarz,by_step,check2}/``) are drawn only with ``--all``
or ``--diag``.  Only the running moments and per-task scalars are read;
never raw state.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

import _path  # noqa: F401

from mup import plotting as P
from mup.config import is_gated
from mup.observe import cv_exponent, half_life
from mup.results import COND_KEYS, by_kappa, condition_mhalf, group_by, half_stats, kappa_min, load_runs, sem

#: per-task scalars compared between halves for check 2 (PLAN.md §6)
CHECK2_KEYS = ("Q_all_post", "a2_all_post", "aQ_all_post", "aQ_max_post", "E0")


def collect(records):
    """Pool the records of each condition over seeds.

    Parameters
    ----------
    records : list of RunRecord

    Returns
    -------
    dict
        ``{condition key: dict}`` with ``ages``, ``S``, ``E`` (seed-pooled
        curves), ``S_se``, ``E_se`` (s.e.m. across seeds), ``S_seeds``,
        ``m_half``, ``censored``, ``m_half_se``, ``m_half_0``, ``m_half_1``
        (per measurement half), ``E0``, ``E0_se``, ``cv_pooled``,
        ``lam_real``, ``abar2``, ``gauge_delta``, ``gauge_uc``, ``flags``
        (check-2 flag counts per key), ``records``.
    """
    conds = {}
    for key, recs in group_by(records, COND_KEYS).items():
        r0 = recs[0]
        n = sum(r.moments["count"].sum(-1) for r in recs)
        with np.errstate(invalid="ignore", divide="ignore"):
            S = sum(r.moments["sum_s"].sum(-1) for r in recs) / n
            E = sum(r.moments["sum_e"].sum(-1) for r in recs) / n
            S2 = sum(r.moments["sum_s2"].sum(-1) for r in recs) / n
            cv = np.sqrt(np.maximum(S2 - S**2, 0)) / np.abs(S)
        cs = [r.curve() for r in recs]
        S_seeds = np.stack([c["S"] for c in cs])
        E_seeds = np.stack([c["E"] for c in cs])
        mh = np.array([half_life(r0.ages, s)[0] for s in S_seeds])
        m_half, cens = half_life(r0.ages, S)
        flags = {k: 0 for k in CHECK2_KEYS}
        for r in recs:
            for k in CHECK2_KEYS:
                if k in r.scalars and half_stats(r.scalars[k], r.T_burn, r.T_meas)["z"] > 3.0:
                    flags[k] += 1
        d = {
            "ages": r0.ages, "S": S, "E": E, "cv_pooled": cv, "n_pooled": n, "S_seeds": S_seeds, "E_seeds": E_seeds,
            "S_se": sem(S_seeds), "E_se": sem(E_seeds), "n_seeds": len(recs), "records": recs, "run": r0.run,
            "m_half": m_half, "censored": cens,
            "m_half_se": float(sem(mh[np.isfinite(mh)])) if np.isfinite(mh).sum() > 1 else np.nan,
            "m_half_0": condition_mhalf(recs, (0,))[0], "m_half_1": condition_mhalf(recs, (1,))[0],
            "E0": float(E_seeds[:, 0].mean()), "E0_se": float(sem(E_seeds[:, 0])),
            "abar2": float(np.nanmean([r.scalars["abar2"][0] for r in recs])),
            "lam_real": float(np.nanmean([r.run.eta * r.alpha_task * r.scalars["abar2"][0] for r in recs])),
            "gauge_delta": float(max(np.nanmax(r.scalars["gauge_delta"]) for r in recs)),
            "gauge_uc": float(max(np.nanmax(r.scalars["gauge_uc"]) for r in recs)),
            "flags": flags,
        }
        conds[key] = d
    return conds


def write_summary(conds, out_dir):
    """Write ``summary.csv`` (one row per condition) and ``curves.json``.

    Parameters
    ----------
    conds : dict
        Output of ``collect``.
    out_dir : str

    Returns
    -------
    list of dict
        The rows written.
    """
    rows = []
    for key, d in sorted(conds.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        row = dict(zip(COND_KEYS, key))
        row.update(
            n_seeds=d["n_seeds"], S0=d["S"][0], S0_se=d["S_se"][0], E0=d["E0"], E0_se=d["E0_se"],
            m_half=d["m_half"], m_half_se=d["m_half_se"], censored=d["censored"],
            m_half_0=d["m_half_0"], m_half_1=d["m_half_1"], dm_half=d["m_half_1"] - d["m_half_0"],
            lam_real=d["lam_real"], abar2=d["abar2"], gauge_delta_max=d["gauge_delta"], gauge_uc_max=d["gauge_uc"],
            cv_s_age1=float(d["cv_pooled"][1]) if len(d["ages"]) > 1 else np.nan,
            **{f"flag_{k}": v for k, v in d["flags"].items()},
        )
        rows.append(row)
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with open(os.path.join(out_dir, "curves.json"), "w") as f:
        json.dump([{**dict(zip(COND_KEYS, k)), "ages": d["ages"].tolist(), "S": d["S"].tolist(), "E": d["E"].tolist(),
                    "S_se": np.nan_to_num(d["S_se"], nan=-1).tolist(), "E_se": np.nan_to_num(d["E_se"], nan=-1).tolist()}
                   for k, d in conds.items()], f)
    return rows


def _cells(conds, fields):
    """
    Group conditions by κ label (ungated repeated) and then by ``fields``.

    Returns
    -------
    dict
        ``{(kappa_label, cell tuple): {condition key: condition dict}}``.
    """
    out = {}
    for kappa, recs in by_kappa([d["records"][0] for d in conds.values()]).items():
        keys = {r.key() for r in recs}
        for cell, sub in group_by([conds[k]["records"][0] for k in keys], fields).items():
            out[(kappa, cell)] = {r.key(): conds[r.key()] for r in sub}
    return out


_LAST_AGE_WARNED = set()


def _thin_mask(n):
    """Ages whose pooled count is below 25 % of the trace's maximum (drawn thin, kept out of autoscaling; §12.7 G14)."""
    n = np.asarray(n, float)
    return n < 0.25 * np.nanmax(n)


def _draw_curve(ax, x, y, se, n, color, ls, lw=P.LW_TRACE):
    """A trace with its ±1 s.e.m. band; few-sample ages thin and outside the autoscale.  Returns the thinned (x, y)."""
    thin = _thin_mask(n)
    ax.plot(x, np.where(thin, np.nan, y), color=color, ls=ls, lw=lw, zorder=4)
    ax.plot(x, y, color=color, ls=ls, lw=P.LW_OVERLAY, alpha=0.5, zorder=3, scaley=False)
    if se is not None:
        P.band(ax, x, np.where(thin, np.nan, y - se), np.where(thin, np.nan, y + se), color)
    return x[thin], y[thin]


def fig_curves(conds, fig_dir, index, figdata):
    """F3: S(m), E(m) vs age; rows S, E; columns N; traces = readouts ± s.e.m. over seeds (PLAN.md §5, §12.3)."""
    for (kappa, cell), members in _cells(conds, ("delta0", "lam", "solver", "eta")).items():
        d0, lam, solver, eta = cell
        Ns = sorted({k[0] for k in members})
        fig, axes = P.grid(2, len(Ns), sharex="col", sharey="row")
        data, max_age, notes = {}, 1.0, []
        for j, N in enumerate(Ns):
            edges = {0: ([], []), 1: ([], [])}
            last_age = max(float(members[k]["ages"].max()) for k in members if k[0] == N)
            for ro in P.READOUT_COLOR:
                key = next((k for k in members if k[0] == N and k[2] == ro), None)
                if key is None:
                    continue
                d = members[key]
                a = d["ages"].astype(float)
                max_age = max(max_age, float(a.max()))
                if a.max() < last_age and (ro, N) not in _LAST_AGE_WARNED:
                    _LAST_AGE_WARNED.add((ro, N))
                    print(f"warning: {P.READOUT_LABEL[ro]} at N={N} has no data at the panel's last age {P.fmt(last_age)} (ends at {P.fmt(a.max())}; PLAN.md §5)")
                for i, (q, qse) in enumerate((("S", "S_se"), ("E", "E_se"))):
                    tx, ty = _draw_curve(axes[i, j], a, d[q], d[qse], d["n_pooled"], P.READOUT_COLOR[ro], P.READOUT_STYLE[ro])
                    edges[i][0].extend(tx); edges[i][1].extend(ty)
                if _thin_mask(d["n_pooled"]).any():
                    notes.append(f"{P.READOUT_LABEL[ro]} N = {P.fmt(N)}: {int(d['n_pooled'][-1])} at age {P.fmt(a[-1])}")
                data[f"{ro}_N{N}_S"], data[f"{ro}_N{N}_E"], data[f"{ro}_N{N}_ages"] = d["S"], d["E"], d["ages"]
            for i in range(2):
                P.age_axis(axes[i, j], max_age)
                P.edge_markers(axes[i, j], *edges[i])
            axes[1, j].set_xlabel("age m (tasks)")
        P.label_cols_N(axes, Ns)
        axes[0, 0].set_ylabel("S(m)"); axes[1, 0].set_ylabel("E(m)")
        P.fig_legend(fig, {k[2] for k in members})
        P.cell_title(fig, kappa, d0, lam, solver, eta)
        cap = "dashed: ungated; band: ±1 s.e.m. over seeds; thin: ages with < 25 % of the trace's maximum count"
        if notes:
            cap += " (count at the last age: " + "; ".join(notes[:2]) + (", …" if len(notes) > 2 else "") + ")"
        P.caption(fig, cap)
        stem = P.cell_stem(kappa, d0, lam, solver, eta)
        P.save(fig, os.path.join(fig_dir, "curves", "curves_" + stem), data if figdata else None, index, "curves", stem)


def fig_curvesN(conds, fig_dir, index, figdata):
    """F4: S(m), E(m) vs age; rows S, E; columns = readouts; traces = N.  ``curvesN_lt_`` plots S/S(0) and E against m κ_s²/N."""
    kmin = kappa_min([d["records"][0] for d in conds.values()])
    for (kappa, cell), members in _cells(conds, ("delta0", "lam", "solver", "eta")).items():
        d0, lam, solver, eta = cell
        Ns = sorted({k[0] for k in members})
        ros = [ro for ro in P.READOUT_COLOR if any(k[2] == ro for k in members)]
        for lt in (False, True):
            fig, axes = P.grid(2, len(ros), sharex="all", sharey="row")
            data, max_x, min_pos, notes = {}, 1.0, np.inf, []
            for j, ro in enumerate(ros):
                edges = {0: ([], []), 1: ([], [])}
                for i_N, N in enumerate(Ns):
                    key = next((k for k in members if k[0] == N and k[2] == ro), None)
                    if key is None:
                        continue
                    d = members[key]
                    rec = d["records"][0]
                    scale = rec.kappa_sched(kmin) ** 2 / N if lt else 1.0
                    x = d["ages"].astype(float) * scale
                    max_x = max(max_x, float(x.max()))
                    if lt and len(x) > 1:
                        min_pos = min(min_pos, float(x[1]))
                    col = P.n_shade(i_N, len(Ns))
                    with np.errstate(invalid="ignore", divide="ignore"):
                        S = d["S"] / d["S"][0] if lt else d["S"]
                        S_se = d["S_se"] / abs(d["S"][0]) if lt else d["S_se"]
                    for i, (y, se) in enumerate(((S, S_se), (d["E"], d["E_se"]))):
                        tx, ty = _draw_curve(axes[i, j], x, y, se, d["n_pooled"], col, P.READOUT_STYLE[ro])
                        edges[i][0].extend(tx); edges[i][1].extend(ty)
                    if _thin_mask(d["n_pooled"]).any():
                        notes.append(f"{P.READOUT_LABEL[ro]} N = {P.fmt(N)}: {int(d['n_pooled'][-1])}")
                    data[f"{ro}_N{N}_S"], data[f"{ro}_N{N}_E"], data[f"{ro}_N{N}_x"] = S, d["E"], x
                axes[0, j].set_title(P.READOUT_LABEL[ro], color=P.READOUT_COLOR[ro])
                axes[1, j].set_xlabel(r"$m\kappa_s^2/N$ (lifetimes)" if lt else "age m (tasks)")
                for i in range(2):
                    P.edge_markers(axes[i, j], *edges[i])
            for ax in axes.ravel():
                if lt:
                    P.lifetime_axis(ax, max_x, min_pos if np.isfinite(min_pos) else 1e-3)
                else:
                    P.age_axis(ax, max_x)
            if lt:
                P.ref_line(axes[0, 0], 1.0)
            axes[0, 0].set_ylabel("S(m) / S(0)" if lt else "S(m)"); axes[1, 0].set_ylabel("E(m)")
            P.fig_legend(fig, None, handles=P.n_handles(Ns))
            P.cell_title(fig, kappa, d0, lam, solver, eta, "traces = N" + (", x in lifetimes" if lt else ""))
            cap = "band: ±1 s.e.m. over seeds; dashed columns: ungated; thin: ages with < 25 % of the trace's maximum count"
            if lt:
                cap += "; collapse across N = the lifetime scales as N/κ_s² (check 4)"
            if notes:
                cap += " (count at the last age: " + "; ".join(notes[:2]) + (", …" if len(notes) > 2 else "") + ")"
            P.caption(fig, cap)
            stem = P.cell_stem(kappa, d0, lam, solver, eta)
            P.save(fig, os.path.join(fig_dir, "curves", ("curvesN_lt_" if lt else "curvesN_") + stem), data if figdata else None, index, "curves", stem)


def fig_mhalf_vs_N(conds, fig_dir, index, figdata):
    """m_half vs N (log–log): one figure per (κ, δ₀, solver, η); panels per λ; traces = readouts."""
    for (kappa, cell), members in _cells(conds, ("delta0", "solver", "eta")).items():
        d0, solver, eta = cell
        lams = sorted({k[4] for k in members})
        fig, axes = P.grid(1, len(lams), sharey=True)
        data = {}
        for j, lam in enumerate(lams):
            ax = axes[0, j]
            for ro in P.READOUT_COLOR:
                pts = sorted((k[0], d["m_half"], d["censored"], d["m_half_se"]) for k, d in members.items() if k[2] == ro and k[4] == lam)
                if not pts:
                    continue
                N, m, c, mse = (np.array(z, dtype=float) for z in zip(*pts))
                P.plot_readout(ax, N, m, ro, marker="o")
                ax.errorbar(N, m, yerr=np.nan_to_num(mse), fmt="none", ecolor=P.GREY_REF, elinewidth=P.LW_REF, capsize=2)
                if c.any():
                    top = np.nanmax(m) if np.isfinite(m).any() else 1.0
                    ax.plot(N[c > 0], [top] * int(c.sum()), "v", color=P.READOUT_COLOR[ro], ms=5)
                data[f"{ro}_lam{lam}"] = np.stack([N, m, mse])
            P.logaxis(ax, "x"); P.logaxis(ax, "y"); ax.set_title(rf"$\lambda$ = {P.fmt(lam)}"); ax.set_xlabel("N")
            P.discrete_ticks(ax, sorted({k[0] for k in members}), "x")
        axes[0, 0].set_ylabel(r"$m_{1/2}$ (tasks)")
        P.fig_legend(fig, {k[2] for k in members})
        P.cell_title(fig, kappa, d0, None, solver, eta)
        P.caption(fig, "▼ censored; dashed: ungated; bars: s.e.m. over seeds")
        stem = P.cell_stem(kappa, d0, None, solver, eta)
        P.save(fig, os.path.join(fig_dir, "diag", "mhalf", "mhalf_vs_N_" + stem), data if figdata else None, index, "diag/mhalf", stem)


def fig_mhalf_vs_E0(conds, fig_dir, index, figdata):
    """m_half against E(0): readouts compared at matched learning per task; one figure per (κ, solver, η), panels per N."""
    for (kappa, cell), members in _cells(conds, ("solver", "eta")).items():
        solver, eta = cell
        Ns = sorted({k[0] for k in members})
        fig, axes = P.grid(1, len(Ns), sharey=True)
        data = {}
        for j, N in enumerate(Ns):
            ax = axes[0, j]
            for ro in P.READOUT_COLOR:
                pts = [(d["E0"], d["m_half"], d["E0_se"], d["m_half_se"]) for k, d in members.items()
                       if k[0] == N and k[2] == ro and not d["censored"]]
                if not pts:
                    continue
                x, y, xe, ye = (np.array(z, dtype=float) for z in zip(*pts))
                ax.errorbar(x, y, xerr=np.nan_to_num(xe), yerr=np.nan_to_num(ye), fmt="o" if is_gated(ro) else "s", ms=4.5,
                            color=P.READOUT_COLOR[ro], ecolor=P.GREY_REF, elinewidth=P.LW_REF, capsize=2, lw=0)
                data[f"{ro}_N{N}"] = np.stack([x, y, xe, ye])
            ax.set_xlabel("E(0)"); P.logaxis(ax, "y")
        P.label_cols_N(axes, Ns)
        axes[0, 0].set_ylabel(r"$m_{1/2}$ (tasks)")
        P.fig_legend(fig, {k[2] for k in members}, marker=True)
        P.cell_title(fig, kappa, None, None, solver, eta)
        P.caption(fig, "all δ₀, λ; squares: ungated; bars: s.e.m. over seeds")
        stem = P.cell_stem(kappa, None, None, solver, eta)
        P.save(fig, os.path.join(fig_dir, "diag", "mhalf", "mhalf_vs_E0_" + stem), data if figdata else None, index, "diag/mhalf", stem)


def fig_kaczmarz(conds, fig_dir, index, figdata):
    """Check 4: frozen readouts, S(m)/S(0) vs κ²m/N should collapse across N (m/N for ungated frozen)."""
    for ro in ("gated_frozen", "frozen"):
        members = {k: d for k, d in conds.items() if k[2] == ro}
        for cell, sub in group_by([d["records"][0] for d in members.values()], ("kappa_label", "lam", "solver", "eta")).items():
            kappa, lam, solver, eta = cell
            keys = [r.key() for r in sub]
            Ns = sorted({k[0] for k in keys})
            fig, axes = P.grid(1, 1)
            ax = axes[0, 0]
            data = {}
            for i, N in enumerate(Ns):
                first = True
                for k in keys:
                    if k[0] != N:
                        continue
                    d = conds[k]
                    x = (d["run"].kappa ** 2 if is_gated(ro) else 1.0) * d["ages"] / N
                    norm = d["S_seeds"] / d["S_seeds"][:, :1]
                    y, se = norm.mean(0), sem(norm)
                    col = P.n_shade(i, len(Ns))
                    ax.plot(x[1:], y[1:], color=col)
                    P.band(ax, x[1:], (y - se)[1:], (y + se)[1:], col)
                    data[f"N{N}_d{k[3]}"] = np.stack([x, y])
                    first = False
            ax.set_xlabel(r"$\kappa^2 m / N$" if is_gated(ro) else r"$m / N$"); ax.set_ylabel("S(m) / S(0)"); P.logaxis(ax, "x")
            P.fig_legend(fig, None, handles=P.n_handles(Ns))
            P.cell_title(fig, kappa if is_gated(ro) else None, None, lam, solver, eta, P.READOUT_LABEL[ro])
            P.caption(fig, "all δ₀ pooled; band: ±1 s.e.m. over seeds")
            stem = P.cell_stem(kappa if is_gated(ro) else None, None, lam, solver, eta, readout=ro)
            P.save(fig, os.path.join(fig_dir, "diag", "kaczmarz", "kaczmarz_" + stem), data if figdata else None, index, "diag/kaczmarz", stem)


def report_cv(conds, age_index=1):
    """Check 3: CV of s_k at a fixed age vs N; exponent per (readout, κ, δ₀, λ, solver, η).

    Cells whose pooled mean s at that age is within 3 s.e. of zero are excluded
    from the fit (CV = sd/|mean| is undefined as the mean → 0, §12.7.3) and
    listed under ``excluded``.

    Returns
    -------
    list of dict
        ``readout, kappa, delta0, lam, solver, eta, gamma, Ns, cv, excluded`` (``Ns``/``cv`` of the fitted widths).
    """
    out = []
    for gk, sub in group_by([d["records"][0] for d in conds.values()], ("readout", "kappa_label", "delta0", "lam", "solver", "eta")).items():
        pts, excl = [], []
        for r in sorted(sub, key=lambda r: r.run.N):
            d = conds[r.key()]
            if len(r.ages) <= age_index:
                continue
            S, cv, n = d["S"][age_index], d["cv_pooled"][age_index], d["n_pooled"][age_index]
            se = cv * abs(S) / np.sqrt(n) if n > 0 else np.inf
            (excl if abs(S) <= 3 * se or not np.isfinite(cv) else pts).append((r.run.N, cv))
        row = dict(zip(("readout", "kappa", "delta0", "lam", "solver", "eta"), gk), excluded=excl)
        if len(pts) >= 2:
            Ns, cvs = zip(*pts)
            row.update(gamma=cv_exponent(Ns, cvs)[0], Ns=Ns, cv=cvs)
        else:
            row.update(gamma=np.nan, Ns=tuple(p[0] for p in pts), cv=tuple(p[1] for p in pts))
        out.append(row)
    return out


def fig_cv(cv_rows, fig_dir, index, figdata):
    """Check 3 figure (§12.7.3): one panel per readout, log–log; thin line per (κ, δ₀, λ) cell, geometric mean bold, N^{−1/2} reference."""
    for (solver, eta), rows in group_by_rows(cv_rows, ("solver", "eta")).items():
        ros = [ro for ro in P.READOUT_COLOR if any(r["readout"] == ro for r in rows)]
        Ns_all = sorted({N for r in rows for N in r["Ns"]} | {N for r in rows for N, _ in r["excluded"]})
        fig, axes = P.grid(1, len(ros), sharex=True, sharey="row")
        data, two_point = {}, len(Ns_all) == 2
        for j, ro in enumerate(ros):
            ax = axes[0, j]
            sub = [r for r in rows if r["readout"] == ro]
            per_N, slopes, n_excl = {}, [], 0
            for r in sub:
                if len(r["Ns"]) >= 2:
                    P.plot_readout(ax, r["Ns"], r["cv"], ro, lw=P.LW_OVERLAY, alpha=0.5, zorder=2)
                    slopes.append(r["gamma"])
                    for N, c in zip(r["Ns"], r["cv"]):
                        per_N.setdefault(N, []).append(c)
                for N, c in r["excluded"]:
                    if np.isfinite(c):
                        ax.plot([N], [c], "o", mfc="white", mec=P.READOUT_COLOR[ro], ms=4.5, zorder=3)
                    n_excl += 1
                data[f"{ro}_k{r['kappa']}_d{r['delta0']}_lam{r['lam']}"] = np.stack([r["Ns"], r["cv"]]) if r["Ns"] else np.zeros((2, 0))
            Ns = sorted(per_N)
            if len(Ns) >= 2:
                gm = np.array([np.exp(np.mean(np.log(per_N[N]))) for N in Ns])
                P.plot_readout(ax, Ns, gm, ro, lw=P.LW_MEAN, zorder=5)
                ax.plot(Ns, gm[0] * (np.array(Ns, float) / Ns[0]) ** -0.5, color=P.GREY_REF, lw=P.LW_REF, zorder=1)
                data[f"{ro}_geomean"] = np.stack([Ns, gm])
            sl = np.array(slopes, float); sl = sl[np.isfinite(sl)]
            title = P.READOUT_LABEL[ro] + (f"\nslope {sl.mean():+.2f} ± {sl.std():.2f}" if len(sl) else "\nslope n/a")
            ax.set_title(title, color=P.READOUT_COLOR[ro])
            if n_excl:
                ax.text(0.03, 0.03, f"{n_excl} cell{'s' if n_excl > 1 else ''} excluded: mean ≈ 0", transform=ax.transAxes, fontsize=P.FS_MIN, color=P.GREY_CAPTION)
            P.logaxis(ax, "x"); P.logaxis(ax, "y"); ax.set_xlabel("N")
            P.discrete_ticks(ax, Ns_all, "x")
            ax._mup_no_const_check = True
        axes[0, 0].set_ylabel("CV of s at m = 1")
        P.fig_legend(fig, ros)
        P.cell_title(fig, None, None, None, solver, eta, "check 3: self-averaging")
        P.caption(fig, "thin: one (κ, δ₀, λ) cell; bold: geometric mean over cells; grey: ∝ N^(−1/2) through the mean's first point; "
                       "hollow: cells with mean s within 3 s.e. of 0 (excluded from the slope)"
                  + ("; slopes are two-point slopes" if two_point else ""))
        stem = P.cell_stem(None, None, None, solver, eta)
        P.save(fig, os.path.join(fig_dir, "diag", "check2", "cv_exponent_" + stem), data if figdata else None, index, "diag/check2", stem)


def group_by_rows(rows, fields):
    """
    Group plain dicts by the given fields.

    Parameters
    ----------
    rows : list of dict
    fields : tuple of str

    Returns
    -------
    dict
        ``{tuple: [rows]}``.
    """
    out = {}
    for r in rows:
        out.setdefault(tuple(r[f] for f in fields), []).append(r)
    return out


DIAG_FAMILIES = ("mhalf", "kaczmarz", "by_step", "check2")


def parse_cells(spec):
    """``"d0=0,lam=1,N=10"`` → ``{"delta0": {0.0}, "lam": {1.0}, "N": {10}}`` (values comma- or slash-separated per key).

    Parameters
    ----------
    spec : str or None

    Returns
    -------
    dict or None
    """
    if not spec:
        return None
    alias = {"d0": "delta0", "delta0": "delta0", "lam": "lam", "N": "N", "k": "kappa_label", "kappa": "kappa_label",
             "readout": "readout", "eta": "eta", "solver": "solver"}
    out = {}
    for part in spec.split(","):
        k, v = part.split("=")
        key = alias[k.strip()]
        vals = {x.strip() for x in v.split("/")}
        out[key] = {int(x) for x in vals} if key == "N" else (vals if key in ("kappa_label", "readout", "solver") else {float(x) for x in vals})
    return out


def select_cells(records, cells):
    """Records whose run matches every field of ``cells`` (``parse_cells`` output); all records when ``cells`` is None."""
    if not cells:
        return list(records)
    return [r for r in records if all(getattr(r.run, k) in v for k, v in cells.items())]


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
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--save-figdata", action="store_true", help="write the plotted arrays next to each figure as .npz")
    ap.add_argument("--cells", default=None, help="restrict F1–F4 to these cells, e.g. d0=0,lam=1 (values may be a/b lists)")
    ap.add_argument("--all", action="store_true", help="also draw every diagnostic family (diag/)")
    ap.add_argument("--diag", nargs="*", default=None, choices=DIAG_FAMILIES, help="draw these diagnostic families (diag/)")
    ap.add_argument("--levels", action="store_true", help="with the check2 diagnostics, also the absolute half-window levels")
    args = ap.parse_args(argv)
    diag = set(DIAG_FAMILIES) if args.all else set(args.diag or [])

    records, skipped = load_runs(args.out_dir)
    if not records:
        print("no batch files found"); return 1
    if skipped:
        print(f"{len(skipped)} runs were skipped as infeasible (δ₀ ≤ −min_i Q0_i), e.g. {skipped[0]}")
    conds = collect(records)
    rows = write_summary(conds, args.out_dir)
    print(f"{len(rows)} conditions from {len(records)} runs")
    proj = [r for r in rows if "frozen" not in r["readout"]]
    if proj:
        print(f"check 1  max gauge residuals (projected readouts): Δ {max(r['gauge_delta_max'] for r in proj):.2e}, "
              f"u_c {max(r['gauge_uc_max'] for r in proj):.2e}")
    print("m_half (tasks), NaN = censored; Δm½ = half 1 − half 0:")
    for r in rows:
        print(f"  N={r['N']:4d} κ={r['kappa_label']:5s} {r['readout']:14s} δ₀={P.fmt(r['delta0']):>5s} λ={P.fmt(r['lam']):>4s}  "
              f"E0={r['E0']:.3f}  S0={r['S0']:.3f}  m½={r['m_half']:7.2f}  Δm½={r['dm_half']:+6.2f}  λ_real={r['lam_real']:.2f}")
    if args.no_figures:
        return 0
    fig_dir = args.fig_dir or os.path.join(args.out_dir, "figures")
    index = []
    import plot_check2
    import plot_norms
    # the checks table (and check2 diagnostics) from every record
    index += plot_check2.run(args.out_dir, fig_dir, records, args.save_figdata, args.levels, "check2" in diag, skipped)
    # F1–F4 from the selected cells
    sel = select_cells(records, parse_cells(args.cells))
    if not sel:
        print(f"--cells {args.cells!r} selects no runs"); return 1
    fconds = collect(sel) if sel is not records else conds
    index += plot_norms.run(args.out_dir, fig_dir, sel, args.save_figdata, "by_step" in diag)
    fig_curves(fconds, fig_dir, index, args.save_figdata)
    fig_curvesN(fconds, fig_dir, index, args.save_figdata)
    if "mhalf" in diag:
        fig_mhalf_vs_N(fconds, fig_dir, index, args.save_figdata)
        fig_mhalf_vs_E0(fconds, fig_dir, index, args.save_figdata)
    if "kaczmarz" in diag:
        fig_kaczmarz(fconds, fig_dir, index, args.save_figdata)
    if "check2" in diag:
        fig_cv(report_cv(fconds), fig_dir, index, args.save_figdata)
    P.write_index(fig_dir, index)
    print(f"{len(index)} figures → {fig_dir} (INDEX.md)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
