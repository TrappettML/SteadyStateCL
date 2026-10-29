#!/usr/bin/env python
"""Merge the batch files of a sweep: S(m), E(m), m_half vs N at matched E(0), checks 1–4, figures.

    python scripts/analyze.py results/pilot [--fig-dir DIR] [--no-figures] [--no-norms] [--no-check2] [--save-figdata]

Writes ``summary.csv`` (one row per condition), ``curves.json`` and
``figures/{curves,mhalf,kaczmarz,checks}/`` (PLAN.md §12.4), then calls
``plot_norms.py`` and ``plot_check2.py`` for ``figures/norms/`` and
``figures/check2/``.  Only the running moments and per-task scalars are
read; never raw state.
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
from mup.results import COND_KEYS, by_kappa, condition_mhalf, group_by, half_stats, load_runs, sem

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
            "ages": r0.ages, "S": S, "E": E, "cv_pooled": cv, "S_seeds": S_seeds, "E_seeds": E_seeds,
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


def fig_curves(conds, fig_dir, index, figdata):
    """S(m), E(m): one figure per (κ, δ₀, λ, solver, η); rows S, E; columns N; traces = readouts ± s.e.m. (PLAN.md §5)."""
    for (kappa, cell), members in _cells(conds, ("delta0", "lam", "solver", "eta")).items():
        d0, lam, solver, eta = cell
        Ns = sorted({k[0] for k in members})
        fig, axes = P.grid(2, len(Ns), sharex="col", sharey="row")
        data = {}
        for j, N in enumerate(Ns):
            for ro in P.READOUT_COLOR:
                key = next((k for k in members if k[0] == N and k[2] == ro), None)
                if key is None:
                    continue
                d = members[key]
                a = d["ages"][1:]
                for i, (q, qse) in enumerate((("S", "S_se"), ("E", "E_se"))):
                    P.plot_readout(axes[i, j], a, d[q][1:], ro)
                    P.band(axes[i, j], a, (d[q] - d[qse])[1:], (d[q] + d[qse])[1:], P.READOUT_COLOR[ro])
                data[f"{ro}_N{N}_S"], data[f"{ro}_N{N}_E"], data[f"{ro}_N{N}_ages"] = d["S"], d["E"], d["ages"]
            P.logaxis(axes[1, j], "x"); axes[1, j].set_xlabel("age m (tasks)")
        P.label_cols_N(axes, Ns)
        axes[0, 0].set_ylabel("S(m)"); axes[1, 0].set_ylabel("E(m)")
        P.readout_legend(axes[0, 0], {k[2] for k in members})
        P.cell_title(fig, kappa, d0, lam, solver, eta)
        P.caption(fig, "dashed: ungated; band: ±1 s.e.m. over seeds")
        stem = P.cell_stem(kappa, d0, lam, solver, eta)
        P.save(fig, os.path.join(fig_dir, "curves", "curves_" + stem), data if figdata else None, index, "curves", stem)


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
            P.logaxis(ax, "x"); P.logaxis(ax, "y"); ax.set_title(rf"$\lambda$ = {lam}"); ax.set_xlabel("N")
        axes[0, 0].set_ylabel(r"$m_{1/2}$ (tasks)")
        P.readout_legend(axes[0, 0], {k[2] for k in members})
        P.cell_title(fig, kappa, d0, None, solver, eta)
        P.caption(fig, "▼ censored; dashed: ungated; bars: s.e.m. over seeds")
        stem = P.cell_stem(kappa, d0, None, solver, eta)
        P.save(fig, os.path.join(fig_dir, "mhalf", "mhalf_vs_N_" + stem), data if figdata else None, index, "mhalf", stem)


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
        P.readout_legend(axes[0, 0], {k[2] for k in members})
        P.cell_title(fig, kappa, None, None, solver, eta)
        P.caption(fig, "all δ₀, λ; squares: ungated; bars: s.e.m. over seeds")
        stem = P.cell_stem(kappa, None, None, solver, eta)
        P.save(fig, os.path.join(fig_dir, "mhalf", "mhalf_vs_E0_" + stem), data if figdata else None, index, "mhalf", stem)


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
                    ax.plot(x[1:], y[1:], color=col, label=f"N = {N}" if first else None)
                    P.band(ax, x[1:], (y - se)[1:], (y + se)[1:], col)
                    data[f"N{N}_d{k[3]}"] = np.stack([x, y])
                    first = False
            ax.set_xlabel(r"$\kappa^2 m / N$" if is_gated(ro) else r"$m / N$"); ax.set_ylabel("S(m) / S(0)"); P.logaxis(ax, "x")
            ax.legend()
            P.cell_title(fig, kappa if is_gated(ro) else None, None, lam, solver, eta, P.READOUT_LABEL[ro])
            P.caption(fig, "all δ₀ pooled; band: ±1 s.e.m. over seeds")
            stem = P.cell_stem(kappa if is_gated(ro) else None, None, lam, solver, eta, readout=ro)
            P.save(fig, os.path.join(fig_dir, "kaczmarz", "kaczmarz_" + stem), data if figdata else None, index, "kaczmarz", stem)


def report_cv(conds, age_index=1):
    """Check 3: CV of s_k at a fixed age vs N; exponent per (readout, κ, δ₀, λ, solver, η).

    Returns
    -------
    list of dict
        ``readout, kappa, delta0, lam, solver, eta, gamma, Ns, cv``.
    """
    out = []
    for gk, sub in group_by([d["records"][0] for d in conds.values()], ("readout", "kappa_label", "delta0", "lam", "solver", "eta")).items():
        pts = sorted((r.run.N, conds[r.key()]["cv_pooled"][age_index]) for r in sub if len(r.ages) > age_index)
        if len(pts) >= 2:
            Ns, cvs = zip(*pts)
            g, _ = cv_exponent(Ns, cvs)
            out.append(dict(zip(("readout", "kappa", "delta0", "lam", "solver", "eta"), gk), gamma=g, Ns=Ns, cv=cvs))
    return out


def fig_cv(cv_rows, fig_dir, index, figdata):
    """Check 3 figure: CV(s, m=1) vs N per readout, one figure per (solver, η); points = (κ, δ₀, λ) cells."""
    for (solver, eta), rows in group_by_rows(cv_rows, ("solver", "eta")).items():
        fig, axes = P.grid(1, 1)
        ax = axes[0, 0]
        data = {}
        for row in rows:
            P.plot_readout(ax, row["Ns"], row["cv"], row["readout"], marker="o", alpha=0.7, lw=P.LW_OVERLAY)
            data[f"{row['readout']}_k{row['kappa']}_d{row['delta0']}_lam{row['lam']}"] = np.stack([row["Ns"], row["cv"]])
        gam = {ro: np.mean([r["gamma"] for r in rows if r["readout"] == ro]) for ro in {r["readout"] for r in rows}}
        P.logaxis(ax, "x"); P.logaxis(ax, "y"); ax.set_xlabel("N"); ax.set_ylabel("CV of s at m = 1")
        P.readout_legend(ax, gam.keys())
        P.cell_title(fig, None, None, None, solver, eta, "check 3")
        P.caption(fig, "one line per (κ, δ₀, λ); mean γ: " + ", ".join(f"{P.READOUT_LABEL[r]} {g:+.2f}" for r, g in gam.items()))
        stem = P.cell_stem(None, None, None, solver, eta)
        P.save(fig, os.path.join(fig_dir, "checks", "cv_exponent_" + stem), data if figdata else None, index, "checks", stem)


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
    ap.add_argument("--no-norms", action="store_true", help="skip plot_norms.py")
    ap.add_argument("--no-check2", action="store_true", help="skip plot_check2.py")
    ap.add_argument("--save-figdata", action="store_true", help="write the plotted arrays next to each figure as .npz")
    args = ap.parse_args(argv)

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
    flagged = [r for r in rows if any(v for k, v in r.items() if k.startswith("flag_"))]
    print(f"check 2  conditions with a half-window flag in ≥1 seed: {len(flagged)}/{len(rows)}  (details: figures/check2/check2_cells.csv)")
    cv_rows = report_cv(conds)
    for c in cv_rows:
        print(f"check 3  CV(s, m=1) ∝ N^γ  γ={c['gamma']:+.2f}  {c['readout']:14s} κ={c['kappa']} δ₀={c['delta0']} λ={c['lam']}")
    print("m_half (tasks), NaN = censored; Δm½ = half 1 − half 0:")
    for r in rows:
        print(f"  N={r['N']:4d} κ={r['kappa_label']:5s} {r['readout']:14s} δ₀={r['delta0']:6.2f} λ={r['lam']:4.1f}  "
              f"E0={r['E0']:.3f}  S0={r['S0']:.3f}  m½={r['m_half']:7.2f}  Δm½={r['dm_half']:+6.2f}  λ_real={r['lam_real']:.2f}")
    if args.no_figures:
        return 0
    fig_dir = args.fig_dir or os.path.join(args.out_dir, "figures")
    index = []
    fig_curves(conds, fig_dir, index, args.save_figdata)
    fig_mhalf_vs_N(conds, fig_dir, index, args.save_figdata)
    fig_mhalf_vs_E0(conds, fig_dir, index, args.save_figdata)
    fig_kaczmarz(conds, fig_dir, index, args.save_figdata)
    fig_cv(cv_rows, fig_dir, index, args.save_figdata)
    if not args.no_norms:
        import plot_norms
        index += plot_norms.run(args.out_dir, fig_dir, records, args.save_figdata)
    if not args.no_check2:
        import plot_check2
        index += plot_check2.run(args.out_dir, fig_dir, records, args.save_figdata)
    P.write_index(fig_dir, index)
    print(f"{len(index)} figures → {fig_dir} (INDEX.md)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
