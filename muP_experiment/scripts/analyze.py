#!/usr/bin/env python
"""Merge batch files of a sweep: S(m), E(m), m_half vs N at matched E(0), checks, figures.

    python scripts/analyze.py results/pilot [--fig-dir results/pilot/figures]

Never reads raw state: only the running moments and per-task scalars.
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np

import _path  # noqa: F401

from mup import io
from mup.observe import Moments, check_bounded, check_gauge, curves, cv_exponent, half_life
from mup.config import is_gated
from mup.plotting import READOUT_COLOR, READOUT_LABEL, READOUT_STYLE, n_shade, plt

COND_KEYS = ("N", "kappa_label", "readout", "delta0", "lam", "solver", "eta")


def collect(out_dir):
    """→ dict condition → {ages, sums (pooled over seeds and halves), per-seed curves, scalars}."""
    conds = {}
    skipped = []
    for res, runs, meta in io.load_sweep(out_dir):
        skipped += meta.get("skipped", [])
        if not runs:
            continue
        mom = {f: res[f] for f in Moments.FIELDS}
        c = curves(mom)
        T_burn, T_meas = int(res["T_burn"]), int(res["T_meas"])
        gauge = check_gauge(res)
        bounded = check_bounded(res, T_burn, T_meas)
        for i, r in enumerate(runs):
            key = tuple(getattr(r, k) for k in COND_KEYS)
            d = conds.setdefault(key, {
                "ages": res["ages"], "sum": np.zeros((5, len(res["ages"]))), "S_seeds": [], "E_seeds": [],
                "E0_seeds": [], "cv_s": [], "gauge_delta": [], "gauge_uc": [], "flags": defaultdict(int),
                "abar2": [], "lam_real": [], "n_seeds": 0, "run": r,
            })
            d["sum"] += mom_sum(mom, i)
            d["S_seeds"].append(c["S"][i]); d["E_seeds"].append(c["E"][i]); d["E0_seeds"].append(c["E"][i][0])
            d["cv_s"].append(c["cv_s"][i])
            d["gauge_delta"].append(gauge["max_delta_resid"][i]); d["gauge_uc"].append(gauge["max_uc_resid"][i])
            for name, f in bounded.items():
                d["flags"][name] += int(bool(f["flag"][i]))
            ab = float(res["abar2"][0, i])
            d["abar2"].append(ab)
            d["lam_real"].append(r.eta * r.alpha_task * ab)
            d["n_seeds"] += 1
    for d in conds.values():
        n = d["sum"][4]
        with np.errstate(invalid="ignore", divide="ignore"):
            d["S"] = d["sum"][0] / n
            d["E"] = d["sum"][2] / n
            d["cv_pooled"] = np.sqrt(np.maximum(d["sum"][1] / n - d["S"] ** 2, 0)) / np.abs(d["S"])
        d["S_se"] = np.std(d["S_seeds"], axis=0, ddof=1) / np.sqrt(d["n_seeds"]) if d["n_seeds"] > 1 else np.full_like(d["S"], np.nan)
        d["m_half"], d["censored"] = half_life(d["ages"], d["S"])
        d["E0"] = float(np.mean(d["E0_seeds"]))
    if skipped:
        print(f"{len(skipped)} runs were skipped as infeasible (δ₀ ≤ −min_i Q0_i), e.g. {skipped[0]}")
    return conds


def mom_sum(mom, i):
    return np.stack([mom[f][i].sum(-1) for f in Moments.FIELDS])


def write_summary(conds, out_dir):
    rows = []
    for key, d in sorted(conds.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        row = dict(zip(COND_KEYS, key))
        row.update(
            n_seeds=d["n_seeds"], S0=d["S"][0], E0=d["E0"], m_half=d["m_half"], censored=d["censored"],
            lam_real=float(np.mean(d["lam_real"])), abar2=float(np.mean(d["abar2"])),
            gauge_delta_max=float(np.max(d["gauge_delta"])), gauge_uc_max=float(np.max(d["gauge_uc"])),
            cv_s_age1=float(d["cv_pooled"][1]) if len(d["ages"]) > 1 else np.nan,
            **{f"flag_{k}": v for k, v in sorted(d["flags"].items())},
        )
        rows.append(row)
    path = os.path.join(out_dir, "summary.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with open(os.path.join(out_dir, "curves.json"), "w") as f:
        json.dump([{**dict(zip(COND_KEYS, k)), "ages": d["ages"].tolist(), "S": d["S"].tolist(), "E": d["E"].tolist(),
                    "S_se": np.nan_to_num(d["S_se"], nan=-1).tolist()} for k, d in conds.items()], f)
    return rows


def group(conds, by):
    g = defaultdict(dict)
    for key, d in conds.items():
        kd = dict(zip(COND_KEYS, key))
        g[tuple(kd[b] for b in by)][key] = d
    return g


def by_kappa(conds):
    """{κ_label: members} for the gated conditions, with the ungated ones (no κ) repeated in every group."""
    ungated = {k: d for k, d in conds.items() if not is_gated(k[2])}
    out = defaultdict(dict)
    for k, d in conds.items():
        if is_gated(k[2]):
            out[k[1]][k] = d
    if not out:
        out["sqrtN"] = {}
    for members in out.values():
        members.update(ungated)
    return out


def _plot(ax, x, y, ro, **kw):
    ax.plot(x, y, READOUT_STYLE[ro], color=READOUT_COLOR[ro], label=READOUT_LABEL[ro], **kw)


def fig_curves(conds, fig_dir):
    """S(m), E(m): one figure per (κ, δ₀, λ, solver, η); columns = N; traces = readouts (PLAN.md §5)."""
    for kappa, kmembers in by_kappa(conds).items():
        for gk, members in group(kmembers, ("delta0", "lam", "solver", "eta")).items():
            d0, lam, solver, eta = gk
            Ns = sorted({k[0] for k in members})
            fig, axes = plt.subplots(2, len(Ns), figsize=(3.0 * len(Ns), 5.2), squeeze=False, sharex="col")
            for j, N in enumerate(Ns):
                for ro in READOUT_COLOR:
                    key = next((k for k in members if k[0] == N and k[2] == ro), None)
                    if key is None:
                        continue
                    d = members[key]
                    a = d["ages"]
                    _plot(axes[0, j], a[1:], d["S"][1:], ro)
                    _plot(axes[1, j], a[1:], d["E"][1:], ro)
                axes[0, j].set_title(f"N={N}"); axes[0, j].set_xscale("log"); axes[1, j].set_xscale("log")
                axes[1, j].set_xlabel("age m (tasks)")
            axes[0, 0].set_ylabel("S(m)"); axes[1, 0].set_ylabel("E(m)")
            axes[0, -1].legend(fontsize=7)
            fig.suptitle(f"κ={kappa}  δ₀={d0}  λ={lam}  {solver} η={eta}   (dashed = ungated)", fontsize=9)
            fig.tight_layout()
            fig.savefig(os.path.join(fig_dir, f"curves_k{kappa}_d{d0}_lam{lam}_{solver}_eta{eta}.png"))
            plt.close(fig)


def fig_mhalf_vs_N(conds, fig_dir):
    """m_half vs N (log–log): one figure per (κ, δ₀, solver, η); panels per λ; traces = readouts."""
    for kappa, kmembers in by_kappa(conds).items():
        for gk, members in group(kmembers, ("delta0", "solver", "eta")).items():
            d0, solver, eta = gk
            lams = sorted({k[4] for k in members})
            fig, axes = plt.subplots(1, len(lams), figsize=(3.2 * len(lams), 3.0), squeeze=False, sharey=True)
            for j, lam in enumerate(lams):
                ax = axes[0, j]
                for ro in READOUT_COLOR:
                    pts = sorted((k[0], d["m_half"], d["censored"]) for k, d in members.items() if k[2] == ro and k[4] == lam)
                    if not pts:
                        continue
                    N, m, c = zip(*pts)
                    _plot(ax, N, m, ro, marker="o", ms=4)
                    cens = [n for n, mm, cc in pts if cc]
                    if cens:
                        ax.plot(cens, [np.nanmax([mm for _, mm, _ in pts if np.isfinite(mm)] or [1.0])] * len(cens), "v",
                                color=READOUT_COLOR[ro], ms=5)
                ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(f"λ={lam}"); ax.set_xlabel("N")
            axes[0, 0].set_ylabel("m½ (tasks)"); axes[0, -1].legend(fontsize=7)
            fig.suptitle(f"κ={kappa}  δ₀={d0}  {solver} η={eta}   (▼ censored, dashed = ungated)", fontsize=9)
            fig.tight_layout(); fig.savefig(os.path.join(fig_dir, f"mhalf_vs_N_k{kappa}_d{d0}_{solver}_eta{eta}.png")); plt.close(fig)


def fig_mhalf_vs_E0(conds, fig_dir):
    """m_half against E(0): readouts compared at matched learning per task; one figure per (κ, solver, η), panels per N."""
    for kappa, kmembers in by_kappa(conds).items():
        for gk, members in group(kmembers, ("solver", "eta")).items():
            solver, eta = gk
            Ns = sorted({k[0] for k in members})
            fig, axes = plt.subplots(1, len(Ns), figsize=(3.0 * len(Ns), 3.0), squeeze=False, sharey=True)
            for j, N in enumerate(Ns):
                ax = axes[0, j]
                for ro in READOUT_COLOR:
                    pts = [(d["E0"], d["m_half"]) for k, d in members.items() if k[0] == N and k[2] == ro and not d["censored"]]
                    if pts:
                        x, y = zip(*pts)
                        ax.plot(x, y, "o" if is_gated(ro) else "s", ms=4, color=READOUT_COLOR[ro], label=READOUT_LABEL[ro], alpha=0.85)
                ax.set_title(f"N={N}"); ax.set_xlabel("E(0)"); ax.set_yscale("log")
            axes[0, 0].set_ylabel("m½ (tasks)"); axes[0, -1].legend(fontsize=7)
            fig.suptitle(f"κ={kappa}  {solver} η={eta}  (all δ₀, λ; squares = ungated)", fontsize=9)
            fig.tight_layout(); fig.savefig(os.path.join(fig_dir, f"mhalf_vs_E0_k{kappa}_{solver}_eta{eta}.png")); plt.close(fig)


def fig_kaczmarz(conds, fig_dir):
    """Check 4: frozen readouts, S(m)/S(0) vs κ²m/N should collapse across N (m/N for ungated frozen)."""
    for ro in ("gated_frozen", "frozen"):
        members = {k: d for k, d in conds.items() if k[2] == ro}
        for gk, sub in group(members, ("kappa_label", "lam", "solver", "eta")).items():
            kappa, lam, solver, eta = gk
            Ns = sorted({k[0] for k in sub})
            fig, ax = plt.subplots(figsize=(4.0, 3.0))
            for i, N in enumerate(Ns):
                first = True
                for k, d in sub.items():
                    if k[0] != N:
                        continue
                    x = d["run"].kappa ** 2 * d["ages"] / N
                    ax.plot(x, d["S"] / d["S"][0], color=n_shade(i, len(Ns)), alpha=0.8, label=f"N={N}" if first else None)
                    first = False
            ax.set_xlabel("κ² m / N" if is_gated(ro) else "m / N"); ax.set_ylabel("S(m) / S(0)"); ax.set_xscale("log")
            ax.set_title(f"{READOUT_LABEL[ro]}, κ={kappa}, λ={lam} (all δ₀)", color=READOUT_COLOR[ro]); ax.legend(fontsize=7)
            fig.tight_layout(); fig.savefig(os.path.join(fig_dir, f"kaczmarz_{ro}_k{kappa}_lam{lam}_{solver}_eta{eta}.png")); plt.close(fig)


def report_cv(conds, age_index=1):
    """Check 3: CV of s_k at a fixed age vs N; exponent per (readout, κ, δ₀, λ)."""
    out = []
    for gk, members in group(conds, ("readout", "kappa_label", "delta0", "lam", "solver", "eta")).items():
        pts = sorted((k[0], d["cv_pooled"][age_index]) for k, d in members.items() if len(d["ages"]) > age_index)
        if len(pts) >= 2:
            Ns, cvs = zip(*pts)
            g, _ = cv_exponent(Ns, cvs)
            out.append({"readout": gk[0], "kappa": gk[1], "delta0": gk[2], "lam": gk[3], "gamma": g, "Ns": Ns, "cv": cvs})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--fig-dir", default=None)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)
    conds = collect(args.out_dir)
    if not conds:
        print("no batch files found"); return 1
    rows = write_summary(conds, args.out_dir)
    print(f"{len(rows)} conditions from {sum(d['n_seeds'] for d in conds.values())} runs")
    proj = [r for r in rows if "frozen" not in r["readout"]]
    if proj:
        print(f"check 1  max gauge residuals (projected readouts): Δ {max(r['gauge_delta_max'] for r in proj):.2e}, "
              f"u_c {max(r['gauge_uc_max'] for r in proj):.2e}")
    flagged = [r for r in rows if any(v for k, v in r.items() if k.startswith('flag_'))]
    print(f"check 2  conditions with a half-window flag in ≥1 seed: {len(flagged)}/{len(rows)}")
    for cvr in report_cv(conds):
        print(f"check 3  CV(s, m=1) ∝ N^γ  γ={cvr['gamma']:+.2f}  {cvr['readout']:14s} κ={cvr['kappa']} δ₀={cvr['delta0']} λ={cvr['lam']}")
    print("m_half (tasks), NaN = censored:")
    for r in rows:
        print(f"  N={r['N']:4d} κ={r['kappa_label']:5s} {r['readout']:14s} δ₀={r['delta0']:6.2f} λ={r['lam']:4.1f}  "
              f"E0={r['E0']:.3f}  S0={r['S0']:.3f}  m½={r['m_half']:7.2f}  λ_real={r['lam_real']:.2f}")
    if not args.no_figures:
        fig_dir = args.fig_dir or os.path.join(args.out_dir, "figures")
        os.makedirs(fig_dir, exist_ok=True)
        fig_curves(conds, fig_dir); fig_mhalf_vs_N(conds, fig_dir); fig_mhalf_vs_E0(conds, fig_dir); fig_kaczmarz(conds, fig_dir)
        print(f"figures → {fig_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
