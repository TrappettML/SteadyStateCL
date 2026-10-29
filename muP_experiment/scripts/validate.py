#!/usr/bin/env python
"""Validation runs (PLAN.md §7.7): η-halving, Kaczmarz collapse, CV exponent.

    python scripts/validate.py [--N-max 40] [--seeds 4] [--out results/validate] [--parts eta kaczmarz cv]

Each part runs its own small sweep in-process, prints a verdict and writes
``<out>/figures/{eta_halving,kaczmarz_collapse,cv_exponent}.png`` in
the style of PLAN.md §12.0.  Nothing here is needed by the Stage-1 sweep.
"""
import argparse
import os
import sys

import numpy as np

import _path  # noqa: F401

from mup import plotting as P
from mup.config import Config, make_batches, make_runs
from mup.observe import Moments, curves, cv_exponent
from mup.protocol import run_stream


def pooled(cfg, log):
    """Run a config in-process and pool each (N, readout, η) over seeds.

    Parameters
    ----------
    cfg : Config
    log : callable
        Progress sink.

    Returns
    -------
    dict
        ``{(N, readout, eta): (ages, S, E, cv_s, S_se)}`` with ``S``, ``E`` the
        seed-pooled curves, ``cv_s`` the pooled coefficient of variation of
        s_k and ``S_se`` the s.e.m. of ``S`` across seeds (NaN for one seed).
    """
    out = {}
    for batch in make_batches(make_runs(cfg), cfg.batch_size):
        res = run_stream(batch, log=log)
        mom = {f: res[f] for f in Moments.FIELDS}
        per_seed = curves(mom)["S"]
        for key in {(r.N, r.readout, r.eta) for r in batch}:
            idx = [i for i, r in enumerate(batch) if (r.N, r.readout, r.eta) == key]
            s = {f: mom[f][idx].sum((0, 2)) for f in Moments.FIELDS}
            n = s["count"]
            with np.errstate(invalid="ignore", divide="ignore"):
                S, E = s["sum_s"] / n, s["sum_e"] / n
                cv = np.sqrt(np.maximum(s["sum_s2"] / n - S**2, 0)) / np.abs(S)
            se = per_seed[idx].std(0, ddof=1) / np.sqrt(len(idx)) if len(idx) > 1 else np.full_like(S, np.nan)
            out[key] = (res["ages"], S, E, cv, se)
    return out


def eta_halving(args, fig_dir, log):
    """SGD at η, η/2, η/4 and the population scheme at h ā² = 0.1, 0.01 against a fine-step reference.

    At fixed λ the η² term in dQ/dα acts over α_task ∝ 1/η, so the SGD gap
    should be ∝ η (ratio ≈ 2 per halving); the population scheme's Euler
    error is ∝ h (ratio ≈ 10 per decade in h).  Reference: h ā² = 0.001.

    Parameters
    ----------
    args : argparse.Namespace
        ``seeds``.
    fig_dir : str
    log : callable
    """
    N, lam, d0 = 10, 1.0, 0.0
    readouts = ["gated_shared", "gated_frozen"]
    contractions = [0.1, 0.01]
    ref = pooled(Config(N=[N], readout=readouts, delta0=[d0], lam=[lam], seeds=args.seeds, solver="pop", pop_contraction=0.001), log)
    pops = {c: pooled(Config(N=[N], readout=readouts, delta0=[d0], lam=[lam], seeds=args.seeds, solver="pop", pop_contraction=c), log)
            for c in contractions}
    etas = [0.05, 0.025, 0.0125]
    sgd = pooled(Config(N=[N], readout=readouts, delta0=[d0], lam=[lam], seeds=args.seeds, solver="sgd", eta=etas), log)
    fig, axes = P.grid(1, 2, sharey=True)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color="k", lw=P.LW_MEAN, label="population (fine h)")]
    print("\n[η-halving]  offsets from the fine-step population reference (h ā² = 0.001), same task streams per seed:")
    print("  population Euler error ∝ h (expect ×0.1 per row); SGD gap ∝ η (expect ×0.5 per row)")
    for j, ro in enumerate(readouts):
        ax = axes[0, j]
        ages, S_ref, E_ref, _, se_ref = ref[(N, ro, 1.0)]
        ax.plot(ages[1:], S_ref[1:], color="k", lw=P.LW_MEAN)
        P.band(ax, ages[1:], (S_ref - se_ref)[1:], (S_ref + se_ref)[1:], "k")
        for c in contractions:
            _, S, E, _, _ = pops[c][(N, ro, 1.0)]
            print(f"  {ro:14s} pop h ā²={c:<6}   ΔS(0)={S[0] - S_ref[0]:+.5f}  ΔE(0)={E[0] - E_ref[0]:+.5f}  max_m|ΔS|={np.nanmax(np.abs(S - S_ref)):.5f}")
        gaps = []
        for i, eta in enumerate(etas):
            _, S, E, _, se = sgd[(N, ro, eta)]
            gaps.append(E[0] - E_ref[0])
            print(f"  {ro:14s} sgd η={eta:<8}   ΔS(0)={S[0] - S_ref[0]:+.5f}  ΔE(0)={E[0] - E_ref[0]:+.5f}  max_m|ΔS|={np.nanmax(np.abs(S - S_ref)):.5f}")
            col = P.n_shade(i + 1, len(etas) + 1)
            ax.plot(ages[1:], S[1:], color=col)
            if j == 0:
                handles.append(Line2D([], [], color=col, lw=P.LW_TRACE, label=f"SGD η = {P.fmt(eta)}"))
            P.band(ax, ages[1:], (S - se)[1:], (S + se)[1:], col)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratios = ["%.2f" % (gaps[i] / gaps[i + 1]) for i in range(len(gaps) - 1)]
        print(f"  {ro:14s} ΔE(0) ratios per halving: {ratios}")
        P.logaxis(ax, "x"); ax.set_title(P.READOUT_LABEL[ro], color=P.READOUT_COLOR[ro]); ax.set_xlabel("age m (tasks)")
    axes[0, 0].set_ylabel("S(m)"); P.fig_legend(fig, None, handles=handles)
    P.cell_title(fig, None, d0, lam, None, None, f"η-halving, N = {N}")
    P.caption(fig, "band: ±1 s.e.m. over seeds")
    P.save(fig, os.path.join(fig_dir, "eta_halving"))


def kaczmarz(args, fig_dir, log):
    """Gated frozen readout: S(m)/S(0) vs κ²m/N across N should collapse (check 4).

    Parameters
    ----------
    args : argparse.Namespace
        ``seeds``, ``N_max``.
    fig_dir : str
    log : callable
    """
    Ns = [n for n in (10, 20, 40, 80) if n <= args.N_max]
    res = pooled(Config(N=Ns, readout=["gated_frozen"], delta0=[0.0], lam=[1.0], seeds=args.seeds, solver="pop"), log)
    fig, axes = P.grid(1, 1)
    ax = axes[0, 0]
    print("\n[Kaczmarz]  gated frozen readout, S/S0 at κ²m/N ≈ 0.5 for each N (collapse ⇒ equal):")
    for i, N in enumerate(Ns):
        ages, S, _, _, se = res[(N, "gated_frozen", 1.0)]
        x = ages / N
        col = P.n_shade(i, len(Ns))
        ax.plot(x[1:], (S / S[0])[1:], color=col)
        P.band(ax, x[1:], ((S - se) / S[0])[1:], ((S + se) / S[0])[1:], col)
        print(f"  N={N:3d}  S/S0 = {np.interp(0.5, x, S / S[0]):.3f}")
    P.logaxis(ax, "x"); ax.set_xlabel(r"$\kappa^2 m / N$"); ax.set_ylabel("S(m) / S(0)"); P.fig_legend(fig, None, handles=P.n_handles(Ns))
    P.cell_title(fig, 1.0, 0.0, 1.0, "pop", None, "gated frozen")
    P.caption(fig, "band: ±1 s.e.m. over seeds")
    P.save(fig, os.path.join(fig_dir, "kaczmarz_collapse"))


def cv_scaling(args, fig_dir, log):
    """CV of s_k at a fixed age vs N; fit the exponent (gated-frozen prediction: −1/2; check 3).

    Parameters
    ----------
    args : argparse.Namespace
        ``seeds``, ``N_max``.
    fig_dir : str
    log : callable
    """
    Ns = [n for n in (10, 20, 40, 80) if n <= args.N_max]
    readouts = ["gated_shared", "gated_frozen", "task_heads"]
    res = pooled(Config(N=Ns, readout=readouts, delta0=[0.0], lam=[1.0], seeds=args.seeds, solver="pop"), log)
    fig, axes = P.grid(1, 1)
    ax = axes[0, 0]
    print("\n[self-averaging]  CV(s_k) at age m=1 ∝ N^γ:")
    gam = {}
    for ro in readouts:
        cvs = [res[(N, ro, 1.0)][3][1] for N in Ns]
        gam[ro], _ = cv_exponent(Ns, cvs)
        P.plot_readout(ax, Ns, cvs, ro, marker="o")
        print(f"  {ro:14s} γ = {gam[ro]:+.3f}   CV = {['%.3f' % v for v in cvs]}")
    P.logaxis(ax, "x"); P.logaxis(ax, "y"); ax.set_xlabel("N"); ax.set_ylabel("CV of s at m = 1")
    P.discrete_ticks(ax, Ns, "x")
    P.fig_legend(fig, readouts)
    P.cell_title(fig, 1.0, 0.0, 1.0, "pop", None, "check 3")
    P.caption(fig, "γ: " + ", ".join(f"{P.READOUT_LABEL[r]} {g:+.2f}" for r, g in gam.items()))
    P.save(fig, os.path.join(fig_dir, "cv_exponent"))


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
    ap.add_argument("--out", default=os.path.join(_path.ROOT, "results", "validate"))
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--N-max", type=int, default=40)
    ap.add_argument("--parts", nargs="+", default=["eta", "kaczmarz", "cv"])
    args = ap.parse_args(argv)
    fig_dir = os.path.join(args.out, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    log = lambda m: print("  " + m, flush=True)
    if "eta" in args.parts:
        eta_halving(args, fig_dir, log)
    if "kaczmarz" in args.parts:
        kaczmarz(args, fig_dir, log)
    if "cv" in args.parts:
        cv_scaling(args, fig_dir, log)
    print(f"figures → {fig_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
