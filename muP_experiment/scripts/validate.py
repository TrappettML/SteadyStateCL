#!/usr/bin/env python
"""Validation runs (PLAN.md §7.7): η-halving, Kaczmarz collapse, CV exponent.

    python scripts/validate.py [--N-max 40] [--seeds 4] [--out results/validate]

Each part runs its own small sweep in-process and writes a figure plus a short
printed verdict; nothing here is needed by the Stage-1 sweep.
"""
import argparse
import os
import sys

import numpy as np

import _path  # noqa: F401

from mup.config import Config, make_batches, make_runs
from mup.observe import Moments, curves, cv_exponent
from mup.plotting import READOUT_COLOR, READOUT_LABEL, n_shade, plt
from mup.protocol import run_stream


def pooled(cfg, log):
    """Run a config; → {(N, readout, eta): (ages, S, E, cv_s pooled over seeds, S_se across seeds)}."""
    out = {}
    for batch in make_batches(make_runs(cfg), cfg.batch_size):
        res = run_stream(batch, log=log)
        mom = {f: res[f] for f in Moments.FIELDS}
        per_seed = curves(mom)["S"]
        for key in {(r.N, r.readout, r.eta) for r in batch}:
            idx = [i for i, r in enumerate(batch) if (r.N, r.readout, r.eta) == key]
            s = {f: mom[f][idx].sum((0, 2)) for f in Moments.FIELDS}
            n = s["count"]
            S, E = s["sum_s"] / n, s["sum_e"] / n
            cv = np.sqrt(np.maximum(s["sum_s2"] / n - S**2, 0)) / np.abs(S)
            se = per_seed[idx].std(0, ddof=1) / np.sqrt(len(idx)) if len(idx) > 1 else np.full_like(S, np.nan)
            out[key] = (res["ages"], S, E, cv, se)
    return out


def eta_halving(args, fig_dir, log):
    """SGD at η, η/2, η/4 and the population scheme at h ā² = 0.1, 0.01 vs a fine-step reference.

    At fixed λ the η² term in dQ/dα acts over α_task ∝ 1/η, so the SGD gap
    should be ∝ η (ratio ≈ 2 per halving); the population scheme's Euler
    error is ∝ h (ratio ≈ 10 per decade in h).  Reference: h ā² = 0.001.
    """
    N, lam, d0 = 10, 1.0, 0.0
    readouts = ["learned", "frozen"]
    contractions = [0.1, 0.01]
    ref = pooled(Config(N=[N], readout=readouts, delta0=[d0], lam=[lam], seeds=args.seeds, solver="pop", pop_contraction=0.001), log)
    pops = {c: pooled(Config(N=[N], readout=readouts, delta0=[d0], lam=[lam], seeds=args.seeds, solver="pop", pop_contraction=c), log)
            for c in contractions}
    etas = [0.05, 0.025, 0.0125]
    sgd = pooled(Config(N=[N], readout=readouts, delta0=[d0], lam=[lam], seeds=args.seeds, solver="sgd", eta=etas), log)
    fig, axes = plt.subplots(1, 2, figsize=(7, 3), sharey=True)
    print("\n[η-halving]  offsets from the fine-step population reference (h ā² = 0.001), same seeds:")
    print("  population Euler error ∝ h (expect ×0.1 per row); SGD gap ∝ η (expect ×0.5 per row)")
    for j, ro in enumerate(readouts):
        ages, S_ref, E_ref, _, _ = ref[(N, ro, 1.0)]
        axes[j].plot(ages[1:], S_ref[1:], color="k", lw=1.2, label="population (fine h)")
        for c in contractions:
            _, S, E, _, _ = pops[c][(N, ro, 1.0)]
            print(f"  {ro:8s} pop h ā²={c:<6}   ΔS(0)={S[0] - S_ref[0]:+.5f}  ΔE(0)={E[0] - E_ref[0]:+.5f}  max_m|ΔS|={np.nanmax(np.abs(S - S_ref)):.5f}")
        gaps = []
        for i, eta in enumerate(etas):
            _, S, E, _, _ = sgd[(N, ro, eta)]
            gaps.append(E[0] - E_ref[0])
            print(f"  {ro:8s} sgd η={eta:<8}   ΔS(0)={S[0] - S_ref[0]:+.5f}  ΔE(0)={E[0] - E_ref[0]:+.5f}  max_m|ΔS|={np.nanmax(np.abs(S - S_ref)):.5f}")
            axes[j].plot(ages[1:], S[1:], color=n_shade(i + 1, len(etas) + 1), label=f"SGD η={eta}")
        print(f"  {ro:8s} ΔE(0) ratios per halving: {['%.2f' % (gaps[i] / gaps[i + 1]) for i in range(len(gaps) - 1)]}")
        axes[j].set_xscale("log"); axes[j].set_title(READOUT_LABEL[ro], color=READOUT_COLOR[ro]); axes[j].set_xlabel("age m")
    axes[0].set_ylabel("S(m)"); axes[1].legend(fontsize=7)
    fig.suptitle(f"η-halving at fixed λ={lam}, δ₀={d0}, N={N}", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "eta_halving.png")); plt.close(fig)


def kaczmarz(args, fig_dir, log):
    """Frozen readout: S(m)/S(0) vs κ²m/N across N should collapse."""
    Ns = [n for n in (10, 20, 40, 80) if n <= args.N_max]
    res = pooled(Config(N=Ns, readout=["frozen"], delta0=[0.0], lam=[1.0], seeds=args.seeds, solver="pop"), log)
    fig, ax = plt.subplots(figsize=(4, 3))
    print("\n[Kaczmarz]  frozen readout, S/S0 at κ²m/N ≈ 0.5 for each N (collapse ⇒ equal):")
    for i, N in enumerate(Ns):
        ages, S, _, _, _ = res[(N, "frozen", 1.0)]
        x = ages / N
        ax.plot(x[1:], (S / S[0])[1:], color=n_shade(i, len(Ns)), label=f"N={N}")
        print(f"  N={N:3d}  S/S0 = {np.interp(0.5, x, S / S[0]):.3f}")
    ax.set_xscale("log"); ax.set_xlabel("κ² m / N"); ax.set_ylabel("S(m)/S(0)"); ax.legend(fontsize=7)
    ax.set_title("frozen readout (κ=1, λ=1, δ₀=0)")
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "kaczmarz_collapse.png")); plt.close(fig)


def cv_scaling(args, fig_dir, log):
    """CV of s_k at a fixed age vs N; fit the exponent (frozen prediction: −1/2)."""
    Ns = [n for n in (10, 20, 40, 80) if n <= args.N_max]
    readouts = ["learned", "frozen", "heads"]
    res = pooled(Config(N=Ns, readout=readouts, delta0=[0.0], lam=[1.0], seeds=args.seeds, solver="pop"), log)
    fig, ax = plt.subplots(figsize=(4, 3))
    print("\n[self-averaging]  CV(s_k) at age m=1 ∝ N^γ:")
    for ro in readouts:
        cvs = [res[(N, ro, 1.0)][3][1] for N in Ns]
        g, c = cv_exponent(Ns, cvs)
        ax.plot(Ns, cvs, "o-", ms=4, color=READOUT_COLOR[ro], label=f"{READOUT_LABEL[ro]} (γ={g:+.2f})")
        print(f"  {ro:8s} γ = {g:+.3f}   CV = {['%.3f' % v for v in cvs]}")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("N"); ax.set_ylabel("CV of s at m=1"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "cv_exponent.png")); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(_path.ROOT, "results", "validate"))
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--N-max", type=int, default=40)
    ap.add_argument("--parts", nargs="+", default=["eta", "kaczmarz", "cv"])
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    log = lambda m: print("  " + m, flush=True)
    if "eta" in args.parts:
        eta_halving(args, args.out, log)
    if "kaczmarz" in args.parts:
        kaczmarz(args, args.out, log)
    if "cv" in args.parts:
        cv_scaling(args, args.out, log)
    print(f"figures → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
