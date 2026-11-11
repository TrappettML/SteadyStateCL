#!/usr/bin/env python
"""Check 5 of the write-up: the initial scale σ₀ must not matter after the burn-in.

    python old/run_sigma0.py [--N 100] [--kappa 1] [--delta0 0] [--sigma0 0.5 1 2] [--seeds 8] [--gpus]

One condition per readout (every readout of ``--readout``, at one N, κ, δ₀
and the config's λ) is rerun with W⁽⁰⁾_ij ~ N(0, σ₀²) for each ``--sigma0``,
on the same task streams (σ₀ is not part of the run key).  The initial
readout is drawn from the rescaled Q as always, so δ₀ and λ are held fixed.
After the burn-in the signal and error curves S(m), E(m) should not depend
on σ₀; if they do, the ratio of the two learning rates is a genuine
parameter (write-up, "One learning rate suffices") and must be swept.

Output: ``<out>/figures/sigma0_N{N}_k{κ}_d0{δ₀}.png`` — rows S(m)/S(0) and
E(m), columns = readouts, traces = σ₀ (shades), bands ±1 s.e.m. over seeds —
and a printed table of the largest relative deviation of S(m) and E(m)
from the σ₀ = 1 curve over the ages, per readout (each seed against its
own σ₀ = 1 run, averaged over seeds).

Placement as ``validate.py``: every σ₀'s batches run as jobs over the GPUs
(``--gpus``), ``--workers`` CPU processes or this process (``mup.pool``).
"""
import argparse
import os
import sys
import warnings

import numpy as np

import _path  # noqa: F401

from mup import experiments as X
from mup import plotting as P
from mup import pool
from mup.config import Config, READOUT_ORDER, make_batches, make_runs


def configs(args, cfg: Config) -> dict:
    """``{sigma0: Config}``: the requested condition at each initial scale."""
    out = {}
    for s0 in args.sigma0:
        out[float(s0)] = Config(**{**cfg.__dict__, "name": "sigma0", "N": [args.N], "kappa": [args.kappa],
                                   "readout": list(args.readout), "delta0": [args.delta0],
                                   "lam": [args.lam if args.lam is not None else cfg.lam[0]],
                                   "seeds": args.seeds, "sigma0": [float(s0)], "extra": (), "trace": False})
    return out


def batches_of(cfgs: dict) -> list:
    """``[(sigma0, i, batch, T)]`` over every config's batches."""
    out = []
    for s0, cfg in cfgs.items():
        for i, batch in enumerate(make_batches(make_runs(cfg), cfg.batch_size)):
            out.append((s0, i, batch, batch[0].T_burn + batch[0].T_meas))
    return out


def run_all(cfgs: dict, place: dict, log) -> dict:
    """``{sigma0: {(N, readout, eta): curves}}`` (``experiments.seed_curves``) through ``pool.run_jobs``."""
    jobs = [((s0, i), {"runs": batch}, X.job_cost(batch, T)) for s0, i, batch, T in batches_of(cfgs)]
    res = pool.run_jobs("mup.experiments:stream_batch", jobs, log=log, **place)
    return {s0: X.seed_curves([(kw["runs"], r) for ((s, _), kw, _), r in zip(jobs, res) if s == s0]) for s0 in cfgs}


def deviations(res: dict, N: int, readouts, ref: float = 1.0) -> dict:
    """``{readout: {sigma0: (max_m |S/S_ref − 1|, max_m |E − E_ref|)}}``: each seed against its own σ₀ = 1 run (same
    task stream), max over the ages, then the mean over seeds (E is absolute: it crosses 0)."""
    out = {}
    for ro in readouts:
        c1 = res[ref][(N, ro, 1.0)]
        out[ro] = {}
        for s0, r in res.items():
            c = r[(N, ro, 1.0)]
            assert c["seeds"] == c1["seeds"], "each σ₀ run is compared with the σ₀ = 1 run of the same seed"
            with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                dS = np.nanmax(np.abs(c["S_seeds"] / c1["S_seeds"] - 1.0), 1)     # per seed, over the ages
                dE = np.nanmax(np.abs(c["E_seeds"] - c1["E_seeds"]), 1)
                out[ro][s0] = (float(np.nanmean(dS)), float(np.nanmean(dE)))
    return out


def figure(res: dict, N: int, readouts, run0, path: str) -> str:
    """Rows S(m)/S(0), E(m); columns readouts; traces σ₀ in shades of the readout colour; bands ±1 s.e.m."""
    P.apply_style()
    sig = sorted(res)
    fig, axes = P.grid(2, len(readouts), sharex="col", sharey="row", panel=(3.0, 2.3))
    figdata = {}
    for j, ro in enumerate(readouts):
        for i, s0 in enumerate(sig):
            c = res[s0][(N, ro, 1.0)]
            ages, E = c["ages"], c["E"]
            with np.errstate(invalid="ignore", divide="ignore"):
                Sn = c["S_seeds"] / c["S_seeds"][:, :1]                 # each seed normalised by its own S(0)
            y = np.nanmean(Sn, 0)
            se = Sn.std(0, ddof=1) / np.sqrt(len(Sn)) if len(Sn) > 1 else np.zeros_like(y)
            col = P.kappa_shades(i, len(sig), ro)
            x = ages + 1                                                # log axis with age 0 at 1
            axes[0, j].plot(x, y, color=col)
            P.band(axes[0, j], x, y - se, y + se, col)
            axes[1, j].plot(x, E, color=col)
            figdata[f"S_{ro}_s0{P.fmt(s0)}"], figdata[f"E_{ro}_s0{P.fmt(s0)}"] = c["S"], E
        figdata["ages"] = ages
        axes[0, j].set_title(P.READOUT_LABEL[ro], color=P.READOUT_COLOR[ro])
        P.logaxis(axes[1, j], "x"); axes[1, j].set_xlabel("age m + 1")
    axes[0, 0].set_ylabel("S(m) / S(0)"); axes[1, 0].set_ylabel("E(m)")
    handles = [P.Line2D([], [], color=P.kappa_shades(i, len(sig)), lw=1.6, label=f"σ₀ = {P.fmt(s0)}") for i, s0 in enumerate(sig)]
    P.fig_legend(fig, None, handles=handles)
    P.cell_title(fig, kappa=run0.kappa_label, d0=run0.delta0, lam=run0.lam, solver=run0.solver, eta=run0.eta,
                 extra=f"check 5: initial scale, N = {N}")
    P.caption(fig, "line: seed mean of each seed's S(m)/S(0), and of E(m), over the measurement window at each initial scale σ₀ "
                   "(W⁽⁰⁾ ~ N(0, σ₀²), same task streams); band: ±1 s.e.m. over seeds.  Curves that depend on σ₀ "
                   "mean the ratio of the two learning rates is a genuine parameter.")
    return P.save(fig, path, figdata=figdata)


def main(argv=None):
    """Command-line entry point (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(_path.ROOT, "configs", "stage1.yaml"))
    ap.add_argument("--N", type=int, default=100)
    ap.add_argument("--kappa", default="1.0")
    ap.add_argument("--delta0", type=float, default=0.0)
    ap.add_argument("--lam", type=float, default=None, help="default: the config's first λ")
    ap.add_argument("--sigma0", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--readout", nargs="*", default=list(READOUT_ORDER))
    ap.add_argument("--out", default=os.path.join(_path.ROOT, "results", "sigma0"))
    pool.add_args(ap)
    args = ap.parse_args(argv)
    if 1.0 not in [float(s) for s in args.sigma0]:
        args.sigma0 = [1.0] + list(args.sigma0)
    cfgs = configs(args, Config.from_yaml(args.config))
    if not make_runs(cfgs[1.0]):
        print("no runs left after skipping κ outside [1, √N]; nothing to do", flush=True)
        return 0
    plan = pool.plan(args, [(b, T) for _, _, b, T in batches_of(cfgs)], out_per_task=8 * 64, split=False)
    run0 = make_runs(cfgs[1.0])[0]
    print(f"check 5 (initial scale): N={args.N} κ={args.kappa} δ₀={args.delta0} λ={run0.lam} {run0.solver} "
          f"σ₀={args.sigma0} seeds={args.seeds}  [{plan.describe()}]", flush=True)
    res = run_all(cfgs, plan.place, lambda m: print("  " + m, flush=True))
    dev = deviations(res, args.N, args.readout)
    print("largest deviation from σ₀ = 1 over the ages, each seed against its own σ₀ = 1 run, seed mean (S relative, E absolute):")
    for ro, d in dev.items():
        print(f"  {ro:14s} " + "  ".join(f"σ₀={P.fmt(s0)}: ΔS {dS:6.3f} ΔE {dE:6.3f}" for s0, (dS, dE) in sorted(d.items())))
    fdir = os.path.join(args.out, "figures")
    f = figure(res, args.N, args.readout, run0, os.path.join(fdir, f"sigma0_N{args.N}_k{P.fmt(args.kappa)}_d0{P.fmt(args.delta0)}"))
    np.savez_compressed(os.path.join(fdir, f"sigma0_N{args.N}_k{P.fmt(args.kappa)}_d0{P.fmt(args.delta0)}.npz"),
                        **{f"{ro}_s0{P.fmt(s0)}_{q}": res[s0][(args.N, ro, 1.0)][q] for ro in dev for s0 in res
                           for q in ("S", "E", "S_seeds", "E_seeds")})
    print(f"wrote {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
