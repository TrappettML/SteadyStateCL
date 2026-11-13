#!/usr/bin/env python
"""Nested-model test: does the two-layer student forget like a simpler pattern memory, and if not, which
ingredient makes the difference — SGD noise, the second layer, or error-correcting updates?

    python scripts/run_nested.py [--N 256] [--D-factor 2] [--delta0 0 1 10] [--overlap 0.1 0.5] [--rho 0 0.7]
                                 [--lam 1] [--seeds 10] [--models sgd gf delta_matched delta_raw hebb]
                                 [--no-ref-checks] [--reuse] [--workers K] [--gpus [ids]]
    python scripts/run_nested.py --gpus --workers 8                                    # GPU node
    python scripts/run_nested.py --N 32 --seeds 2 --delta0 0 10 --overlap 0.2 --rho 0 --no-ref-checks \\
                                 --out /tmp/nested                                     # laptop smoke run

Five models (``mup/nested.py``), each removing one ingredient from the one above, on the same task stream
from the same forked weights: SGD → gradient flow (no sampling noise) → Δ-matched (readout frozen, one
relaxed projection per task at gradient flow's measured rate) → Δ-raw (the bare rate, nothing fitted) →
Hebb (fixed-size writes with decay, no error correction; (b, γ) tuned once at the reference condition to
gradient flow's ε(0) and τ½ and held fixed).  ``--overlap`` is the gate density p, the fraction of units two
tasks share (the project's κ = p√N).

Grid: ``--delta0`` × ``--overlap`` × ``--rho`` at one N, D = ``--D-factor``·N and λ.  At the reference
condition (``--ref-delta0``, ``--ref-overlap``, ``--ref-rho``) three extra checks run unless
``--no-ref-checks``: ``--ref-lam`` (Hebb against task length), ``--ref-N`` (is τ½ linear in N) and
``--ref-D-factor`` (does the SGD–gradient-flow gap shrink with D).

Measured per condition and model (seed mean ± s.e.): the memory curve M(τ) = 1 − E_c(τ)/E_∞ over the age τ of
a task, its lifetime τ½ (M = M(0)/2, jackknife s.e.), ε(0) = E_c(0), the best of an exponential, stretched
exponential and power-law fit, and for SGD and gradient flow the key-drift fraction φ_key and the rate ratio ϱ.

The decision rules are fixed in ``mup.nested`` (``coincide``: max |ΔM|/M_A(0) < 0.05, τ½ within 10 %, and the
difference within 2 s.e.m. on the age-averaged curve and on τ½) and applied to SGD vs gradient flow, gradient flow vs Δ-matched, Δ-matched vs
Δ-raw and Δ-matched vs Hebb.

Output, in ``results/nested/``: ``nested_report.md`` (every number and verdict), ``nested_summary.json``,
``nested.npz`` (per condition and model the per-seed curves ``{cond}_{model}_M|E|S|Ev``, ``_Einf``, ``_ages``),
``figures/nested_curves.png`` (M(τ), rows = (p, ρ), columns = δ₀), ``figures/nested_summary.png`` (τ½, ε(0),
φ_key and ϱ against δ₀), ``figures/nested_reference.png`` and ``figures/nested_tau_vs_N.png`` (the reference
checks), and ``raw/`` (one pickle per condition and seed, read back by ``--reuse``).

Placement.  ``--gpus [ids]`` runs SGD and gradient flow on the GPUs (``--backend jax``, chosen automatically):
the seeds of a condition are one ``vmap`` batch through ``model.train_block``, one device call per task, and
the conditions are spread over the cards by ``mup.pool`` (``--per-gpu`` processes per card, ``--split k`` to
cut every condition's seeds into k batches).  Without ``--gpus`` the NumPy kernel runs one (condition, seed)
per CPU process (``--workers``).  The delta rules, Hebb and the probes are NumPy on the host in both; the Hebb
stage always uses ``--workers``.

Cost: gradient flow, the delta rules and Hebb take seconds per run; SGD takes n = λD/c steps per task
(10 240 at D = 512, c = 0.05), which is nearly all of the wall time.
``--models gf delta_matched delta_raw hebb`` is a first pass without it.
"""
import argparse
import dataclasses
import hashlib
import itertools
import json
import math
import os
import pickle
import sys
import time
from typing import Optional

import numpy as np

import _path  # noqa: F401

from mup import driver
from mup import nested as X
from mup import plotting as P
from mup import pool

_T0 = time.perf_counter()


# ----------------------------------------------------------------------------
# conditions and jobs
# ----------------------------------------------------------------------------
def conditions(args) -> tuple[list, X.Cond, dict]:
    """(every condition, the reference condition, {check: conditions}) from the command line."""
    mk = lambda **kw: X.Cond(**{"N": args.N, "D": args.D_factor * args.N, "lam": args.lam, "latent": args.latent,
                                "rank": args.rank, **kw})
    main = [mk(delta0=d, p=p, rho=r) for p, r, d in itertools.product(args.overlap, args.rho, args.delta0)]
    ref = mk(delta0=args.ref_delta0, p=args.ref_overlap, rho=args.ref_rho)
    checks = {"main": main, "lam": [], "N": [], "D": []}
    if not args.no_ref_checks:
        rk = dict(delta0=ref.delta0, p=ref.p, rho=ref.rho)
        checks["lam"] = [mk(lam=l, **rk) for l in args.ref_lam]
        checks["N"] = [mk(N=n, D=args.D_factor * n, **rk) for n in args.ref_N]
        checks["D"] = [mk(D=f * args.N, **rk) for f in args.ref_D_factor]
    every = list(dict.fromkeys(main + [ref] + checks["lam"] + checks["N"] + checks["D"]))
    return every, ref, checks


def placement(args, n_jobs: int) -> tuple[dict, dict]:
    """``pool.run_jobs`` placement of the training jobs and of the (NumPy, host-only) Hebb jobs.

    With ``--gpus`` the training jobs go to the cards, ``--per-gpu`` processes each (default: as many as there
    are jobs to fill, at most ``pool.MAX_PER_GPU``); the Hebb jobs always run on the CPU (``--workers``).
    """
    cpu = {"gpus": None, "per_gpu": 1, "workers": int(args.workers or 0), "chunk": None}
    if args.gpus is None:
        return cpu, cpu
    ids = [int(g) for g in args.gpus if str(g).lstrip("-").isdigit()]
    gpus = ids or driver.visible_gpus() or list(range(pool.GPU_COUNT))
    per_gpu = int(args.per_gpu) if args.per_gpu else min(pool.MAX_PER_GPU, max(1, -(-n_jobs // len(gpus))))
    return {"gpus": gpus, "per_gpu": per_gpu, "workers": 0, "chunk": None}, cpu


def cost(c: X.Cond, st: X.Settings, n_seeds: int = 1) -> float:
    """Relative cost of a job: SGD steps × active block when SGD runs, else gradient-flow steps.  On the device
    a batch of seeds costs about what one run does (launch-latency bound), so the seeds enter weakly."""
    T = st.schedule(c)[1]
    steps = c.lam * c.D / st.sgd_contraction if "sgd" in st.models else c.lam / st.gf_step
    return T * steps * (math.sqrt(c.pN * c.D) * n_seeds ** 0.25 if st.backend == "jax" else c.pN * c.D * n_seeds)


def fork_jobs(conds, seeds, st: X.Settings, raw_dir: str, reuse: bool, pieces: int) -> tuple[dict, list]:
    """(results read back from ``raw_dir``, jobs for the rest).  A job is one condition with a group of seeds:
    every missing seed on its own for the NumPy backend, all of them in ``pieces`` vmap batches for JAX."""
    sd = dataclasses.asdict(st)
    tag = hashlib.sha256(json.dumps(sd, sort_keys=True).encode()).hexdigest()[:8]
    res, jobs = {}, []
    for c in conds:
        path = {s: os.path.join(raw_dir, f"{c.stem()}_s{s}_{tag}.pkl") for s in seeds}
        todo = []
        for s in seeds:
            if reuse and os.path.exists(path[s]):
                with open(path[s], "rb") as f:
                    res[(c, s)] = pickle.load(f)
            else:
                todo.append(s)
        k = len(todo) if st.backend == "numpy" else min(len(todo), max(1, pieces))
        for i in range(k):
            part = todo[i::k]
            jobs.append(((c.stem(), tuple(part)), {"cond": dataclasses.asdict(c), "seeds": part, "settings": sd,
                                                   "save": [path[s] for s in part]}, cost(c, st, len(part))))
    return res, jobs


def run_forks(conds, seeds, res: dict, jobs: list, place: dict) -> dict:
    """``{cond: [run_seeds result per seed]}`` from the reused results and the jobs of ``fork_jobs``."""
    by_stem = {c.stem(): c for c in conds}
    for ((stem, part), _, _), outs in zip(jobs, pool.run_jobs("mup.nested:run_seeds", jobs, log=print, **place)):
        for s, out in zip(part, outs):
            res[(by_stem[stem], s)] = out
    return {c: [res[(c, s)] for s in seeds] for c in conds}


def run_hebbs(forks: dict, conds, gammas, st: X.Settings, place: dict) -> dict:
    """``{(cond, γ): [unit-b run_hebb result per seed]}`` on the streams and forked readouts of ``forks``."""
    sd = dataclasses.asdict(st)
    jobs = [((c, g, i), {"cond": dataclasses.asdict(c), "seed": r["seed"], "settings": sd, "a_fork": r["a_fork"],
                         "t_fork": r["t_fork"], "gamma": float(g)}, 1.0 / g)
            for c in conds for g in gammas for i, r in enumerate(forks[c])]
    out = {}
    for (key, _, _), r in zip(jobs, pool.run_jobs("mup.nested:run_hebb", jobs, **place)):
        out.setdefault(key[:2], []).append(r)
    return out


def tune(forks: dict, ref: X.Cond, st: X.Settings, place: dict) -> Optional[dict]:
    """(b, γ) of the palimpsest at the reference condition (``nested.tune_hebb``): a geometric grid of γ around
    ln 2/τ½ of gradient flow, then a finer one around the best."""
    ages = forks[ref][0]["ages"]
    cv = X.curves(forks[ref], "gf")
    eps0, (tau, _) = float(cv["eps0"].mean()), X.tau_half(cv["M"], ages)
    if not np.isfinite(tau):
        print("Hebb: gradient flow never reaches half at the reference condition; nothing to tune to", flush=True)
        return None
    by_gamma = {}
    grid = math.log(2.0) / tau * np.geomspace(1 / 8, 8, 13)
    for _ in range(2):
        for (_, g), res in run_hebbs(forks, [ref], grid, st, place).items():
            by_gamma[g] = res
        best = X.tune_hebb(by_gamma, ages, ref.lam, eps0, tau)
        if not np.isfinite(best["loss"]):
            print("Hebb: no (b, γ) on the grid gives a positive memory with a half-life", flush=True)
            return None
        grid = best["gamma"] * np.geomspace(1 / 1.5, 1.5, 9)
    best.update(target_eps0=eps0, target_tau=tau)
    return best


# ----------------------------------------------------------------------------
# analysis
# ----------------------------------------------------------------------------
def sem(x: np.ndarray, axis: int = -1) -> np.ndarray:
    n = x.shape[axis]
    return x.std(axis, ddof=1) / math.sqrt(n) if n > 1 else np.zeros(np.delete(x.shape, axis))


def analyse(c: X.Cond, results: list, hebb: Optional[list], b: float) -> dict:
    """Curves, scalars and the four pre-registered comparisons of one condition."""
    ages = results[0]["ages"]
    rs = [{**r, "models": {**r["models"], **({"hebb": hebb[i]} if hebb else {})}} for i, r in enumerate(results)]
    out = {"cond": c, "ages": ages, "models": {}, "compare": {}, "n_seeds": len(rs),
           "drift": float(np.nanmax([r["burn"]["drift"] for r in rs])), "t_fork": int(np.mean([r["t_fork"] for r in rs])),
           "a2": float(np.mean([r["a2"] for r in rs])), "Q": float(np.mean([r["Q"] for r in rs]))}
    for m in X.MODELS:
        if m not in rs[0]["models"]:
            continue
        cv = X.curves(rs, m, b if m == "hebb" else 1.0)
        M = cv["M"]
        tau, tau_se = X.tau_half(M, ages)
        d = {**cv, "M_mean": M.mean(1), "M_sem": sem(M), "tau": tau, "tau_se": tau_se,
             "eps0_mean": float(cv["eps0"].mean()), "eps0_sem": float(sem(cv["eps0"])),
             "plateau": float(M.mean(1)[-1] / M.mean(1)[0]), "fit": X.fit_shapes(ages, M.mean(1)),
             "theta": float(np.mean([r["models"][m]["theta"] for r in rs])),
             "theta_raw": float(np.mean([r["models"][m]["theta_raw"] for r in rs])),
             "rate": float(np.mean([r["models"][m]["rate"] for r in rs])),
             "rate_sem": float(sem(np.array([r["models"][m]["rate"] for r in rs])))}
        if cv["Ev"] is not None:
            phi = X.key_fraction(cv["E"], cv["Ev"])                    # (n_ages, n_seeds)
            at = tau if np.isfinite(tau) else float(ages[-1])
            col = np.array([np.interp(at, ages[1:], phi[1:, j]) for j in range(phi.shape[1])])
            d.update(phi=phi, phi_tau=float(col.mean()), phi_tau_sem=float(sem(col)))
        out["models"][m] = d
    for A, B in X.COMPARISONS:
        if A in out["models"] and B in out["models"]:
            out["compare"][(A, B)] = X.coincide(out["models"][A]["M"], out["models"][B]["M"], ages)
    return out


def f3(v) -> str:
    return "n/a" if v is None or not np.isfinite(v) else f"{v:.3g}"


def verdict(k: dict) -> str:
    return "coincide" if k["coincide"] else "differ"


def report(an: dict, ref: X.Cond, checks: dict, hebb: Optional[dict], st: X.Settings, args) -> str:
    """The markdown report: settings, rules, per-condition tables, the decision table applied, the reference checks."""
    L = P.MODEL_LABEL
    out = ["# Nested-model test: is Stage 1 more than a palimpsest?", "",
           f"Settings: seeds {args.seeds}, clock {st.clock}, gradient-flow step hηā² = {st.gf_step:g}, SGD ηā² = "
           f"{st.sgd_contraction:g}, burn-in ≥ {st.burn:g} lifetime units (≤ {st.burn_max:g}), probed tasks {st.learn:g} "
           f"unit, ages to {st.tau_max:g} units (unit: half-life of eq. S_learned); teachers: latent {args.latent}"
           + (f", rank {args.rank}" if args.latent == "sphere" else "") + ".", "",
           f"Rules (fixed before running): two curves coincide if max_τ |M_A − M_B|/M_A(0) < {X.TOL_M:g}, τ½ agrees "
           f"within {100 * X.TOL_TAU:g} %, and the difference is within {X.N_SEM:g} s.e.m. across seeds (unpaired, tested "
           f"on the age-averaged curve and on τ½; the paired z of the age-averaged difference is listed).", ""]
    if hebb:
        out += [f"Hebb, tuned at the reference condition ({ref.label()}): b = {hebb['b']:.4g}, γ = {hebb['gamma']:.4g}; "
                f"ε(0) = {hebb['eps0']:.3g} (gradient flow {hebb['target_eps0']:.3g}), τ½ = {hebb['tau']:.4g} "
                f"({hebb['target_tau']:.4g}); mismatch ln²ε + ln²τ = {hebb['loss']:.3g}.", ""]
    for c, a in an.items():
        out += [f"## {c.label()}", "",
                f"fork after {a['t_fork']} tasks, burn-in drift {a['drift']:.3f}"
                + (" (**not stationary**)" if a["drift"] > st.burn_tol else "")
                + f"; mean a² = {a['a2']:.3g}, mean Q = {a['Q']:.3g}; {a['n_seeds']} seeds", "",
                "| model | ε(0) | M(0) | τ½ | M(τ_max)/M(0) | best shape | φ_key(τ½) | θ | ϱ |", "|---|---|---|---|---|---|---|---|---|"]
        for m, d in a["models"].items():
            out.append(f"| {L[m]} | {d['eps0_mean']:.3g} ± {d['eps0_sem']:.2g} | {d['M_mean'][0]:.3f} | {f3(d['tau'])} ± "
                       f"{f3(d['tau_se'])} | {d['plateau']:.3f} | {d['fit']['best']} | "
                       + (f"{d['phi_tau']:.3f} ± {d['phi_tau_sem']:.2g}" if "phi_tau" in d else "—")
                       + f" | {d['theta']:.3f} | {d['rate']:.3f} |")
        out += ["", "| comparison | max dev | at τ | Δτ½/τ½ | within 2 s.e.m. | paired z | verdict |", "|---|---|---|---|---|---|---|"]
        for (A, B), k in a["compare"].items():
            out.append(f"| {L[A]} vs {L[B]} | {k['max_dev']:.3f} | {k['age_dev']} | {f3(k['dtau'])} | "
                       f"{'n/a' if k['sem_ok'] is None else 'yes' if k['sem_ok'] else 'no'} | {f3(k['z_paired'])} | **{verdict(k)}** |")
        out.append("")
    out += ["## Decision rules applied", ""] + decisions(an, ref, checks)
    return "\n".join(out) + "\n"


def decisions(an: dict, ref: X.Cond, checks: dict) -> list:
    """The decision table of the design, read off the comparisons of the main grid and the reference checks."""
    main = [c for c in checks["main"] if c in an]
    out = []

    def tally(pair, conds=main):
        ks = [(c, an[c]["compare"][pair]) for c in conds if pair in an[c]["compare"]]
        return ks, [c for c, k in ks if k["coincide"]]

    ks, yes = tally(("sgd", "gf"))
    if ks:
        out.append(f"- **SGD vs gradient flow**: coincide in {len(yes)}/{len(ks)} conditions; largest deviation "
                   f"{max(k['max_dev'] for _, k in ks):.3f}.  "
                   + ("SGD adds nothing: describe the model as gradient flow." if len(yes) == len(ks) else
                      "A gap is only meaningful if it does not shrink with D (below)."))
        Dc = sorted((c for c in checks["D"] if c in an and ("sgd", "gf") in an[c]["compare"]), key=lambda c: c.D)
        if len(Dc) >= 2:
            dev = [an[c]["compare"][("sgd", "gf")]["max_dev"] for c in Dc]
            e0 = [an[c]["models"]["sgd"]["eps0_mean"] / an[c]["models"]["gf"]["eps0_mean"] - 1 for c in Dc]
            out.append("  - gap against D: " + ", ".join(f"D = {c.D}: max dev {d:.4f}, ε(0) excess {100 * e:.2f} %"
                                                           for c, d, e in zip(Dc, dev, e0))
                       + f"; ratio of max dev {dev[-1] / dev[0]:.2f} for D × {Dc[-1].D / Dc[0].D:g} (1/D predicts "
                         f"{Dc[0].D / Dc[-1].D:.2f}).")
    ks, yes = tally(("gf", "delta_matched"))
    if ks:
        d0s = sorted({c.delta0 for c, _ in ks})
        by = {d: [k["coincide"] for c, k in ks if c.delta0 == d] for d in d0s}
        out.append("- **Gradient flow vs Δ-matched**: " + "; ".join(f"δ₀ = {d:g}: {sum(v)}/{len(v)} coincide" for d, v in by.items()) + ".")
        phi0 = [an[c]["models"]["gf"]["phi_tau"] for c, _ in ks if c.delta0 == d0s[0]]
        if all(all(v) for v in by.values()):
            out.append("  - They coincide at every δ₀: Stage 1 is a delta-rule associative memory with a rescaled rate.")
        elif all(by[d0s[-1]]) and not any(any(by[d]) for d in d0s[:-1]):
            out.append(f"  - They coincide only at δ₀ = {d0s[-1]:g}: the second layer matters only in the rich regime, through "
                       f"key drift.  φ_key at δ₀ = {d0s[0]:g}: {min(phi0):.3f} … {max(phi0):.3f} (the claim needs > 0.2: "
                       f"{'met' if min(phi0) > 0.2 else 'NOT met'}).")
        else:
            out.append(f"  - Mixed pattern (neither row of the decision table); φ_key at δ₀ = {d0s[0]:g}: "
                       f"{min(phi0):.3f} … {max(phi0):.3f}.")
    ks, yes = tally(("delta_matched", "delta_raw"))
    if ks:
        rate = {d: np.mean([an[c]["models"]["gf"]["rate"] for c, _ in ks if c.delta0 == d]) for d in sorted({c.delta0 for c, _ in ks})}
        out.append(f"- **Δ-matched vs Δ-raw**: coincide in {len(yes)}/{len(ks)} conditions"
                   + (": even the rate rescaling is negligible.  " if len(yes) == len(ks) else ".  ")
                   + "Rate ratio ϱ (gradient flow's exponent over the bare hidden-channel one) against δ₀: "
                   + ", ".join(f"{d:g}: {r:.3f}" for d, r in rate.items()) + ".")
    ks, yes = tally(("delta_matched", "hebb"))
    if ks:
        out.append(f"- **Δ vs Hebb**: coincide in {len(yes)}/{len(ks)} conditions.  "
                   + ("Error correction does not matter here." if len(yes) == len(ks) else
                      "The comparison literature is sequential projection and delta-rule memory, not fixed-write palimpsests."))
        lam = [c for c in checks["lam"] if c in an and ("delta_matched", "hebb") in an[c]["compare"]]
        if lam:
            out.append("  - task length: " + "; ".join(
                f"λ = {c.lam:g}: ε(0) Δ {an[c]['models']['delta_matched']['eps0_mean']:.3g} / Hebb "
                f"{an[c]['models']['hebb']['eps0_mean']:.3g}, {verdict(an[c]['compare'][('delta_matched', 'hebb')])}" for c in lam) + ".")
        hi = [c for c, _ in ks if c.rho > 0]
        if hi:
            out.append("  - similar teachers, M(τ_max)/M(0): " + "; ".join(
                f"{c.label()}: Δ {an[c]['models']['delta_matched']['plateau']:.3f}, Hebb {an[c]['models']['hebb']['plateau']:.3f}"
                for c in hi) + ".")
    Nc = sorted((c for c in checks["N"] if c in an), key=lambda c: c.N)
    if len(Nc) >= 2:
        out.append("- **τ½ against N** at the reference condition (log–log slope; 1 = linear): " + ", ".join(
            f"{P.MODEL_LABEL[m]} {slope:.2f}" for m in X.MODELS if all(m in an[c]["models"] for c in Nc)
            for slope in [np.polyfit(np.log([c.N for c in Nc]), np.log([an[c]["models"][m]["tau"] for c in Nc]), 1)[0]]) + ".")
    return out or ["(no comparison available)"]


# ----------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------
def draw_curves(ax, a: dict, figdata: dict) -> None:
    """M(τ) of every model of one condition on a log age axis (τ + 1), seed mean with a ±1 s.e. band."""
    x = a["ages"] + 1
    for m, d in a["models"].items():
        P.band(ax, x, d["M_mean"] - d["M_sem"], d["M_mean"] + d["M_sem"], P.MODEL_COLOR[m])
        ax.plot(x, d["M_mean"], color=P.MODEL_COLOR[m], ls=P.MODEL_STYLE[m], zorder=4)
        figdata[f"{a['cond'].stem()}_{m}_M"] = d["M_mean"]
    figdata[f"{a['cond'].stem()}_ages"] = a["ages"]
    P.ref_line(ax, 0.0)
    P.logaxis(ax, "x")


def fig_curves(an: dict, conds: list, path: str, title: str) -> str:
    """Memory curves of the main grid: rows = (p, ρ), columns = δ₀."""
    rows = list(dict.fromkeys((c.p, c.rho) for c in conds))
    cols = sorted({c.delta0 for c in conds})
    fig, axes = P.grid(len(rows), len(cols), sharex="col", sharey="row", panel=(3.0, 2.2))
    figdata, models = {}, []
    for i, (p, rho) in enumerate(rows):
        for j, d0 in enumerate(cols):
            c = next((c for c in conds if (c.p, c.rho, c.delta0) == (p, rho, d0)), None)
            if c is not None:
                draw_curves(axes[i, j], an[c], figdata)
                models = models or list(an[c]["models"])
            if i == 0:
                axes[i, j].set_title(rf"$\delta_0$ = {d0:g}", fontsize=P.FS_MIN + 1)
            if i == len(rows) - 1:
                axes[i, j].set_xlabel(r"age $\tau + 1$ (tasks)")
        axes[i, 0].set_ylabel(f"memory $M(\\tau)$\n$p$ = {p:g}, $\\rho$ = {rho:g}")
    fig.suptitle(title)
    P.caption(fig, r"$M(\tau) = 1 - E_c(\tau)/E_\infty$ of a task $\tau$ tasks after it was learned, mean over the tasks "
                   "learned after the fork and over seeds; band: ±1 s.e. over seeds; all models share the forked "
                   "weights and the task stream.")
    P.fig_legend(fig, handles=P.model_handles(models))
    return P.save(fig, path, figdata=figdata)


def fig_summary(an: dict, conds: list, path: str, title: str) -> str:
    """τ½, ε(0), φ_key(τ½) and the rate ratio ϱ against δ₀: columns = (p, ρ), one trace per model."""
    cells = list(dict.fromkeys((c.p, c.rho) for c in conds))
    d0s = sorted({c.delta0 for c in conds})
    rows = [("tau", "tau_se", r"lifetime $\tau_{1/2}$ (tasks)"), ("eps0_mean", "eps0_sem", r"$\varepsilon(0)$"),
            ("phi_tau", "phi_tau_sem", "key-drift fraction\n" + r"$\varphi_{\rm key}(\tau_{1/2})$"),
            ("rate", "rate_sem", "rate ratio $\\varrho$")]
    fig, axes = P.grid(len(rows), len(cells), sharex="col", sharey="row", panel=(2.8, 2.0))
    figdata, models = {"delta0": np.array(d0s)}, []
    xs = np.arange(len(d0s))
    for j, (p, rho) in enumerate(cells):
        cs = [next((c for c in conds if (c.p, c.rho, c.delta0) == (p, rho, d)), None) for d in d0s]
        for i, (key, err, ylabel) in enumerate(rows):
            ax = axes[i, j]
            for m in X.MODELS:
                if key == "rate" and m not in X.LEARNED:
                    continue                                           # ϱ is set, not measured, for the frozen keys
                y = np.array([an[c]["models"].get(m, {}).get(key, np.nan) if c else np.nan for c in cs], float)
                e = np.array([an[c]["models"].get(m, {}).get(err, np.nan) if c else np.nan for c in cs], float)
                if not np.isfinite(y).any():
                    continue
                models += [m] if m not in models else []
                ax.errorbar(xs, y, yerr=np.nan_to_num(e), color=P.MODEL_COLOR[m], ls=P.MODEL_STYLE[m],
                            marker=P.MODEL_MARKER[m], capsize=2, zorder=4)
                figdata[f"p{p:g}_rho{rho:g}_{m}_{key}"] = y
            if key in ("phi_tau",):
                P.ref_line(ax, 0.0)
            if key == "rate":
                P.ref_line(ax, 1.0)
            ax.set_xticks(xs)
            ax.set_xticklabels([f"{d:g}" for d in d0s])
            ax.set_xlim(-0.4, len(d0s) - 0.6)
            if j == 0:
                ax.set_ylabel(ylabel)
        axes[0, j].set_title(rf"$p$ = {p:g}, $\rho$ = {rho:g}", fontsize=P.FS_MIN + 1)
        axes[-1, j].set_xlabel(r"imbalance $\delta_0$")
    fig.suptitle(title)
    P.caption(fig, r"seed mean ± 1 s.e. (jackknife for $\tau_{1/2}$); $\varphi_{\rm key}$: share of forgetting from the keys "
                   r"moving, read at $\tau_{1/2}$; $\varrho$: realized exponent over the bare hidden-channel exponent "
                   "(1 = no second-layer rescaling).")
    P.fig_legend(fig, handles=P.model_handles([m for m in X.MODELS if m in models], marker=True))
    return P.save(fig, path, figdata=figdata)


def fig_reference(an: dict, conds: list, path: str, title: str) -> str:
    """Memory curves of the reference checks, one panel per condition."""
    fig, axes = P.grid(1, len(conds), sharex=False, sharey="row", panel=(3.0, 2.2))
    figdata, models = {}, []
    for ax, c in zip(axes[0], conds):
        draw_curves(ax, an[c], figdata)
        models = models or list(an[c]["models"])
        ax.set_title(rf"$N$ = {c.N}, $D$ = {c.D}, $\lambda$ = {c.lam:g}", fontsize=P.FS_MIN + 1)
        ax.set_xlabel(r"age $\tau + 1$ (tasks)")
    axes[0, 0].set_ylabel(r"memory $M(\tau)$")
    fig.suptitle(title)
    P.caption(fig, "the reference condition at each value of the three checks (task length, width, input dimension); "
                   "line: seed mean, band: ±1 s.e.")
    P.fig_legend(fig, handles=P.model_handles(models))
    return P.save(fig, path, figdata=figdata)


def fig_tau_N(an: dict, conds: list, path: str, title: str) -> str:
    """τ½ against N at the reference condition, log–log, one trace per model; grey line: slope 1 through gradient flow."""
    conds = sorted(conds, key=lambda c: c.N)
    Ns = np.array([c.N for c in conds], float)
    fig, axes = P.grid(1, 1, panel=(3.4, 2.6))
    ax, figdata, models = axes[0, 0], {"N": Ns}, []
    for m in X.MODELS:
        if not all(m in an[c]["models"] for c in conds):
            continue
        y = np.array([an[c]["models"][m]["tau"] for c in conds])
        e = np.nan_to_num(np.array([an[c]["models"][m]["tau_se"] for c in conds]))
        ax.errorbar(Ns, y, yerr=e, color=P.MODEL_COLOR[m], ls=P.MODEL_STYLE[m], marker=P.MODEL_MARKER[m], capsize=2, zorder=4)
        figdata[f"{m}_tau"] = y
        models.append(m)
    if "gf_tau" in figdata and np.isfinite(figdata["gf_tau"][0]):
        P.ref_curve(ax, Ns, figdata["gf_tau"][0] * Ns / Ns[0])
    P.logaxis(ax, "x")
    P.logaxis(ax, "y")
    P.discrete_ticks(ax, Ns, "x")
    ax.set_xlabel("width $N$")
    ax.set_ylabel(r"lifetime $\tau_{1/2}$ (tasks)")
    fig.suptitle(title)
    P.caption(fig, r"$\tau_{1/2}$ ± jackknife s.e.; grey: proportional to $N$ through the first gradient-flow point.")
    P.fig_legend(fig, handles=P.model_handles(models, marker=True))
    return P.save(fig, path, figdata=figdata)


# ----------------------------------------------------------------------------
def jsonable(an: dict, hebb: Optional[dict]) -> dict:
    keep = ("tau", "tau_se", "eps0_mean", "eps0_sem", "plateau", "theta", "theta_raw", "rate", "rate_sem", "phi_tau", "phi_tau_sem")
    out = {"hebb": hebb, "thresholds": {"max_dev": X.TOL_M, "tau": X.TOL_TAU, "n_sem": X.N_SEM}, "conditions": {}}
    for c, a in an.items():
        out["conditions"][c.stem()] = {
            "cond": dataclasses.asdict(c), "t_fork": a["t_fork"], "drift": a["drift"], "n_seeds": a["n_seeds"],
            "models": {m: {**{k: d[k] for k in keep if k in d}, "M0": float(d["M_mean"][0]), "fit": d["fit"]}
                       for m, d in a["models"].items()},
            "compare": {f"{A}|{B}": k for (A, B), k in a["compare"].items()}}
    return out


def main(argv: Optional[list] = None) -> int:
    """Command-line entry point (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--N", type=int, default=256)
    ap.add_argument("--D-factor", type=int, default=2, help="D = D_factor·N (default 2: D = 512 at N = 256)")
    ap.add_argument("--delta0", type=float, nargs="+", default=[0.0, 1.0, 10.0], help="imbalance, rich (0) to lazy")
    ap.add_argument("--overlap", type=float, nargs="+", default=[0.1, 0.5],
                    help="gate density p: the fraction of units two tasks share (κ = p√N)")
    ap.add_argument("--rho", type=float, nargs="+", default=[0.0, 0.7], help="teacher similarity")
    ap.add_argument("--lam", type=float, default=1.0, help="hidden-layer learning per task λ of the main grid")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--models", nargs="+", default=list(X.MODELS), choices=X.MODELS,
                    help="models to run; gradient flow always runs (it makes the fork)")
    ap.add_argument("--latent", choices=["common", "sphere"], default="common",
                    help="teachers at ρ > 0: 'common' = one shared direction, cosine ρ between any two teachers; "
                         "'sphere' = the write-up's z_k uniform on the unit sphere of R^rank")
    ap.add_argument("--rank", type=int, default=4, help="latent rank r for --latent sphere")
    ap.add_argument("--ref-delta0", type=float, default=1.0)
    ap.add_argument("--ref-overlap", type=float, default=0.1)
    ap.add_argument("--ref-rho", type=float, default=0.0)
    ap.add_argument("--ref-lam", type=float, nargs="+", default=[1.0, 5.0])
    ap.add_argument("--ref-N", type=int, nargs="+", default=[128, 256, 512])
    ap.add_argument("--ref-D-factor", type=int, nargs="+", default=[2, 4], help="D/N of the D check at the main N")
    ap.add_argument("--no-ref-checks", action="store_true", help="skip the λ, N and D checks at the reference condition")
    ap.add_argument("--clock", choices=["realized", "nominal"], default="realized",
                    help="task clock: 'realized' = (ii), Λ_c = λ/ā²_c, so θ = 1 − e^{−λ} at every δ₀ (default); "
                         "'nominal' = (i), Λ = λ/(1 + δ₀)")
    ap.add_argument("--gf-step", type=float, default=0.02, help="hηā² of a gradient-flow step")
    ap.add_argument("--sgd-eta", type=float, default=0.05, help="ηā² of SGD; n = λD/this steps per task")
    ap.add_argument("--burn", type=float, default=10.0, help="minimum burn-in, in lifetime units (the half-life of eq. S_learned)")
    ap.add_argument("--burn-max", type=float, default=15.0, help="longest burn-in, in lifetimes")
    ap.add_argument("--learn", type=float, default=1.0, help="tasks probed after the fork, in lifetimes")
    ap.add_argument("--tau-max", type=float, default=3.0, help="largest age, in lifetimes")
    ap.add_argument("--ages", type=int, default=96, help="geometric ages beyond the first 32")
    ap.add_argument("--backend", choices=["auto", "numpy", "jax"], default="auto",
                    help="how SGD and gradient flow are stepped: 'jax' = the seeds of a condition as one vmap batch "
                         "on the device (the GPU path); 'numpy' = one run per CPU process; 'auto' = jax with --gpus, "
                         "else numpy")
    ap.add_argument("--reuse", action="store_true", help="read back the raw runs of an earlier call with the same settings")
    ap.add_argument("--out", default=os.path.join(_path.ROOT, "results", "nested"))
    pool.add_args(ap)
    args = ap.parse_args(argv)

    models = tuple(m for m in X.MODELS if m in set(args.models) | {"gf"})
    st = X.Settings(gf_step=args.gf_step, sgd_contraction=args.sgd_eta, burn=args.burn, burn_max=args.burn_max,
                    learn=args.learn, tau_max=args.tau_max, n_ages=args.ages, clock=args.clock,
                    backend=("jax" if args.gpus is not None else "numpy") if args.backend == "auto" else args.backend,
                    models=models)
    conds, ref, checks = conditions(args)
    seeds = list(range(args.seeds))
    reused, jobs = fork_jobs(conds, seeds, st, os.path.join(args.out, "raw"), args.reuse, int(args.split or 1))
    place, place_cpu = placement(args, len(jobs))
    for c in conds:
        if c.D < 2 * c.N:
            print(f"note: D = {c.D} < 2N at {c.label()}", flush=True)
    where = (f"GPUs {place['gpus']} × {place['per_gpu']} process(es)" if place["gpus"]
             else f"{max(place['workers'], 1)} CPU process(es)")
    print(f"nested-model test: {len(conds)} conditions × {len(seeds)} seeds, models {list(models)}, {st.clock} clock, "
          f"{st.backend} backend, {len(jobs)} training jobs ({len(reused)} runs reused)  [{where}]", flush=True)
    if st.backend == "numpy" and place["gpus"]:
        print("note: the numpy backend does not use the GPUs; drop --backend numpy to run on them", flush=True)
    for c in conds:
        T_learn, T, ages = st.schedule(c)
        n_sgd = int(round(c.lam * c.D / st.sgd_contraction))
        print(f"  {c.label()}: lifetime unit {c.unit()}, {T} tasks after the fork ({T_learn} probed, {len(ages)} ages to "
              f"{ages[-1]})" + (f", SGD {n_sgd} steps/task" if "sgd" in models else ""), flush=True)

    forks = run_forks(conds, seeds, reused, jobs, place)
    hebb, hebbs = None, {}
    if "hebb" in models:
        hebb = tune(forks, ref, st, place_cpu)
        if hebb:
            print(f"Hebb tuned at {ref.label()}: b = {hebb['b']:.4g}, γ = {hebb['gamma']:.4g} → ε(0) {hebb['eps0']:.3g} "
                  f"(target {hebb['target_eps0']:.3g}), τ½ {hebb['tau']:.4g} (target {hebb['target_tau']:.4g})", flush=True)
            hebbs = {c: r for (c, _), r in run_hebbs(forks, conds, [hebb["gamma"]], st, place_cpu).items()}

    an = {c: analyse(c, forks[c], hebbs.get(c), hebb["b"] if hebb else 1.0) for c in conds}
    for c, a in an.items():
        print(f"{c.label()}  (fork at {a['t_fork']}, drift {a['drift']:.3f})")
        for m, d in a["models"].items():
            print(f"  {P.MODEL_LABEL[m]:14s} ε(0) {d['eps0_mean']:.4f}  M(0) {d['M_mean'][0]:.3f}  τ½ {f3(d['tau'])} ± {f3(d['tau_se'])}"
                  f"  shape {d['fit']['best']}" + (f"  φ_key {d['phi_tau']:.3f}" if "phi_tau" in d else "")
                  + (f"  ϱ {d['rate']:.3f}" if m in X.LEARNED else ""))
        for (A, B), k in a["compare"].items():
            print(f"  {P.MODEL_LABEL[A]} vs {P.MODEL_LABEL[B]}: max dev {k['max_dev']:.3f} at τ = {k['age_dev']}, "
                  f"Δτ½ {f3(k['dtau'])} → {verdict(k)}", flush=True)
    print("\n".join(decisions(an, ref, checks)), flush=True)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "nested_report.md"), "w") as f:
        f.write(report(an, ref, checks, hebb, st, args))
    with open(os.path.join(args.out, "nested_summary.json"), "w") as f:
        json.dump(jsonable(an, hebb), f, indent=1, default=lambda o: None if isinstance(o, float) and not np.isfinite(o) else str(o))
    arrays = {}
    for c, a in an.items():
        arrays[f"{c.stem()}_ages"] = a["ages"]
        for m, d in a["models"].items():
            arrays.update({f"{c.stem()}_{m}_{k}": d[k] for k in ("M", "E", "S", "Ev", "phi") if d.get(k) is not None})
            arrays[f"{c.stem()}_{m}_Einf"] = d["E_inf"]
    np.savez_compressed(os.path.join(args.out, "nested.npz"), **arrays)
    print(f"wrote {os.path.join(args.out, 'nested_report.md')}, nested_summary.json, nested.npz")

    P.apply_style()
    figs = os.path.join(args.out, "figures")
    head = f"N = {args.N}, D = {args.D_factor * args.N}, λ = {args.lam:g}"
    print("wrote", fig_curves(an, checks["main"], os.path.join(figs, "nested_curves"), P.mathify(f"Memory curves of the model ladder: {head}")))
    print("wrote", fig_summary(an, checks["main"], os.path.join(figs, "nested_summary"), P.mathify(f"Lifetime, error, key drift and rate: {head}")))
    extra = list(dict.fromkeys(checks["lam"] + checks["N"] + checks["D"]))
    if extra:
        rt = P.mathify(f"Reference condition: p = {ref.p:g}, ρ = {ref.rho:g}, δ₀ = {ref.delta0:g}")
        print("wrote", fig_reference(an, extra, os.path.join(figs, "nested_reference"), rt))
        if len(checks["N"]) >= 2:
            print("wrote", fig_tau_N(an, checks["N"], os.path.join(figs, "nested_tau_vs_N"), rt))
    return 0


if __name__ == "__main__":
    _rc = main()
    _dt = time.perf_counter() - _T0
    print(f"run_nested: total {_dt:.1f} s ({_dt / 60:.2f} min) from after the imports to the end", flush=True)
    sys.exit(_rc)
