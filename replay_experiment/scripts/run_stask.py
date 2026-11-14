#!/usr/bin/env python
"""Steady-state teacher signal against age, one trace per replay method: S_T(τ) = u_{T+τ}·w_T*/|w_T*|².

    python scripts/run_stask.py [--N 100] [--capacity 0.5D 4D] [--delta0 0] [--lam 4] [--seeds 4] [--T 1400] [--tau 200]
                                [--mode sleep|interleaved] [--stat mean|seed] [--gpus]

The replay counterpart of ``muP_experiment/scripts/run_stask.py`` and the companion of ``run_szero.py`` with
the probe taken at steady state: train T tasks (default the config's T_burn + T_meas = 14N), by which the
buffer has long been full, then follow the teacher of task T for τ = 0 … ``--tau`` more tasks.  One figure per
cell (N, C, δ₀), ``results/stask/figures/stask_N{N}_C{C}_d0{δ₀}_lam{λ}_{mode}.png``: x = τ + 1 (log), y =
S_T(τ)/S_T(0), one trace per replay method (``none`` dashed grey), ±1 s.e. band, half-lives in the caption; the
per-τ arrays ``ST_{method}`` (τ + 1, n_seeds) go to the ``.npz``.  A second figure per cell,
``stask_area_N{N}_…png``: the area under S_T(m)/S_T(0) from m = 0 to n against n (``run_szero.area_curve``), the
tasks' worth of the memory kept over the first n tasks, with ``area_{method}`` (τ + 1,) in its ``.npz``.

Placement as ``run_szero.py`` (``replay.signal_after_tasks`` with ``T_pre = T, k = T``).
"""
import argparse
import sys
import time
from typing import Optional

import _path  # noqa: F401

import common as C
from sscl_core import halflife as H
from run_szero import report_and_draw, signals

_T0 = time.perf_counter()        # wall clock from after the imports to the end of the script


def main(argv: Optional[list[str]] = None) -> int:
    """Command-line entry point (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    C.add_args(ap, "stask", N=[100], delta0_list=True)
    ap.add_argument("--T", type=int, default=None, help="tasks before the probed one (default: the config's T_burn + T_meas)")
    ap.add_argument("--tau", type=int, default=200, help="tasks followed after task T")
    H.add_arg(ap)
    args = ap.parse_args(argv)
    runs = C.build(args, "stask")
    T = C.stream_T(runs, args.T, "--T")
    r0 = runs[0]
    print(f"S_T(τ) vs τ: N={args.N} C={C.cap_labels(runs)} methods={C.methods_of(runs)} δ₀={args.delta0} λ={r0.lam} "
          f"{r0.solver} n_steps={r0.n_steps} T={T} τ≤{args.tau} seeds={args.seeds} statistic={args.stat}", flush=True)
    res = signals(runs, T, args.tau + 1, T, args)
    report_and_draw(res, runs, args.out, "stask", args.stat, T)
    return 0


if __name__ == "__main__":
    _rc = main()
    _dt = time.perf_counter() - _T0
    print(f"run_stask: total {_dt:.1f} s ({_dt / 60:.2f} min) from after the imports to the end", flush=True)
    sys.exit(_rc)
