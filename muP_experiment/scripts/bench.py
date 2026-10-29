#!/usr/bin/env python
"""Steps/s vs batch size B at fixed N on this machine (PLAN.md §8, step 1).

    python scripts/bench.py --N 20 100 --B 1 8 32 64 128 [--solver pop|sgd]

Run on the Xeon with the thread-pinning environment of mup.driver.PINNED_ENV to
choose B and the worker count.
"""
import argparse
import os
import sys
import time

import _path  # noqa: F401

import jax.numpy as jnp

from mup.config import Config, make_runs
from mup.observe import age_schedule
from mup.protocol import batched_run_task, init_batch


def bench(N, B, solver, n_tasks, eta):
    cfg = Config(N=[N], readout=["learned"], delta0=[0.0], lam=[1.0], seeds=B, solver=solver, eta=[eta], T_min=8)
    runs = make_runs(cfg)
    T = runs[0].T_burn + runs[0].T_meas
    ages = jnp.asarray(age_schedule(N, runs[0].kappa, T))
    st, p = init_batch(runs)
    f = batched_run_task(runs[0].n_steps, solver)
    t0 = time.time(); st, out = f(st, p, ages); out["E0"].block_until_ready(); t_compile = time.time() - t0
    t0 = time.time()
    for _ in range(n_tasks):
        st, out = f(st, p, ages)
    out["E0"].block_until_ready()
    dt = time.time() - t0
    steps = n_tasks * runs[0].n_steps * B
    return steps / dt, dt / n_tasks, t_compile, runs[0].n_steps


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, nargs="+", default=[20, 100])
    ap.add_argument("--B", type=int, nargs="+", default=[1, 8, 32, 64, 128])
    ap.add_argument("--solver", default="pop")
    ap.add_argument("--eta", type=float, default=None)
    ap.add_argument("--tasks", type=int, default=20)
    args = ap.parse_args(argv)
    eta = args.eta if args.eta is not None else (1.0 if args.solver == "pop" else 0.05)
    print(f"XLA_FLAGS={os.environ.get('XLA_FLAGS', '')!r}  OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', '')!r}")
    print(f"{'N':>5} {'B':>5} {'n_steps':>8} {'run-steps/s':>12} {'s/task':>8} {'compile s':>10}")
    for N in args.N:
        for B in args.B:
            sps, spt, tc, ns = bench(N, B, args.solver, args.tasks, eta)
            print(f"{N:5d} {B:5d} {ns:8d} {sps:12.0f} {spt:8.4f} {tc:10.2f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
