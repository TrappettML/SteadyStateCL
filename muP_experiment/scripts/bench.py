#!/usr/bin/env python
"""Steps/s vs batch size B at fixed N on this machine (PLAN.md §8, step 1).

    python scripts/bench.py --N 20 100 --B 1 8 32 64 128 [--solver pop|sgd] [--chunk 1 16]

Run on the Xeon with the thread-pinning environment of mup.driver.PINNED_ENV to
choose B and the worker count.  On a GPU (``CUDA_VISIBLE_DEVICES=0``) sweep B
up to ~1024 and ``--chunk`` (tasks per device call) to pick ``--batch-size``
and ``--chunk`` for ``run_sweep.py``; ``scripts/gpu_check.py --bench`` runs
this on every GPU at once.
"""
import argparse
import os
import sys
import time

import _path  # noqa: F401

import jax.numpy as jnp

from mup import device
from mup.config import Config, make_runs
from mup.observe import age_schedule
from mup.protocol import batched_run_task, batched_run_tasks, init_batch


def bench(N, B, solver, n_tasks, eta, chunk=1):
    """
    Time ``run_task`` for one batch shape after a warm-up compile.

    Parameters
    ----------
    N : int
    B : int
        Batch size (as seeds of one condition).
    solver : {"pop", "sgd"}
    n_tasks : int
        Tasks timed after the compile call.
    eta : float
    chunk : int
        Tasks per device call (``batched_run_tasks``); 1 times ``batched_run_task``.

    Returns
    -------
    (run_steps_per_s, s_per_task, compile_s, n_steps)
    """
    cfg = Config(N=[N], readout=["gated_shared"], delta0=[0.0], lam=[1.0], seeds=B, solver=solver, eta=[eta], T_min=8)
    runs = make_runs(cfg)
    T = runs[0].T_burn + runs[0].T_meas
    ages = jnp.asarray(age_schedule(N, runs[0].kappa, T))
    st, p = init_batch(runs)
    f = batched_run_task(runs[0].n_steps, solver) if chunk == 1 else batched_run_tasks(runs[0].n_steps, solver, False, chunk)
    t0 = time.time(); st, out = f(st, p, ages); out["E0"].block_until_ready(); t_compile = time.time() - t0
    n_calls = max(1, n_tasks // chunk)
    t0 = time.time()
    for _ in range(n_calls):
        st, out = f(st, p, ages)
    out["E0"].block_until_ready()
    dt = time.time() - t0
    done = n_calls * chunk
    steps = done * runs[0].n_steps * B
    return steps / dt, dt / done, t_compile, runs[0].n_steps


def main(argv=None):
    """
    Command-line entry point: prints a table of run-steps/s vs (N, B).

    Parameters
    ----------
    argv : list of str, optional

    Returns
    -------
    int
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, nargs="+", default=[20, 100])
    ap.add_argument("--B", type=int, nargs="+", default=[1, 8, 32, 64, 128])
    ap.add_argument("--solver", default="pop")
    ap.add_argument("--eta", type=float, default=None)
    ap.add_argument("--tasks", type=int, default=20, help="tasks timed per shape (rounded down to whole chunks)")
    ap.add_argument("--chunk", type=int, nargs="+", default=[1], help="tasks per device call")
    args = ap.parse_args(argv)
    eta = args.eta if args.eta is not None else (1.0 if args.solver == "pop" else 0.05)
    print(f"backend: {device.describe()}  CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')!r}")
    print(f"XLA_FLAGS={os.environ.get('XLA_FLAGS', '')!r}  OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', '')!r}")
    print(f"{'N':>5} {'B':>5} {'chunk':>5} {'n_steps':>8} {'run-steps/s':>12} {'s/task':>8} {'compile s':>10}")
    for N in args.N:
        for B in args.B:
            for chunk in args.chunk:
                sps, spt, tc, ns = bench(N, B, args.solver, max(args.tasks, chunk), eta, chunk)
                print(f"{N:5d} {B:5d} {chunk:5d} {ns:8d} {sps:12.0f} {spt:8.4f} {tc:10.2f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
