#!/usr/bin/env python
"""Check the GPU node for the sweep: devices, float64, the kernels on every GPU at once.

    source ~/opt/jax-env.sh
    python old/gpu_check.py                 # devices, x64, one task of the pilot batch per GPU
    python old/gpu_check.py --bench         # bench.py (B × chunk table) on every GPU concurrently
    python old/gpu_check.py --bench --N 100 200 --B 64 256 1024 --chunk 1 16

Without ``--bench`` the script runs in this process: it lists the devices,
checks that arrays default to float64, times a float64 matmul on each GPU,
and runs one chunk of the pilot batch on each GPU through
``protocol.batched_run_tasks`` (the same kernel ``run_sweep.py`` uses),
comparing every GPU's result with the CPU's.  With ``--bench`` it starts
one ``old/bench.py`` subprocess per GPU (``CUDA_VISIBLE_DEVICES=i``,
the environment of ``sscl_core.driver.worker_env``) so all cards run at the same
time, and prints each table.  Exit status is nonzero when a check fails.
"""
import argparse
import os
import subprocess
import sys
import time

import _path  # noqa: F401

from sscl_core import driver

HERE = os.path.dirname(os.path.abspath(__file__))


def check_inprocess(N: int, B: int, chunk: int, self_test: bool = False) -> int:
    """
    Device listing, x64, matmul and one kernel chunk per GPU; returns the number of failures.

    Parameters
    ----------
    N, B, chunk : int
        Width, batch size and tasks per call of the kernel check.
    self_test : bool
        Run the per-GPU checks against the CPU device itself (exercises the
        script on a machine without a GPU; the backend check is skipped).
    """
    import jax
    import jax.numpy as jnp
    import numpy as np

    from sscl_core import device
    from sscl_core.config import Config, make_runs
    from sscl_core.observe import age_schedule
    from sscl_core.protocol import batched_run_tasks, init_batch

    fails = 0

    def report(name, ok, detail=""):
        nonlocal fails
        fails += (not ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""), flush=True)

    print(f"jax {jax.__version__}  backend: {device.describe()}")
    for d in jax.devices():
        print(f"  {d}  kind={getattr(d, 'device_kind', '?')}")
    report("x64 default dtype", jnp.zeros(1).dtype == jnp.float64, str(jnp.zeros(1).dtype))
    gpus = [d for d in jax.devices() if d.platform in ("gpu", "cuda", "rocm")]
    if self_test:
        gpus = [jax.devices("cpu")[0]]
        print("  --self-test: using the CPU in place of the GPUs")
    else:
        report("GPU backend", bool(gpus), f"{len(gpus)} device(s)")
    if not gpus:
        return fails

    n = 4096
    for d in gpus:
        x = jax.device_put(jnp.ones((n, n), dtype=jnp.float64), d)
        (x @ x).block_until_ready()
        t0 = time.perf_counter(); y = (x @ x).block_until_ready(); dt = time.perf_counter() - t0
        ok = y.dtype == jnp.float64 and float(y[0, 0]) == n
        report(f"float64 matmul on {d}", ok, f"{2 * n**3 / dt / 1e12:.2f} TFLOP/s, y[0,0]={float(y[0, 0])}")

    cfg = Config(N=[N], readout=["gated_shared", "learned_shared", "gated_frozen", "frozen", "task_heads"],
                 delta0=[0.0], lam=[1.0], seeds=max(1, B // 5), T_min=8)
    from conftest_free import one_batch
    runs = one_batch(make_runs(cfg))
    r0 = runs[0]
    T = r0.T_burn + r0.T_meas
    ages = jnp.asarray(age_schedule(N, r0.kappa_s, T, r0.n_ages_geom, r0.m_max_factor))
    f = batched_run_tasks(r0.n_steps, r0.solver, False, chunk)
    ref = None
    for d in [jax.devices("cpu")[0]] + gpus:
        st, p = init_batch(runs)
        st, p, a = jax.device_put((st, p, ages), d)
        t0 = time.perf_counter()
        st, out = f(st, p, a)
        out["E0"].block_until_ready()
        t_first = time.perf_counter() - t0
        t0 = time.perf_counter()
        st, out = f(st, p, a)
        out["E0"].block_until_ready()
        dt = time.perf_counter() - t0
        E0 = np.asarray(out["E0"])
        if ref is None:
            ref = E0
            print(f"  CPU reference: {len(runs)} runs × {chunk} tasks × {r0.n_steps} steps: {dt:.3f} s/chunk (compile {t_first:.1f} s)")
            continue
        err = float(np.nanmax(np.abs(E0 - ref)))
        report(f"kernel chunk on {d}", np.isfinite(E0).all() and err < 1e-8,
               f"{dt * 1e3:.1f} ms/chunk = {chunk * r0.n_steps * len(runs) / dt:.0f} run-steps/s, |ΔE0| vs CPU {err:.1e}")
    return fails


def bench_all_gpus(argv_rest: list[str]) -> int:
    """One ``bench.py`` subprocess per visible GPU, concurrently; returns the number that failed."""
    gpus = driver.visible_gpus()
    if not gpus:
        print("no GPU visible (CUDA_VISIBLE_DEVICES / nvidia-smi)")
        return 1
    procs = []
    for s, g in enumerate(gpus):
        env = driver.worker_env(s, gpus, 1, 1)
        cmd = [sys.executable, os.path.join(HERE, "bench.py")] + argv_rest
        procs.append((g, subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)))
    print(f"bench.py on GPUs {gpus} concurrently …", flush=True)
    fails = 0
    for g, pr in procs:
        out, _ = pr.communicate()
        fails += pr.returncode != 0
        print(f"\n=== GPU {g} (rc={pr.returncode}) ===\n{out.rstrip()}")
    return fails


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", action="store_true", help="run old/bench.py on every GPU at once")
    ap.add_argument("--N", type=int, nargs="+", default=[100])
    ap.add_argument("--B", type=int, nargs="+", default=[64, 256, 1024])
    ap.add_argument("--chunk", type=int, nargs="+", default=[1, 16])
    ap.add_argument("--solver", default="pop")
    ap.add_argument("--tasks", type=int, default=32)
    ap.add_argument("--self-test", action="store_true", help="run the GPU checks on the CPU (no GPU needed)")
    args = ap.parse_args(argv)
    if args.bench:
        rest = ["--N", *map(str, args.N), "--B", *map(str, args.B), "--chunk", *map(str, args.chunk),
                "--solver", args.solver, "--tasks", str(args.tasks)]
        return 1 if bench_all_gpus(rest) else 0
    fails = check_inprocess(args.N[0], args.B[0], args.chunk[-1], args.self_test)
    print(f"{fails} failure(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
