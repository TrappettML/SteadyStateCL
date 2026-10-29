"""Multiprocessing launcher for the Xeon (PLAN.md §0.5, §8).

Launches ``scripts/run_sweep.py --batch-id i`` subprocesses from the sweep
manifest with a bounded pool, thread-pinning environment variables and,
optionally, ``numactl`` binding by NUMA node.  Each batch writes its own
``.npz`` so the sweep is resumable: rerunning skips finished batches.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from mup import io

PINNED_ENV = {
    "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "JAX_PLATFORMS": "cpu",
}

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_SWEEP = os.path.join(os.path.dirname(HERE), "scripts", "run_sweep.py")


def _command(config_path: str, out_dir: str, batch_id: int, numa_node: Optional[int]) -> list[str]:
    cmd = [sys.executable, RUN_SWEEP, "--config", config_path, "--out", out_dir, "--batch-id", str(batch_id)]
    if numa_node is not None and shutil.which("numactl"):
        cmd = ["numactl", f"--cpunodebind={numa_node}", f"--membind={numa_node}"] + cmd
    return cmd


def launch(
    config_path: str,
    out_dir: str,
    workers: int,
    numa_nodes: int = 0,
    threads_per_worker: int = 1,
    batch_ids: Optional[list[int]] = None,
    log=print,
) -> dict:
    """Run every pending batch of ``out_dir``'s manifest with ``workers`` subprocesses."""
    ids = batch_ids if batch_ids is not None else io.pending_batches(out_dir)
    if not ids:
        log("nothing to do")
        return {"done": [], "failed": []}
    env = dict(os.environ)
    env.update(PINNED_ENV)
    if threads_per_worker != 1:
        env["XLA_FLAGS"] = f"--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads={threads_per_worker}"
        env["OMP_NUM_THREADS"] = str(threads_per_worker)
    lock = threading.Lock()
    done, failed = [], []

    def worker(k: int, bid: int):
        node = (k % numa_nodes) if numa_nodes > 0 else None
        cmd = _command(config_path, out_dir, bid, node)
        with lock:
            io.set_status(out_dir, bid, "running")
        logfile = os.path.join(out_dir, "logs", f"batch_{bid:05d}.log")
        os.makedirs(os.path.dirname(logfile), exist_ok=True)
        with open(logfile, "w") as lf:
            rc = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
        with lock:
            io.set_status(out_dir, bid, "done" if rc == 0 else "failed")
        return bid, rc

    log(f"launching {len(ids)} batches on {workers} workers")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(worker, k, bid): bid for k, bid in enumerate(ids)}
        for fut in as_completed(futs):
            bid, rc = fut.result()
            (done if rc == 0 else failed).append(bid)
            log(f"batch {bid}: {'ok' if rc == 0 else f'FAILED rc={rc}'}  ({len(done)+len(failed)}/{len(ids)})")
    return {"done": sorted(done), "failed": sorted(failed)}
