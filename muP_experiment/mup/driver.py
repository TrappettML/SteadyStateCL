"""Multiprocessing launcher for the Xeon and the GPU node (PLAN.md §0.5, §8).

Launches ``scripts/run_sweep.py --batch-id i`` subprocesses from the sweep
manifest with a bounded pool.  Each batch writes its own ``.npz`` so the
sweep is resumable: rerunning skips finished batches.

Two placements:

CPU (``gpus=None``)
    One thread-pinned process per worker slot, optionally bound to a NUMA
    node with ``numactl`` (the Xeon).
GPU (``gpus=[0, 1, 2, 3]``)
    ``per_gpu`` processes per listed GPU; slot ``s`` sees only
    ``CUDA_VISIBLE_DEVICES=gpus[s % len(gpus)]``, so every process holds one
    batch on one device and no collectives are needed.  JAX preallocates
    a fraction of the device memory per process, so the fraction is set to
    share the card between the ``per_gpu`` processes.  Batches are launched
    largest first so the long ones do not trail at the end.
"""
from __future__ import annotations

import os
import queue
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from mup import io

#: Thread pinning for CPU workers (PLAN.md decision 5): XLA and BLAS single-threaded.
PINNED_ENV = {
    "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "JAX_PLATFORMS": "cpu",
}

#: Host-side threads of a GPU worker: the Python loop and the NumPy moment
#: sums are tiny, so one thread each keeps ``per_gpu`` processes from fighting.
GPU_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "JAX_PLATFORMS": "cuda,cpu",
    "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
}
#: Share of a GPU's memory the ``per_gpu`` processes may preallocate between them.
GPU_MEM_SHARE = 0.9

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_SWEEP = os.path.join(os.path.dirname(HERE), "scripts", "run_sweep.py")


def visible_gpus() -> list[int]:
    """
    GPU ids on this host, from ``CUDA_VISIBLE_DEVICES`` or ``nvidia-smi`` (empty when neither says).

    Returns
    -------
    list of int
    """
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env is not None:
        return [int(x) for x in env.split(",") if x.strip().lstrip("-").isdigit() and int(x) >= 0]
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=30).stdout
            return [int(x) for x in out.split() if x.strip().isdigit()]
        except (subprocess.SubprocessError, ValueError):
            return []
    return []


def _command(config_path: str, out_dir: str, batch_id: int, numa_node: Optional[int]) -> list[str]:
    """
    Command line of one batch subprocess, optionally bound to a NUMA node.

    Parameters
    ----------
    config_path, out_dir : str
    batch_id : int
    numa_node : int or None

    Returns
    -------
    list of str
    """
    cmd = [sys.executable, RUN_SWEEP, "--config", config_path, "--out", out_dir, "--batch-id", str(batch_id)]
    if numa_node is not None and shutil.which("numactl"):
        cmd = ["numactl", f"--cpunodebind={numa_node}", f"--membind={numa_node}"] + cmd
    return cmd


def worker_env(slot: int, gpus: Optional[list[int]], per_gpu: int, threads_per_worker: int,
               chunk: Optional[int] = None) -> dict:
    """
    Environment of the subprocess in worker slot ``slot``.

    Parameters
    ----------
    slot : int
        0 … workers − 1; on a GPU node slot ``s`` uses ``gpus[s % len(gpus)]``.
    gpus : list of int or None
        None: CPU placement (``PINNED_ENV``).
    per_gpu : int
        Processes sharing one GPU (sets the preallocation fraction).
    threads_per_worker : int
        XLA/BLAS threads of a CPU worker (1 recommended, PLAN.md decision 5).
    chunk : int, optional
        Tasks per device call (``MUP_CHUNK``); None keeps the backend default.

    Returns
    -------
    dict
        A full environment (a copy of ``os.environ`` with the overrides).
    """
    env = dict(os.environ)
    if gpus:
        env.update(GPU_ENV)
        env["CUDA_VISIBLE_DEVICES"] = str(gpus[slot % len(gpus)])
        env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = f"{GPU_MEM_SHARE / max(per_gpu, 1):.3f}"
    else:
        env.update(PINNED_ENV)
        if threads_per_worker != 1:
            env["XLA_FLAGS"] = f"--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads={threads_per_worker}"
            env["OMP_NUM_THREADS"] = str(threads_per_worker)
    if chunk is not None and chunk > 0:
        env["MUP_CHUNK"] = str(int(chunk))
    return env


def batch_cost(entry: dict) -> float:
    """
    Rough device-time of a manifest batch entry: runs × N·D · steps per task · tasks.

    Parameters
    ----------
    entry : dict
        One item of ``manifest["batches"]`` (``n_runs``, ``N``, ``n_steps``, ``T``).

    Returns
    -------
    float
    """
    return float(entry["n_runs"]) * entry["N"] ** 2 * entry["n_steps"] * entry["T"]


def order_largest_first(out_dir: str, ids: list[int]) -> list[int]:
    """
    ``ids`` sorted by ``batch_cost`` descending (stable), from the manifest in ``out_dir``.

    Parameters
    ----------
    out_dir : str
    ids : list of int

    Returns
    -------
    list of int
    """
    man = io.read_manifest(out_dir)
    cost = {b["id"]: batch_cost(b) for b in man["batches"]}
    return sorted(ids, key=lambda i: -cost.get(i, 0.0))


def launch(
    config_path: str,
    out_dir: str,
    workers: int,
    numa_nodes: int = 0,
    threads_per_worker: int = 1,
    batch_ids: Optional[list[int]] = None,
    log=print,
    gpus: Optional[list[int]] = None,
    per_gpu: int = 1,
    chunk: Optional[int] = None,
) -> dict:
    """
    Run every pending batch of ``out_dir``'s manifest with ``workers`` subprocesses.

    Parameters
    ----------
    config_path : str
        YAML config the manifest was made from.
    out_dir : str
    workers : int
        Pool size.  0 with ``gpus``: ``len(gpus) * per_gpu``.
    numa_nodes : int
        CPU only: bind workers round-robin to this many NUMA nodes with ``numactl`` (0: no binding).
    threads_per_worker : int
        CPU only: XLA/BLAS threads per process (1 recommended, PLAN.md decision 5).
    batch_ids : list of int, optional
        Batches to run (default: the manifest's pending ones).
    log : callable
    gpus : list of int, optional
        GPU ids to spread the workers over (``visible_gpus()`` for all of them);
        None runs on the CPU.
    per_gpu : int
        Processes per GPU.  More than one overlaps the host-side bookkeeping of
        one batch with the kernels of another; the device memory is split
        between them (``GPU_MEM_SHARE / per_gpu`` each).
    chunk : int, optional
        Tasks per device call for the subprocesses (``MUP_CHUNK``).

    Returns
    -------
    dict
        ``done`` and ``failed`` lists of batch ids.
    """
    ids = batch_ids if batch_ids is not None else io.pending_batches(out_dir)
    if not ids:
        log("nothing to do")
        return {"done": [], "failed": []}
    gpus = list(gpus) if gpus else None
    if gpus:
        numa_nodes = 0
        if workers <= 0:
            workers = len(gpus) * max(per_gpu, 1)
        per_gpu = max(1, -(-workers // len(gpus)))       # ceil: processes that may share a card
    elif workers <= 0:
        raise ValueError("workers must be > 0 on the CPU")
    if numa_nodes > 0 and not shutil.which("numactl"):
        log("warning: numactl not found; running without NUMA binding")
        numa_nodes = 0
    ids = order_largest_first(out_dir, ids)
    envs = [worker_env(s, gpus, per_gpu, threads_per_worker, chunk) for s in range(workers)]
    slots: "queue.Queue[int]" = queue.Queue()
    for s in range(workers):
        slots.put(s)
    lock = threading.Lock()
    done, failed = [], []

    def worker(bid: int):
        """Run one batch in a free slot, logging to ``logs/batch_<id>.log``; returns (id, return code)."""
        slot = slots.get()
        try:
            node = (slot % numa_nodes) if numa_nodes > 0 else None
            cmd = _command(config_path, out_dir, bid, node)
            with lock:
                io.set_status(out_dir, bid, "running")
            logfile = os.path.join(out_dir, "logs", f"batch_{bid:05d}.log")
            os.makedirs(os.path.dirname(logfile), exist_ok=True)
            with open(logfile, "w") as lf:
                where = f"GPU {envs[slot]['CUDA_VISIBLE_DEVICES']}" if gpus else f"slot {slot}"
                lf.write(f"# batch {bid} on {where}\n")
                lf.flush()
                rc = subprocess.run(cmd, env=envs[slot], stdout=lf, stderr=subprocess.STDOUT).returncode
            with lock:
                io.set_status(out_dir, bid, "done" if rc == 0 else "failed")
            return bid, rc
        finally:
            slots.put(slot)

    where = f"{len(gpus)} GPUs {gpus} × {per_gpu} process(es)" if gpus else "CPU"
    log(f"launching {len(ids)} batches on {workers} workers ({where}, largest first)")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(worker, bid): bid for bid in ids}
        for fut in as_completed(futs):
            bid, rc = fut.result()
            (done if rc == 0 else failed).append(bid)
            log(f"batch {bid}: {'ok' if rc == 0 else f'FAILED rc={rc}'}  ({len(done)+len(failed)}/{len(ids)})")
    return {"done": sorted(done), "failed": sorted(failed)}
