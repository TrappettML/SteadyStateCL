"""Independent jobs over the GPUs (one process per card), CPU processes, or in-process.

The experiment scripts (``run_szero.py``, ``run_stask.py``, ``run_Nscaling.py``,
``run_weight_norms.py``, ``run_gauge_comp.py``, ``validate.py``) each reduce to a
list of independent jobs — a cell, an (N, T) group of runs, or a sweep batch —
whose results are merged on the host.  ``run_jobs`` places them:

GPU (``gpus=[0, 1, 2, 3]``)
    ``per_gpu`` worker processes per card, each pinned to one device through
    ``driver.worker_env`` (``CUDA_VISIBLE_DEVICES``, memory fraction, chunk).
CPU (``gpus=None, workers > 1``)
    ``workers`` thread-pinned processes (``driver.PINNED_ENV``), as the sweep.
in-process (``gpus=None, workers <= 1``)
    The jobs run here, in order.

Jobs are claimed dynamically, largest first (``cost``), through a directory
of claim files, so a long job never trails behind short ones and a card that
finishes early takes the next job.  Arguments and results travel as pickles;
a job function is named ``"module:function"`` and must be importable in the
worker (``mup.experiments`` holds the ones of the scripts).  Workers are
``python -m mup.pool <dir>``.
"""
from __future__ import annotations

import argparse
import importlib
import os
import pickle
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Callable, Optional, Sequence

from mup import driver

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def resolve_func(name: str) -> Callable:
    """``"module:function"`` → the function."""
    mod, _, fn = name.partition(":")
    return getattr(importlib.import_module(mod), fn)


def n_slots(gpus: Optional[list[int]], per_gpu: int = 1, workers: int = 0) -> int:
    """Worker processes a placement starts: ``len(gpus) · per_gpu`` on GPUs, ``workers`` on the CPU (1 = in-process)."""
    if gpus:
        return len(gpus) * max(per_gpu, 1)
    return max(workers, 1)


def run_jobs(func: str, jobs: Sequence[tuple], gpus: Optional[list[int]] = None, per_gpu: int = 1,
             workers: int = 0, chunk: Optional[int] = None, log: Optional[Callable[[str], None]] = None,
             keep_dir: Optional[str] = None) -> list:
    """
    Run ``func(**kwargs)`` for every job and return the results in job order.

    Parameters
    ----------
    func : str
        ``"module:function"``; its return value must pickle (NumPy arrays, dicts).
    jobs : sequence of (key, kwargs, cost)
        ``key`` names the job in the log, ``cost`` orders the queue (largest first).
    gpus : list of int, optional
        GPU ids; None places on the CPU.
    per_gpu : int
        Processes per GPU.
    workers : int
        CPU processes when ``gpus`` is None; ≤ 1 runs in this process.
    chunk : int, optional
        Tasks per device call for the workers (``MUP_CHUNK``).
    log : callable, optional
        Progress sink.
    keep_dir : str, optional
        Spool directory to use (and keep) instead of a temporary one.

    Returns
    -------
    list
        One result per job, in the order of ``jobs``.

    Raises
    ------
    RuntimeError
        A job raised (its traceback is in the message) or a worker died.
    """
    jobs = list(jobs)
    if not jobs:
        return []
    slots = n_slots(gpus, per_gpu, workers)
    if not gpus and slots <= 1:
        if chunk:
            os.environ["MUP_CHUNK"] = str(int(chunk))
        f = resolve_func(func)
        out = []
        t0 = time.time()
        for i, (key, kw, _) in enumerate(jobs):
            t1 = time.time()
            out.append(f(**kw))
            if log:
                log(f"job {i + 1}/{len(jobs)} {key} done in {time.time() - t1:.1f}s (in-process, {time.time() - t0:.0f}s total)")
        return out

    order = sorted(range(len(jobs)), key=lambda i: -float(jobs[i][2]))
    tmp = keep_dir or tempfile.mkdtemp(prefix="mup_pool_")
    os.makedirs(tmp, exist_ok=True)
    with open(os.path.join(tmp, "spec.pkl"), "wb") as fh:
        pickle.dump({"func": func, "order": order, "keys": [str(j[0]) for j in jobs]}, fh)
    for i, (_, kw, _) in enumerate(jobs):
        with open(os.path.join(tmp, f"job_{i}.pkl"), "wb") as fh:
            pickle.dump(kw, fh)

    procs = []
    for s in range(slots):
        env = driver.worker_env(s, gpus, per_gpu, 1, chunk)
        where = f"GPU {gpus[s % len(gpus)]}" if gpus else f"slot {s}"
        env["MUP_POOL_WHERE"] = where
        logf = open(os.path.join(tmp, f"worker_{s}.log"), "w")
        procs.append((subprocess.Popen([sys.executable, "-m", "mup.pool", tmp], env=env, cwd=ROOT,
                                       stdout=logf, stderr=subprocess.STDOUT), logf, where))
    if log:
        log(f"{len(jobs)} jobs on {slots} worker{'s' if slots != 1 else ''} "
            f"({'GPUs ' + ','.join(map(str, gpus)) if gpus else 'CPU'}), largest first")

    seen = set()
    t0 = time.time()
    while True:
        alive = any(p.poll() is None for p, _, _ in procs)
        for i in range(len(jobs)):
            if i in seen:
                continue
            done = os.path.join(tmp, f"done_{i}.txt")
            if os.path.exists(done):
                seen.add(i)
                if log:
                    log(f"job {len(seen)}/{len(jobs)} {jobs[i][0]} {open(done).read().strip()} ({time.time() - t0:.0f}s total)")
        if not alive:
            break
        time.sleep(0.5)
    for _, logf, _ in procs:
        logf.close()

    results, errors = [], []
    for i in range(len(jobs)):
        res, err = os.path.join(tmp, f"res_{i}.pkl"), os.path.join(tmp, f"err_{i}.txt")
        if os.path.exists(res):
            with open(res, "rb") as fh:
                results.append(pickle.load(fh))
        elif os.path.exists(err):
            errors.append(f"job {jobs[i][0]} raised:\n{open(err).read()}")
            results.append(None)
        else:
            errors.append(f"job {jobs[i][0]} has no result (a worker died?)")
            results.append(None)
    dead = [(p.returncode, lf.name, where) for p, lf, where in procs if p.returncode != 0]
    if errors or dead:
        tails = []
        for rc, name, where in dead:
            with open(name) as fh:
                tails.append(f"worker on {where} exited {rc}; last output:\n{fh.read()[-2000:]}")
        raise RuntimeError("\n".join(errors + tails))
    if keep_dir is None:
        for name in os.listdir(tmp):
            os.unlink(os.path.join(tmp, name))
        os.rmdir(tmp)
    return results


def _worker(tmp: str) -> int:
    """Claim and run jobs from the spool directory ``tmp`` until none is left."""
    with open(os.path.join(tmp, "spec.pkl"), "rb") as fh:
        spec = pickle.load(fh)
    where = os.environ.get("MUP_POOL_WHERE", "worker")
    f = resolve_func(spec["func"])
    for i in spec["order"]:
        try:
            fd = os.open(os.path.join(tmp, f"claim_{i}"), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            continue
        os.close(fd)
        t0 = time.time()
        with open(os.path.join(tmp, f"job_{i}.pkl"), "rb") as fh:
            kw = pickle.load(fh)
        try:
            out = f(**kw)
            part = os.path.join(tmp, f"res_{i}.pkl.part")
            with open(part, "wb") as fh:
                pickle.dump(out, fh)
            os.replace(part, os.path.join(tmp, f"res_{i}.pkl"))
            status = f"done on {where} in {time.time() - t0:.1f}s"
        except Exception:                                       # noqa: BLE001 — reported to the parent
            with open(os.path.join(tmp, f"err_{i}.txt"), "w") as fh:
                fh.write(traceback.format_exc())
            status = f"FAILED on {where} after {time.time() - t0:.1f}s"
        with open(os.path.join(tmp, f"done_{i}.txt"), "w") as fh:
            fh.write(status)
        print(f"{spec['keys'][i]}: {status}", flush=True)
    return 0


# ----------------------------------------------------------------------------
# command-line placement shared by the scripts
# ----------------------------------------------------------------------------
def add_args(ap: argparse.ArgumentParser, workers_default: int = 0) -> None:
    """Add ``--gpus``, ``--per-gpu``, ``--workers``, ``--chunk`` and ``--split`` to a parser."""
    g = ap.add_argument_group("placement")
    g.add_argument("--gpus", nargs="*", default=["auto"],
                   help="GPU ids for one process each; 'auto' (default) uses every visible GPU, 'none' the CPU")
    g.add_argument("--per-gpu", type=int, default=1, help="processes per GPU (default 1)")
    g.add_argument("--workers", type=int, default=workers_default,
                   help=f"CPU processes when there is no GPU (default {workers_default}; ≤ 1: in-process)")
    g.add_argument("--chunk", type=int, default=None, help="tasks per device call (default 16 on a GPU, 1 on the CPU)")
    g.add_argument("--split", type=int, default=None,
                   help="pieces each group of runs is split into (default: the number of worker processes)")


def placement(args: argparse.Namespace) -> dict:
    """``run_jobs`` keyword arguments from ``add_args`` options; ``gpus`` resolved against the host."""
    gpus = None
    if args.gpus and args.gpus != ["none"]:
        gpus = driver.visible_gpus() if args.gpus == ["auto"] else [int(g) for g in args.gpus]
        gpus = gpus or None
    return {"gpus": gpus, "per_gpu": args.per_gpu, "workers": args.workers, "chunk": args.chunk}


def pieces(args: argparse.Namespace, place: dict) -> int:
    """Pieces per group of runs: ``--split`` or the number of worker processes."""
    return max(1, args.split or n_slots(place["gpus"], place["per_gpu"], place["workers"]))


def describe(place: dict) -> str:
    """One line naming the placement for a script's banner."""
    if place["gpus"]:
        return f"GPUs {place['gpus']} × {place['per_gpu']} process(es)"
    w = max(place["workers"], 1)
    return f"{w} CPU process(es)" if w > 1 else "in-process"


if __name__ == "__main__":
    sys.exit(_worker(sys.argv[1]))
