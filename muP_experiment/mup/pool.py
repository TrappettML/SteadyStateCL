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
# command-line placement shared by the scripts: ``--gpus`` alone plans the rest
# ----------------------------------------------------------------------------
#: The GPU node: four cards (``--gpus`` with no ids and no ``nvidia-smi`` answer).
GPU_COUNT = 4
#: Device memory per card when ``nvidia-smi`` cannot be asked (Tesla V100-SXM2-32GB).
GPU_MEM_GB = 32.0
#: Processes per card at most.  The kernels here are tiny (N ≤ a few hundred)
#: and launch-latency bound, so a card is idle most of the time and several
#: processes time-slice into the gaps; beyond three they contend for the host
#: and compile together.
MAX_PER_GPU = 3
#: Device bytes of a job per byte of its batched state: the scan keeps the
#: state, its donated copy and XLA temporaries of the vmap'd step.
MEM_FACTOR = 4
#: Chunk sizes tried, largest first, for the tasks per device call.
CHUNKS = (64, 32, 16)
#: Device bytes the stacked per-chunk observable may take.
OUT_BUDGET = 512 * 2 ** 20


def add_args(ap: argparse.ArgumentParser, workers_default: int = 0) -> None:
    """Add ``--gpus`` and the optional overrides ``--per-gpu``, ``--workers``, ``--chunk``, ``--split``."""
    g = ap.add_argument_group("placement")
    g.add_argument("--gpus", "--gpu", nargs="*", default=None,
                   help="run on the GPUs: no ids (or 'auto') for every card of the node, or a list of ids; "
                        "absent: the CPU.  Processes per card, chunk and the split of each batch are planned "
                        "from the runs' sizes and the cards' memory")
    g.add_argument("--per-gpu", type=int, default=None, help=f"override: processes per GPU (planned ≤ {MAX_PER_GPU})")
    g.add_argument("--workers", type=int, default=workers_default,
                   help=f"CPU processes when there is no GPU (default {workers_default}; ≤ 1: in-process)")
    g.add_argument("--chunk", type=int, default=None, help="override: tasks per device call (planned 16–64 on a GPU, 1 on the CPU)")
    g.add_argument("--split", type=int, default=None,
                   help="override: pieces every group of runs is split into (planned: as few as fill the workers)")


class Plan:
    """Where the jobs of a script go and how each group of runs is cut (``plan``).

    Attributes
    ----------
    gpus : list of int or None
    per_gpu, workers, chunk : int
    pieces : list of int
        Pieces per group, aligned with the ``groups`` given to ``plan``.
    mem_gb : float
        Device memory per card assumed (GPU placement).
    """

    def __init__(self, gpus, per_gpu, workers, chunk, pieces, mem_gb=0.0):
        self.gpus, self.per_gpu, self.workers, self.chunk, self.pieces, self.mem_gb = gpus, per_gpu, workers, chunk, list(pieces), mem_gb

    @property
    def place(self) -> dict:
        """``run_jobs`` keyword arguments."""
        return {"gpus": self.gpus, "per_gpu": self.per_gpu, "workers": self.workers, "chunk": self.chunk}

    @property
    def slots(self) -> int:
        return n_slots(self.gpus, self.per_gpu, self.workers)

    def describe(self) -> str:
        """One line for a script's banner."""
        if self.gpus:
            where = f"GPUs {self.gpus} × {self.per_gpu} process(es), {self.mem_gb:.0f} GB/card, chunk {self.chunk}"
        else:
            w = max(self.workers, 1)
            where = f"{w} CPU process(es)" if w > 1 else "in-process"
        pieces = sorted(set(self.pieces))
        if pieces and pieces != [1]:
            where += f", groups split into {pieces[0]}" + (f"–{pieces[-1]}" if len(pieces) > 1 else "") + " piece(s)"
        return where


def job_bytes(runs: Sequence, T: int, chunk: int, out_per_task: float) -> int:
    """Device bytes a job of ``runs`` over ``T`` tasks needs: the batched state (``protocol.state_bytes``
    with ``T`` head slots) times ``MEM_FACTOR``, plus the stacked observable of one chunk."""
    from mup.protocol import state_bytes
    r = runs[0]
    P = max(T, r.T_burn + r.T_meas)
    per_run = state_bytes(r, 1) + 8 * r.N * (P - (r.T_burn + r.T_meas))
    return int(MEM_FACTOR * per_run * len(runs) + chunk * len(runs) * out_per_task)


def plan(args: argparse.Namespace, groups: Sequence[tuple], out_per_task: float = 8.0, split: bool = True) -> Plan:
    """
    Placement and split of a script's groups of runs from the ``add_args`` options.

    With ``--gpus`` the cards are the ids given, else ``driver.visible_gpus()``,
    else ``GPU_COUNT``; their memory comes from ``nvidia-smi`` or ``GPU_MEM_GB``.
    A job's wall time is set by its sequential steps (tasks × n_steps), not by
    how many runs it batches, so the plan keeps batches whole and fills the
    workers with distinct groups: ``per_gpu`` processes per card (as many as
    ``MAX_PER_GPU`` and the memory allow, no more than there are runs to
    occupy them), a group split only when there are fewer groups than workers
    or its batch does not fit a process's share of the memory, and the largest
    chunk whose stacked observable stays under ``OUT_BUDGET``.  ``--per-gpu``,
    ``--chunk`` and ``--split`` override the respective choices.

    Parameters
    ----------
    args : argparse.Namespace
    groups : sequence of (runs, T)
        The script's groups: each a list of ``config.Run`` sharing
        ``experiments.group_key`` that one job (or its pieces) trains for ``T`` tasks.
    out_per_task : float
        Bytes of the job's observable per run per task (8 for a scalar signal).
    split : bool
        False when every group is one job whatever its size (fixed batches:
        ``validate.py``, ``run_gauge_comp.py``); a batch beyond a process's
        memory share is then an error.

    Returns
    -------
    Plan
    """
    groups = [(list(runs), int(T)) for runs, T in groups if len(runs)]
    B = [len(r) for r, _ in groups]
    B_tot = sum(B)
    if args.gpus is None:
        workers = max(int(args.workers or 0), 1)
        slots = workers
        gpus, per_gpu, mem_slot, mem_gb, chunk = None, 1, float("inf"), 0.0, int(args.chunk or 1)
    else:
        ids = [int(g) for g in args.gpus if str(g).lstrip("-").isdigit()]
        gpus = ids or driver.visible_gpus() or list(range(GPU_COUNT))
        mem_gb = driver.gpu_memory_gb() or GPU_MEM_GB
        per_gpu = int(args.per_gpu) if args.per_gpu else min(MAX_PER_GPU, max(1, -(-B_tot // len(gpus))))
        single = max((job_bytes(r[:1], T, CHUNKS[-1], out_per_task) for r, T in groups), default=0)
        while not args.per_gpu and per_gpu > 1 and single > driver.GPU_MEM_SHARE * mem_gb * 2 ** 30 / per_gpu:
            per_gpu -= 1
        mem_slot = driver.GPU_MEM_SHARE * mem_gb * 2 ** 30 / per_gpu
        if single > mem_slot:
            raise SystemExit(f"one run needs {single / 2**30:.1f} GB on the device, more than a process's "
                             f"{mem_slot / 2**30:.1f} GB share of a {mem_gb:.0f} GB card")
        workers, slots = 0, len(gpus) * per_gpu
        chunk = int(args.chunk) if args.chunk else 0
    pieces = []
    for (runs, T), b in zip(groups, B):
        need = 1
        while job_bytes(runs[: -(-b // need)], T, CHUNKS[-1], out_per_task) > mem_slot:
            need += 1
        if not split:
            if need > 1:
                raise SystemExit(f"a batch of {b} runs needs {job_bytes(runs, T, CHUNKS[-1], out_per_task) / 2**30:.1f} GB "
                                 f"on the device, more than a process's {mem_slot / 2**30:.1f} GB share; use a smaller batch")
            pieces.append(1)
            continue
        fill = -(-slots * b // B_tot) if len(groups) < slots else 1
        want = max(need, int(args.split) if args.split else fill)
        pieces.append(min(b, want))
    if gpus and not chunk:
        B_max = max((-(-b // k) for b, k in zip(B, pieces)), default=1)
        T_max = max((T for _, T in groups), default=1)
        chunk = next((c for c in CHUNKS if c * B_max * out_per_task <= OUT_BUDGET and c <= T_max), CHUNKS[-1])
    return Plan(gpus, per_gpu, workers, chunk, pieces, mem_gb)


if __name__ == "__main__":
    sys.exit(_worker(sys.argv[1]))
