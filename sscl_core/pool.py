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

Every job is assigned to a card before the workers start: largest first,
each to the card with the least load so far (``assign``: the usual
longest-first greedy, which keeps the cards' loads within one job of each
other), so where the big jobs land no longer depends on which worker wins
the race.  Worker slot ``s`` serves card ``s % len(gpus)`` and claims its
card's jobs largest first through a directory of claim files; a worker
whose card has nothing left takes another card's remaining jobs (largest
first), so a card that finishes early still helps.  On the CPU every
slot is its own "card".  Arguments and results travel as pickles;
a job function is named ``"module:function"`` and must be importable in the
worker (``sscl_core.experiments`` holds the ones of the scripts).  Workers are
``python -m sscl_core.pool <dir>``.
"""
from __future__ import annotations

import argparse
import importlib
import math
import os
import pickle
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Callable, Optional, Sequence

from sscl_core import driver

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


# ----------------------------------------------------------------------------
# progress of the running job: tenths of its tasks, relayed to the script's console
# ----------------------------------------------------------------------------
_REPORT: Optional[Callable[[str], None]] = None     # set per job by run_jobs (in-process) or _worker (a file)


def report(msg: str) -> None:
    """Send a progress line of the running job to the console (in-process) or to the parent through the spool."""
    if _REPORT is not None:
        _REPORT(msg)


class Progress:
    """Reports a job's advance through ``total`` tasks at every tenth (and at the end), with the elapsed time.

    The runs of a job advance together in one ``vmap``, so a block of tasks
    is the unit that completes; ``add(n)`` after every device call of ``n``
    tasks is enough.
    """

    def __init__(self, total: int, label: str = "task", tenths: int = 10):
        self.total, self.label, self.tenths, self.done, self.next, self.t0 = max(int(total), 1), label, tenths, 0, 1, time.time()

    def add(self, n: int) -> None:
        self.done += int(n)
        while self.next <= self.tenths and self.done * self.tenths >= self.next * self.total:
            report(f"{self.label} {self.done}/{self.total} ({100 * self.next // self.tenths}%) {time.time() - self.t0:.0f}s")
            self.next += 1


def resolve_func(name: str) -> Callable:
    """``"module:function"`` → the function."""
    mod, _, fn = name.partition(":")
    return getattr(importlib.import_module(mod), fn)


def n_slots(gpus: Optional[list[int]], per_gpu: int = 1, workers: int = 0) -> int:
    """Worker processes a placement starts: ``len(gpus) · per_gpu`` on GPUs, ``workers`` on the CPU (1 = in-process)."""
    if gpus:
        return len(gpus) * max(per_gpu, 1)
    return max(workers, 1)


def assign(costs: Sequence[float], units: int) -> list[int]:
    """Unit (card, or CPU slot) of every job: jobs largest first, each to the unit with the least load so far.

    Parameters
    ----------
    costs : sequence of float
        One per job.
    units : int

    Returns
    -------
    list of int
        ``unit_of[i]`` for job ``i``; ties go to the lowest unit.
    """
    units = max(int(units), 1)
    load = [0.0] * units
    unit_of = [0] * len(costs)
    for i in sorted(range(len(costs)), key=lambda i: -float(costs[i])):
        u = min(range(units), key=lambda u: (load[u], u))
        unit_of[i] = u
        load[u] += float(costs[i])
    return unit_of


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
    global _REPORT
    if not gpus and slots <= 1:
        if chunk:
            os.environ["MUP_CHUNK"] = str(int(chunk))
        f = resolve_func(func)
        out = []
        t0 = time.time()
        if log:
            log(f"{len(jobs)} jobs in-process, largest first")
        for i, (key, kw, _) in enumerate(jobs):
            t1 = time.time()
            _REPORT = (lambda m, key=key: log(f"  {key}: {m}")) if log else None
            try:
                out.append(f(**kw))
            finally:
                _REPORT = None
            if log:
                log(f"job {i + 1}/{len(jobs)} {key} done in {time.time() - t1:.1f}s ({time.time() - t0:.0f}s total)")
        return out

    order = sorted(range(len(jobs)), key=lambda i: -float(jobs[i][2]))
    units = len(gpus) if gpus else slots
    unit_of = assign([j[2] for j in jobs], units)                      # card (or CPU slot) of every job
    tmp = keep_dir or tempfile.mkdtemp(prefix="mup_pool_")
    os.makedirs(tmp, exist_ok=True)
    with open(os.path.join(tmp, "spec.pkl"), "wb") as fh:
        pickle.dump({"func": func, "order": order, "keys": [str(j[0]) for j in jobs], "unit_of": unit_of}, fh)
    for i, (_, kw, _) in enumerate(jobs):
        with open(os.path.join(tmp, f"job_{i}.pkl"), "wb") as fh:
            pickle.dump(kw, fh)

    procs = []
    for s in range(slots):
        env = driver.worker_env(s, gpus, per_gpu, 1, chunk)
        where = f"GPU {gpus[s % len(gpus)]}" if gpus else f"slot {s}"
        env["MUP_POOL_WHERE"] = where
        env["MUP_POOL_SLOT"] = str(s)
        # the worker imports what the script can: sscl_core (ROOT) and the experiment's own modules
        env["PYTHONPATH"] = os.pathsep.join([p for p in sys.path if p] + [p for p in (env.get("PYTHONPATH"),) if p])
        logf = open(os.path.join(tmp, f"worker_{s}.log"), "w")
        procs.append((subprocess.Popen([sys.executable, "-m", "sscl_core.pool", tmp], env=env, cwd=ROOT,
                                       stdout=logf, stderr=subprocess.STDOUT), logf, where))
    if log:
        log(f"{len(jobs)} jobs on {slots} worker{'s' if slots != 1 else ''} "
            f"({'GPUs ' + ','.join(map(str, gpus)) if gpus else 'CPU'}): each job on the least-loaded "
            f"{'card' if gpus else 'slot'}, largest first; a {'card' if gpus else 'slot'} that runs out takes "
            f"the others' remaining jobs")
        total = sum(float(j[2]) for j in jobs) or 1.0
        for u in range(units):
            mine = [i for i in order if unit_of[i] == u]
            share = sum(float(jobs[i][2]) for i in mine) / total
            log(f"  {procs[u][2]}: {len(mine)} job{'s' if len(mine) != 1 else ''}, {100 * share:.0f}% of the cost: "
                + ", ".join(str(jobs[i][0]) for i in mine))

    seen = set()
    shown: dict[int, str] = {}
    t0 = time.time()
    while True:
        alive = any(p.poll() is None for p, _, _ in procs)
        for i in range(len(jobs)):
            if i in seen:
                continue
            prog = os.path.join(tmp, f"prog_{i}.txt")
            if log and os.path.exists(prog):                 # the worker's latest progress line, once per change
                try:
                    msg = open(prog).read().strip()
                except OSError:
                    msg = ""
                if msg and msg != shown.get(i):
                    shown[i] = msg
                    log(f"  {jobs[i][0]}: {msg}")
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
    slot = int(os.environ.get("MUP_POOL_SLOT", "-1"))
    unit_of = spec["unit_of"]
    units = max(unit_of, default=0) + 1
    unit = slot % units if slot >= 0 else -1
    mine = [i for i in spec["order"] if unit_of[i] == unit]            # this card's jobs, largest first
    left = [i for i in spec["order"] if unit_of[i] != unit]            # the others', once this card is done
    f = resolve_func(spec["func"])
    for i in mine + left:
        try:
            fd = os.open(os.path.join(tmp, f"claim_{i}"), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            continue
        os.close(fd)
        t0 = time.time()
        with open(os.path.join(tmp, f"job_{i}.pkl"), "rb") as fh:
            kw = pickle.load(fh)

        def to_parent(msg, i=i):
            """Latest progress line of this job, for the parent to relay (atomic replace)."""
            part = os.path.join(tmp, f"prog_{i}.txt.part")
            with open(part, "w") as fh:
                fh.write(f"{msg} on {where}")
            os.replace(part, os.path.join(tmp, f"prog_{i}.txt"))
            print(f"{spec['keys'][i]}: {msg}", flush=True)

        global _REPORT
        _REPORT = to_parent
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
        _REPORT = None
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
    from sscl_core.protocol import state_bytes
    r = runs[0]
    P = max(T, r.T_burn + r.T_meas)
    per_run = state_bytes(r, 1) + 8 * r.N * (P - (r.T_burn + r.T_meas))
    return int(MEM_FACTOR * per_run * len(runs) + chunk * len(runs) * out_per_task)


def plan(args: argparse.Namespace, groups: Sequence[tuple], out_per_task: float = 8.0, split: bool = True) -> Plan:
    """
    Placement and split of a script's groups of runs from the ``add_args`` options.

    With ``--gpus`` the cards are the ids given, else ``driver.visible_gpus()``,
    else ``GPU_COUNT``; their memory comes from ``nvidia-smi`` or ``GPU_MEM_GB``.
    ``run_jobs`` places every job on the least-loaded card, so a group is
    cut only as far as balance needs: into ``ceil(cards · its share of the
    total cost)`` pieces (``experiments.job_cost``: a lone widest group
    becomes one piece per card; one of many equal groups stays whole, since
    every piece pays the per-step launch overhead again), never more than
    its runs, and more only when a batch does not fit a process's share of
    the memory — then rounded up to a multiple of the card count so the
    pieces spread evenly over the cards.  ``per_gpu``
    processes per card (as many as ``MAX_PER_GPU`` and the memory allow, no
    more than there are runs to occupy them) interleave the shorter chains
    with the long ones.  The chunk is the largest whose stacked observable
    stays under ``OUT_BUDGET``.  On CPU workers (compute-bound) the groups are
    instead split to fill the workers by run count.  ``--per-gpu``,
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
    if gpus:
        from sscl_core.experiments import job_cost
        cost = [job_cost(runs, T) for runs, T in groups]
        total = sum(cost) or 1.0
    pieces = []
    for gi, ((runs, T), b) in enumerate(zip(groups, B)):
        need = 1
        while job_bytes(runs[: -(-b // need)], T, CHUNKS[-1], out_per_task) > mem_slot:
            need += 1
        if gpus and need > 1:
            need = -(-need // len(gpus)) * len(gpus)                 # a multiple of the card count
        if not split:
            if need > 1:
                raise SystemExit(f"a batch of {b} runs needs {job_bytes(runs, T, CHUNKS[-1], out_per_task) / 2**30:.1f} GB "
                                 f"on the device, more than a process's {mem_slot / 2**30:.1f} GB share; use a smaller batch")
            pieces.append(1)
            continue
        if gpus:
            fill = max(1, math.ceil(len(gpus) * cost[gi] / total - 1e-9))
        else:
            fill = -(-slots * b // B_tot) if len(groups) < slots else 1
        want = max(need, int(args.split) if args.split else fill)
        pieces.append(min(b, want))
    if gpus and not chunk:
        B_max = max((-(-b // k) for b, k in zip(B, pieces)), default=1)
        T_max = max((T for _, T in groups), default=1)
        chunk = next((c for c in CHUNKS if c * B_max * out_per_task <= OUT_BUDGET and c <= T_max), CHUNKS[-1])
    return Plan(gpus, per_gpu, workers, chunk, pieces, mem_gb)


if __name__ == "__main__":
    # run the worker of the imported module, not of this ``__main__`` copy: ``report`` in the job functions
    # reads ``sscl_core.pool._REPORT``, which the worker sets
    from sscl_core import pool as _pool
    sys.exit(_pool._worker(sys.argv[1]))
