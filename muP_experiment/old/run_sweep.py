#!/usr/bin/env python
"""Run a sweep from a YAML config.

    python old/run_sweep.py --config configs/pilot.yaml --N 20            # in-process, sequential
    python old/run_sweep.py --config configs/stage1.yaml --workers 90     # subprocess pool (Xeon)
    python old/run_sweep.py --config configs/stage1.yaml --gpus auto      # one process per GPU (cyclops)
    python old/run_sweep.py --config configs/stage1.yaml --gpus 0 1 2 3 --per-gpu 2 --batch-size 256
    python old/run_sweep.py --config configs/stage1.yaml --batch-id 7     # one batch (used by the driver)

Results go to results/<name>/ (or --out); the manifest makes the sweep resumable:
a rerun does only the missing and stale (older-schema) batches.  ``--fresh``
deletes the existing batch files, traces, logs and figures first and reruns
every batch.

GPU node: ``--gpus auto`` (or explicit ids) runs ``--per-gpu`` batch processes
on each GPU, each seeing one device.  A GPU wants big ``vmap`` batches, so
``--batch-size`` overrides the YAML's (it is recorded in the manifest: keep it
fixed for the life of a results directory, or use ``--out``/``--fresh``).
``--chunk`` sets the tasks scanned per device call (default 16 on a GPU, 1
on the CPU; see ``sscl_core.device``).
"""
import argparse
import dataclasses
import os
import sys
import time

import _path  # noqa: F401

from sscl_core import io
from sscl_core.config import Config

_T0 = time.perf_counter()        # wall clock from after the imports to the end of the script


def main(argv=None):
    """
    Command-line entry point (see the module docstring).

    Parameters
    ----------
    argv : list of str, optional

    Returns
    -------
    int
        Exit code (1 if any batch failed under the driver).
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None, help="results dir (default results/<config name>)")
    ap.add_argument("--N", type=int, nargs="*", default=None, help="restrict to these widths")
    ap.add_argument("--workers", type=int, default=0, help="subprocess pool size (0: run in-process)")
    ap.add_argument("--numa-nodes", type=int, default=0, help="bind workers round-robin to this many NUMA nodes")
    ap.add_argument("--threads", type=int, default=1, help="XLA threads per worker (CPU)")
    ap.add_argument("--gpus", nargs="*", default=None,
                    help="GPU ids for the worker pool, or 'auto' for every visible GPU (default: CPU)")
    ap.add_argument("--per-gpu", type=int, default=1, help="batch processes per GPU (with --gpus)")
    ap.add_argument("--chunk", type=int, default=None,
                    help="tasks per device call (default: 1 on CPU, 16 on GPU, or $MUP_CHUNK)")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="override the config's vmap batch size (fixed per results directory)")
    ap.add_argument("--batch-id", type=int, default=None, help="run exactly this batch in-process")
    ap.add_argument("--dry-run", action="store_true", help="only write the manifest and print the batches")
    ap.add_argument("--fresh", action="store_true",
                    help="delete the existing batch files, traces, logs and figures, then rerun every batch (default: resume, rerunning only missing or stale batches)")
    ap.add_argument("--log-every", type=int, default=0)
    args = ap.parse_args(argv)

    cfg = Config.from_yaml(args.config)
    if args.batch_size is not None:
        cfg = dataclasses.replace(cfg, batch_size=int(args.batch_size))
    out = args.out or os.path.join(_path.ROOT, "results", cfg.name)
    config_path = os.path.abspath(args.config)
    if args.chunk is not None:
        os.environ["MUP_CHUNK"] = str(args.chunk)       # read by sscl_core.device in this process

    if args.batch_id is not None:
        # one batch under the driver: the driver owns the manifest, so only write the batch file
        batches = io.batches_from_manifest(out)
        run_batch(out, args.batch_id, batches[args.batch_id], args.log_every, update_manifest=False)
        return 0

    gpus = None
    if args.gpus is not None:
        from sscl_core import driver
        if len(args.gpus) == 1 and args.gpus[0].lower() == "auto" or len(args.gpus) == 0:
            gpus = driver.visible_gpus()
            if not gpus:
                print("--gpus auto: no GPU found (CUDA_VISIBLE_DEVICES / nvidia-smi); falling back to the CPU pool")
        else:
            gpus = [int(g) for g in args.gpus]

    if args.fresh and not args.dry_run:
        removed = io.clear_sweep(out)
        print(f"--fresh: removed {len(removed)} existing files/directories in {out}")
    batches = io.plan_sweep(cfg, out, args.N, config_path)
    n_runs = sum(len(b) for b in batches)
    print(f"{cfg.name}: {n_runs} runs in {len(batches)} batches → {out}")
    if args.dry_run:
        from sscl_core.protocol import feasible, state_bytes
        n_skip = 0
        for i, b in enumerate(batches):
            r = b[0]
            bad = int((~feasible(b)).sum())
            n_skip += bad
            print(f"  batch {i:4d}: {len(b):3d} runs  N={r.N} D={r.D} κ={r.kappa_label} {r.solver} n_steps={r.n_steps} "
                  f"T={r.T_burn + r.T_meas}  state {state_bytes(r, len(b)) / 2**30:.2f} GiB"
                  + (f"  ({bad} infeasible δ₀, will be skipped)" if bad else ""))
        print(f"{n_skip} of {n_runs} runs are infeasible (δ₀ ≤ −min_i Q0_i) and will be skipped")
        return 0

    pending = io.pending_batches(out)          # after --fresh every batch is pending
    if args.workers > 0 or gpus:
        from sscl_core import driver
        res = driver.launch(config_path, out, args.workers, args.numa_nodes, args.threads, pending,
                            gpus=gpus, per_gpu=args.per_gpu, chunk=args.chunk)
        print(f"done: {len(res['done'])}, failed: {len(res['failed'])}")
        return 1 if res["failed"] else 0

    for i in pending:
        run_batch(out, i, batches[i], args.log_every)
    return 0


def run_batch(out, i, batch, log_every, update_manifest=True):
    """
    Run one batch in-process and write its batch file (and trace files for trace runs).

    Infeasible runs (δ₀ ≤ −min_i Q0_i) are skipped and listed under ``skipped``
    in the batch metadata.

    Parameters
    ----------
    out : str
        Results directory.
    i : int
        Batch id.
    batch : list of Run
    log_every : int
    update_manifest : bool
        False under the driver, which owns the manifest.
    """
    from sscl_core.protocol import feasible, run_stream
    from sscl_core import device
    r = batch[0]
    print(f"batch {i}: {len(batch)} runs  N={r.N} κ={r.kappa_label} {r.solver} n_steps={r.n_steps} T={r.T_burn + r.T_meas}"
          f"  [{device.describe()}]", flush=True)
    ok = feasible(batch)
    skipped = [batch[j].label() for j in range(len(batch)) if not ok[j]]
    if skipped:
        print(f"  skipping {len(skipped)} infeasible runs (δ₀ ≤ −min_i Q0_i): {skipped[0]} …", flush=True)
    kept = [batch[j] for j in range(len(batch)) if ok[j]]
    res = run_stream(kept, log=lambda m: print("  " + m, flush=True), log_every=log_every) if kept else {}
    if "trace" in res:
        io.save_traces(out, i, res, kept)          # per-run trace files; the batch file stays small
        res = {k: v for k, v in res.items() if k not in ("trace", "trace_idx")}
    io.save_batch(out, i, res, kept, extra={"skipped": skipped})
    if update_manifest:
        io.set_status(out, i, "done")


if __name__ == "__main__":
    _rc = main()
    _dt = time.perf_counter() - _T0
    print(f"run_sweep: total {_dt:.1f} s ({_dt / 60:.2f} min) from after the imports to the end", flush=True)
    sys.exit(_rc)
