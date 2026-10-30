#!/usr/bin/env python
"""Run a sweep from a YAML config.

    python scripts/run_sweep.py --config configs/pilot.yaml --N 20            # in-process, sequential
    python scripts/run_sweep.py --config configs/stage1.yaml --workers 90     # subprocess pool (Xeon)
    python scripts/run_sweep.py --config configs/stage1.yaml --batch-id 7     # one batch (used by the driver)

Results go to results/<name>/ (or --out); the manifest makes the sweep resumable:
a rerun does only the missing and stale (older-schema) batches.  ``--fresh``
deletes the existing batch files, traces, logs and figures first and reruns
every batch.
"""
import argparse
import os
import sys

import _path  # noqa: F401

from mup import io
from mup.config import Config


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
    ap.add_argument("--threads", type=int, default=1, help="XLA threads per worker")
    ap.add_argument("--batch-id", type=int, default=None, help="run exactly this batch in-process")
    ap.add_argument("--dry-run", action="store_true", help="only write the manifest and print the batches")
    ap.add_argument("--fresh", action="store_true",
                    help="delete the existing batch files, traces, logs and figures, then rerun every batch (default: resume, rerunning only missing or stale batches)")
    ap.add_argument("--log-every", type=int, default=0)
    args = ap.parse_args(argv)

    cfg = Config.from_yaml(args.config)
    out = args.out or os.path.join(_path.ROOT, "results", cfg.name)
    config_path = os.path.abspath(args.config)

    if args.batch_id is not None:
        # one batch under the driver: the driver owns the manifest, so only write the batch file
        batches = io.batches_from_manifest(out)
        run_batch(out, args.batch_id, batches[args.batch_id], args.log_every, update_manifest=False)
        return 0

    if args.fresh and not args.dry_run:
        removed = io.clear_sweep(out)
        print(f"--fresh: removed {len(removed)} existing files/directories in {out}")
    batches = io.plan_sweep(cfg, out, args.N, config_path)
    n_runs = sum(len(b) for b in batches)
    print(f"{cfg.name}: {n_runs} runs in {len(batches)} batches → {out}")
    if args.dry_run:
        from mup.protocol import feasible
        n_skip = 0
        for i, b in enumerate(batches):
            r = b[0]
            bad = int((~feasible(b)).sum())
            n_skip += bad
            print(f"  batch {i:4d}: {len(b):3d} runs  N={r.N} D={r.D} κ={r.kappa_label} {r.solver} n_steps={r.n_steps} "
                  f"T={r.T_burn + r.T_meas}" + (f"  ({bad} infeasible δ₀, will be skipped)" if bad else ""))
        print(f"{n_skip} of {n_runs} runs are infeasible (δ₀ ≤ −min_i Q0_i) and will be skipped")
        return 0

    pending = io.pending_batches(out)          # after --fresh every batch is pending
    if args.workers > 0:
        from mup import driver
        res = driver.launch(config_path, out, args.workers, args.numa_nodes, args.threads, pending)
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
    from mup.protocol import feasible, run_stream
    r = batch[0]
    print(f"batch {i}: {len(batch)} runs  N={r.N} κ={r.kappa_label} {r.solver} n_steps={r.n_steps} T={r.T_burn + r.T_meas}", flush=True)
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
    sys.exit(main())
