#!/usr/bin/env python
"""Run a sweep from a YAML config.

    python scripts/run_sweep.py --config configs/pilot.yaml --N 20            # in-process, sequential
    python scripts/run_sweep.py --config configs/stage1.yaml --workers 90     # subprocess pool (Xeon)
    python scripts/run_sweep.py --config configs/stage1.yaml --batch-id 7     # one batch (used by the driver)

Results go to results/<name>/ (or --out); the manifest makes the sweep resumable.
"""
import argparse
import os
import sys

import _path  # noqa: F401

from mup import io
from mup.config import Config


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None, help="results dir (default results/<config name>)")
    ap.add_argument("--N", type=int, nargs="*", default=None, help="restrict to these widths")
    ap.add_argument("--workers", type=int, default=0, help="subprocess pool size (0: run in-process)")
    ap.add_argument("--numa-nodes", type=int, default=0, help="bind workers round-robin to this many NUMA nodes")
    ap.add_argument("--threads", type=int, default=1, help="XLA threads per worker")
    ap.add_argument("--batch-id", type=int, default=None, help="run exactly this batch in-process")
    ap.add_argument("--dry-run", action="store_true", help="only write the manifest and print the batches")
    ap.add_argument("--log-every", type=int, default=0)
    args = ap.parse_args(argv)

    cfg = Config.from_yaml(args.config)
    out = args.out or os.path.join(_path.ROOT, "results", cfg.name)
    config_path = os.path.abspath(args.config)

    if args.batch_id is not None:
        # single batch; the manifest already exists (written by the launcher)
        man = io.read_manifest(out)
        batches = io.plan_sweep(Config(**man["config"]), out, None, man.get("config_path", config_path))
        run_batch(out, args.batch_id, batches[args.batch_id], args.log_every)
        return 0

    batches = io.plan_sweep(cfg, out, args.N, config_path)
    n_runs = sum(len(b) for b in batches)
    print(f"{cfg.name}: {n_runs} runs in {len(batches)} batches → {out}")
    if args.dry_run:
        for i, b in enumerate(batches):
            r = b[0]
            print(f"  batch {i:4d}: {len(b):3d} runs  N={r.N} D={r.D} κ={r.kappa_label} {r.solver} n_steps={r.n_steps} T={r.T_burn + r.T_meas}")
        return 0

    pending = io.pending_batches(out)
    if args.workers > 0:
        from mup import driver
        res = driver.launch(config_path, out, args.workers, args.numa_nodes, args.threads, pending)
        print(f"done: {len(res['done'])}, failed: {len(res['failed'])}")
        return 1 if res["failed"] else 0

    for i in pending:
        run_batch(out, i, batches[i], args.log_every)
    return 0


def run_batch(out, i, batch, log_every):
    from mup.protocol import feasible, run_stream
    r = batch[0]
    print(f"batch {i}: {len(batch)} runs  N={r.N} κ={r.kappa_label} {r.solver} n_steps={r.n_steps} T={r.T_burn + r.T_meas}", flush=True)
    ok = feasible(batch)
    skipped = [batch[j].label() for j in range(len(batch)) if not ok[j]]
    if skipped:
        print(f"  skipping {len(skipped)} infeasible runs (δ₀ ≤ −min_i Q0_i): {skipped[0]} …", flush=True)
    kept = [batch[j] for j in range(len(batch)) if ok[j]]
    res = run_stream(kept, log=lambda m: print("  " + m, flush=True), log_every=log_every) if kept else {}
    io.save_batch(out, i, res, kept, extra={"skipped": skipped})
    io.set_status(out, i, "done")


if __name__ == "__main__":
    sys.exit(main())
