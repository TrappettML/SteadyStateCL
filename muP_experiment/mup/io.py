"""Result schema: one .npz per batch plus a JSON manifest for resumable sweeps."""
from __future__ import annotations

import json
import os
import time
from typing import Optional

import numpy as np

from mup.config import Config, Run, make_batches, make_runs

SCHEMA_VERSION = 1


def batch_path(out_dir: str, batch_id: int) -> str:
    return os.path.join(out_dir, f"batch_{batch_id:05d}.npz")


def save_batch(out_dir: str, batch_id: int, results: dict, runs: list[Run], extra: Optional[dict] = None) -> str:
    """Write the running moments, per-task scalars and the run metadata of a batch."""
    os.makedirs(out_dir, exist_ok=True)
    meta = {
        "schema": SCHEMA_VERSION,
        "batch_id": batch_id,
        "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "runs": [r.to_dict() for r in runs],
        **(extra or {}),
    }
    arrays = {k: np.asarray(v) for k, v in results.items()}
    arrays["meta_json"] = np.array(json.dumps(meta))
    path = batch_path(out_dir, batch_id)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)
    return path


def load_batch(path: str) -> tuple[dict, list[Run], dict]:
    """→ (results dict of arrays, runs, meta)."""
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta_json"]))
        results = {k: z[k] for k in z.files if k != "meta_json"}
    runs = [Run.from_dict(d) for d in meta["runs"]]
    return results, runs, meta


def load_sweep(out_dir: str) -> list[tuple[dict, list[Run], dict]]:
    files = sorted(f for f in os.listdir(out_dir) if f.startswith("batch_") and f.endswith(".npz"))
    return [load_batch(os.path.join(out_dir, f)) for f in files]


# ----------------------------------------------------------------------------
# manifest
# ----------------------------------------------------------------------------
def manifest_path(out_dir: str) -> str:
    return os.path.join(out_dir, "manifest.json")


def write_manifest(out_dir: str, cfg: Config, batches: list[list[Run]], config_path: str = "") -> dict:
    os.makedirs(out_dir, exist_ok=True)
    man = {
        "schema": SCHEMA_VERSION,
        "name": cfg.name,
        "config_path": config_path,
        "config": cfg.to_dict(),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "batches": [
            {
                "id": i,
                "n_runs": len(b),
                "N": b[0].N, "kappa": b[0].kappa, "solver": b[0].solver,
                "n_steps": b[0].n_steps, "T": b[0].T_burn + b[0].T_meas,
                "status": "done" if os.path.exists(batch_path(out_dir, i)) else "pending",
            }
            for i, b in enumerate(batches)
        ],
    }
    with open(manifest_path(out_dir), "w") as f:
        json.dump(man, f, indent=1)
    return man


def read_manifest(out_dir: str) -> dict:
    with open(manifest_path(out_dir)) as f:
        return json.load(f)


def set_status(out_dir: str, batch_id: int, status: str) -> None:
    man = read_manifest(out_dir)
    man["batches"][batch_id]["status"] = status
    man["batches"][batch_id]["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    tmp = manifest_path(out_dir) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(man, f, indent=1)
    os.replace(tmp, manifest_path(out_dir))


def plan_sweep(cfg: Config, out_dir: str, N_filter=None, config_path: str = "") -> list[list[Run]]:
    """Expand the config into batches and (re)write the manifest, keeping 'done' marks."""
    runs = make_runs(cfg, N_filter)
    batches = make_batches(runs, cfg.batch_size)
    write_manifest(out_dir, cfg, batches, config_path)
    return batches


def pending_batches(out_dir: str) -> list[int]:
    man = read_manifest(out_dir)
    return [b["id"] for b in man["batches"] if b["status"] != "done" and not os.path.exists(batch_path(out_dir, b["id"]))]
