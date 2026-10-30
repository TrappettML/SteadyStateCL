"""Result schema: one .npz per batch plus a JSON manifest for resumable sweeps."""
from __future__ import annotations

import json
import os
import re
import time
from typing import Optional

import numpy as np

from mup.config import Config, Run, make_batches, make_runs

SCHEMA_VERSION = 2   # 2: PLAN.md §12.1 norm scalars (pre/post, logf, quantiles); 1: the original set


def batch_path(out_dir: str, batch_id: int) -> str:
    """Path of batch ``batch_id``'s ``.npz`` inside ``out_dir``."""
    return os.path.join(out_dir, f"batch_{batch_id:05d}.npz")


def save_batch(out_dir: str, batch_id: int, results: dict, runs: list[Run], extra: Optional[dict] = None) -> str:
    """
    Write the running moments, per-task scalars and the run metadata of a batch.

    Parameters
    ----------
    out_dir : str
    batch_id : int
    results : dict
        ``run_stream`` output (without the trace arrays).
    runs : list of Run
    extra : dict, optional
        Extra JSON metadata (e.g. ``skipped`` run labels).

    Returns
    -------
    str
        Path written (atomically, via a temp name that ``load_sweep`` ignores).
    """
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
    tmp = os.path.join(out_dir, f".tmp_{os.getpid()}_batch_{batch_id:05d}.npz")   # never matches load_sweep's filter
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)
    return path


def batch_schema(path: str) -> int:
    """Schema version recorded in a batch file (1 for files written before the version was stored)."""
    with np.load(path, allow_pickle=False) as z:
        return int(json.loads(str(z["meta_json"])).get("schema", 1))


def is_stale(path: str) -> bool:
    """Whether a batch file was written under an older result schema than ``SCHEMA_VERSION``.

    Such a batch lacks scalars the figure scripts expect (e.g. the §12.1
    norms), so a resumed sweep must rerun it rather than keep it.
    """
    return os.path.exists(path) and batch_schema(path) < SCHEMA_VERSION


def load_batch(path: str) -> tuple[dict, list[Run], dict]:
    """
    Read one batch file.

    Parameters
    ----------
    path : str

    Returns
    -------
    (results, runs, meta)
        Arrays, the runs and the JSON metadata.
    """
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta_json"]))
        results = {k: z[k] for k in z.files if k != "meta_json"}
    runs = [Run.from_dict(d) for d in meta["runs"]]
    return results, runs, meta


def traces_dir(out_dir: str) -> str:
    """``<out_dir>/traces``."""
    return os.path.join(out_dir, "traces")


def save_traces(out_dir: str, batch_id: int, res: dict, runs: list[Run]) -> list[str]:
    """Write one ``traces/run_<batch>_<j>.npz`` per run of a trace batch and update ``traces/manifest.json``.

    Parameters
    ----------
    out_dir : str
        The sweep's results directory.
    batch_id : int
    res : dict
        ``run_stream`` output with ``trace`` (B, T, n_trace, n_quant) and ``trace_idx``.
    runs : list of Run
        The runs of the batch, in the order of the leading axis of ``trace``.

    Returns
    -------
    list of str
        Paths written.
    """
    from mup.protocol import TRACE_QUANT
    tdir = traces_dir(out_dir)
    os.makedirs(tdir, exist_ok=True)
    paths = []
    man_path = os.path.join(tdir, "manifest.json")
    man = json.load(open(man_path)) if os.path.exists(man_path) else {"schema": SCHEMA_VERSION, "quantities": list(TRACE_QUANT), "runs": []}
    man["runs"] = [m for m in man["runs"] if m.get("batch_id") != batch_id]
    for j, r in enumerate(runs):
        name = f"run_{batch_id:05d}_{j:03d}.npz"
        path = os.path.join(tdir, name)
        meta = {"schema": SCHEMA_VERSION, "batch_id": batch_id, "index": j, "quantities": list(TRACE_QUANT),
                "T_burn": int(res["T_burn"]), "T_meas": int(res["T_meas"]), "n_steps": int(res["n_steps"]), "run": r.to_dict()}
        tmp = os.path.join(tdir, f".tmp_{os.getpid()}_{name}")
        np.savez_compressed(tmp, trace=res["trace"][j], trace_idx=res["trace_idx"], meta_json=np.array(json.dumps(meta)))
        os.replace(tmp, path)
        paths.append(path)
        man["runs"].append({"file": name, "batch_id": batch_id, "index": j, "label": r.label()})
    _write_json(man_path, man)
    return paths


def load_traces(out_dir: str) -> list[tuple[np.ndarray, np.ndarray, Run, dict]]:
    """Load every trace file of a sweep.

    Returns
    -------
    list of (trace (T, n_trace, n_quant) float32, trace_idx (n_trace,), Run, meta dict)
    """
    tdir = traces_dir(out_dir)
    if not os.path.isdir(tdir):
        return []
    out = []
    for f in sorted(os.listdir(tdir)):
        if not re.fullmatch(r"run_\d+_\d+\.npz", f):
            continue
        with np.load(os.path.join(tdir, f), allow_pickle=False) as z:
            meta = json.loads(str(z["meta_json"]))
            out.append((z["trace"], z["trace_idx"], Run.from_dict(meta["run"]), meta))
    return out


def clear_sweep(out_dir: str) -> list[str]:
    """Delete a sweep's batch files, traces, logs and derived outputs (``--fresh``); the manifest is rewritten by the caller.

    Returns
    -------
    list of str
        The paths removed.
    """
    import shutil
    removed = []
    if not os.path.isdir(out_dir):
        return removed
    for f in sorted(os.listdir(out_dir)):
        path = os.path.join(out_dir, f)
        if re.fullmatch(r"batch_\d+\.npz", f) or f in ("summary.csv", "curves.json", "validate.log"):
            os.remove(path); removed.append(path)
        elif f in ("traces", "logs", "figures") and os.path.isdir(path):
            shutil.rmtree(path); removed.append(path)
    return removed


def load_sweep(out_dir: str) -> list[tuple[dict, list[Run], dict]]:
    """
    Load every finished batch file of a sweep, in batch order.

    Parameters
    ----------
    out_dir : str

    Returns
    -------
    list of (results, runs, meta)
    """
    files = sorted(f for f in os.listdir(out_dir) if re.fullmatch(r"batch_\d+\.npz", f))
    return [load_batch(os.path.join(out_dir, f)) for f in files]


# ----------------------------------------------------------------------------
# manifest
# ----------------------------------------------------------------------------
def manifest_path(out_dir: str) -> str:
    """``<out_dir>/manifest.json``."""
    return os.path.join(out_dir, "manifest.json")


def write_manifest(out_dir: str, cfg: Config, batches: list[list[Run]], config_path: str = "",
                   N_filter=None) -> dict:
    """
    Write the sweep manifest listing every batch and its status.

    A batch is "done" when its file exists with the current schema, "stale"
    when its file was written under an older schema (rerun on resume), else
    "pending"; the driver sets "running"/"failed" while it works.

    Parameters
    ----------
    out_dir : str
    cfg : Config
    batches : list of list of Run
    config_path : str
    N_filter : sequence of int, optional
        The ``--N`` filter the batches were expanded with (needed to re-expand
        them identically in a subprocess).

    Returns
    -------
    dict
        The manifest written.
    """
    os.makedirs(out_dir, exist_ok=True)
    man = {
        "schema": SCHEMA_VERSION,
        "name": cfg.name,
        "config_path": config_path,
        "config": cfg.to_dict(),
        "N_filter": None if N_filter is None else [int(n) for n in N_filter],
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "batches": [
            {
                "id": i,
                "n_runs": len(b),
                "N": b[0].N, "kappa": b[0].kappa, "solver": b[0].solver,
                "n_steps": b[0].n_steps, "T": b[0].T_burn + b[0].T_meas,
                "status": _file_status(batch_path(out_dir, i)),
            }
            for i, b in enumerate(batches)
        ],
    }
    _write_json(manifest_path(out_dir), man)
    return man


def _file_status(path: str) -> str:
    """``done`` / ``stale`` / ``pending`` from a batch file's presence and schema."""
    if not os.path.exists(path):
        return "pending"
    return "stale" if batch_schema(path) < SCHEMA_VERSION else "done"


def _write_json(path: str, obj: dict) -> None:
    """Atomic replace with a per-process temp name (the driver's threads serialise their own writes)."""
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def read_manifest(out_dir: str) -> dict:
    """
    Read ``manifest.json``.

    Parameters
    ----------
    out_dir : str

    Returns
    -------
    dict
    """
    with open(manifest_path(out_dir)) as f:
        return json.load(f)


def set_status(out_dir: str, batch_id: int, status: str) -> None:
    """
    Update one batch's status in the manifest (atomic replace).

    Parameters
    ----------
    out_dir : str
    batch_id : int
    status : str
    """
    man = read_manifest(out_dir)
    man["batches"][batch_id]["status"] = status
    man["batches"][batch_id]["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    _write_json(manifest_path(out_dir), man)


def plan_sweep(cfg: Config, out_dir: str, N_filter=None, config_path: str = "") -> list[list[Run]]:
    """Expand the config into batches and (re)write the manifest ('done' = batch file exists)."""
    runs = make_runs(cfg, N_filter)
    batches = make_batches(runs, cfg.batch_size)
    write_manifest(out_dir, cfg, batches, config_path, N_filter)
    return batches


def batches_from_manifest(out_dir: str) -> list[list[Run]]:
    """Re-expand the batches exactly as the manifest was written, without touching it.

    Used by ``run_sweep.py --batch-id`` (the driver's subprocesses): the driver
    owns the manifest, and a subprocess must neither rewrite it nor race its
    status updates.
    """
    man = read_manifest(out_dir)
    cfg = Config(**man["config"])
    batches = make_batches(make_runs(cfg, man.get("N_filter")), cfg.batch_size)
    if len(batches) != len(man["batches"]) or any(len(b) != m["n_runs"] for b, m in zip(batches, man["batches"])):
        raise RuntimeError("manifest does not match the config it was written from; rerun with --dry-run")
    return batches


def pending_batches(out_dir: str) -> list[int]:
    """
    Ids of batches not marked done and without a current-schema batch file.

    A batch file written under an older schema (``is_stale``) counts as
    pending, so resuming a sweep after a schema change reruns it.

    Parameters
    ----------
    out_dir : str

    Returns
    -------
    list of int
    """
    man = read_manifest(out_dir)
    out = []
    for b in man["batches"]:
        path = batch_path(out_dir, b["id"])
        if is_stale(path) or (b["status"] != "done" and not os.path.exists(path)):
            out.append(b["id"])
    return out
