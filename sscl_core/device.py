"""Which backend JAX is running on, and the per-backend defaults of the task loop.

Nothing here imports ``jax`` at module load beyond ``sscl_core._x64`` (which only
sets the x64 flag), so the config and plotting modules stay JAX-free.

Environment variables read here:

``MUP_CHUNK``
    Tasks per device call in ``protocol.run_stream`` (overrides the backend
    default: 1 on CPU, ``GPU_CHUNK`` on a GPU).
"""
from __future__ import annotations

import os
from typing import Optional

import sscl_core._x64  # noqa: F401  (enables x64 before any array exists)

#: Tasks per device call on a GPU.  One call per task leaves a V100/A100 idle
#: between kernel launches and host syncs at the Stage-1 sizes (N ≤ 200);
#: scanning this many tasks inside one executable removes that overhead.
GPU_CHUNK = 16


def platform() -> str:
    """``"cpu"``, ``"gpu"`` (CUDA) or whatever ``jax.default_backend()`` reports."""
    import jax
    return jax.default_backend()


def is_gpu() -> bool:
    """True when the default backend is a GPU."""
    return platform() in ("gpu", "cuda", "rocm")


def devices() -> list:
    """The devices of the default backend."""
    import jax
    return jax.devices()


def default_chunk(override: Optional[int] = None) -> int:
    """
    Tasks per device call for ``protocol.run_stream``.

    Parameters
    ----------
    override : int, optional
        An explicit value (``run_sweep.py --chunk``) wins over the
        ``MUP_CHUNK`` environment variable, which wins over the backend
        default.

    Returns
    -------
    int
        ≥ 1.  1 reproduces the one-call-per-task loop of the CPU sweeps
        bit for bit.
    """
    if override is not None and override > 0:
        return int(override)
    env = os.environ.get("MUP_CHUNK")
    if env:
        return max(1, int(env))
    return GPU_CHUNK if is_gpu() else 1


def describe() -> str:
    """One line naming the backend and its devices, for logs."""
    import jax
    devs = jax.devices()
    kinds = sorted({getattr(d, "device_kind", str(d)) for d in devs})
    return f"{jax.default_backend()} × {len(devs)} ({', '.join(kinds)}), x64={jax.config.jax_enable_x64}"
