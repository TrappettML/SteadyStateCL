"""A 5-task run at N = 10 under clock (i) hashes to a stored value.

The stored hash predates the task clock (2026-11-11): it is taken under ``clock="nominal"`` and over the
result keys of that date, so it proves that clock (i) still runs the dynamics it always did, bit for bit.

Regenerate ``tests/reference_hash.txt`` deliberately (delete it and rerun) when the
kernels change.  If the hash differs on another machine, that is FMA/compiler
behaviour, not a bug: compare curves instead (PLAN.md §10).
"""
import hashlib
import os

import jax
import numpy as np
import pytest

from sscl_core.config import Config, make_runs
from sscl_core.protocol import CLOCK_KEYS, run_stream

REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference_hash.txt")


def _hash():
    cfg = Config(solver="pop", N=[10], readout=["gated_shared", "learned_shared", "gated_frozen", "frozen", "task_heads"],
                 delta0=[1.0], lam=[1.0], seeds=1, T_factor=0.25, T_min=5,
                 clock="nominal")      # clock (i): the dynamics the stored hash was made with
    from conftest import one_batch
    runs = one_batch(make_runs(cfg))
    assert runs[0].T_burn + runs[0].T_meas == 10
    res = run_stream(runs)
    h = hashlib.sha256()
    for k in sorted(res):
        if k in CLOCK_KEYS:            # scalars added on 2026-11-11; left out so the stored hash still pins the dynamics
            continue
        h.update(k.encode())
        h.update(np.ascontiguousarray(np.asarray(res[k], dtype=np.float64)).tobytes())
    return h.hexdigest()


def test_hash_matches_reference():
    if jax.default_backend() != "cpu":
        pytest.skip("reference hash is for the CPU backend (GPU FMA order differs)")
    if not jax.config.jax_threefry_partitionable:
        pytest.skip("reference hash is for jax >= 0.5's PRNG stream (jax_threefry_partitionable)")
    h = _hash()
    if not os.path.exists(REF):
        with open(REF, "w") as f:
            f.write(h + "\n")
        pytest.skip(f"reference hash written to {REF}; rerun to compare")
    with open(REF) as f:
        ref = f.read().strip()
    assert h == ref, f"hash {h} != reference {ref} (kernels changed? or FMA differences on this machine)"


def test_same_seed_twice_is_identical():
    assert _hash() == _hash()
