"""A 5-task run at N = 10 hashes to a stored value.

Regenerate ``tests/reference_hash.txt`` deliberately (delete it and rerun) when the
kernels change.  If the hash differs on another machine, that is FMA/compiler
behaviour, not a bug: compare curves instead (PLAN.md §10).
"""
import hashlib
import os

import numpy as np
import pytest

from mup.config import Config, make_runs
from mup.protocol import run_stream

REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference_hash.txt")


def _hash():
    cfg = Config(N=[10], readout=["learned", "frozen", "heads"], delta0=[1.0], lam=[1.0], seeds=1,
                 T_factor=0.25, T_min=5)
    runs = make_runs(cfg)
    assert runs[0].T_burn + runs[0].T_meas == 10
    res = run_stream(runs)
    h = hashlib.sha256()
    for k in sorted(res):
        h.update(k.encode())
        h.update(np.ascontiguousarray(np.asarray(res[k], dtype=np.float64)).tobytes())
    return h.hexdigest()


def test_hash_matches_reference():
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
