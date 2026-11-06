"""Scanning several tasks per device call (``run_stream(chunk=k)``) gives the same stream.

``chunk`` only changes how many tasks one jitted call advances (the GPU
setting); draws and integer outputs must match exactly, floats to 1e-10.
"""
import numpy as np
from conftest import small_runs  # noqa: F401

from mup.protocol import run_stream


def test_chunked_stream_matches(small_runs):
    T = small_runs[0].T_burn + small_runs[0].T_meas
    ref = run_stream(small_runs, chunk=1)
    for chunk in (3, T, 2 * T):               # remainder chunk, exactly T, capped at T
        res = run_stream(small_runs, chunk=chunk)
        assert set(res) == set(ref)
        for k in ref:
            a, b = np.asarray(ref[k]), np.asarray(res[k])
            assert a.shape == b.shape, k
            if k in ("count", "n_active", "ages", "T_burn", "T_meas", "n_steps"):
                assert np.array_equal(a, b), k
            else:
                np.testing.assert_allclose(a, b, atol=1e-10, rtol=0, equal_nan=True, err_msg=k)


def test_chunked_trace_matches():
    import dataclasses
    from conftest import one_batch
    from mup.config import Config, make_runs
    cfg = Config(N=[10], readout=["gated_shared", "task_heads"], delta0=[1.0], lam=[1.0], seeds=1,
                 T_factor=0.5, T_min=5, trace=True, trace_points=4)
    runs = one_batch(make_runs(cfg))
    ref = run_stream(runs, chunk=1)
    res = run_stream(runs, chunk=4)
    assert ref["trace"].shape == res["trace"].shape
    np.testing.assert_allclose(ref["trace"], res["trace"], atol=1e-6, rtol=0, equal_nan=True)
    assert np.array_equal(ref["trace_idx"], res["trace_idx"])
