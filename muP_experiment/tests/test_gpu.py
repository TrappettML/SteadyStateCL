"""The sweep on the GPU node (cyclops, 4× V100): skipped wherever JAX sees no GPU.

    python -m pytest tests/test_gpu.py -v

What is checked, in order:

1. JAX's default backend is a GPU and float64 arrays live there (x64 on).
2. One chunk of the sweep kernel on every GPU gives the CPU's result
   (integer fields exact, floats to 1e-8; the teacher draw can differ by an ulp
   between backends through ``erfinv``).
3. Scanning several tasks per call (``chunk``) equals one task per call on the GPU.
4. ``run_stream`` of the same batch in one subprocess per GPU, all at the same
   time, gives the same moments and scalars on every card (and matches this
   process's CPU run).
5. ``driver.launch`` over every GPU runs the pilot sweep end to end: every batch
   file is written, each log names its GPU, and the batches are spread over the cards.

The subprocesses of 4 and 5 get ``GPU_MEM_SHARE = 0.2`` instead of the driver's 0.9:
this pytest process already holds memory on every card (``tests/conftest.py`` turns
preallocation off, but the compiled kernels still live there), and the pilot is tiny.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from conftest import one_batch, small_runs  # noqa: F401
from mup import device, driver, io
from mup.config import Config, make_runs
from mup.observe import age_schedule
from mup.protocol import SCALARS, batched_run_tasks, init_batch, run_stream

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GPUS = [d for d in jax.devices() if d.platform in ("gpu", "cuda", "rocm")]
pytestmark = pytest.mark.skipif(not GPUS, reason="no GPU visible to JAX")


def _runs(N=20, seeds=2):
    cfg = Config(N=[N], readout=["gated_shared", "learned_shared", "gated_frozen", "frozen", "task_heads"],
                 delta0=[1.0], lam=[1.0], seeds=seeds, T_factor=0.5, T_min=6)
    return one_batch(make_runs(cfg))


def _on(dev, runs, chunk):
    """(state, out) of one chunk of ``runs`` on device ``dev``."""
    r0 = runs[0]
    T = r0.T_burn + r0.T_meas
    ages = jnp.asarray(age_schedule(r0.N, r0.kappa_s, T, r0.n_ages_geom, r0.m_max_factor))
    st, p = init_batch(runs)
    st, p, ages = jax.device_put((st, p, ages), dev)
    f = batched_run_tasks(r0.n_steps, r0.solver, False, chunk)
    st, out = f(st, p, ages)
    return jax.device_get(st), jax.device_get(out)


def test_backend_is_gpu_with_x64():
    assert device.is_gpu(), jax.default_backend()
    x = jnp.ones(3)
    assert x.dtype == jnp.float64
    assert list(x.devices())[0].platform in ("gpu", "cuda", "rocm") if hasattr(x, "devices") else x.device().platform != "cpu"
    y = jnp.ones((256, 256)) @ jnp.ones((256, 256))
    assert y.dtype == jnp.float64 and float(y[0, 0]) == 256.0


@pytest.mark.parametrize("gpu", GPUS, ids=[str(d) for d in GPUS])
def test_each_gpu_matches_cpu(gpu):
    runs = _runs()
    st_c, out_c = _on(jax.devices("cpu")[0], runs, 4)
    st_g, out_g = _on(gpu, runs, 4)
    for name, a, b in zip(st_c._fields, st_c, st_g):
        if name in ("key", "t"):
            assert np.array_equal(np.asarray(a), np.asarray(b)), name
        else:
            np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-8, rtol=0, err_msg=name)
    for k in out_c:
        if k in ("valid", "n_active"):
            assert np.array_equal(out_c[k], out_g[k]), k
        else:
            np.testing.assert_allclose(out_c[k], out_g[k], atol=1e-8, rtol=0, equal_nan=True, err_msg=k)


def test_chunked_equals_unchunked_on_gpu(small_runs):
    ref = run_stream(small_runs, chunk=1)
    res = run_stream(small_runs, chunk=4)
    for k in ref:
        a, b = np.asarray(ref[k]), np.asarray(res[k])
        if k in ("count", "n_active", "ages", "T_burn", "T_meas", "n_steps"):
            assert np.array_equal(a, b), k
        else:
            np.testing.assert_allclose(a, b, atol=1e-10, rtol=0, equal_nan=True, err_msg=k)


_CHILD = r'''
import json, sys
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[1] + "/tests")
import numpy as np, jax
from conftest import one_batch
from mup.config import Config, make_runs
from mup.protocol import run_stream
cfg = Config(N=[20], readout=["gated_shared", "learned_shared", "gated_frozen", "frozen", "task_heads"],
             delta0=[1.0], lam=[1.0], seeds=2, T_factor=0.5, T_min=6)
res = run_stream(one_batch(make_runs(cfg)))
print(json.dumps({"backend": jax.default_backend(), "devices": [str(d) for d in jax.devices()],
                  "E0": np.asarray(res["E0"]).tolist(), "sum_s": np.asarray(res["sum_s"]).tolist(),
                  "count": np.asarray(res["count"]).tolist()}))
'''


@pytest.fixture
def small_share(monkeypatch):
    """Subprocesses preallocate 20 % of a card, leaving room beside this process."""
    monkeypatch.setattr(driver, "GPU_MEM_SHARE", 0.2)


def test_all_gpus_concurrently(tmp_path, small_share):
    """One ``run_stream`` per GPU at the same time, each process pinned to its card."""
    gpus = driver.visible_gpus() or list(range(len(GPUS)))
    script = tmp_path / "child.py"
    script.write_text(_CHILD)
    procs = []
    for s in range(len(gpus)):
        env = driver.worker_env(s, gpus, 1, 1)
        procs.append(subprocess.Popen([sys.executable, str(script), ROOT], env=env,
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
    outs = []
    for pr in procs:
        o, e = pr.communicate(timeout=600)
        assert pr.returncode == 0, e[-2000:]
        outs.append(json.loads(o.strip().splitlines()[-1]))
    for o in outs:
        assert o["backend"] in ("gpu", "cuda", "rocm"), o["backend"]
        assert len([d for d in o["devices"] if "cpu" not in d.lower()]) == 1, o["devices"]   # one card per process
    ref = run_stream(_runs())
    for o in outs:
        np.testing.assert_allclose(np.array(o["E0"]), ref["E0"], atol=1e-8, rtol=0)
        np.testing.assert_allclose(np.array(o["sum_s"]), ref["sum_s"], atol=1e-6, rtol=0)
        assert np.array_equal(np.array(o["count"]), ref["count"])


def test_driver_spreads_batches_over_gpus(tmp_path, small_share):
    """The pilot sweep (N = 10) through ``driver.launch`` with one process per GPU."""
    gpus = driver.visible_gpus() or list(range(len(GPUS)))
    config = os.path.join(ROOT, "configs", "pilot.yaml")
    cfg = Config.from_yaml(config)
    out = str(tmp_path / "pilot")
    batches = io.plan_sweep(cfg, out, [10], config)
    res = driver.launch(config, out, 0, batch_ids=list(range(len(batches))), log=lambda m: None,
                        gpus=gpus, per_gpu=1, chunk=8)
    logs = {i: open(os.path.join(out, "logs", f"batch_{i:05d}.log")).read()[-1500:]
            for i in res["failed"] if os.path.exists(os.path.join(out, "logs", f"batch_{i:05d}.log"))}
    assert res["failed"] == [] and len(res["done"]) == len(batches), logs
    used = set()
    for i in range(len(batches)):
        assert os.path.exists(io.batch_path(out, i)), i
        log = open(os.path.join(out, "logs", f"batch_{i:05d}.log")).read()
        assert log.startswith(f"# batch {i} on GPU "), log[:80]
        used.add(log.splitlines()[0].split("GPU ")[1])
        assert "8 tasks per device call" in log and "cpu)" not in log, log[-300:]
    assert len(used) == min(len(gpus), len(batches)), used
    man = json.load(open(io.manifest_path(out)))
    assert all(b["status"] == "done" for b in man["batches"])
