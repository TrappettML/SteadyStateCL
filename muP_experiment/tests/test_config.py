import math

import pytest

from mup.config import Config, make_batches, make_runs
from mup.observe import age_schedule, half_life
import numpy as np


def test_pop_duration_and_contraction():
    cfg = Config(N=[20], delta0=[-0.75, 0.0, 16.0], lam=[0.5, 4.0], seeds=1, readout=["gated_shared"])
    for r in make_runs(cfg):
        assert abs(r.h * r.eta * (1 + r.delta0) - 0.1) < 1e-12 or r.h <= cfg.h_max
        assert abs(r.h * r.n_steps - r.alpha_task) < 1e-12
        assert r.n_steps == math.ceil(10 * r.lam - 1e-9)


def test_sgd_duration():
    cfg = Config(N=[10], delta0=[0.0], lam=[1.0], seeds=1, solver="sgd", eta=[0.05], readout=["gated_frozen"])
    (r,) = make_runs(cfg)
    assert r.n_steps == round(1.0 / 0.05 * 40)
    with pytest.raises(ValueError):
        make_runs(Config(N=[10], delta0=[4.0], lam=[1.0], seeds=1, solver="sgd", eta=[0.05]))


def test_delta0_guard_and_batching():
    with pytest.raises(ValueError):
        make_runs(Config(delta0=[-1.0]))
    cfg = Config(N=[10, 20], kappa=[1.0, "sqrtN"], delta0=[0.0, 4.0], lam=[1.0], seeds=3, batch_size=4)
    runs = make_runs(cfg)
    batches = make_batches(runs, cfg.batch_size)
    assert sum(len(b) for b in batches) == len(runs)
    for b in batches:
        assert len(b) <= 4 and len({r.batch_key() for r in b}) == 1
    assert any(r.kappa_label == "sqrtN" and abs(r.kappa - math.sqrt(r.N)) < 1e-12 for r in runs)


def test_ungated_listed_once_per_N():
    cfg = Config(N=[10, 20], kappa=[1.0, 2.0], delta0=[0.0], lam=[1.0], seeds=3)
    runs = make_runs(cfg)
    gated = [r for r in runs if r.gated == 1.0]
    ungated = [r for r in runs if r.gated == 0.0]
    assert len(gated) == 2 * 2 * 3 * 3 and len(ungated) == 2 * 2 * 3
    for r in ungated:
        assert r.kappa_label == "sqrtN" and r.pN == r.N and r.T_burn == max(cfg.T_min, round(cfg.T_factor))
    with pytest.raises(ValueError):
        make_runs(Config(readout=["learned"]))


def test_age_schedule_and_half_life():
    ages = age_schedule(100, 1.0, 2000)
    assert ages[0] == 0 and ages[-1] == 1999 and np.all(np.diff(ages) > 0)
    assert list(ages[:21]) == list(range(21))
    small = age_schedule(5, math.sqrt(5), 8)
    assert small[-1] <= 7
    S = np.exp(-np.arange(10) / 3.0)
    m, cens = half_life(np.arange(10), S)
    assert not cens and abs(m - 3 * math.log(2)) < 0.15
    assert half_life(np.arange(3), np.ones(3))[1] is True


def test_legacy_readout_names_load():
    """Batch files of the first code version named readouts learned/frozen/heads (PLAN.md §4 renamed them)."""
    from mup.config import Run, is_gated
    r = Run(N=10, D=40, kappa=1.0, readout="gated_shared", delta0=0.0, lam=1.0, seed=0, solver="pop", eta=1.0,
            h=0.1, n_steps=10, alpha_task=1.0, T_burn=4, T_meas=4)
    d = r.to_dict(); d.pop("gated"); d["readout"] = "learned"       # schema-1 files carry no "gated" key
    assert Run.from_dict(d).readout == "gated_shared"
    d["readout"] = "heads"
    assert Run.from_dict(d).readout == "task_heads"
    d["readout"] = "frozen"
    assert Run.from_dict(d).readout == "gated_frozen"
    d["kappa_label"] = "sqrtN"
    assert Run.from_dict(d).readout == "frozen"
    assert Run.from_dict(r.to_dict()).readout == "gated_shared"
    assert is_gated("frozen") is False and is_gated("learned") is True and is_gated("gated_frozen") is True


def test_stale_schema_batch_is_rerun(tmp_path):
    """A batch file written under an older result schema counts as pending on resume, and load_runs warns about it."""
    import json
    import numpy as np
    import pytest
    from mup import io
    from mup.config import Run
    from mup.results import load_runs
    r = Run(N=10, D=40, kappa=1.0, readout="gated_shared", delta0=0.0, lam=1.0, seed=0, solver="pop", eta=1.0,
            h=0.1, n_steps=10, alpha_task=1.0, T_burn=4, T_meas=4)
    out = str(tmp_path)
    cfg = Config(name="t", N=[10], kappa=[1.0], readout=["gated_shared"], delta0=[0.0], lam=[1.0], seeds=1, batch_size=64)
    batches = io.plan_sweep(cfg, out)
    res = {"ages": np.arange(3), "alpha_task": np.ones(len(batches[0])), "T_burn": np.array(4), "T_meas": np.array(4),
           **{f: np.zeros((len(batches[0]), 3, 2)) for f in ("sum_s", "sum_s2", "sum_e", "sum_e2", "count")},
           "S0": np.zeros((8, len(batches[0])))}
    io.save_batch(out, 0, res, batches[0])
    assert io.pending_batches(out) == []
    # rewrite the file as the older schema
    z = dict(np.load(io.batch_path(out, 0), allow_pickle=False))
    meta = json.loads(str(z["meta_json"])); meta["schema"] = 1
    z["meta_json"] = np.array(json.dumps(meta))
    np.savez(io.batch_path(out, 0), **z)
    assert io.is_stale(io.batch_path(out, 0))
    assert io.pending_batches(out) == [0]
    assert io.plan_sweep(cfg, out) and io.read_manifest(out)["batches"][0]["status"] == "stale"
    with pytest.warns(UserWarning, match="schema"):
        recs, _ = load_runs(out)
    assert recs and recs[0].schema == 1
    del r
