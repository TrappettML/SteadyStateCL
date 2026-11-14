import math

import pytest

from sscl_core.config import Config, make_batches, make_runs
from sscl_core.observe import age_schedule, half_life
import numpy as np


def test_pop_duration_and_contraction():
    cfg = Config(solver="pop", N=[20], delta0=[-0.75, 0.0, 16.0], lam=[0.5, 4.0], seeds=1, readout=["gated_shared"])
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


def test_sgd_batch_divides_the_steps_and_names_the_kernel():
    """B samples per step: n = round(λD/(η ā² B)) steps over the same α, the solver token carries B (so kernels and
    vmap batches are per B), B = 1 is plain "sgd", the population solver ignores it, and B beyond 2D is refused."""
    base = dict(N=[10], delta0=[0.0, 4.0], lam=[1.0], seeds=1, readout=["gated_shared"])
    one = make_runs(Config(**base))
    for B in (1, 8, 40):
        runs = make_runs(Config(**base, sgd_batch=B))
        for r, r1 in zip(runs, one):
            assert r.sgd_batch == B and r.solver == ("sgd" if B == 1 else f"sgd{B}") and r.eta == r1.eta
            assert r.n_steps == round(r1.n_steps / B) and abs(r.alpha_task - r1.alpha_task) < 1e-12
    assert all(r.solver == "sgd" and r.batch_key() == r1.batch_key() for r, r1 in zip(make_runs(Config(**base, sgd_batch=1)), one))
    pop = make_runs(Config(**base, solver="pop", sgd_batch=8))
    assert all(r.solver == "pop" and r.sgd_batch == 0 and r.n_steps == 10 for r in pop)
    make_runs(Config(**base, sgd_batch=80))                                     # B = 2D: (B/D) η ā² = 0.1, the limit
    for B in (0, 81):
        with pytest.raises(ValueError):
            make_runs(Config(**base, sgd_batch=B))


def test_default_is_sgd_at_the_largest_allowed_rate():
    """With no solver or η given a run is online SGD at η = sgd_contraction/(1 + δ₀), so every δ₀ of a grid takes
    n = round(λD/sgd_contraction) steps per task on the nominal clock; "auto" is 1 for the population solver."""
    cfg = Config(N=[10], delta0=[-0.75, 0.0, 4.0, 16.0], lam=[1.0], seeds=1, readout=["gated_shared"])
    assert cfg.solver == "sgd" and tuple(cfg.eta) == ("auto",)
    runs = make_runs(cfg)
    for r in runs:
        assert r.solver == "sgd" and r.clock == "nominal" and r.h == 0.0
        assert r.eta * (1 + r.delta0) <= cfg.sgd_contraction and abs(r.eta * (1 + r.delta0) - 0.05) < 1e-12
        assert r.n_steps == round(1.0 * 40 / 0.05)
    assert all(r.eta == 1.0 for r in make_runs(Config(**{**cfg.__dict__, "solver": "pop"})))
    stage1 = Config.from_yaml("configs/stage1.yaml")
    assert stage1.solver == "sgd"
    assert {r.n_steps for r in make_runs(Config(**{**stage1.__dict__, "N": [20], "extra": ()}))} == {round(4.0 * 80 / 0.05 / stage1.sgd_batch)}
    assert stage1.sgd_batch == 16


def test_delta0_guard_and_batching():
    with pytest.raises(ValueError):
        make_runs(Config(solver="pop", delta0=[-1.0]))
    cfg = Config(solver="pop", N=[10, 20], kappa=[1.0, "sqrtN"], delta0=[0.0, 4.0], lam=[1.0], seeds=3, batch_size=4)
    runs = make_runs(cfg)
    batches = make_batches(runs, cfg.batch_size)
    assert sum(len(b) for b in batches) == len(runs)
    for b in batches:
        assert len(b) <= 4 and len({r.batch_key() for r in b}) == 1
    assert any(r.kappa_label == "sqrtN" and abs(r.kappa - math.sqrt(r.N)) < 1e-12 for r in runs)


def test_ungated_listed_once_per_N():
    cfg = Config(solver="pop", N=[10, 20], kappa=[1.0, 2.0], delta0=[0.0], lam=[1.0], seeds=3)
    runs = make_runs(cfg)
    gated = [r for r in runs if r.gated == 1.0]
    ungated = [r for r in runs if r.gated == 0.0]
    assert len(gated) == 2 * 2 * 3 * 3 and len(ungated) == 2 * 2 * 3
    for r in ungated:
        # PLAN.md §5: an ungated run takes its length, ages and batch from the κ_min = 1 gated run at the same N
        ref = next(g for g in gated if g.N == r.N and g.kappa == 1.0 and g.delta0 == r.delta0 and g.lam == r.lam)
        assert r.kappa_label == "sqrtN" and r.pN == r.N and r.kappa_sched == 1.0
        assert (r.T_burn, r.T_meas) == (ref.T_burn, ref.T_meas) == (round(cfg.T_factor * r.N), round(cfg.T_factor * r.N))
        assert r.batch_key() != ref.batch_key()                   # but keeps its own batch (manifest ids stay put)
    assert all(g.kappa_sched == g.kappa for g in gated)
    with pytest.raises(ValueError):
        make_runs(Config(solver="pop", readout=["learned"]))


def test_T_per_N_and_extra_cells():
    """PLAN.md §12.3 (M4–M8 burn-in): T_burn = T_meas = c·N for every κ; ``extra`` adds gated cells off the N × κ product."""
    cfg = Config(solver="pop", N=[10, 20], kappa=[1.0, 2.0], delta0=[0.0], lam=[1.0], seeds=1, T_per_N=7.0, extra=[{"N": 20, "kappa": 4.0}, {"N": 40, "kappa": 1.0}])
    runs = make_runs(cfg)
    assert all(r.T_burn == r.T_meas == 7 * r.N for r in runs)
    assert sorted({(r.N, r.kappa_label) for r in runs if r.gated}) == [(10, "1.0"), (10, "2.0"), (20, "1.0"), (20, "2.0"), (20, "4.0"), (40, "1.0")]
    assert sorted({r.N for r in runs if not r.gated}) == [10, 20]       # an extra N gets no ungated runs
    assert all(r.kappa_sched == 1.0 for r in runs if not r.gated)      # κ_min from the product grid, not the extra cell
    stage1 = Config.from_yaml("configs/stage1.yaml")
    assert len(make_runs(stage1)) == 2448



def test_kappa_range(capsys):
    """Gated runs outside 1 ≤ κ ≤ √N are skipped, never raised on, and listed on stdout (over the whole grid,
    not just N_filter); κ = √N (typed or rounded) is kept; ungated runs are unaffected."""
    G = ["gated_shared", "frozen"]
    runs = make_runs(Config(solver="pop", N=[5, 10, 20], kappa=["1", "4"], readout=G, seeds=1))
    assert sorted({(r.N, r.kappa_label) for r in runs if r.gated}) == [(5, "1"), (10, "1"), (20, "1"), (20, "4")]
    assert sorted(r.N for r in runs if not r.gated) == [5, 10, 20]
    assert all(r.pN <= r.N for r in runs)
    out = capsys.readouterr().out
    assert "(N=5, κ=4: κ > √N" in out and "(N=10, κ=4: κ > √N" in out and "N=20" not in out
    assert {r.N for r in make_runs(Config(solver="pop", N=[5, 20], kappa=["4"], readout=G, seeds=1), N_filter=[5])} == {5}
    assert "(N=5, κ=4" in capsys.readouterr().out
    for cfg, n_gated in ((dict(N=[5, 10], kappa=[4.0]), 0), (dict(N=[100], kappa=[0.5, 1.0]), 1),
                         (dict(N=[100], kappa=[1.0], extra=[{"N": 10, "kappa": 4.0}]), 1)):
        runs = make_runs(Config(solver="pop", readout=G, seeds=1, **cfg))
        assert sum(r.gated == 1.0 for r in runs) == n_gated and "skipped" in capsys.readouterr().out
    runs = make_runs(Config(solver="pop", N=[10], kappa=["sqrtN", 3.1623], readout=["gated_shared"], seeds=1))
    assert len(runs) == 2 and all(abs(r.pN - 10) < 1e-3 for r in runs) and capsys.readouterr().out == ""
    assert len(make_runs(Config(solver="pop", N=[10], kappa=[4.0], readout=["frozen"], seeds=1))) == 1
    assert capsys.readouterr().out == ""


def test_age_schedule_and_half_life():
    ages = age_schedule(100, 1.0, 2000)
    assert ages[0] == 0 and ages[-1] == round(4 * math.log(2) * 100) and np.all(np.diff(ages) > 0)
    assert age_schedule(100, 4.0, 2000)[-1] == ages[-1]                 # κ-free: 4 half-lives of N ln 2 at every κ
    assert age_schedule(100, 1.0, 200)[-1] == 199                       # capped at T − 1
    assert list(ages[:21]) == list(range(21))
    small = age_schedule(5, math.sqrt(5), 8)
    assert small[-1] <= 7
    S = np.exp(-np.arange(10) / 3.0)
    m, cens = half_life(np.arange(10), S)
    assert not cens and abs(m - 3 * math.log(2)) < 0.15
    assert half_life(np.arange(3), np.ones(3))[1] is True


def test_legacy_readout_names_load():
    """Batch files of the first code version named readouts learned/frozen/heads (PLAN.md §4 renamed them)."""
    from sscl_core.config import Run, is_gated
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
    from sscl_core import io
    from sscl_core.config import Run
    from sscl_core.results import load_runs
    r = Run(N=10, D=40, kappa=1.0, readout="gated_shared", delta0=0.0, lam=1.0, seed=0, solver="pop", eta=1.0,
            h=0.1, n_steps=10, alpha_task=1.0, T_burn=4, T_meas=4)
    out = str(tmp_path)
    cfg = Config(solver="pop", name="t", N=[10], kappa=[1.0], readout=["gated_shared"], delta0=[0.0], lam=[1.0], seeds=1, batch_size=64)
    batches = io.plan_sweep(cfg, out)
    b0 = batches[0][0]
    res = {"ages": np.arange(3), "alpha_task": np.ones(len(batches[0])), "T_burn": np.array(b0.T_burn), "T_meas": np.array(b0.T_meas),
           **{f: np.zeros((len(batches[0]), 3, 2)) for f in ("sum_s", "sum_s2", "sum_e", "sum_e2", "count")},
           "S0": np.zeros((b0.T_burn + b0.T_meas, len(batches[0])))}
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


def test_wrong_length_batch_is_rerun(tmp_path):
    """A batch file run under another stream length than the manifest plans (the pre-§5 ungated runs) is pending on resume."""
    import numpy as np
    from sscl_core import io
    out = str(tmp_path)
    cfg = Config(solver="pop", name="t", N=[10], kappa=[1.0], readout=["frozen"], delta0=[0.0], lam=[1.0], seeds=1, batch_size=64)
    batches = io.plan_sweep(cfg, out)
    T_plan = batches[0][0].T_burn + batches[0][0].T_meas
    assert T_plan == 2 * round(cfg.T_factor * 10)                   # 20N/κ_min², not 20
    res = {"ages": np.arange(3), "alpha_task": np.ones(1), "T_burn": np.array(10), "T_meas": np.array(10),
           **{f: np.zeros((1, 3, 2)) for f in ("sum_s", "sum_s2", "sum_e", "sum_e2", "count")}, "S0": np.zeros((20, 1))}
    io.save_batch(out, 0, res, batches[0])                          # the old 20-task file
    assert io.batch_T(io.batch_path(out, 0)) == 20
    assert io.is_stale(io.batch_path(out, 0), T_plan) and not io.is_stale(io.batch_path(out, 0))
    assert io.pending_batches(out) == [0]
    assert io.plan_sweep(cfg, out) and io.read_manifest(out)["batches"][0]["status"] == "stale"
