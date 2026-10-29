"""Test 10 (PLAN.md §7, §12.7.1): F1–F4 and the checks table render from a small synthetic record set without a ``plotting.save`` warning.

The diagnostic families (``--diag``) are rendered too, so they stay usable
for locating a failed check.
"""
import os
import sys
import warnings

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from mup import io  # noqa: E402
from mup.config import READOUT_ORDER, Run, is_gated  # noqa: E402
from mup.observe import age_schedule  # noqa: E402
from mup.protocol import SCALARS, TRACE_QUANT  # noqa: E402
from mup.results import RunRecord  # noqa: E402


def _record(rng, N, readout, d0, lam, seed, T_factor=2.0):
    """One synthetic run: gated and ungated readouts, plausible S(m), E(m) and norm scalars."""
    gated = is_gated(readout)
    kappa = 1.0 if gated else float(np.sqrt(N))
    T_half = int(T_factor * N / kappa ** 2) if gated else 10
    T = 2 * T_half
    run = Run(N=N, D=4 * N, kappa=kappa, readout=readout, delta0=d0, lam=lam, seed=seed, solver="pop", eta=1.0,
              h=0.1, n_steps=10, alpha_task=lam, T_burn=T_half, T_meas=T_half, kappa_label="1.0" if gated else "sqrtN")
    ages = age_schedule(N, kappa, T)
    t = np.arange(T)
    sc = {k: 1.0 + 0.05 * rng.standard_normal(T) for k in SCALARS}
    sc["S0"] = 0.7 + 0.05 * rng.standard_normal(T)
    sc["E0"] = 0.2 + 0.02 * rng.standard_normal(T)
    sc["gauge_delta"] = np.full(T, 1e-14); sc["gauge_uc"] = np.full(T, 1e-14)
    sc["n_active"] = np.full(T, float(N if not gated else max(1, int(np.sqrt(N)))))
    sc["abar2"] = np.full(T, 1.0 + d0)
    sc["Q_q05"], sc["Q_q95"] = sc["Q_act_pre"] - 0.3, sc["Q_act_pre"] + 0.3
    sc["aQ_q05"], sc["aQ_q95"] = sc["aQ_act_post"] - 0.3, sc["aQ_act_post"] + 0.3
    if "frozen" in readout:
        sc["logf_sum"] = np.zeros(T); sc["logf_max"] = np.zeros(T)
        if readout == "frozen":
            sc["a2_act_post"] = np.full(T, 1.0 + d0)
    else:
        sc["logf_sum"] = -0.01 + 0.002 * rng.standard_normal(T)
    if readout == "task_heads":
        sc["a2_act_post"] = sc["a2_act_post"] + 0.002 * t
    S = 0.7 * np.exp(-ages / (0.5 * N / kappa ** 2 + 1.0))
    E = 0.2 + 1.3 * (1 - np.exp(-ages / (0.5 * N / kappa ** 2 + 1.0)))
    n = np.array([max(1.0, min(T_half / 2, T - a)) for a in ages])
    noise = 0.02 * rng.standard_normal((len(ages), 2))
    mom = {"count": np.stack([n, n], 1)}
    mom["sum_s"] = (S[:, None] + noise) * mom["count"]
    mom["sum_s2"] = ((S[:, None] + noise) ** 2 + 0.01) * mom["count"]
    mom["sum_e"] = (E[:, None] + noise) * mom["count"]
    mom["sum_e2"] = ((E[:, None] + noise) ** 2 + 0.01) * mom["count"]
    return RunRecord(run, sc, mom, ages, T_half, T_half, float(lam))


@pytest.fixture
def records():
    rng = np.random.default_rng(0)
    recs = []
    for N in (10, 20):
        for ro in READOUT_ORDER:
            for d0 in (0.0, 4.0):
                for seed in (0, 1):
                    recs.append(_record(rng, N, ro, d0, 1.0, seed))
    return recs


def _trace_dir(tmp_path, records):
    """Write synthetic trace files for the by-step diagnostics (one run per readout)."""
    out = str(tmp_path / "sweep")
    rng = np.random.default_rng(1)
    for i, rec in enumerate(r for r in records if r.run.N == 10 and r.run.seed == 0 and r.run.delta0 == 0.0):
        n_steps, T = 10, rec.T
        tr = np.cumsum(0.01 * rng.standard_normal((T, n_steps, len(TRACE_QUANT))), axis=1).astype(np.float32)
        res = {"trace": tr[None], "trace_idx": np.arange(n_steps, dtype=np.int32), "T_burn": rec.T_burn, "T_meas": rec.T_meas, "n_steps": n_steps}
        io.save_traces(out, i, res, [rec.run])
    return out


def test_default_families_render_without_warnings(records, tmp_path):
    """F1, F2 (+ _lt), F3, F4 (+ _lt) and the checks table; nothing else is written by default."""
    import analyze
    import plot_check2
    import plot_norms
    from mup import plotting as P
    fig_dir = str(tmp_path / "figures")
    out_dir = str(tmp_path / "sweep")
    skipped = ["N=10 k=1.0 gated_frozen d0=-0.75 lam=1.0 eta=1.0 seed=3"]
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        conds = analyze.collect(records)
        index = []
        index += plot_check2.run(out_dir, fig_dir, records, True, skipped=skipped)
        index += plot_norms.run(out_dir, fig_dir, records, True)
        analyze.fig_curves(conds, fig_dir, index, True)
        analyze.fig_curvesN(conds, fig_dir, index, True)
        P.write_index(fig_dir, index)
    assert {f for f, _, _ in index} == {"norms", "curves", "checks"}
    assert sorted(os.listdir(fig_dir)) == ["INDEX.md", "checks", "curves", "norms"]
    stems = {os.path.basename(p) for _, _, p in index}
    for s in ("norms_k1.0_d0.0_lam1.0_pop_eta1.0.png", "normsN_k1.0_d0.0_lam1.0_pop_eta1.0.png", "normsN_lt_k1.0_d0.0_lam1.0_pop_eta1.0.png",
              "curves_k1.0_d0.0_lam1.0_pop_eta1.0.png", "curvesN_k1.0_d0.0_lam1.0_pop_eta1.0.png", "curvesN_lt_k1.0_d0.0_lam1.0_pop_eta1.0.png",
              "checks_pop_eta1.0.png"):
        assert s in stems, s
    for f in ("checks_pop_eta1.0.csv", "check2_cells.csv", "checks_pop_eta1.0.npz"):
        assert os.path.exists(os.path.join(fig_dir, "checks", f)), f
    # every table row has a mark or text for every column
    import csv
    rows = list(csv.DictReader(open(os.path.join(fig_dir, "checks", "checks_pop_eta1.0.csv"))))
    assert len(rows) == len(plot_check2.CHECK_ROWS) * 10
    feas = [r for r in rows if r["row"].startswith("feasibility") and r["readout"] == "gated_frozen" and r["N"] == "10"]
    assert feas and feas[0]["mark"] == "fail"           # one skipped run of five = 20 % > 10 %
    gauge_na = [r for r in rows if r["row"].startswith("1 gauge") and "frozen" in r["readout"]]
    assert gauge_na and all(r["mark"] == "na" for r in gauge_na)


def test_diag_families_render_without_warnings(records, tmp_path):
    import analyze
    import plot_check2
    import plot_norms
    fig_dir = str(tmp_path / "figures")
    out_dir = _trace_dir(tmp_path, records)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        conds = analyze.collect(records)
        index = []
        index += plot_check2.run(out_dir, fig_dir, records, False, levels=True, diag=True)
        index += plot_norms.run(out_dir, fig_dir, records, False, diag=True)
        analyze.fig_mhalf_vs_N(conds, fig_dir, index, False)
        analyze.fig_mhalf_vs_E0(conds, fig_dir, index, False)
        analyze.fig_kaczmarz(conds, fig_dir, index, False)
        analyze.fig_cv(analyze.report_cv(conds), fig_dir, index, False)
    fams = {f for f, _, _ in index}
    assert {"diag/check2", "diag/by_step", "diag/mhalf", "diag/kaczmarz"} <= fams
    assert sorted(os.listdir(os.path.join(fig_dir, "diag"))) == ["by_step", "check2", "kaczmarz", "mhalf"]


def test_na_pairs_have_no_rows(records):
    import plot_check2
    rows = plot_check2._grid_stats(records)
    for (q, ro) in plot_check2.NA_BY_CONSTRUCTION:
        assert not any(r["quantity"] == q and r["readout"] == ro for r in rows)
    assert any(r["quantity"] == "dm_half" for r in rows)


def test_cells_filter():
    import analyze
    cells = analyze.parse_cells("d0=0/4,lam=1,N=10,k=1.0")
    assert cells == {"delta0": {0.0, 4.0}, "lam": {1.0}, "N": {10}, "kappa_label": {"1.0"}}
    assert analyze.parse_cells(None) is None


def test_helpers():
    from mup import plotting as P
    assert P.fmt(4.0) == "4" and P.fmt("1.0") == "1" and P.fmt("sqrtN") == "sqrtN" and P.fmt(0.5) == "0.5"
    x = np.arange(20.0)
    Y = np.stack([x, x + 1])
    xb, mb, sb = P.block_mean(x, Y, 5, cuts=(10, 15))
    assert len(xb) == 4 and np.allclose(mb[0], 2.5) and np.isfinite(sb).all()
    xb, _, _ = P.block_mean(x, Y, 7, cuts=(10,))
    assert np.all(np.diff(xb) > 0) and xb[1] < 10 < xb[2]           # a block never straddles a cut
