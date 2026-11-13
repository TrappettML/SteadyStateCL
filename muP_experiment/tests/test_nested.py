"""``mup.nested``: the five-model ladder of ``scripts/run_nested.py`` — stream, update rules, statistics, one run."""
import dataclasses
import math

import numpy as np
import pytest

from mup import nested as X


def cond(**kw):
    return X.Cond(**{"N": 24, "D": 48, "p": 0.3, "rho": 0.0, "delta0": 1.0, "lam": 1.0, **kw})


def test_stream_is_a_function_of_the_task_index():
    s1, s2 = X.Stream(cond(), 3), X.Stream(cond(delta0=10.0, lam=5.0), 3)
    for t in (-5, 0, 7):
        (g1, w1), (g2, w2) = s1.task(t), s2.task(t)
        assert (g1 == g2).all() and (w1 == w2).all()          # δ₀ and λ do not move the stream
    assert not np.array_equal(s1.task(0)[1], s1.task(1)[1])
    assert not np.array_equal(s1.task(0)[1], X.Stream(cond(), 4).task(0)[1])
    W, a = s1.init()
    assert np.allclose(a * a - (W * W).sum(1) / W.shape[1], 1.0)        # a_i² − Q_ii = δ₀


@pytest.mark.parametrize("latent,want", [("common", 0.7), ("sphere", 0.0)])
def test_teacher_similarity(latent, want):
    c = cond(D=400, rho=0.7, latent=latent, rank=4)
    s = X.Stream(c, 0)
    Wt = np.stack([s.task(t)[1] for t in range(300)])
    T = Wt @ Wt.T / c.D
    off = T[~np.eye(len(T), dtype=bool)]
    assert abs(np.diag(T).mean() - 1.0) < 0.02
    assert abs(off.mean() - want) < 0.02                       # common: cosine ρ; sphere: ρ z·z′, zero mean
    if latent == "sphere":
        assert abs((off ** 2).mean() - (0.7 ** 2 / 4 + 1 / c.D)) < 0.02


@pytest.mark.parametrize("clock", ["realized", "nominal"])
def test_delta_step_is_a_relaxed_projection(clock):
    c = cond()
    s = X.Stream(c, 1)
    W, a = s.init()
    m = X.DeltaModel(c, W, a, rate=1.3, clock=clock)
    (g, w), (g2, _) = s.task(0), s.task(1)
    k, k2 = g * a, g2 * a
    e0, u2 = m.u(k[None])[0] - w, m.u(k2[None])[0]
    m.train(0, g, w)
    kk = float(k @ k)
    theta = m.theta(kk)
    lam_c = 1.3 * (c.lam if clock == "realized" else c.lam / (1 + c.delta0) * kk / c.pN)
    assert theta == pytest.approx(1 - math.exp(-lam_c))
    assert np.allclose(m.u(k[None])[0] - w, (1 - theta) * e0)                  # the task's own error shrinks by 1 − θ
    assert np.allclose(m.u(k2[None])[0] - u2, -theta * (k @ k2) / kk * e0)     # another task moves by its key overlap
    assert (m.W[~g] == W[~g]).all() and (m.a == a).all()


@pytest.mark.parametrize("clock", ["realized", "nominal"])
def test_raw_theta_is_the_frozen_population_flow(clock):
    """θ_c = 1 − exp(−Λ_c ā²_c) is what the project's population kernel removes with the readout frozen."""
    import jax
    import jax.numpy as jnp
    import mup._x64  # noqa: F401
    from mup.model import Params, train_block
    c = cond()
    s = X.Stream(c, 2)
    W, a = s.init()
    g, w = s.task(0)
    n = 400
    f = lambda v: jnp.asarray(v, dtype=jnp.float64)
    p = Params(Delta0=f(c.delta0), eta=f(1.0), h=f(c.lam / (1 + c.delta0) / n), pN=f(c.pN), lam=f(c.lam), gated=f(1.0),
               learn_a=f(0.0), do_proj=f(0.0), use_heads=f(0.0), sigma0=f(1.0), clock=f(1.0 if clock == "realized" else 0.0))
    out = train_block(W, a, g.astype(float), w, p, jax.random.PRNGKey(0), n_steps=n, solver="pop", K=c.N)
    m = X.DeltaModel(c, W, a, clock=clock)
    m.train(0, g, w)
    assert np.allclose(np.asarray(out["W"]), m.W, atol=5e-3)
    k = g * a
    assert np.linalg.norm(m.u(k[None])[0] - w) / np.linalg.norm(k @ W / c.pN - w) == pytest.approx(1 - m.theta(float(k @ k)))


def test_hebb_lazy_decay_and_linearity_in_b():
    c = cond()
    s = X.Stream(c, 0)
    _, a = s.init()
    gamma, b = 0.3, 0.37
    m = X.HebbModel(c, a, gamma)
    W = np.zeros((c.N, c.D))
    for t in range(400):                                       # long enough for the scale to be folded in
        g, w = s.task(t)
        m.train(t, g, w)
        k = g * a
        W = (1 - gamma) * W + b * (c.lam * c.pN / (k @ k)) * np.outer(k, w)
    assert np.allclose(b * m.W, W)
    r = X.follow(X.HebbModel(c, a, gamma), s, 0, 30, 10, np.arange(5))
    assert (r["count"] == 10).all()
    g, w = s.task(0)
    assert X.errors(r["post"][:, 0], b) == pytest.approx(((b * c.lam - 1) ** 2) * (w @ w) / c.D)   # first write: u = bλ w


def test_follow_probes_every_learned_task_at_every_age():
    c = cond()
    s = X.Stream(c, 0)
    W, a = s.init()
    ages = np.array([0, 1, 3, 6])
    r = X.follow(X.DeltaModel(c, W, a), s, 5, 4 + 6, 4, ages)
    assert (r["count"] == 4).all()
    post = X.errors(r["post"])
    assert X.errors(r["age"])[0] == pytest.approx(post[:4].mean())      # age 0 is the error just after learning
    m = X.DeltaModel(c, W, a)                                           # by hand: task 5 at age 3
    for t in range(5, 9):
        m.train(t, *s.task(t))
    g, w = s.task(5)
    e = m.u((g * a)[None])[0] - w
    r1 = X.follow(X.DeltaModel(c, W, a), s, 5, 4, 1, np.array([3]))
    assert X.errors(r1["age"])[0] == pytest.approx(e @ e / c.D)


def test_half_age_and_tau_half():
    ages = np.array([0, 1, 2, 4, 8, 16, 32])
    assert X.half_age(np.exp(-ages / 10.0), ages) == pytest.approx(10 * math.log(2), rel=0.03)
    assert math.isnan(X.half_age(np.full(7, 0.8), ages)) and math.isnan(X.half_age(-np.ones(7), ages))
    M = np.exp(-ages[:, None] / np.array([9.0, 10.0, 11.0])[None, :])
    t, se = X.tau_half(M, ages)
    assert t == pytest.approx(10 * math.log(2), rel=0.05) and 0 < se < 1.0


def test_coincide_rule():
    ages = np.arange(0, 200, 2)
    rng = np.random.default_rng(0)
    base = lambda tau, n=10: 0.9 * np.exp(-ages[:, None] / tau) + 0.004 * rng.standard_normal((len(ages), n))
    same = X.coincide(base(40.0), base(40.0), ages)
    assert same["coincide"] and same["max_dev"] < X.TOL_M and same["dtau"] < X.TOL_TAU
    slow = X.coincide(base(40.0), base(52.0), ages)             # τ½ off by 30 %
    assert not slow["coincide"] and not slow["tau_ok"]
    low = X.coincide(base(40.0), base(40.0) - 0.1, ages)        # shifted by more than 5 % of M(0)
    assert not low["coincide"] and not low["m_ok"]
    A = base(40.0)
    tight = X.coincide(A, A + 0.01 * np.exp(-ages[:, None] / 40.0), ages)   # 1 %: inside 5 % and inside the noise
    assert tight["m_ok"] and tight["tau_ok"]
    one = X.coincide(A[:, :1], A[:, :1], ages)
    assert one["coincide"] and one["sem_ok"] is None


def test_key_fraction_and_fits():
    E = np.array([[0.2], [0.5], [0.8]])
    Ev = np.array([[0.2], [0.4], [0.5]])
    phi = X.key_fraction(E, Ev)[:, 0]
    assert math.isnan(phi[0]) and phi[1:] == pytest.approx([1 / 3, 0.5])
    t = X.age_grid(300, 40)
    assert t[0] == 0 and t[-1] == 300 and (np.diff(t) > 0).all()
    assert X.fit_shapes(t, 0.8 * np.exp(-t / 50.0))["exponential"]["params"] == pytest.approx([0.8, 50.0], rel=1e-3)
    assert X.fit_shapes(t, 0.8 * (1 + t / 20.0) ** -1.5)["best"] == "power"
    assert X.fit_shapes(t, 0.8 * np.exp(-(t / 50.0) ** 0.6))["best"] == "stretched"


def test_run_seed_and_hebb_end_to_end():
    c = cond(N=16, D=32)
    st = X.Settings(burn=1.0, burn_max=1.0, learn=0.5, tau_max=1.0, n_ages=8, gf_step=0.05)
    cd, sd = dataclasses.asdict(c), dataclasses.asdict(st)
    r = X.run_seed(cd, 0, sd)
    assert set(r["models"]) == {"sgd", "gf", "delta_matched", "delta_raw"}
    cv = {m: X.curves([r], m) for m in r["models"]}
    for m, v in cv.items():
        assert np.isfinite(v["M"]).all() and 0 < v["eps0"][0] < v["E_inf"][0]
    assert r["models"]["delta_raw"]["rate"] == pytest.approx(1.0)                       # nothing fitted
    assert r["models"]["delta_raw"]["theta"] == pytest.approx(1 - math.exp(-1.0))       # θ = 1 − e^{−λ}, clock (ii)
    assert r["models"]["delta_matched"]["rate"] == pytest.approx(r["models"]["gf"]["rate"], rel=1e-6)
    assert r["models"]["gf"]["rate"] > 1.0                                              # the readout channel only adds
    assert abs(cv["sgd"]["eps0"][0] / cv["gf"]["eps0"][0] - 1) < 0.3                     # same model up to SGD noise
    assert cv["gf"]["Ev"] is not None and cv["delta_raw"]["Ev"] is None
    again = X.run_seed(cd, 0, sd)
    assert np.array_equal(again["models"]["sgd"]["age"], r["models"]["sgd"]["age"])     # reproducible
    h = [X.run_hebb(cd, 0, sd, r["a_fork"], r["t_fork"], g) for g in (0.05, 0.2)]
    gf = cv["gf"]
    best = X.tune_hebb({0.05: [h[0]], 0.2: [h[1]]}, r["ages"], c.lam, float(gf["eps0"].mean()), 8.0)
    assert np.isfinite(best["loss"]) and best["gamma"] in (0.05, 0.2) and best["b"] > 0


def test_jax_backend_matches_numpy_backend():
    """The device path (seeds of a condition as one vmap batch of ``model.train_block``) against the NumPy path."""
    c = cond(N=16, D=32)
    kw = dict(burn=1.0, burn_max=1.0, learn=0.5, tau_max=1.0, n_ages=8, gf_step=0.05)
    cd = dataclasses.asdict(c)
    nps = X.run_seeds(cd, [0, 1], dataclasses.asdict(X.Settings(backend="numpy", **kw)))
    jxs = X.run_seeds(cd, [0, 1], dataclasses.asdict(X.Settings(backend="jax", **kw)))
    for n, j in zip(nps, jxs):
        assert n["t_fork"] == j["t_fork"]
        assert np.allclose(n["a_fork"], j["a_fork"], rtol=1e-8, atol=1e-10)          # same population steps
        for m in ("gf", "delta_matched", "delta_raw"):
            for k in ("age", "pre", "post"):
                assert np.allclose(n["models"][m][k], j["models"][m][k], rtol=1e-6, atol=1e-9), (m, k)
        assert np.allclose(n["models"]["gf"]["age_value"], j["models"]["gf"]["age_value"], rtol=1e-6, atol=1e-9)
        # SGD: different input draws of the same process
        en, ej = X.errors(n["models"]["sgd"]["age"]), X.errors(j["models"]["sgd"]["age"])
        assert np.isfinite(ej).all() and abs(ej[0] / en[0] - 1) < 0.2
        assert j["models"]["sgd"]["rate"] == pytest.approx(n["models"]["sgd"]["rate"], rel=0.05)
    one = X.run_seeds(cd, [1], dataclasses.asdict(X.Settings(backend="jax", **kw)))[0]   # batching does not change a seed
    assert np.allclose(one["models"]["sgd"]["age"], jxs[1]["models"]["sgd"]["age"], rtol=1e-8, atol=1e-10)
