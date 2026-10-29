"""Config dataclass, YAML loader, grid → runs → batches (PLAN.md §3.6, §4)."""
from __future__ import annotations

import dataclasses
import hashlib
import itertools
import math
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import yaml

READOUTS = {
    #                  gated, learn_a, do_proj, use_heads        (PLAN.md §4)
    "gated_shared":   (1.0, 1.0, 1.0, 0.0),   # one a, learned every task; κ-sparse gates
    "learned_shared": (0.0, 1.0, 1.0, 0.0),   # one a, learned every task; no gating
    "gated_frozen":   (1.0, 0.0, 0.0, 0.0),   # a = a⁽⁰⁾ forever, Q unpinned; κ-sparse gates
    "frozen":         (0.0, 0.0, 0.0, 0.0),   # a = a⁽⁰⁾ forever, Q unpinned; no gating
    "task_heads":     (1.0, 1.0, 1.0, 1.0),   # fresh head per task, stored heads rescaled; κ-sparse gates
}
READOUT_ORDER = tuple(READOUTS)
#: readout names of the first code version (batch files written before PLAN.md §4 renamed them).
#: All three were gated then; a legacy "frozen" at κ = √N is the ungated point and is mapped to
#: today's "frozen" by ``Run.from_dict`` (a legacy file has no "gated" key).
LEGACY_READOUTS = {"learned": "gated_shared", "heads": "task_heads"}


def is_gated(readout: str) -> bool:
    """
    Whether a readout key draws κ-sparse gates (True) or activates every unit (False).

    Legacy names of schema-1 batch files (``LEGACY_READOUTS``) are accepted.

    Parameters
    ----------
    readout : str
        One of ``READOUTS``.

    Returns
    -------
    bool
    """
    return READOUTS[LEGACY_READOUTS.get(readout, readout)][0] == 1.0


@dataclass
class Config:
    """
    A sweep's grid and solver settings, loaded from YAML (PLAN.md §3.6, §4, §12.2).

    Attributes
    ----------
    name : results subfolder
    N : widths; D = D_factor·N
    kappa : gate scales for the gated readouts (floats or "sqrtN")
    readout : readout keys (``READOUTS``)
    delta0 : dimensionless imbalances δ₀ = ηΔ₀ (> −1)
    lam : learning per task λ = Λ ā²
    seeds : int (→ range) or explicit list
    solver : "pop" or "sgd"
    eta : per-run η (pop: 1; sgd: values with η ā² ≤ sgd_contraction)
    h_max, pop_contraction, sgd_contraction : step-size rules of §3.6
    D_factor, T_factor, T_min : D = D_factor·N; T_burn = T_meas = max(T_min, round(T_factor·N/κ²))
    batch_size : max runs per vmap batch
    n_ages_geom, m_max_factor : probe-age schedule (§5)
    trace, trace_points : per-step traces (§12.2), separate configs only
    """
    name: str = "pilot"
    N: Sequence[int] = (10, 20)
    kappa: Sequence[Any] = (1.0,)          # floats or the string "sqrtN"
    readout: Sequence[str] = READOUT_ORDER
    delta0: Sequence[float] = (0.0,)
    lam: Sequence[float] = (1.0,)
    seeds: Any = 2                          # int (→ range) or explicit list
    solver: str = "pop"                     # "pop" | "sgd"
    eta: Sequence[float] = (1.0,)           # per-run η (pop: 1; sgd: values with η ā² ≤ 0.05)
    h_max: float = 1.0                      # cap on the population step in α
    pop_contraction: float = 0.1            # h η ā² ≤ this
    sgd_contraction: float = 0.05           # η ā² ≤ this
    D_factor: int = 4                       # D = D_factor · N
    T_factor: float = 10.0                  # T_burn = T_meas = T_factor · N / κ²
    T_min: int = 4                          # floor on T_burn and T_meas
    batch_size: int = 64
    n_ages_geom: int = 24
    m_max_factor: float = 4.0
    trace: bool = False                     # per-step norm traces (PLAN.md §12.2); separate configs only
    trace_points: int = 200                 # stored steps per task in a trace run

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """
        Load a config; unknown keys are an error.

        Parameters
        ----------
        path : str

        Returns
        -------
        Config
        """
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**raw)

    def to_dict(self) -> dict:
        """Plain-dict form (what the manifest stores)."""
        return dataclasses.asdict(self)

    def seed_list(self) -> list[int]:
        """
        Seeds as a list: ``range(seeds)`` for an int, else the given list.

        Returns
        -------
        list of int
        """
        return list(range(int(self.seeds))) if isinstance(self.seeds, int) else [int(s) for s in self.seeds]


@dataclass
class Run:
    """
    One (seed, κ, δ₀, λ, readout, η) run; every field is a host-side constant (PLAN.md §3.6).

    The derived properties (``pN``, ``Delta0``, ``gated``, ``learn_a``, ``do_proj``,
    ``use_heads``) are what ``protocol.make_params`` packs into ``Params``.
    """

    N: int
    D: int
    kappa: float
    readout: str
    delta0: float
    lam: float
    seed: int
    solver: str
    eta: float
    h: float
    n_steps: int
    alpha_task: float
    T_burn: int
    T_meas: int
    kappa_label: str = ""
    n_ages_geom: int = 24
    m_max_factor: float = 4.0
    trace: bool = False
    trace_points: int = 200

    @property
    def pN(self) -> float:
        """Nominal active count κ√N; exactly N for an ungated run (PLAN.md §2)."""
        return float(self.N) if not is_gated(self.readout) else self.kappa * math.sqrt(self.N)

    @property
    def Delta0(self) -> float:
        """Raw imbalance Δ₀ = δ₀/η."""
        return self.delta0 / self.eta

    @property
    def gated(self) -> float:
        """1.0 for κ-sparse gates, 0.0 for every unit active."""
        return READOUTS[self.readout][0]

    @property
    def learn_a(self) -> float:
        """1.0 if the readout learns, 0.0 if frozen."""
        return READOUTS[self.readout][1]

    @property
    def do_proj(self) -> float:
        """1.0 if the projection onto Δ_i = Δ₀ runs after every step."""
        return READOUTS[self.readout][2]

    @property
    def use_heads(self) -> float:
        """1.0 for task-specific heads."""
        return READOUTS[self.readout][3]

    def rng_seed(self) -> int:
        """Seed of the run's PRNGKey: a hash of (seed, N, κ, readout, δ₀, λ).

        Every run of the grid draws its own weights, teachers, gates and SGD
        inputs; nothing is shared across seeds or across conditions.  The
        solver, step size and η are left out so that η-halving and the
        population-vs-SGD validation (scripts/validate.py) compare the same
        task streams.
        """
        tag = f"{self.seed}|{self.N}|{self.kappa_label or self.kappa}|{self.readout}|{self.delta0}|{self.lam}"
        return int.from_bytes(hashlib.sha256(tag.encode()).digest()[:4], "little")

    def batch_key(self) -> tuple:
        """Fields a vmap batch must share: (N, D, κ, solver, n_steps, T_burn, T_meas, trace)."""
        return (self.N, self.D, self.kappa, self.solver, self.n_steps, self.T_burn, self.T_meas, self.trace)

    def label(self) -> str:
        """Short human-readable identifier (used in messages and trace manifests)."""
        return (f"N={self.N} k={self.kappa_label or self.kappa} {self.readout} "
                f"d0={self.delta0} lam={self.lam} eta={self.eta} seed={self.seed}")

    def to_dict(self) -> dict:
        """Fields plus the derived properties, JSON-serialisable (stored in every batch file)."""
        d = dataclasses.asdict(self)
        d.update(pN=self.pN, Delta0=self.Delta0, gated=self.gated, learn_a=self.learn_a,
                 do_proj=self.do_proj, use_heads=self.use_heads)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Run":
        """
        Rebuild a Run from ``to_dict`` output (derived keys are ignored).

        Parameters
        ----------
        d : dict

        Returns
        -------
        Run
        """
        names = {f.name for f in dataclasses.fields(cls)}
        legacy = "gated" not in d          # schema-1 batch file
        d = {k: v for k, v in d.items() if k in names}
        if legacy:
            ro = d.get("readout")
            if ro in LEGACY_READOUTS:
                d["readout"] = LEGACY_READOUTS[ro]
            elif ro == "frozen" and d.get("kappa_label") != "sqrtN":
                d["readout"] = "gated_frozen"
        return cls(**d)


def task_duration(solver: str, lam: float, eta: float, delta0: float, cfg: Config, D: int) -> tuple[int, float, float]:
    """
    Steps per task from the deterministic ā² = 1 + δ₀ (PLAN.md §3.6).

    Parameters
    ----------
    solver : {"pop", "sgd"}
    lam, eta, delta0 : float
    cfg : Config
        Supplies ``h_max``, ``pop_contraction``, ``sgd_contraction``.
    D : int
        Input dimension (SGD: n_steps = round(α_task·D)).

    Returns
    -------
    (n_steps, h, alpha_task)
        ``h`` is 0 for SGD; for the population solver ``h·n_steps = alpha_task``
        exactly.

    Raises
    ------
    ValueError
        If SGD would violate η ā² ≤ sgd_contraction, or the solver is unknown.
    """
    abar2 = 1.0 + delta0
    alpha_task = lam / (eta * abar2)
    if solver == "sgd":
        if eta * abar2 > cfg.sgd_contraction:
            raise ValueError(f"SGD needs η ā² ≤ {cfg.sgd_contraction}; got η={eta}, δ₀={delta0}")
        n_steps = max(1, int(round(alpha_task * D)))
        return n_steps, 0.0, n_steps / D
    if solver == "pop":
        h = min(cfg.h_max, cfg.pop_contraction / (eta * abar2))
        n_steps = int(math.ceil(alpha_task / h - 1e-12))
        return n_steps, alpha_task / n_steps, alpha_task
    raise ValueError(f"unknown solver {solver!r}")


def resolve_kappa(k: Any, N: int) -> tuple[float, str]:
    """
    Interpret a YAML κ entry.

    Parameters
    ----------
    k : float or str
        A number, or "sqrtN" (κ = √N, every unit active on average).
    N : int

    Returns
    -------
    (kappa, label)
        The numeric value and the label used in file names and conditions.
    """
    if isinstance(k, str):
        if k.lower() in ("sqrtn", "sqrt_n", "sqrt(n)"):
            return math.sqrt(N), "sqrtN"
        return float(k), k
    return float(k), str(k)


def make_runs(cfg: Config, N_filter: Optional[Sequence[int]] = None) -> list[Run]:
    """
    Expand the grid of a config into a list of runs.

    Gated readouts iterate over ``cfg.kappa``; ungated readouts have no κ
    (every unit is active, pN = N, i.e. the κ = √N point) and are listed once
    per (N, δ₀, λ, η, seed) with ``kappa_label = "sqrtN"`` (PLAN.md §4).

    Parameters
    ----------
    cfg : Config
    N_filter : sequence of int, optional
        Keep only these widths.

    Returns
    -------
    list of Run

    Raises
    ------
    ValueError
        Unknown readout, δ₀ ≤ −1, or an SGD η that violates the contraction rule.
    """
    for readout in cfg.readout:
        if readout not in READOUTS:
            raise ValueError(f"unknown readout {readout!r}; known: {list(READOUTS)}")
    runs = []
    for N in cfg.N:
        if N_filter is not None and N not in N_filter:
            continue
        D = cfg.D_factor * N
        kappas = [resolve_kappa(k, N) for k in cfg.kappa]
        for readout in cfg.readout:
            for kappa, klabel in (kappas if is_gated(readout) else [(math.sqrt(N), "sqrtN")]):
                T_half = max(cfg.T_min, int(round(cfg.T_factor * N / kappa**2)))
                for delta0, lam, eta, seed in itertools.product(cfg.delta0, cfg.lam, cfg.eta, cfg.seed_list()):
                    if delta0 <= -1.0:
                        raise ValueError(f"δ₀ = {delta0} ≤ −1 makes the initial readout imaginary")
                    n_steps, h, alpha = task_duration(cfg.solver, lam, eta, delta0, cfg, D)
                    runs.append(Run(
                        N=N, D=D, kappa=kappa, readout=readout, delta0=float(delta0), lam=float(lam),
                        seed=int(seed), solver=cfg.solver, eta=float(eta), h=float(h), n_steps=int(n_steps),
                        alpha_task=float(alpha), T_burn=T_half, T_meas=T_half, kappa_label=klabel,
                        n_ages_geom=int(cfg.n_ages_geom), m_max_factor=float(cfg.m_max_factor),
                        trace=bool(cfg.trace), trace_points=int(cfg.trace_points),
                    ))
    return runs


def make_batches(runs: Sequence[Run], batch_size: int) -> list[list[Run]]:
    """
    Bucket runs by ``batch_key`` and split each bucket into chunks of ≤ ``batch_size``.

    Parameters
    ----------
    runs : sequence of Run
    batch_size : int

    Returns
    -------
    list of list of Run
        Batches in sorted key order; the order is what the manifest records.
    """
    buckets: dict[tuple, list[Run]] = {}
    for r in runs:
        buckets.setdefault(r.batch_key(), []).append(r)
    batches = []
    for key in sorted(buckets):
        b = buckets[key]
        for i in range(0, len(b), batch_size):
            batches.append(b[i:i + batch_size])
    return batches
