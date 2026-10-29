"""Config dataclass, YAML loader, grid → runs → batches (PLAN.md §3.6, §4)."""
from __future__ import annotations

import dataclasses
import itertools
import math
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import yaml

READOUTS = {
    #            learn_a, do_proj, use_heads
    "learned": (1.0, 1.0, 0.0),
    "frozen":  (0.0, 0.0, 0.0),
    "heads":   (1.0, 1.0, 1.0),
}


@dataclass
class Config:
    name: str = "pilot"
    N: Sequence[int] = (10, 20)
    kappa: Sequence[Any] = (1.0,)          # floats or the string "sqrtN"
    readout: Sequence[str] = ("learned", "frozen", "heads")
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

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**raw)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def seed_list(self) -> list[int]:
        return list(range(int(self.seeds))) if isinstance(self.seeds, int) else [int(s) for s in self.seeds]


@dataclass
class Run:
    """One (seed, κ, δ₀, λ, readout, η) run; all fields are host-side constants."""

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

    @property
    def pN(self) -> float:
        return self.kappa * math.sqrt(self.N)

    @property
    def Delta0(self) -> float:
        return self.delta0 / self.eta

    @property
    def learn_a(self) -> float:
        return READOUTS[self.readout][0]

    @property
    def do_proj(self) -> float:
        return READOUTS[self.readout][1]

    @property
    def use_heads(self) -> float:
        return READOUTS[self.readout][2]

    def batch_key(self) -> tuple:
        return (self.N, self.D, self.kappa, self.solver, self.n_steps, self.T_burn, self.T_meas)

    def label(self) -> str:
        return (f"N={self.N} k={self.kappa_label or self.kappa} {self.readout} "
                f"d0={self.delta0} lam={self.lam} eta={self.eta} seed={self.seed}")

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d.update(pN=self.pN, Delta0=self.Delta0, learn_a=self.learn_a, do_proj=self.do_proj, use_heads=self.use_heads)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Run":
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


def task_duration(solver: str, lam: float, eta: float, delta0: float, cfg: Config, D: int = 1) -> tuple[int, float, float]:
    """(n_steps, h, alpha_task) from the deterministic ā² = 1 + δ₀ (PLAN.md §3.6)."""
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
    if isinstance(k, str):
        if k.lower() in ("sqrtn", "sqrt_n", "sqrt(n)"):
            return math.sqrt(N), "sqrtN"
        return float(k), k
    return float(k), str(k)


def make_runs(cfg: Config, N_filter: Optional[Sequence[int]] = None) -> list[Run]:
    """Expand the grid of a config into a list of runs."""
    runs = []
    for N in cfg.N:
        if N_filter is not None and N not in N_filter:
            continue
        D = cfg.D_factor * N
        for kraw in cfg.kappa:
            kappa, klabel = resolve_kappa(kraw, N)
            T_half = max(cfg.T_min, int(round(cfg.T_factor * N / kappa**2)))
            for readout, delta0, lam, eta, seed in itertools.product(
                cfg.readout, cfg.delta0, cfg.lam, cfg.eta, cfg.seed_list()
            ):
                if readout not in READOUTS:
                    raise ValueError(f"unknown readout {readout!r}")
                if delta0 <= -1.0:
                    raise ValueError(f"δ₀ = {delta0} ≤ −1 makes the initial readout imaginary")
                n_steps, h, alpha = task_duration(cfg.solver, lam, eta, delta0, cfg, D)
                runs.append(Run(
                    N=N, D=D, kappa=kappa, readout=readout, delta0=float(delta0), lam=float(lam),
                    seed=int(seed), solver=cfg.solver, eta=float(eta), h=float(h), n_steps=int(n_steps),
                    alpha_task=float(alpha), T_burn=T_half, T_meas=T_half, kappa_label=klabel,
                ))
    return runs


def make_batches(runs: Sequence[Run], batch_size: int) -> list[list[Run]]:
    """Bucket runs by ``batch_key`` and split each bucket into chunks of ≤ batch_size."""
    buckets: dict[tuple, list[Run]] = {}
    for r in runs:
        buckets.setdefault(r.batch_key(), []).append(r)
    batches = []
    for key in sorted(buckets):
        b = buckets[key]
        for i in range(0, len(b), batch_size):
            batches.append(b[i:i + batch_size])
    return batches
