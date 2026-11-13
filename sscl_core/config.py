"""Config dataclass, YAML loader, grid → runs → batches (PLAN.md §3.6, §4)."""
from __future__ import annotations

import dataclasses
import hashlib
import itertools
import math

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import yaml

from sscl_core import theory

#: Task clocks of the write-up ("Decision: the task clock") → ``Params.clock``.
#: "nominal": (i) every task lasts Λ = λ/ā²_nom, ā²_nom = 1 + δ₀.
#: "realized": (ii), the default: every task lasts Λ_c = λ/ā²_c, ā²_c its key norm at the start of the task.
CLOCKS = {"nominal": 0.0, "realized": 1.0}

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
    kappa : gate scales for the gated readouts (floats or "sqrtN"), 1 ≤ κ ≤ √N (``kappa_ok``); gated
        runs at an (N, κ) outside that range are skipped and listed on stdout by ``make_runs``
    readout : readout keys (``READOUTS``)
    delta0 : dimensionless imbalances δ₀ = ηΔ₀ (> −1)
    lam : hidden-layer learning per task λ = Λ ā²_nom (the frozen-readout exponent; the readout exponent
        λᵃ_c = Λ‖M_c‖ is derived from (λ, δ₀) and the state, never set)
    clock : task clock, "realized" (clock (ii) of the write-up, the default) or "nominal" (clock (i)); SGD runs
        always take the nominal clock (its step count n = round(λD/(η ā²_nom)) is the write-up's)
    burn_halflives : burn-in the config must cover, in half-lives of the slowest readout (task heads,
        N ln 2/(θν)); ``make_runs`` notes the runs that fall short (``theory.burn_in_halflives``)
    seeds : int (→ range) or explicit list
    solver : "sgd" (online SGD, the default) or "pop" (population-gradient steps)
    eta : per-run η, numbers or "auto" (the default; ``resolve_eta``: sgd → sgd_contraction/(1 + δ₀), the largest
        rate with η ā²_nom ≤ sgd_contraction; pop → 1)
    sgd_batch : samples per SGD step B (default 1, online SGD); a step sums the B single-sample updates at the
        same η, so a task takes n = round(λD/(η ā²_nom B)) steps (``model.sgd_step``); ignored by the population solver
    h_max, pop_contraction, sgd_contraction : step-size rules of §3.6
    D_factor, T_factor, T_per_N, T_min : D = D_factor·N; T_burn = T_meas = max(T_min, round(T_per_N·N)) when
        T_per_N > 0 (a fixed number of measured half-lives, PLAN.md §12.3), else max(T_min, round(T_factor·N/κ_s²))
    extra : extra gated (N, κ) cells, each ``{N: …, kappa: …}``; skipped like ``kappa`` outside 1 ≤ κ ≤ √N
    batch_size : max runs per vmap batch
    n_ages_geom, m_max_factor : probe-age schedule (§5)
    trace, trace_points : per-step traces (§12.2), separate configs only
    sigma0 : initial scales W⁽⁰⁾_ij ~ N(0, σ₀²) (check 5; (1.0,) in every sweep)
    """
    name: str = "pilot"
    N: Sequence[int] = (10, 20)
    kappa: Sequence[Any] = (1.0,)          # floats or the string "sqrtN"
    readout: Sequence[str] = READOUT_ORDER
    delta0: Sequence[float] = (0.0,)
    lam: Sequence[float] = (1.0,)
    seeds: Any = 2                          # int (→ range) or explicit list
    solver: str = "sgd"                     # "sgd" (default) | "pop"
    eta: Sequence[Any] = ("auto",)          # per-run η: numbers or "auto" (sgd: sgd_contraction/(1 + δ₀); pop: 1)
    h_max: float = 1.0                      # cap on the population step in α
    pop_contraction: float = 0.1            # h η ā² ≤ this
    sgd_contraction: float = 0.05           # η ā² ≤ this
    sgd_batch: int = 1                      # samples per SGD step (their updates summed at the same η); 1 = online SGD
    D_factor: int = 4                       # D = D_factor · N
    T_factor: float = 10.0                  # T_burn = T_meas = T_factor · N / κ_s² (when T_per_N is 0)
    T_per_N: float = 0.0                    # > 0: T_burn = T_meas = T_per_N · N, independent of κ (PLAN.md §12.3 M4–M8, burn-in)
    T_min: int = 4                          # floor on T_burn and T_meas
    batch_size: int = 64
    n_ages_geom: int = 24
    m_max_factor: float = 4.0
    trace: bool = False                     # per-step norm traces (PLAN.md §12.2); separate configs only
    trace_points: int = 200                 # stored steps per task in a trace run
    #: extra gated (N, κ) cells off the N × κ product, e.g. ``[{N: 100, kappa: 4.0}]``; an N not in ``N`` gets no ungated runs
    extra: Sequence[dict] = ()
    sigma0: Sequence[float] = (1.0,)        # initial scale of W (check 5 of the write-up sweeps it)
    clock: str = "realized"                 # task clock: "realized" = (ii) Λ_c = λ/ā²_c (default), "nominal" = (i) Λ = λ/ā²_nom
    #: burn-in in half-lives of the slowest readout that a sweep should cover: the write-up's T_burn = 7N is
    #: 7θν/ln 2 ≈ 10 of them (9.9ν at λ = 4, i.e. 9.4 at N = 5, κ = 1 and 9.9 from N = 20), so the note fires below 9
    burn_halflives: float = 9.0

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
    #: schedule κ of PLAN.md §5: the run's κ when gated, the sweep's smallest gated κ when ungated (sets T and the ages);
    #: 0.0 in batch files of older versions, which ``kappa_s`` then reads as ``kappa``
    kappa_sched: float = 0.0
    n_ages_geom: int = 24
    m_max_factor: float = 4.0
    trace: bool = False
    trace_points: int = 200
    sigma0: float = 1.0                     # W⁽⁰⁾_ij ~ N(0, σ₀²); 1 unless check 5 sweeps it
    #: task clock (``CLOCKS``); "nominal" for batch files written before 2026-11-11, which ran clock (i)
    clock: str = "nominal"
    #: samples per SGD step; ``solver`` is then the kernel token ``"sgd{B}"`` (``solver_token``).  0 for the population solver
    sgd_batch: int = 1

    @property
    def clock_id(self) -> float:
        """``Params.clock``: 0.0 for the nominal clock (i), 1.0 for the realized clock (ii)."""
        return CLOCKS[self.clock]

    @property
    def theta(self) -> float:
        """θ = 1 − e^{−λ}: the fraction of the residual a task removes through the hidden layer."""
        return theory.theta(self.lam)

    @property
    def p(self) -> float:
        """Gate density pN/N: κ/√N when gated, 1 when ungated."""
        return self.pN / self.N

    def popstab(self) -> float:
        """Left side of the write-up's sufficient contraction condition, eq. (popstab), before the run:
        hη (n_c/pN)(δ₀ + 2Q̄_c) at n_c = pN and Q̄_c the larger of σ₀² and the steady-state estimate of
        eq. (abarss) (``theory.popstab_steady``; the hidden channel alone for a frozen readout).  The step
        contracts the residual in every direction when this is < 2; NaN for SGD.  The value a run realizes
        task by task is the recorded scalar ``popstab``."""
        if self.solver != "pop":
            return float("nan")
        return theory.popstab_steady(self.h, self.eta, self.delta0, self.pN, self.lam, self.sigma0,
                                     projected=self.do_proj == 1.0, realized=self.clock == "realized",
                                     n_steps=self.n_steps)

    @property
    def pN(self) -> float:
        """Nominal active count κ√N; exactly N for an ungated run (PLAN.md §2)."""
        return float(self.N) if not is_gated(self.readout) else self.kappa * math.sqrt(self.N)

    @property
    def kappa_s(self) -> float:
        """Schedule κ (PLAN.md §5): ``kappa_sched`` when recorded, else ``kappa`` (older batch files)."""
        return float(self.kappa_sched) if self.kappa_sched else float(self.kappa)

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

    @property
    def closed(self) -> bool:
        """Whether the task's steps have a closed form: population solver with a frozen, unprojected readout
        (``model._closed_form``); such runs batch apart from the others (``batch_key``)."""
        return self.solver == "pop" and self.learn_a == 0.0 and self.do_proj == 0.0

    def block_rows(self) -> int:
        """Rows of the active block the steps run on (``model.block_rows``)."""
        from sscl_core.model import block_rows
        return block_rows(self.N, self.pN, self.gated == 1.0)

    def rng_seed(self) -> int:
        """Seed of the run's PRNGKey: a hash of (seed, N, κ, readout, δ₀, λ).

        Every run of the grid draws its own weights, teachers, gates and SGD
        inputs; nothing is shared across seeds or across conditions.  The
        solver, step size and η are left out so that η-halving and the
        population-vs-SGD validation (old/validate.py) compare the same
        task streams, and so is σ₀ (check 5 compares initial scales on the
        same streams).
        """
        tag = f"{self.seed}|{self.N}|{self.kappa_label or self.kappa}|{self.readout}|{self.delta0}|{self.lam}"
        return int.from_bytes(hashlib.sha256(tag.encode()).digest()[:4], "little")

    def batch_key(self) -> tuple:
        """Fields a vmap batch must share: (N, D, κ, solver, n_steps, T_burn, T_meas, trace).

        κ, not the schedule κ_s: ungated runs (κ = √N) keep their own batches, so
        their batch ids in a manifest do not move when their length changes
        (PLAN.md §5; the Stage-1 gated batch files stay valid across that change).
        ``make_batches`` further keeps the closed-form runs (``closed``) in
        batches of their own, so their tasks take the one-line update; a batch
        that mixes them with stepped runs is allowed and steps every run.
        """
        return (self.N, self.D, self.kappa, self.solver, self.n_steps, self.T_burn, self.T_meas, self.trace)

    def label(self) -> str:
        """Short human-readable identifier (used in messages and trace manifests)."""
        return (f"N={self.N} k={self.kappa_label or self.kappa} {self.readout} "
                f"d0={self.delta0} lam={self.lam} eta={self.eta} seed={self.seed}"
                + (f" sigma0={self.sigma0}" if self.sigma0 != 1.0 else "")
                + (" clock=nominal" if self.clock == "nominal" and self.solver == "pop" else ""))

    def to_dict(self) -> dict:
        """Fields plus the derived properties, JSON-serialisable (stored in every batch file)."""
        d = dataclasses.asdict(self)
        d.update(pN=self.pN, Delta0=self.Delta0, gated=self.gated, learn_a=self.learn_a,
                 do_proj=self.do_proj, use_heads=self.use_heads, clock_id=self.clock_id)
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


def solver_token(solver: str, sgd_batch: int = 1) -> str:
    """``Run.solver``, the static solver argument of every kernel: ``"pop"``, ``"sgd"`` (online, one sample per
    step) or ``"sgd{B}"`` for B > 1 samples per step (``model.sgd_batch`` reads B back).  The batch is part of the
    token because it sets array shapes, so kernels are compiled, cached and batched (``group_key``) per B."""
    return f"sgd{int(sgd_batch)}" if solver == "sgd" and int(sgd_batch) > 1 else solver


def resolve_eta(eta: Any, solver: str, delta0: float, cfg: Config) -> float:
    """
    The rate of a run from an entry of ``Config.eta``.

    A number is returned as it is.  ``"auto"`` is the default rate of the solver: 1 for the population solver
    (η only sets the unit of time there) and, for SGD, the largest rate ``task_duration`` accepts at this δ₀,
    η = sgd_contraction/(1 + δ₀) (nudged down when rounding puts η ā²_nom above the bound), so every δ₀ of a
    grid runs the same n = round(λD/sgd_contraction) steps per task.
    """
    if not (isinstance(eta, str) and eta.lower() == "auto"):
        return float(eta)
    if solver != "sgd":
        return 1.0
    abar2 = 1.0 + delta0
    out = cfg.sgd_contraction / abar2
    while out * abar2 > cfg.sgd_contraction:
        out = float(np.nextafter(out, 0.0))
    return out


def task_duration(solver: str, lam: float, eta: float, delta0: float, cfg: Config, D: int) -> tuple[int, float, float]:
    """
    Steps per task from the deterministic ā²_nom = 1 + δ₀ (PLAN.md §3.6; write-up "Population step h").

    Population: h = min(h_max, 0.1/(η ā²_nom)), n = ⌈α_task/h⌉, α_task = λ/(η ā²_nom); h is then set to
    α_task/n so that n steps span the nominal task exactly.  This is the step of clock (i) and the first-task
    step of clock (ii); under clock (ii) the step count n is kept and each task takes its own
    h_c = λ/(η n ā²_c) (``model.task_h``), so hη ā²_c = λ/n ≤ 0.1 holds for the realized key norm.
    The rule resolves the hidden channel only; the readout channel is covered by eq. (popstab)
    (``Run.popstab``, checked in ``make_runs``).  SGD: n = round(λD/(η ā²_nom B)) with B = ``cfg.sgd_batch`` samples
    per step, η ā²_nom ≤ ``sgd_contraction``; a step of B summed samples spans B/D of α, a population step of
    h = B/D with sampling noise, so B must also keep (B/D) η ā²_nom ≤ ``pop_contraction`` (B ≤ 2D at the defaults).

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
        B = int(cfg.sgd_batch)
        if B < 1 or B / D * eta * abar2 > cfg.pop_contraction * (1 + 1e-12):
            raise ValueError(f"sgd_batch = {B} needs 1 ≤ B and (B/D) η ā² ≤ {cfg.pop_contraction}; got D={D}, η={eta}, δ₀={delta0}")
        n_steps = max(1, int(round(alpha_task * D / B)))
        return n_steps, 0.0, n_steps * B / D
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


#: relative slack on the κ ≤ √N bound, so a κ typed as a rounded √N still passes
KAPPA_RTOL = 1e-4


def kappa_ok(kappa: float, N: int) -> bool:
    """Whether 1 ≤ κ ≤ √N: the gate density κ/√N is at most 1 and the mean active count κ√N at least √N."""
    return 1.0 - KAPPA_RTOL <= kappa <= math.sqrt(N) * (1.0 + KAPPA_RTOL)


def kappa_skip_reason(kappa: float, N: int) -> str:
    """
    Why a gated run at (N, κ) is skipped, or "" when 1 ≤ κ ≤ √N.

    Above √N the gate g_i ~ Bernoulli(κ/√N) saturates (every unit active, as
    ungated) while the readout still divides by pN = κ√N > N, so the run would
    be an ungated network with its output and step size shrunk by √N/κ, not a
    gated one.  Below 1 the mean active count κ√N drops under √N and, for
    κ√N < 1, most tasks draw no active unit at all.
    """
    if kappa > math.sqrt(N) * (1.0 + KAPPA_RTOL):
        return "κ > √N, gate saturated"
    if kappa < 1.0 - KAPPA_RTOL:
        return "κ < 1"
    return ""


def make_runs(cfg: Config, N_filter: Optional[Sequence[int]] = None) -> list[Run]:
    """
    Expand the grid of a config into a list of runs.

    Gated readouts iterate over ``cfg.kappa``; ungated readouts have no κ
    (every unit is active, pN = N, i.e. the κ = √N point) and are listed once
    per (N, δ₀, λ, η, seed) with ``kappa_label = "sqrtN"`` (PLAN.md §4).
    Their stream length and age schedule come from the schedule κ_s = the
    sweep's smallest gated κ (PLAN.md §5), so they run as long as the κ_min
    gated runs (20N/κ_min² tasks, or 2·T_per_N·N for every run when
    ``T_per_N`` is set); ``kappa_sched`` records κ_s on every run.  The
    gated cells of ``cfg.extra`` are added to the N × κ product.

    Gated runs need 1 ≤ κ ≤ √N (``kappa_skip_reason``).  Those at an (N, κ)
    of the product or of ``cfg.extra`` outside that range are skipped, never
    raised on, and one line on stdout lists the skipped cells of the whole
    grid (not just ``N_filter``), so a κ list can be swept over widths too
    small for its largest values.  The ungated runs are unaffected.

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
        Unknown readout or clock, δ₀ ≤ −σ₀², or an SGD η that violates the contraction rule.
    """
    for readout in cfg.readout:
        if readout not in READOUTS:
            raise ValueError(f"unknown readout {readout!r}; known: {list(READOUTS)}")
    if cfg.clock not in CLOCKS:
        raise ValueError(f"unknown clock {cfg.clock!r}; known: {list(CLOCKS)}")
    clock = "nominal" if cfg.solver == "sgd" else cfg.clock     # SGD: η is the noise amplitude, n is nominal
    if any(map(is_gated, cfg.readout)):
        _note_skipped_kappas(cfg)
    runs = []
    extra_N = [int(e["N"]) for e in cfg.extra if int(e["N"]) not in cfg.N]
    for N in list(cfg.N) + sorted(set(extra_N)):
        if N_filter is not None and N not in N_filter:
            continue
        D = cfg.D_factor * N
        kappas = [kl for kl in (resolve_kappa(k, N) for k in cfg.kappa) if kappa_ok(kl[0], N)] if N in cfg.N else []
        kappas += [kl for kl in (resolve_kappa(e["kappa"], N) for e in cfg.extra if int(e["N"]) == N) if kappa_ok(kl[0], N)]
        kappa_min = min(k for k, _ in kappas) if kappas else math.sqrt(N)
        for readout in cfg.readout:
            if not is_gated(readout) and N not in cfg.N:
                continue                                    # an extra cell is gated only
            for kappa, klabel in (kappas if is_gated(readout) else [(math.sqrt(N), "sqrtN")]):
                kappa_s = kappa if is_gated(readout) else kappa_min
                if cfg.T_per_N > 0:
                    T_half = max(cfg.T_min, int(round(cfg.T_per_N * N)))
                else:
                    T_half = max(cfg.T_min, int(round(cfg.T_factor * N / kappa_s**2)))
                for delta0, lam, eta, seed, sigma0 in itertools.product(cfg.delta0, cfg.lam, cfg.eta, cfg.seed_list(), cfg.sigma0):
                    if delta0 <= -sigma0 ** 2:
                        raise ValueError(f"δ₀ = {delta0} ≤ −σ₀² = {-sigma0 ** 2:g} makes the initial readout imaginary "
                                         f"(a_i² = δ₀ + Q_ii with Q_ii ≈ σ₀²)")
                    eta = resolve_eta(eta, cfg.solver, delta0, cfg)
                    n_steps, h, alpha = task_duration(cfg.solver, lam, eta, delta0, cfg, D)
                    runs.append(Run(
                        N=N, D=D, kappa=kappa, readout=readout, delta0=float(delta0), lam=float(lam),
                        seed=int(seed), solver=solver_token(cfg.solver, cfg.sgd_batch), eta=float(eta), h=float(h), n_steps=int(n_steps),
                        alpha_task=float(alpha), T_burn=T_half, T_meas=T_half, kappa_label=klabel, kappa_sched=float(kappa_s),
                        n_ages_geom=int(cfg.n_ages_geom), m_max_factor=float(cfg.m_max_factor),
                        trace=bool(cfg.trace), trace_points=int(cfg.trace_points), sigma0=float(sigma0),
                        clock=clock, sgd_batch=int(cfg.sgd_batch) if cfg.solver == "sgd" else 0,
                    ))
    _note_checks(cfg, runs)
    return runs


#: Conditions already reported by ``_note_checks`` in this process (a script calls ``make_runs`` once per cell).
_NOTED: set = set()


def unstable_runs(runs: Sequence[Run]) -> list[Run]:
    """Runs whose population step fails the sufficient contraction condition of the write-up, eq. (popstab):
    ``Run.popstab`` ≥ 2 (one run per condition; seeds share the value)."""
    seen, out = set(), []
    for r in runs:
        key = (r.N, r.kappa_label, r.readout, r.delta0, r.lam, r.eta, r.sigma0, r.clock)
        if key in seen or r.solver != "pop":
            continue
        seen.add(key)
        if r.popstab() >= 2.0:
            out.append(r)
    return out


def short_burn_in(runs: Sequence[Run], n_half: float) -> list[tuple[Run, float]]:
    """(run, half-lives) of the conditions whose burn-in is shorter than ``n_half`` half-lives of the slowest
    readout, the task heads' N ln 2/(θν) at the run's gate density (``theory.burn_in_halflives``; write-up,
    Protocol: T_burn = 7N is 7θν/ln 2 ≈ 10).  One entry per (N, κ, λ, T_burn)."""
    seen, out = set(), []
    for r in runs:
        key = (r.N, r.kappa_label, r.lam, r.T_burn)
        if key in seen:
            continue
        seen.add(key)
        x = theory.burn_in_halflives(r.T_burn, r.N, r.p, r.lam)
        if x < n_half:
            out.append((r, x))
    return out


def _note_checks(cfg: Config, runs: Sequence[Run]) -> None:
    """One line on stdout per derived check of the write-up that some runs fail: the contraction condition
    eq. (popstab) and the burn-in length in half-lives.  Notes, not errors: eq. (popstab) is sufficient and
    rests on a leading-order norm estimate, and scripts that pass their own stream length (``--T``) do not
    use ``T_burn``.  The burn-in is checked only under the rule T_burn = ``T_per_N``·N.  A condition is
    reported once per process."""
    bad = [r for r in unstable_runs(runs) if ("stab", r.label()) not in _NOTED]
    if bad:
        _NOTED.update(("stab", r.label()) for r in bad)
        txt = ", ".join(f"(N={r.N}, κ={r.kappa_label}, {r.readout}, δ₀={r.delta0:g}, λ={r.lam:g}: {r.popstab():.2f})"
                        for r in bad[:6])
        print(f"note: eq. (popstab) hη(δ₀ + 2Q̄) < 2 fails at the estimated steady state for {len(bad)} conditions "
              f"under the {bad[0].clock} clock: {txt}{', …' if len(bad) > 6 else ''}; the recorded scalar "
              f"'popstab' is the realized value", flush=True)
    # the burn-in rule under test is the write-up's T_burn = T_per_N·N; the legacy N/κ² form (T_per_N = 0) is not checked
    short = [(r, x) for r, x in (short_burn_in(runs, cfg.burn_halflives) if cfg.T_per_N > 0 else [])
             if ("burn", r.N, r.kappa_label, r.lam, r.T_burn) not in _NOTED]
    if short:
        _NOTED.update(("burn", r.N, r.kappa_label, r.lam, r.T_burn) for r, _ in short)
        lo = min(short, key=lambda t: t[1])
        print(f"note: T_burn covers fewer than {cfg.burn_halflives:g} half-lives of the slowest readout "
              f"(N ln 2/(θν)) for {len(short)} (N, κ, λ) cells; shortest: N={lo[0].N}, κ={lo[0].kappa_label}, "
              f"λ={lo[0].lam:g}, T_burn={lo[0].T_burn} = {lo[1]:.1f} half-lives", flush=True)


def _note_skipped_kappas(cfg: Config) -> list[tuple[int, str, str]]:
    """
    Print one line listing the gated (N, κ) cells of the whole grid that ``make_runs`` skips.

    Returns
    -------
    list of (N, kappa_label, reason)
    """
    cells = [(N, k) for N in cfg.N for k in cfg.kappa] + [(int(e["N"]), e["kappa"]) for e in cfg.extra]
    skipped = []
    for N, k in cells:
        kappa, klabel = resolve_kappa(k, N)
        why = kappa_skip_reason(kappa, N)
        if why and (N, klabel, why) not in skipped:
            skipped.append((N, klabel, why))
    if skipped:
        print("note: skipped gated runs outside 1 ≤ κ ≤ √N: "
              + ", ".join(f"(N={N}, κ={k}: {why})" for N, k, why in sorted(skipped)), flush=True)
    return skipped


def make_batches(runs: Sequence[Run], batch_size: int) -> list[list[Run]]:
    """
    Bucket runs by ``batch_key`` (and ``closed``) and split each bucket into chunks of ≤ ``batch_size``.

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
        buckets.setdefault(r.batch_key() + (r.closed,), []).append(r)
    batches = []
    for key in sorted(buckets):
        b = buckets[key]
        for i in range(0, len(b), batch_size):
            batches.append(b[i:i + batch_size])
    return batches


def add_clock_arg(ap) -> None:
    """Add ``--clock`` (default: the config's, clock (ii) ``realized``) and ``--sgd_batch`` (default: the config's,
    1) to a script's argument parser."""
    ap.add_argument("--sgd_batch", type=int, default=None,
                    help="samples per SGD step, their updates summed at the same η: a task takes B times fewer steps "
                         "(default: the config's, 1 = online SGD; no effect on the population solver)")
    ap.add_argument("--clock", choices=list(CLOCKS), default=None,
                    help="task clock of the write-up: 'realized' = (ii), every task lasts Λ_c = λ/ā²_c with its own "
                         "key norm (the default); 'nominal' = (i), every task lasts Λ = λ/(1 + δ₀) (default: the config's)")


def clock_override(args) -> dict:
    """The ``Config`` overrides of ``add_clock_arg``: ``clock`` when ``--clock`` was given and ``sgd_batch`` when
    ``--sgd_batch`` was."""
    out = {"clock": args.clock} if getattr(args, "clock", None) else {}
    if getattr(args, "sgd_batch", None) is not None:
        out["sgd_batch"] = int(args.sgd_batch)
    return out
