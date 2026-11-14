"""What every script of ``replay_experiment`` shares: the config, the grid flags, runs → jobs, the method style.

The roles of the μP experiment's axes here:

    μP experiment                         replay experiment
    readout (the traces, one colour each)  replay method: none, fifo, random, priority (``METHOD_COLOR``)
    κ (columns / shades / one per figure)  replay size C, the buffer's capacity in samples (``--capacity``)

The network is always the ungated learned shared readout (both layers learned, projection on), so there is no
``--readout`` and no ``--kappa``.  ``none`` has no buffer, hence no C: like the ungated readouts of the μP
figures it is trained once and drawn in every capacity's cell.
"""
from __future__ import annotations

import os
from typing import Callable, Optional, Sequence

import numpy as np
import yaml

import _path  # noqa: F401

from sscl_core import experiments as X
from sscl_core import plotting as P
from sscl_core import pool
from sscl_core import replay as R
from sscl_core.config import Config, make_runs

#: Colour, label and line style of a replay method (bound to the method, never to rank); no replay is the
#: grey dashed baseline.
METHOD_COLOR = {"none": "0.35", "fifo": "#0072B2", "random": "#E69F00", "priority": "#009E73"}
METHOD_LABEL = {"none": "no replay", "fifo": "FIFO", "random": "random replace", "priority": "priority replace"}
METHOD_STYLE = {"none": "--", "fifo": "-", "random": "-", "priority": "-"}
MODE_LABEL = {"sleep": "sleep replay", "interleaved": "interleaved replay", "none": "no replay"}

#: Keys of ``configs/replay.yaml`` that are not ``sscl_core.config.Config`` fields, with their defaults.
REPLAY_DEFAULTS = {"method": list(R.METHODS), "capacity": ["0.5D", "4D"], "write": "0.125D", "replay_ratio": 1.0,
                   "replay_mode": "sleep"}


def load_config(path: str) -> tuple[Config, dict]:
    """``configs/replay.yaml`` → (the μP ``Config`` of its solver and stream keys, the replay keys)."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    rep = {k: raw.pop(k, v) for k, v in REPLAY_DEFAULTS.items()}
    known = set(Config.__dataclass_fields__)
    if set(raw) - known:
        raise ValueError(f"unknown config keys: {sorted(set(raw) - known)}")
    return Config(**raw), rep


def add_args(ap, name: str, N: Sequence[int], delta0_list: bool = False, seeds: int = 4, mode: bool = True) -> None:
    """The grid flags every script takes (README "Grid flags") and the placement flags of ``sscl_core.pool``."""
    ap.add_argument("--config", default=os.path.join(_path.ROOT, "configs", "replay.yaml"),
                    help="config supplying the solver, λ, step-size and stream-length rules and the replay defaults")
    ap.add_argument("--N", type=int, nargs="+", default=list(N), help="widths (D = 4N)")
    ap.add_argument("--capacity", nargs="+", default=None,
                    help="replay sizes C: a number of samples, or a multiple of the input dimension such as 0.5D "
                         "(default: the config's, 0.5D 4D)")
    ap.add_argument("--method", nargs="+", default=None, choices=R.METHODS,
                    help="replay methods (default: the config's, all four)")
    if delta0_list:
        ap.add_argument("--delta0", type=float, nargs="+", default=[0.0])
    else:
        ap.add_argument("--delta0", type=float, default=0.0)
    ap.add_argument("--lam", type=float, default=None, help="default: the config's first λ")
    ap.add_argument("--seeds", type=int, default=seeds)
    if mode:
        ap.add_argument("--mode", choices=R.MODES, default=None,
                        help="when the buffer is replayed: 'sleep' = after each task, 'interleaved' = within every "
                             "step of the task (default: the config's, sleep)")
    ap.add_argument("--replay-ratio", type=float, default=None,
                    help="replayed samples per fresh sample, ρ (default: the config's, 1)")
    ap.add_argument("--write", default=None,
                    help="samples of each task stored after it, as --capacity (default: the config's, 0.125D)")
    ap.add_argument("--sgd_batch", type=int, default=None, help="fresh samples per SGD step (default: the config's, 16)")
    ap.add_argument("--out", default=os.path.join(_path.ROOT, "results", name))
    pool.add_args(ap)


def build(args, name: str, mode: Optional[str] = None) -> list[R.RRun]:
    """The runs of a script's grid: ``config.make_runs`` at readout ``learned_shared`` × method × capacity."""
    cfg, rep = load_config(args.config)
    d0 = args.delta0 if isinstance(args.delta0, (list, tuple)) else [args.delta0]
    over = {"name": name, "N": list(args.N), "readout": ["learned_shared"], "delta0": list(d0),
            "lam": [args.lam if args.lam is not None else cfg.lam[0]], "seeds": args.seeds, "extra": (), "trace": False}
    if args.sgd_batch is not None:
        over["sgd_batch"] = int(args.sgd_batch)
    base = make_runs(Config(**{**cfg.__dict__, **over}))
    return R.expand(base, args.method or rep["method"], args.capacity or rep["capacity"],
                    args.write if args.write is not None else rep["write"],
                    args.replay_ratio if args.replay_ratio is not None else rep["replay_ratio"],
                    mode or getattr(args, "mode", None) or rep["replay_mode"])


# ----------------------------------------------------------------------------
# runs → cells, groups, jobs
# ----------------------------------------------------------------------------
def cap_labels(runs: Sequence[R.RRun]) -> list[str]:
    """Capacity labels of the replay runs, smallest first (by samples at the smallest N)."""
    N0 = min(r.N for r in runs)
    size = {r.cap_label: r.capacity for r in runs if r.method != "none" and r.N == N0}
    return sorted(size, key=size.get)


def methods_of(runs: Sequence[R.RRun]) -> list[str]:
    """Methods present, in display order."""
    return [m for m in R.METHODS if any(r.method == m for r in runs)]


def in_cap(r: R.RRun, cap: Optional[str]) -> bool:
    """Whether a run is drawn in capacity ``cap``'s cell: its own capacity, or any when it has no buffer."""
    return r.method == "none" or r.cap_label == cap


def cells(runs: Sequence[R.RRun]) -> dict[tuple, list[R.RRun]]:
    """``{(N, capacity label, δ₀, λ): runs}``; the ``none`` runs are listed in every capacity's cell."""
    out = {}
    caps = cap_labels(runs) or [None]
    for N in sorted({r.N for r in runs}):
        for cap in caps:
            for d0 in sorted({r.delta0 for r in runs}):
                for lam in sorted({r.lam for r in runs}):
                    cell = [r for r in runs if r.N == N and r.delta0 == d0 and r.lam == lam and in_cap(r, cap)]
                    if cell:
                        out[(N, cap, d0, lam)] = cell
    return out


def groups(runs: Sequence[R.RRun]) -> dict[tuple, list[R.RRun]]:
    """``{replay.group_key: distinct runs}``, the costliest group first."""
    seen, out = set(), {}
    for r in runs:
        if r.label() not in seen:
            seen.add(r.label())
            out.setdefault(R.group_key(r), []).append(r)
    return {k: out[k] for k in sorted(out, key=lambda k: -R.job_cost(out[k], 1))}


def collect(func: str, runs: Sequence[R.RRun], T_total: int, kwargs: dict, pick: Callable, args,
            out_per_task: float = 8.0, log: Callable[[str], None] = print) -> tuple[dict, pool.Plan]:
    """Run ``sscl_core.replay:<func>`` over the groups of ``runs`` and return ``({run label: pick(result, j)}, plan)``.

    Parameters
    ----------
    func : str
        A job function of ``sscl_core.replay``.
    runs : sequence of RRun
    T_total : int
        Tasks every job trains (for the placement plan and the job order).
    kwargs : dict
        The job function's arguments besides ``runs``.
    pick : callable
        ``pick(result, j)`` → what the script keeps for the j-th run of a job.
    args : argparse.Namespace
        Carries the placement flags.
    """
    g = groups(runs)
    plan = pool.plan(args, [(rs, T_total) for rs in g.values()], out_per_task=out_per_task)
    print(f"  placement: {plan.describe()}", flush=True)
    jobs = []
    for (key, rs), k in zip(g.items(), plan.pieces):
        for i, part in enumerate(X.split(rs, k)):
            jobs.append(((key, i), {"runs": part, **kwargs}, R.job_cost(part, T_total)))
    res = pool.run_jobs(f"sscl_core.replay:{func}", jobs, log=log, **plan.place)
    out = {}
    for (_, kw, _), r in zip(jobs, res):
        for j, run in enumerate(kw["runs"]):
            out[run.label()] = pick(r, j)
    return out, plan


def by_method(cell: Sequence[R.RRun], cols: dict, finite: Callable = lambda v: np.isfinite(v).all()) -> dict:
    """``{method: [values of the cell's runs of that method, in seed order]}``, infeasible (non-finite) seeds dropped."""
    out = {}
    for m in R.METHODS:
        vals = [cols[r.label()] for r in sorted((r for r in cell if r.method == m), key=lambda r: r.seed)]
        vals = [v for v in vals if finite(v)]
        if vals:
            out[m] = vals
    return out


def stream_T(runs: Sequence[R.RRun], T: Optional[int], flag: str) -> int:
    """The stream length every run shares: ``T`` when given, else the config's T_burn + T_meas (one N only)."""
    Ts = {T or (r.T_burn + r.T_meas) for r in runs}
    if len(Ts) != 1:
        raise SystemExit(f"the widths differ in stream length; pass {flag}")
    return Ts.pop()


# ----------------------------------------------------------------------------
# style
# ----------------------------------------------------------------------------
def plot_method(ax, x, y, method: str, **kw):
    """Plot a trace in the method's colour and line style."""
    kw.setdefault("color", METHOD_COLOR[method])
    kw.setdefault("ls", METHOD_STYLE[method])
    kw.setdefault("lw", P.LW_TRACE)
    kw.setdefault("zorder", 3 if method == "none" else 4)
    return ax.plot(np.asarray(x, float), np.asarray(y, float), **kw)


def method_handles(methods: Sequence[str], marker: Optional[str] = None) -> list:
    """Legend handles of the methods, in display order."""
    return [P.Line2D([], [], color=METHOD_COLOR[m], ls=METHOD_STYLE[m], lw=P.LW_TRACE, marker=marker, ms=3,
                     label=METHOD_LABEL[m]) for m in R.METHODS if m in methods]


def cap_shade(i: int, n: int, method: Optional[str] = None) -> tuple:
    """Colour of the i-th of n capacity traces: the method's colour blended towards white, lightest = smallest C."""
    from matplotlib.colors import to_rgb
    base = to_rgb(METHOD_COLOR[method]) if method is not None else (0.15, 0.15, 0.15)
    f = 1.0 if n <= 1 else 0.45 + 0.55 * i / (n - 1)
    return tuple(1.0 - f * (1.0 - c) for c in base)


def cap_text(cap: str) -> str:
    """A capacity label for a title: ``C = 4D``."""
    return f"C = {cap}"


def replay_text(r: R.RRun) -> str:
    """One caption clause stating a replay run's fixed settings."""
    _, _, C, m, B_rep, n_sleep = r.spec
    how = (f"sleep replay ({n_sleep} steps of {max(R.model.sgd_batch(r.solver), 1)} buffer samples after each task's "
           f"{r.n_steps} steps)" if r.mode == "sleep" else
           f"interleaved replay ({B_rep} buffer samples in each of a task's {r.n_steps} steps)")
    return f"{how}, {m} samples of each task stored (D = {r.D})"


def seed_mean(Y: np.ndarray, axis: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Seed mean and s.e. of ``Y`` along ``axis`` (s.e. 0 for one seed)."""
    n = Y.shape[axis]
    return Y.mean(axis), (Y.std(axis, ddof=1) / np.sqrt(n) if n > 1 else np.zeros(np.delete(Y.shape, axis)))
