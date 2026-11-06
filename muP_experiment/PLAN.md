# PLAN.md — maintenance notes for the μP continual-learning experiments

This file is for whoever (human or AI) changes the code.  It records what
the experiment scripts compute, which library code they stand on, the
invariants that code relies on, and what has been measured.  How to run
them is in `README.md`; the mathematics is in
`writing/Derivations/muP_experiment/muP_experiment.tex` (the "write-up";
equation labels below refer to it).

**Scope (2026-11-06).** The write-up is being revised around these scripts,
run in this order, every parameter written out, by `run_all.sh`
(`./run_all.sh --dry-run` lists the commands):

| script | figure | here |
|---|---|---|
| `scripts/run_gauge_comp.py` | Δᵢ per step, projection off / on | §6.1 |
| `scripts/run_szero.py` | S₀(t): the first teacher's signal against task | §6.2 |
| `scripts/run_stask.py` | S_T(τ): a steady-state teacher's signal against age | §6.3 |
| `scripts/run_Nscaling.py` | m½ against N and κ, simulation vs N ln 2 | §6.4 |
| `scripts/run_weight_norms.py` | weight norms against task and against N | §6.5 |
| `scripts/run_imbalance_comp.py` | weight norms and m½ across δ₀ | §6.6 |
| `scripts/run_training_verify.py` | per-step loss at the start, middle and end of the stream | §6.7 |

**Old code lives in `old/`.** The scripts of the earlier, sweep-based
version of the experiments (`run_sweep.py`, `analyze.py`, `plot_norms.py`,
`plot_check2.py`, `validate.py`, `bench.py`, `run_sigma0.py`,
`gpu_check.py` with its helper `conftest_free.py`) were moved there on
2026-11-06; they run as `python old/<script>.py` (`old/_path.py` puts the
package root on `sys.path`, as `scripts/_path.py` does).  The `mup/`
modules only they use (`mup/results.py`, `mup/npmodel.py`) and the
sweep-only parts of `mup/observe.py`, `mup/io.py` and `mup/driver.py`
stay in the package, because `experiments.py` and `pool.py` import the
rest of those three modules.  The old code keeps passing its tests
(`tests/test_plotting.py` imports the old scripts from `old/`;
`driver.launch` starts `old/run_sweep.py`) but is outside this document.
Do not extend it for the write-up; add to the scripts in `scripts/` or to
`mup/experiments.py`.

## 1. Stack and machines

| | laptop | Xeon | GPU node (cyclops) |
|---|---|---|---|
| role | develop, test, small grids | CPU grids | the write-up grids |
| hardware | Apple silicon | 4-socket Xeon Gold 6348H, 96 cores | POWER9, 4× V100 32 GB |
| JAX | 0.12 CPU (`../requirements.txt`, venv `code/sscl_env`) | same | 0.4.28 + CUDA, numpy 1.26 (`~/opt/jax-env.sh`, built per `../../jax-cuda-power9.md`) |
| precision | float64 | float64 | float64 |

The same code runs on all three; nothing is backend-specific except the
chunk default (`mup/device.py`: 16 tasks per device call on a GPU, 1 on
the CPU) and the worker environments (`mup/driver.py`).  Python 3.12 on the
CPU machines; the GPU node's Python is older, so the code avoids syntax
newer than 3.9 (`Optional[...]`, not `X | None`, in annotations; `list[...]`
generics are fine).

`mup/_x64.py` sets `jax_enable_x64` and is imported first by every JAX
module (`model`, `protocol`, `experiments`, `device`); `mup/__init__.py`,
`config.py`, `plotting.py` and `pool.py` do not import JAX, so a script's
argument parsing and the pool's parent process stay JAX-free.

## 2. Files the scripts depend on

```
scripts/_path.py            puts the package root on sys.path (import _path first)
scripts/run_*.py            the experiment scripts (§6)
run_all.sh                  runs them in the order of the table above, every parameter explicit
mup/config.py               Config (YAML), Run, READOUTS, task_duration, make_runs
mup/model.py                Params, State, steps, projection, init, train_block
mup/protocol.py             run_task (the reference kernel), make_params, batched_init,
                            feasible, block_of, state_bytes, _step_fn, _step_and_project
mup/experiments.py          train_task, probe_signal, task_norms, imbalance_stats,
                            the chunked kernels, the job functions, group_key, run_cost, split
mup/pool.py                 run_jobs, Progress/report, add_args, plan, Plan
mup/device.py               platform, default_chunk (MUP_CHUNK)
mup/driver.py               PINNED_ENV, GPU_ENV, GPU_MEM_SHARE, worker_env, visible_gpus, gpu_memory_gb
mup/plotting.py             style, READOUT_COLOR/LABEL/STYLE, grid, save, legend, caption, helpers
configs/stage1.yaml         the defaults every script starts from
tests/test_experiments.py   kernels vs run_task; pool; planner
tests/test_block.py         train_block vs the explicit loop; closed form; redraws
tests/test_gpu.py           the cards (skipped without a GPU)
```

## 3. The model as implemented

### 3.1 State and per-run constants (`model.py`)

```python
class State(NamedTuple):   # one run; vmap adds the batch axis
    W: (N, D)   a: (N,)   g: (N,) 0/1   w: (D,)   heads: (P, N)   key: PRNGKey   t: int32
class Params(NamedTuple):  # 0-d float64 each, vmapped to (B,)
    Delta0, eta, h, pN, lam, gated, learn_a, do_proj, use_heads, sigma0
```

Derived: `key_vec = g ⊙ a`, `u_c = key_vec @ W / pN` (eq. student),
`Q_ii = |W_i|²/D`, `Δ_i = (a_i² − Q_ii)/η` (eq. imb).  `pN = κ√N` is the
nominal active count, exactly `N` for an ungated run.  `heads` holds the
task heads of a task-heads run, **gate-masked** (`g_k ⊙ a_k`) at slot `k`;
zeros otherwise.  A probe's teacher and gate are not stored: task `k`'s
draws are a pure function of `(key, k)` (`task_keys`, `probe_task`), so the
probe regenerates them bit for bit.  Only the heads, which depend on the
state at the task they were drawn, are kept — `P` head slots must cover
every task that will be probed (`experiments._batch` allocates
`P = T_pre + n_follow`).

### 3.2 Steps and the projection

Population step (eq. pop; the solver of every script):
```
r   = (g⊙a) @ W / pN − w
W  ← W − hη (g⊙a) rᵀ
a  ← a − learn_a · hη · g ⊙ (W r) / D
```
SGD step (eqs. sgdW–sgda; available, not used by the scripts): the same
with one sample `x ~ N(0, I_D)` per step from `fold_in(k_stream, i)`.
Inactive units receive exactly zero in both.

Projection (eq. proj) after **every** step on the active units when
`do_proj = 1`: with `A = a²/η`, `B = Q/η`, the larger of `A′, B′` is
`½(|Δ₀| + √(Δ₀² + 4AB))` and the other `AB / larger` (no cancellation for
either sign of Δ₀; `A′B′ = AB` keeps `u_c` exact).  `a ← sign(a)√(ηA′)`,
`W_i ← f_i W_i` with `f_i = √(ηB′/Q_i)`.  Stored heads are divided by `f`
so every stored `u_k` is unchanged; `train_block` accumulates the product
of `f` over a task and `train_task` applies it to the head buffer once
(`model.rescale_heads`).  Frozen readouts (`do_proj = 0`) have `f = 1` and
their `Q` drifts freely.

Initialisation: `W⁽⁰⁾_ij ~ N(0, σ₀²)` with σ₀ = 1, rows whose
`Δ₀ + Q_ii/η < 0` redrawn up to 32 times (`draw_W0`), then
`a⁽⁰⁾_i = ±√(η(Δ₀ + Q_ii/η))` (Rademacher sign).  A run still infeasible
after the redraws has NaN in `a` and `protocol.feasible` drops it.

### 3.3 One task (`model.train_block`, `experiments.train_task`)

A task draws `(w_t, g_t)` from `task_keys(key, t)[0]`, a fresh head from
the current `Q` when `use_heads` (else keeps `a`), then runs `n_steps`
steps each followed by the projection.  `train_block` runs them on the
block of the `K` most-active rows (`block_rows`: the gate count's mean plus
8 s.d., capped at N; `N` when ungated) with the projection's row factors
folded in lazily and `Q_i` tracked analytically, so a step reads the block
twice and writes it once; a task with more active units than `K` returns
NaN rather than running short.  For the population solver with a frozen,
unprojected readout (`Run.closed`) the task is the closed form
`W = W⁽⁰⁾ − (g⊙a) r₀ᵀ (1 − qⁿ)/A`, `q = 1 − hηA`, `A = Σ(g_i a_i)²/pN`.
`protocol.block_of(runs)` gives `(K, closed)` for a batch: `K` for the
largest gate, `closed` only if every run is closed-form.
`train_task` then stores the gate-masked head at slot `t` and advances
`t`.  `tests/test_experiments.py::test_train_task_matches_run_task` pins it
to the sweep kernel `protocol.run_task` bit for bit.

### 3.4 Readouts (`config.READOUTS`)

| key | gated | learn_a | do_proj | use_heads |
|---|---|---|---|---|
| `gated_shared` | 1 | 1 | 1 | 0 |
| `learned_shared` | 0 | 1 | 1 | 0 |
| `gated_frozen` | 1 | 0 | 0 | 0 |
| `frozen` | 0 | 0 | 0 | 0 |
| `task_heads` | 1 | 1 | 1 | 1 |

A readout is a per-run mask, not a code path, so the five share a `vmap`
batch.  `READOUT_ORDER` is the display order everywhere.

### 3.5 Runs from a config (`config.Config`, `config.Run`, `make_runs`)

The scripts load `configs/stage1.yaml`, replace the grid keys from the
command line (`Config(**{**cfg.__dict__, **over})` with `name`, `N`,
`kappa`, `readout`, `delta0`, `lam`, `seeds`, `extra=()`, `trace=False`)
and call `make_runs`.  What the config fixes:

- `solver: pop`, `eta: [1.0]`, `lam: [4.0]`, `D_factor: 4` (D = 4N).
- Task duration (`task_duration`): `ā² = 1 + δ₀`, `α_task = λ/(η ā²)`,
  `h = min(h_max, pop_contraction/(η ā²))` with `pop_contraction = 0.1`,
  `n_steps = ⌈α_task/h⌉`, `h = α_task/n_steps`.  At λ = 4: 40 steps per
  task for every δ₀.  Known bias: for the gated learned readouts at
  δ₀ ≤ 0 the settled ā² is 3–6× the nominal `1 + δ₀`, so the step `hη ā²`
  is larger than 0.1 there (a −9 % to −26 % bias of the gated-shared m½ at
  N = 100 was measured in a 2026-11-06 review); left as is, an author's
  decision.
- Stream length: `T_per_N: 7` → `T_burn = T_meas = max(T_min, round(7N))`
  for every κ (so a run is 14N tasks; `T_factor` is the older
  `10N/κ_s²` form, unused when `T_per_N > 0`).  Burn-in is a concept of
  `run_weight_norms` only (its steady-state window is `t ≥ T_burn`); the
  other scripts use `T_burn + T_meas` just as "the Stage-1 stream".
- Gated readouts iterate over `kappa`; ungated readouts get one run per
  (N, δ₀, λ, η, seed) with `kappa = √N`, `kappa_label = "sqrtN"`.
- `Run.rng_seed()` hashes `(seed, N, kappa_label, readout, δ₀, λ)`, so
  every run draws its own weights, teachers and gates, and the solver,
  η and σ₀ do not enter the seed.
- `Run.label()` is the identity the scripts key per-run results by.

**κ labels are the argument text.** `resolve_kappa` keeps a numeric κ's
label as typed (`"1"` vs `"1.0"` are different labels), and the scaling
scripts look results up by `str(k)` of `--kappa`, so a config's
`kappa: [1.0]` and a command line's `--kappa 1` do not mix.  `plotting.fmt`
prints both as `1` in file stems and titles.

Feasibility (`protocol.feasible`): δ₀ must exceed `−min_i Q0_i` after the
row redraws; at N ≤ 20 and δ₀ = −0.75 a seed can still fail.  Infeasible
runs are dropped from the batch and returned as NaN; every script tolerates
NaN columns (skips the readout or seed, counts it in the caption).

## 4. `mup/experiments.py`: kernels and job functions

Every script reduces to jobs for `pool.run_jobs`; a job is a list of
`Run` sharing `group_key(r) = (N, D, solver, n_steps, closed)` (every κ,
readout and seed of a width, the closed-form runs apart from the stepped
ones) and is trained as ONE `vmap` batch with the observable evaluated
inside a `lax.scan` over `chunk` tasks per device call.

Kernels (`functools.lru_cache` on their static arguments
`(n_steps, solver, n_tasks, K, closed)`; `jit` with the state donated, so
never reuse a state after passing it):

| kernel | returns per call |
|---|---|
| `train_kernel` | state advanced `n_tasks` tasks, no output |
| `signal_kernel` | `(state, S)`, `S` (n_tasks, B): `probe_signal` of probe `k` after each task |
| `norms_kernel` | `(state, X)`, `X` (n_tasks, B, 5): `task_norms` (`NORM_KEYS`) after each task |
| `gauge_kernel` | `(state, R)`, `R` (n_tasks, B, n_steps, 4): `imbalance_stats` after every step (`gauge_task`, the explicit `_step_and_project` loop, not `train_block`) |

Observables:
- `probe_signal(state, p, k)`: `S_k = u·w_k*/|w_k*|²` with
  `u = (g_k ⊙ a_k) @ W / pN`, `a_k = heads[k]` for task heads else `a`.
- `task_norms`: `Q_all, Q_act, a2_all, a2_act, u2` with
  `u2 = |(g ⊙ a) @ W / pN|²/D` of the task just trained; active means NaN
  when the gate drew no unit.
- `imbalance_stats`: mean, min, max of `Δ_i` over active units and
  `max|Δ_i − Δ₀|`.

Job functions (named `"mup.experiments:<name>"` for the pool; their
arguments and results must pickle):

| function | arguments | returns |
|---|---|---|
| `signal_after_tasks` | `runs, T_pre, n_follow, k, chunk=None` | (n_follow, B): train `T_pre` tasks, then `n_follow` recording probe `k` after each.  `run_szero`: `T_pre=0, k=0`; `run_stask`, `run_Nscaling`: `T_pre=T, k=T` |
| `norms_over_tasks` | `runs, T, chunk=None` | `{key: (B, T)}` over `NORM_KEYS` |
| `imbalance_over_steps` | `runs, do_proj, n_tasks, chunk=None` | (B, n_tasks·n_steps, 4) with `Params.do_proj` forced per run from `do_proj[i]` |

`_batch(runs, P)` asserts the job is homogeneous (`group_key[:4]`), drops
infeasible runs, builds the batched state with `P` head slots and returns
the mask; `_expand` puts NaN rows back so every result keeps the order of
the job's runs.  `chunks(T, chunk)` yields the device-call sizes
(`device.default_chunk`: `chunk` argument > `MUP_CHUNK` env > backend
default).  `pool.Progress` reports every tenth of the tasks.

Helpers: `run_cost(r, T) = N·D·n_steps·T` orders the queue; `split(runs,
pieces)` cuts a group into ≤ `pieces` contiguous parts.

## 5. `mup/pool.py`: placement

`run_jobs(func, jobs, gpus, per_gpu, workers, chunk, log, keep_dir)` with
`jobs = [(key, kwargs, cost)]` returns results in job order:

- in-process (`gpus=None, workers ≤ 1`): calls the function here, largest
  first; `MUP_CHUNK` set from `chunk`.
- otherwise it writes a spool directory (`spec.pkl`, `job_i.pkl`), starts
  `python -m mup.pool <dir>` workers with `driver.worker_env` (GPU: one
  card each via `CUDA_VISIBLE_DEVICES`, memory fraction
  `GPU_MEM_SHARE/per_gpu`; CPU: `PINNED_ENV`, single-threaded XLA/BLAS),
  and polls `prog_i.txt`/`done_i.txt`.  The `slots` largest jobs are
  pinned one per slot in order (slot `s` → card `s % len(gpus)`), the rest
  are claimed largest first through `claim_i` files.  A failed job's
  traceback (`err_i.txt`) or a dead worker raises `RuntimeError` after all
  workers exit.  Workers run from `pool.ROOT` so `mup` imports.

`plan(args, groups, out_per_task, split)` turns the `add_args` options into
a `Plan` (`gpus, per_gpu, workers, chunk, pieces, mem_gb`; `plan.place`
are the `run_jobs` keywords, `plan.pieces` aligns with `groups`):

- CPU: `workers` slots, no memory limit, chunk 1 (or `--chunk`); groups
  are split to fill the workers by run count only when there are fewer
  groups than workers.
- GPU: cards = `--gpus` ids, else `driver.visible_gpus()`, else
  `GPU_COUNT = 4`; memory = `nvidia-smi` or `GPU_MEM_GB = 32`.  `per_gpu`
  = `min(MAX_PER_GPU = 3, ⌈runs/cards⌉)`, reduced while one run's job
  (`job_bytes`: `MEM_FACTOR = 4` × the batched state + one chunk of
  observable) exceeds a process's share.  A job's wall time is its chain
  `T·n_steps`, so each group is cut into `⌈cards · chain/max chain⌉`
  pieces (the longest group one per card, pinned to different cards),
  more when a piece would not fit its memory share, never more than its
  runs.  Chunk = the largest of `CHUNKS = (64, 32, 16)` whose stacked
  observable stays under `OUT_BUDGET = 512 MB` and ≤ the longest T.
- `split=False` (`run_gauge_comp`): every group is one job; too large is
  a `SystemExit`.
- `--per-gpu`, `--chunk`, `--split` override the respective choices.

Xeon caveat: many workers compiling XLA at once have stalled the node
(`BrokenProcessPool` on 2026-11-02 with 96 workers); keep `--workers`
moderate there, or use the GPU node.

## 6. The scripts

Common shape: `cells`/`groups` (which runs are trained together and which
figure they belong to) → `make_jobs` (one `(key, kwargs, cost)` per piece
of a group) → `collect` (`pool.run_jobs`, then reassemble per condition
from `Run.label()` or `(N, kappa_label, readout, seed)`) → `figure`
(`plotting`) → `main` (argparse with `pool.add_args`, config override,
`pool.plan`, banner, loop over cells).  Every script prints its wall time
from after the imports at exit.  Figures are written by `plotting.save`,
which appends `.png`, writes `figdata` as `.npz` beside it, and **warns**
about illegible panels (> 6 ticks, text < 8 pt, text past the edge, a
missing suptitle, a legend inside an axes, an all-constant panel) — treat
a warning as a bug in the figure.

### 6.1 `run_gauge_comp.py` — Δᵢ per step, projection off and on

- Cells: `(N, κ label, δ₀, λ)` holding every readout × seed, seeds
  `0 … --seeds − 1` as in the other `run_*` scripts (default 1); the
  ungated readouts join every κ cell.  Each readout of a cell appears twice in the job list, with
  `do_proj` forced to 0 and to 1 (`ROWS`), from the same initial state and
  task stream; an ungated run listed in several κ cells is trained once
  per projection setting (`groups` dedups by `(label, do_proj)`).
  `split=False`: a cell is one job.
- Observable: `imbalance_over_steps` → per (run, do_proj) rows
  `(n_tasks·n_steps, 4)`.
- Figure per cell `gauge_comp_N{N}_k{κ}_d0{δ₀}_lam{λ}`: 2 rows
  × 1 column, panel 5.2 × 2.2 in; x = step (log), y = the seed mean of the
  active-unit mean of Δᵢ with a ±1 s.e. band over seeds (`plotting.band`;
  2026-11-06, was one figure per seed with the min–max band over units),
  one trace per readout, `ref_line` at Δ₀; `cell_title` with `N`, number
  of seeds.  `.npz`: `step`, `{proj|noproj}_{readout}_{mean,se,n_seeds}`
  and the seed means `_{min,max}` of the per-run extremes over active
  units.  The printed summary gives the final mean Δ and max|Δ − Δ₀| as
  seed mean ± s.e.
- The "projection" row projects the frozen readouts too (`do_proj` forced
  to 1 overrides their table value): it tests the projection, not a
  readout condition.
- Result (2026-11-02, N = 100, κ = 1, δ₀ = 0, λ = 4, 1400 tasks × 40
  steps): with the projection every readout holds `max|Δᵢ − Δ₀| ≤ 6e-15`
  over all 56 000 steps.  Without it Δᵢ drifts negative (Qᵢᵢ outruns aᵢ²):
  to a mean of about −10 for gated shared and gated frozen and −60 for
  learned shared by the end of the stream; ungated frozen stays near −1,
  task heads within 0.3 (a fresh head is redrawn at Δ₀ every task).  This
  is the O(h) Euler drift of the discrete population step accumulated
  over the stream, and the reason the projection runs after every step.

### 6.2 `run_szero.py` — S₀(t) against task

- Cells `(N, κ label, δ₀, λ)`, ungated readouts in every κ cell, trained
  once (`groups` dedups by label).  Every readout and seed of a cell is one
  batch; `plan` may cut a group into pieces.
- Job: `signal_after_tasks(runs, T_pre=0, n_follow=T, k=0)`; `T` =
  `--tasks` or `T_burn + T_meas`, which must be the same for every cell
  (several `--N` without `--tasks` is a `SystemExit`).  Task 0 is the
  probed task, so nothing is burn-in.
- `by_readout` stacks a cell's columns per readout as `(T, n_seeds)`,
  dropping seeds with any NaN.  `seed_signal(Y)`: each seed's Y/Y[0]
  (seed mean ± s.e. plotted) and each seed's `half_life` (first index
  where that seed's curve ≤ its own value at 0 / 2, linearly
  interpolated, NaN if never); `mhalf_text` gives the mean ± s.e. over
  the seeds that reach ½ and the count of those that do not.
- Figure per cell `szero_N{N}_k{κ}_d0{δ₀}_lam{λ}`, one panel 5.2 × 2.6 in:
  x = t + 1 (log), y = the seed mean of each seed's S₀(t)/S₀(0) with the
  ±1 s.e. band, `ref_line` at 0 and ½, m½ per readout in the caption.
  `.npz`: `task`, `S0_{readout}` (T, n_seeds), `S0norm_{readout}`,
  `mhalf_{readout}` (per seed).
- Result (2026-11-02, N = 100, κ = 1, δ₀ = 0, λ = 4, 4 seeds; m½ then
  read off the seed-mean curve, not yet per seed): S₀(0) ≈ 1
  for every readout.  The gated readouts decay roughly exponentially with
  m½ = 86 (gated shared), 85 (gated frozen), 68 (task heads) tasks, i.e.
  0.7–0.9 N against N ln 2 ≈ 69.  The ungated readouts drop to ≈ 0 after
  ONE task (u_t = a W_t / N is one D-vector shared by every probe, so it
  tracks only the latest teacher): memory here is a property of gating.
  Past ≈ 4 m½ every trace sits at 0 within seed noise.

### 6.3 `run_stask.py` — S_T(τ) against age at steady state

- Reuses `run_szero`'s `cells`, `groups`, `columns`, `per_cell`,
  `by_readout`, `seed_signal`, `mhalf_text` (imported from it; `_path`
  makes `scripts/` importable).
- Job: `signal_after_tasks(runs, T_pre=T, n_follow=tau + 1, k=T)` with
  `T` = `--T` or `T_burn + T_meas` (same for every cell, else
  `SystemExit`), `tau` = `--tau` (1000).  The head buffer is `T + τ + 1`
  slots.
- Figure per cell `stask_N{N}_k{κ}_d0{δ₀}_lam{λ}` as 6.2 with x = τ + 1,
  `T` in the title.  `.npz`: `tau`, `ST_{readout}` (τ + 1, n_seeds),
  `STnorm_{readout}`, `mhalf_{readout}` (per seed).
- Result (2026-11-02, same cell as 6.2, T = 1400, τ ≤ 1000; m½ of the
  seed-mean curve, before the per-seed rule): S_T(0) ≈ 1;
  steady-state half-lives gated shared 75, gated frozen 78, task heads 71
  tasks, within a few tasks of N ln 2 ≈ 69 and closer to it than the
  fresh-start values — once the norms have settled the three gated
  readouts forget at the same rate.  Ungated readouts drop to 0 after one
  task; every trace is at 0 by τ ≈ 300.

### 6.4 `run_Nscaling.py` — m½ against N and κ

- `groups`: `group_key + (T,)`, largest N first (the longest chains are
  pinned first); `make_jobs`: `signal_after_tasks(runs, T_pre=T,
  n_follow=T + 1, k=T)` per piece, cost over `2T + 1` tasks.
- `mhalf_from_signal` applies `half_life` to every column (per seed, NOT
  to the seed mean; NaN when censored or infeasible); `collect` returns
  `{(N, kappa_label, readout): (n_seeds,)}` for conditions with at least
  one finite seed, printing mean ± s.e. and the censored count.
- `fig_readouts` (`mhalf_vs_N_k{κ₁}_{tag}`, panel 4.0 × 2.8 in): one
  trace per readout at `klabels[0]` (`"sqrtN"` for ungated), log–log,
  `discrete_ticks` at the N grid, dotted grey N ln 2, legend with the
  `styles` row.  `fig_theory` (`mhalf_theory_{kind}_{tag}`, one panel per
  gated readout, 3.0 × 2.8 in): traces = κ via `kappa_shades(i, n,
  readout)`, solid simulation with error bars, dotted `theory(N, κ,
  kind)`: `nln2` → N ln 2, `nk2` → N ln 2/κ².  `mhalf_{tag}.npz` holds
  the per-seed arrays by `N{N}_k{κ}_{readout}`.
- Result (2026-11-02, δ₀ = 0, λ = 4, 4 seeds, no seed censored): at κ = 1
  every gated readout grows as m½ ∝ N from N = 5 to 100 (log–log slope
  ≈ 1); task heads sits on N ln 2 at every N (67 ± 7 at N = 100 vs 69),
  gated frozen just below (56 ± 8), gated shared at about 0.6 of it
  (42 ± 3).  Ungated readouts: 0.5 tasks at every N.  Against κ: task
  heads shows no κ dependence (66–76 at N = 100 for κ = 1, 2, 4); gated
  shared and gated frozen are flat between κ = 1 and 2 but drop to ≈ 13
  tasks at κ = 4, N = 100, and to the ungated floor at κ = 4, N ≤ 20,
  where the gate density κ/√N ≥ 0.9 makes the runs effectively ungated.
  So N ln 2 is the right equation for task heads at every κ and for the
  shared readouts at κ/√N ≲ 0.2; N/κ² is not supported.  The per-seed m½
  averaged here is smaller than the crossing of the seed-mean curve in
  6.3 (42 vs 75 for gated shared at N = 100, κ = 1): per-seed curves are
  noisy and noise reaches the threshold early.  The two numbers are not
  interchangeable.

### 6.5 `run_weight_norms.py` — norms against task and against N

- `groups`/`make_jobs` as 6.4 with `norms_over_tasks(runs, T)`;
  `plan(..., out_per_task=8 · 5)`.
- `collect` returns `{(N, kappa_label, readout): {key: (n_seeds, T)}}`
  over `ALL_KEYS = experiments.NORM_KEYS`; the console line gives the
  steady-state means (window `t ≥ T_burn`).
- Rows drawn (`QUANT` + `Q_SCALED`): `Q_all`, `a2_all`, `u2`, and the
  collapse variable `q_scaled`: `Q̄/pN` when `learn_a = 0`, `Q̄²/pN` when
  the readout learns, with `pN` from the condition's `Run`.  (`Q_act`,
  `a2_act` are recorded and saved but not drawn.)
- `fig_vs_task` (`norms_vs_task_k{κ₁}_{tag}`): rows × columns = N,
  traces = readouts at `klabels[0]`, `_mean_se` over finite seeds, band,
  `shade_burnin`; the infeasible-seed count from the top row.
  `fig_vs_N` (`norms_vs_N_{tag}`): rows × columns = κ, x = N (log,
  discrete ticks), per-seed steady-state means then seed mean ± s.e.;
  ungated readouts repeat in every column.  `norms_{tag}.npz` holds every
  per-run curve, `N{N}_k{κ}_{readout}_{key}`.
- The `run0` argument of both figure functions is the full run list; a
  run per condition is looked up for `pN`, `learn_a`, `T_burn`.

### 6.6 `run_imbalance_comp.py` — norms and m½ across δ₀

- Runs: the `run_weight_norms` grid at every `--delta0` (default
  `0 1 2 4`); `groups` as 6.4/6.5; two pools, `norms_over_tasks(runs, T)`
  then `signal_after_tasks(runs, T, T + 1, T)`, each planned with
  `pool.plan` over the same groups.
- `collect` returns `{(N, kappa_label, readout, delta0): {key: (n_seeds,
  T) for ALL_KEYS, "mhalf": (n_seeds,)}}` (`mhalf_from_signal` per seed).
- `weight_norm_by_delta/norms_vs_N_{readout}` (`fig_norms_by_delta`):
  rows `QUANT + Q_SCALED`, columns = κ, x = N (log, discrete ticks),
  per-seed steady-state means then seed mean ± s.e. bars, traces = δ₀ as
  `kappa_shades(i, n, readout)` (lightest = smallest δ₀); an ungated
  readout repeats its κ-free values in every column.
  `weight_norm_by_model/norms_vs_N_d0{δ₀}`: `run_weight_norms.fig_vs_N`
  on the slice at that δ₀.
- `halflife_by_delta0` / `halflife_by_model` (`fig_halflife`): columns
  = κ, x = N (log), y = m½ (log) with `_stat` over uncensored seeds; rows
  = readouts with δ₀ traces (shades), or rows = δ₀ with readout traces
  (`plot_readout`); dotted `N ln 2` in every panel.
- `imbalance_comp.npz`: `N{N}_k{κ}_{readout}_d0{δ₀}_{key}` and `_mhalf`.

### 6.7 `run_training_verify.py` — the per-step loss at start, middle and end

- Cells `(N, κ, δ₀)` from `--cell N,kappa,delta0` (defaults `100,2,1`,
  `50,1,0`, `20,1,4`, `200,2,1`);
  every readout × seed of a cell under the config's stream rules;
  `groups` by `group_key + (T,)`, `pool.plan` over them.
- `experiments.loss_curves(runs, tasks, T)`: trains the stream with
  `train_kernel` between the listed tasks and runs `loss_kernel`
  (`loss_task`: `train_block(traced=True, closed=False)`, the loss column
  of the trace rows, preceded by the loss before the first step) on each
  listed task → (B, len(tasks), n_steps + 1).  `windows(T)` = `[0, 1,
  T/2 − 1, T/2, T − 2, T − 1]`.  `tests/test_experiments.py` pins the
  curve to the stream's `trace` and `L_end`.
- Figure `loss_N{N}_k{κ}_d0{δ₀}`: `grid(1, 3, sharey="row")`, each panel
  two tasks on x = step within task (ticks 0, n, 0, n), the switch a dotted
  `axvline` at n + ½, y log with ≤ 6 ticks, inner spines removed and break
  marks between the panels; `plot_readout` + `band` (seed mean ± s.e.);
  `.npz`: `{readout}` (n_seeds, 6, n_steps + 1), `tasks`, `step`.

## 7. Plotting conventions used (`mup/plotting.py`)

`apply_style()` first in every figure function.  Panels from
`grid(nrows, ncols, sharex, sharey, panel=(w, h))`; one `cell_title`
(suptitle naming κ, δ₀, λ, solver, η, plus `extra`), one `caption`, one
`fig_legend` row above the panels (readout handles in `READOUT_ORDER`,
`styles=[(linestyle, label)]` or `handles=` for extra entries).  Readout
colours are fixed (`READOUT_COLOR`, Okabe–Ito: gated shared `#0072B2`,
learned shared `#56B4E9`, gated frozen `#D55E00`, frozen `#E69F00`, task
heads `#009E73`); ungated readouts dashed (`plot_readout` applies both);
κ as shades of the readout colour (`kappa_shades`); reference lines
`ref_line` in grey; bands `band` at α 0.2; `logaxis`, `discrete_ticks`
for the N grid; parameter values through `fmt` (`4`, never `4.0`).  Save
only through `save(fig, path, figdata=...)`.  No script sets `rcParams` or
picks a colour.

## 8. Tests that cover this code

```
pytest tests                           # everything (~2 min on the laptop)
pytest tests/test_experiments.py       # the kernels of §4 against protocol.run_task, chunking,
                                       # NaN columns, split/group, the pool in-process vs subprocess,
                                       # failure reporting, the planner (CPU and mocked GPU), progress lines
pytest tests/test_block.py             # train_block vs the explicit loop, overflow → NaN, closed form,
                                       # masked heads, infeasible-row redraw, σ₀
python -m pytest tests/test_gpu.py -v  # on the GPU node: backend, each card vs CPU, chunking, one job per card
```

`tests/conftest.py`: `small_runs` (N = 10, every readout, one seed, short
T) via `one_batch`, which gives the ungated runs the gated runs' κ and T
so all five readouts share one batch.  Changing a kernel changes
`tests/reference_hash.txt` (test 6 of the sweep); delete it and rerun to
regenerate.

## 9. Invariants and gotchas when changing things

- **Per seed, then the mean.** Every run draws its own weights,
  teachers, gates, heads and SGD inputs from `Run.rng_seed()`, so no two
  seeds share a stream.  Every statistic a figure or table reports is
  computed on each seed's own data and only then averaged over seeds
  (± s.e.m. over seeds): half-lives from each seed's S(m), normalisation
  by each seed's own S(0) or level, CVs from each seed's moments, ratios
  per seed.  Never compute a nonlinear statistic on a seed-pooled or
  seed-averaged curve.  Helpers: `results.seed_curves`, `seed_mhalf`,
  `mhalf_stats`, `seed_mean`, `seed_sem`; `experiments.seed_curves` for
  `stream_batch` results.  A condition's m½ is NaN and `censored` when
  any seed's curve never reaches ½ (`m_half_lower`, `n_censored` keep
  the rest); the cell scripts' captions give the mean over the seeds
  that cross with the count of those that do not.
- **A job is homogeneous.** Runs in one job share `(N, D, solver,
  n_steps, closed)`; `_batch` asserts it.  Anything that varies per run
  must be a `Params` field (a 0-d float), not a Python branch.
- **Head slots.** `P` must cover the last probed task: `T_pre +
  n_follow` for the signal scripts, `T` for the norms.  Memory is
  `8·(N·D + P·N + …)` per run (`protocol.state_bytes`), and `plan` uses it.
- **Donated state.** Every kernel donates its state argument; keep the
  returned state and never touch the old object.
- **Kernel cache.** `lru_cache` keys are the static arguments; a new
  `n_tasks` (chunk size) compiles anew, which is why `chunks` uses a
  fixed size with one smaller remainder.
- **Ungated runs once.** The cell scripts list each ungated run in every
  κ cell but train it once (`groups` dedup by `Run.label()`); the scaling
  scripts have one ungated condition per (N, readout) at label `"sqrtN"`.
- **κ label lookup.** `--kappa` values are kept as typed and looked up by
  `str(k)`; do not normalise one side only.
- **Pieces align with groups.** `plan.pieces[i]` belongs to the i-th
  group passed to `plan`; `make_jobs` must iterate the same dict in the
  same order.
- **Equal T per call.** `run_szero`/`run_stask` require every cell to
  have the same number of tasks (one kernel shape); with several `--N`
  pass `--tasks`/`--T`.
- **NaN is the infeasible marker** everywhere; use `np.isfinite` masks,
  never `nanmean` on an all-NaN axis without guarding (`_mean_se` does).
- **Seeds and conditions are independent draws** (`rng_seed`); a change
  to the hash changes every result.  The solver and η are not in it on
  purpose.
- **Plotting warnings are failures**: `save` warns about illegible
  panels, and `tests/test_plotting.py` fails on any such warning for the
  families it renders.
- **The GPU node's JAX is 0.4.28 on an older Python**; avoid newer syntax
  and newer `jax` APIs in anything these scripts import.
- **Timing line.** Every script prints `run_<name>: total … s` at exit;
  `_T0` is taken after the imports.

## 10. History (what changed and why)

- 2026-11-02: the four signal/gauge scripts written against the sweep's
  JAX kernels; the first results above.  `run_Nscaling` and
  `run_weight_norms` were then ported to a NumPy kernel (`mup/npmodel.py`)
  after 96 CPU workers compiling XLA at once broke the Xeon pool.
- 2026-11-05: back to JAX for the GPU node: `mup/experiments.py`
  (`train_task` = the sweep kernel without probes, observables inside a
  `scan`) and `mup/pool.py` (jobs over the cards); `npmodel.py` kept as a
  test reference.  Probe teachers and gates regenerated from the key
  rather than stored (device memory).
- 2026-11-06: `pool.plan` — `--gpus` alone plans processes per card,
  chunk and the split from chain lengths and memory.  Review fixes: stored
  heads gate-masked; κ looked up by the label as typed; infeasible rows of
  W⁽⁰⁾ redrawn instead of dropping the run; the active-row block loop
  with lazy row scale and tracked Q (`train_block`) and the frozen
  population closed form; the cell scripts train each ungated run once.
  Documentation narrowed to the five scripts (this file).
- 2026-11-06: per-seed statistics everywhere.  Half-lives, S/S(0), CVs,
  half-window ratios and the M8 level had been computed on seed-pooled
  curves in `results`, `analyze`, `plot_check2`, `plot_norms`,
  `run_szero`, `run_stask`, `run_sigma0` and `validate`; now each seed's
  own value is averaged (§9).  `experiments.pooled_curves` became
  `seed_curves`, which also keeps every seed of a condition split over
  two batches (the old dict update kept only the last batch's seeds).
- 2026-11-06: the sweep-based scripts moved to `old/` (Scope above);
  `run_all.sh` runs the experiment scripts in the note's order with every
  parameter written out.
