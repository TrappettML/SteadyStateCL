# Coding plan: μP continual-learning experiment (JAX, float64, CPU)

Implements `writing/derivations/muP_experiment/muP_experiment.tex` (the
"plan"; equation labels below refer to it). Pinned stack
(`code/SteadyStateCL/requirements.txt`, one level above this folder):
Python 3.12, `jax==0.12.0.1` / `jaxlib==0.12.0.1` (CPU), `numpy==2.5.4`,
`scipy==1.18.1`, `matplotlib==3.11.2`, `PyYAML==6.0.3`, `pytest==9.2.0`.

Two machines:

| | develop / test | full run |
|---|---|---|
| hardware | Apple silicon laptop | 4-socket Xeon Gold 6348H, 96 cores / 192 threads, x86_64 |
| JAX backend | CPU | CPU |
| precision | float64 | float64 |
| role | correctness, pilot at N ≤ 20 | Stage-1 sweep |

Same code path on both; nothing is backend-specific. The venv on the laptop
is `code/sscl_env` (outside the repo); recreate it on the Xeon with
`python3.12 -m venv sscl_env && pip install -r SteadyStateCL/requirements.txt`
(run from `code/`).

## 0. Design decisions

1. **One implementation, float64, JAX on CPU.** `jax.config.update(
   "jax_enable_x64", True)` at import time in `mup/__init__.py`, before any
   array is made. Assert `jnp.zeros(1).dtype == jnp.float64` in a test so a
   forgotten flag can't silently downgrade a run. No MLX, no GPU.
2. **Equations are implemented exactly; nothing is approximated.** The gate
   is carried as a 0/1 float vector `g` of length N and every unit is
   updated with its update multiplied by `g_i`; inactive units therefore
   receive exactly zero, as the plan prescribes. (Touching only the active
   rows would be cheaper by a factor √N/κ ≤ 10 at N ≤ 100, but needs a
   data-dependent shape that `jit` cannot trace; at these sizes it does not
   matter.)
3. **Fixed shapes, `jit` once per N.** All array sizes (N, D, batch B,
   number of probes, age schedule) are static. The inner loop over steps of a
   task is a `lax.fori_loop` (or `lax.scan` when the per-step loss is
   wanted) inside one jitted function `run_task`, so Python overhead is paid
   once per task, not per step.
4. **Batch with `vmap` over runs.** A "run" is (seed, κ, δ₀, λ, readout).
   `run_task` is written for one run and `vmap`ed over a leading axis B.
   Runs in a batch must share N, D and the per-task step count (§3.6); the
   readout condition is a per-run mask (§4), not a Python branch, so the
   five readouts can share a batch.
8. **Gating is a per-run flag, not a separate code path.** A *gated*
   condition draws `g_t` with density `p = κ/√N`, so κ sets the sparsity;
   an *ungated* condition sets `g_t = 1` for every unit (`pN = N`, i.e.
   κ = √N in the plan's notation). Everything downstream (`key_vec = g*a`,
   the projection on active units, the probes) is unchanged, so an ungated
   run is just a gated run whose gate is all ones. In the YAML this is the
   existing `kappa: sqrtN` entry (`kappa_label = "sqrtN"`), which the
   readout name makes explicit.
5. **Parallelism on the Xeon is across processes, not inside XLA.** XLA's
   CPU backend parallelizes large matmuls with Eigen threads, but our ops are
   small ((B, N, D) with N ≤ 100, D ≤ 400) and the step loop is sequential,
   so one process will not use 192 threads well. Plan: a driver that launches
   one process per (N, batch) with a bounded pool (≈ 90 workers, one per
   physical core, leaving headroom), each pinned to a small thread count:
   ```
   XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
   OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
   ```
   and `taskset`/`numactl --cpunodebind` by socket (4 NUMA nodes; keep a
   worker's memory local). Tune B per process so that B × (per-step work)
   keeps one core busy; expect B ≈ 32–128. Measure once on the Xeon with
   `scripts/bench.py` before committing to numbers.
6. **Randomness is JAX-native and keyed.** One `PRNGKey(seed)` per run;
   `jax.random.fold_in(key, task_index)` gives the task key; from it split
   keys for teacher, gate, head, and the online-input stream. Teachers,
   gates and initial weights are drawn on-device inside the jitted code so a
   run is reproducible bit-for-bit from `(seed, config)` on either machine
   (same XLA CPU backend; cross-platform determinism is expected but is
   verified once by comparing a short run's hash across the two machines).
7. **λ = Λ ā² is the learning-per-task axis, eq. (lam).** `α_task =
   λ/(η ā²)` with ā² = 1 + δ₀ (its deterministic value at the start of task
   1), fixed per run. **Stage 1 holds λ = 4 fixed** (decided
   2026-10-30): the hidden-layer channel alone then leaves a residual of
   e⁻⁴ ≈ 2 % per task, so every task is learned well, and Stage 1 answers
   the scaling and weight-norm questions over (readout, κ, δ₀, N) only.
   How the lifetime trades against learning per task (the λ axis, at one
   fixed δ₀) is a later stage (plan, Deliverable); the code keeps λ as a
   sweep axis for it. Runs are grouped into batches by
   `(N, D, κ, solver, n_steps, T_burn, T_meas)` (§3.6): κ sets the stream
   length and the probe arrays, which must be static within a `vmap` batch.

## 1. Repository layout (`muP_experiment/` in this repo)

```
muP_experiment/
  PLAN.md  README.md   (README: how to run, result schema, deviations log)
  mup/
    __init__.py        enables x64; exports version
    config.py          Config dataclass, YAML loader, task duration, grid → runs → batches
    model.py           pure functions on a State pytree: init, sgd_step, pop_step,
                       project, forward, observables; all single-run, vmapped by caller
    protocol.py        run_task (jit, vmap, state donated), run_stream (Python loop over tasks)
    observe.py         age schedule, probes, s_k, ε_k, running moments, half-life, checks 1–3
    io.py              result schema (.npz per batch + JSON manifest), resume
    driver.py          subprocess pool with thread pinning and optional numactl (Xeon)
    plotting.py        figure conventions: one fixed colour per readout
  scripts/
    run_sweep.py       --config configs/stage1.yaml [--N 50] [--workers 90 --numa-nodes 4] [--dry-run] [--batch-id i]
    bench.py           run-steps/s vs B at fixed N on this machine
    validate.py        η-halving, Kaczmarz collapse, CV exponent
    analyze.py         summary.csv, curves.json, the default figures (slices S1/S2, summaries
                       M1–M3, checks table; §12.3, §12.6); per-cell F1–F4 only with --cells;
                       diagnostic figures only with --diag
    plot_norms.py      norms slice S1 and per-cell F1/F2 (norms vs task) from batch files;
                       by_step/ with --diag (§12.3)
    plot_check2.py     check statistics → checks table + csv (§12.6); C2 figures with --diag
  configs/
    pilot.yaml  stage1.yaml  sgd_corner.yaml  trace.yaml  (trace: per-step norms, §12.2)
  tests/
    test_x64.py test_gradients.py test_projection.py test_conservation.py
    test_vmap_vs_loop.py test_reproducible.py test_config.py test_plotting.py conftest.py
    reference_hash.txt (test 6's stored hash)
  results/             (gitignored) one subfolder per sweep:
    <sweep>/batch_*.npz  manifest.json  summary.csv  curves.json  logs/
    <sweep>/figures/     subfolders by figure family (§12.4), never loose files
    <sweep>/traces/      per-step norm traces (§12.2), only for configs with trace: true
```

## 2. State (one run; `vmap` adds the batch axis)

```python
class State(NamedTuple):      # all float64 unless noted
    W:      (N, D)            # hidden rows
    a:      (N,)              # shared readout, or the current task head
    g:      (N,)              # current gate, 0/1 (all ones when ungated)
    w:      (D,)              # current teacher
    heads:  (P, N)            # stored task heads (zeros unless readout == task_heads)
    probes_w: (P, D);  probes_g: (P, N)
    key:    PRNGKey
    t:      int32             # task index
    # per-run constants, passed as a separate Params pytree:
    # Delta0, eta, h, pN (float), lam, n_steps, gated (0/1), learn_a (0/1), do_proj (0/1), use_heads (0/1)
```
Derived per step: `key_vec = g*a`, `u = key_vec @ W / pN`, `r = u − w`,
`L = r·r/(2D)`. `pN = κ√N` is the nominal value from eq. (student), not the
realized gate count; for an ungated run `pN = N` exactly.

## 3. Kernels (`model.py`), each a pure function `(state, params, ...) → state`

### 3.1 Forward
`yhat(x) = (key_vec @ (W @ x)) / (pN √D)`.

### 3.2 Online SGD step, eqs. (sgdW)–(sgda) — simultaneous update
```python
x   = jax.random.normal(k, (D,))
hx  = W @ x / sqrt(D)                       # pre-activations, (N,)
e   = (g*a) @ hx / pN - w @ x / sqrt(D)
W_n = W - eta * (g*a)[:, None] * e * x[None, :] / sqrt(D)
a_n = a - learn_a * eta / D * g * e * hx    # learn_a = 0 → frozen readout
```
Both updates use the pre-step `W`, `a`.

### 3.3 Population step, eq. (pop)
```python
r   = (g*a) @ W / pN - w
W_n = W - h*eta * (g*a)[:, None] * r[None, :]
a_n = a - learn_a * h*eta * g * (W @ r) / D
```

### 3.4 Projection, eq. (proj), active units only, after every step
```python
Q   = (W*W).sum(-1) / D
A   = a*a / eta;  B = Q / eta
Ap  = 0.5 * (Delta0 + jnp.hypot(Delta0, 2*jnp.sqrt(A*B)))   # = ½(Δ₀+√(Δ₀²+4AB)), no cancellation
Bp  = A*B / Ap                                              # = Ap − Δ₀ exactly, product form
a_p = jnp.sign(a) * jnp.sqrt(eta*Ap)
f   = jnp.sqrt(eta*Bp / Q)
act = (g == 1) & (do_proj == 1)
a   = jnp.where(act, a_p, a);  W = jnp.where(act[:, None], W*f[:, None], W)
heads = jnp.where(use_heads, heads / jnp.where(act, f, 1.0)[None, :], heads)
```
`u_c` is invariant because `a_p·|W_p| = a·|W|` (A′B′ = AB). Guard: assert
`a != 0` on init (sign(0) = 0 would pin a unit); with Gaussian init and
float64 this is never hit in practice, so the kernel just asserts in tests.

### 3.5 Initialization
```python
W0 = normal((N, D));  Q0 = (W0*W0).sum(-1)/D
a0 = rademacher(N) * sqrt(eta*(Delta0 + Q0/eta))
```
`config.py` refuses δ₀ ≤ −1 outright. Whether a given seed is feasible
depends on the realized `Q0_i` (need δ₀ > −min_i Q0_i), which is only known
on device: at N ≤ 20 the row norms fluctuate enough that δ₀ = −0.75 is
infeasible for many seeds (624 of the 5760 gated runs of the earlier 4-λ
grid; `run_sweep.py --dry-run` reports the count for the current one). Such runs
are skipped at run time, listed under `skipped` in the batch metadata, and
reported by `analyze.py`. Task heads: a fresh head is drawn this way from
the current `Q` at the start of each task and stored into `heads[t]`.

### 3.6 Task duration (`config.py`, host side, before batching)
```
abar2       = 1 + δ₀                     (deterministic part of mean_{g_i=1} a_i² at task 1)
alpha_task  = lam / (eta * abar2)
SGD:        n_steps = round(alpha_task * D);   require eta*abar2 ≤ sgd_contraction (0.05)
population: h = min(h_max, pop_contraction/(eta*abar2));  n_steps = ceil(alpha_task / h);  h = alpha_task/n_steps
```
`abar2` is taken as `1 + δ₀` so that `n_steps` is known before any device
work; the realized `abar2` of task 1 is recorded and the realized λ
reported. With `pop_contraction = 0.1` (the plan's `h ā² ≤ 0.1`) the
population step count is `n_steps = ⌈10λ⌉` per task, independent of δ₀ and
η, so the Stage-1 sweep is cheap (§8). The price is an O(h) Euler bias:
`validate.py` measures ΔS(0) ≈ +0.017 (gated frozen) and +0.005 (gated
shared) at N = 10, λ = 1 against a fine-step reference, the same size as the
η effects the SGD comparison is meant to detect. `pop_contraction: 0.01`
removes 90 % of it at 10× the steps; use it for any population/SGD
comparison, 0.1 for the lifetime sweep. Runs are bucketed by
`(N, D, κ, solver, n_steps, T_burn, T_meas)` (decision 7); each bucket is
split into `vmap` batches of ≤ `batch_size` (64).

### 3.7 `run_task` (jit, single run)
```
draw (w_t, g_t[, head_t]) from fold_in(key, t);  g_t = bernoulli(p) if gated else ones(N)
lax.fori_loop(0, n_steps, step_and_project, state)        # step = sgd or pop
store w_t, g_t[, head_t] at probes[t]
evaluate probes at the age schedule (§5) → (n_ages,) s and ε, plus validity mask
per-task scalars: abar2, mean Q, mean a², mean/max |a|√Q, L_end, gauge residuals
```
`run_stream` (Python) loops `t = 0 … T−1` calling the vmapped `run_task`,
accumulating the `(B, n_ages)` moments and per-task scalars on host with
`np.asarray` once per task (one device→host copy per task, not per step).

## 4. Readout conditions as masks

Every readout is *shared* across tasks except the task heads, and every
readout is either *gated* (gate drawn with density `p = κ/√N`, so κ induces
the sparsity) or *ungated* (`g = 1` for all units, `pN = N`). The names make
both facts explicit, so that what a mask does is visible in the condition
label:

| readout | gated | learn_a | do_proj | use_heads | notes |
|---|---|---|---|---|---|
| gated shared | 1 | 1 | 1 | 0 | one `a`, learned every task; κ-sparse gates |
| learned shared | 0 | 1 | 1 | 0 | one `a`, learned every task; no gating |
| gated frozen | 1 | 0 | 0 | 0 | a = a⁽⁰⁾ forever; Q_ii unpinned; κ-sparse gates |
| frozen | 0 | 0 | 0 | 0 | a = a⁽⁰⁾ forever; Q_ii unpinned; no gating |
| task heads | 1 | 1 | 1 | 1 | `a` ← fresh head each task; stored heads rescaled by 1/f; κ-sparse gates |

Config keys: `gated_shared`, `learned_shared`, `gated_frozen`, `frozen`,
`task_heads`. The two ungated conditions do not vary with κ (they are the
κ = √N point of the plan's table), so the sweep lists them once per
(N, δ₀, λ, seed). **Their stream length and age schedule are not set from
κ = √N**, see §5 (schedule κ): with κ = √N the current code gives
`T_burn = T_meas = 10` tasks and `m_max = 19` at every N, so the ungated
traces stop at age ≈ 20 in every figure while the gated κ = 1 traces run
to `40N`. An ungated task-heads condition is the same flag flip
and can be added later if wanted. Probe evaluation uses `heads[k]` when
`use_heads` else the current `a`.

## 5. Probes and observables (`observe.py`)

- Every task is a probe (N ≤ 100): `P = T_burn + T_meas`, preallocated,
  with `T_burn = T_meas = max(T_min, round(T_factor·N/κ_s²))`, `T_factor =
  10`, `T_min = 4`, where **`κ_s` is the schedule κ**: the run's own κ for
  a gated run, and `κ_s = min(κ in the sweep)` (= 1 in Stage 1) for an
  ungated run. The ungated lifetime is ∝ N (check 4 uses `m/N`), not
  `N/κ²` with κ = √N, so sizing the stream from κ = √N truncates it at
  20 tasks; using `κ_min` gives the ungated conditions exactly the stream
  length and age schedule of the smallest-κ gated runs they are overlaid
  with, so every dashed trace is evaluated at every age the figure shows.
  `Run` gets a `kappa_sched` field; the bucket key of §3.6 uses it in
  place of `kappa` (so ungated runs batch with the κ_min gated runs of the
  same N, which share shapes). Cost: an ungated run is `20N/κ_min²` tasks
  instead of 20, i.e. the same as a gated κ_min run — at N = 100 that is
  2000 tasks × ≤ 40 steps, still seconds (§8).
- Age schedule (static): `ages = [0..20] ∪ geomspace(24, m_max, 24)`
  rounded to unique ints, `m_max = 4·10N/κ_s²` capped at `T−1`. For task `t`
  the probe indices are `k = t − ages` with validity `k ≥ 0`; invalid
  entries read probe 0 and are masked out of the sums.
- `u_k = (g_k*a_k) @ W / pN`, `s_k = u_k·w_k/|w_k|²`,
  `ε_k = |u_k−w_k|²/|w_k|²`, computed for all ages in one `(n_ages, N) @
  (N, D)` matmul.
- Accumulate `(Σs, Σs², Σε, Σε², count)` per `(run, age, half)` where
  `half ∈ {0,1}` splits the measurement window; burn-in tasks are not
  accumulated. Nothing per-sample is stored. Check 2's standard error treats
  consecutive tasks as independent, so its flags are conservative
  (over-sensitive) for slowly drifting norms: read the `half0`/`half1`
  values, not just the flag.
- `m_half`: first age with `S(m) ≤ S(0)/2`, linear interpolation between
  bracketing ages; `NaN` with `censored=True` if not reached.
- Figures for `S(m)` and `E(m)` (`analyze.py`, §12.3): by default the
  curves slice S2, one figure per (readout, κ, λ, solver, η) with rows
  `S/S(0)` and `E`, **columns = δ₀**, **traces = N** and x = age m in tasks;
  per cell, with `--cells`, F3 (**columns indexed by N**, **traces = the
  readout types** in the fixed colours of `plotting.py`) and F4 (columns
  = readouts, traces = N). The ungated conditions have no κ and are repeated in every
  κ figure; they are run on the κ_min schedule, so in the κ = κ_min
  figure they cover the full age axis, and in a larger-κ figure (shorter
  axis) they are simply truncated to that figure's `m_max`. `analyze.py`
  asserts that every trace in a panel has data at the panel's last age
  and warns otherwise, so a half-plotted trace is caught, not shipped.
- Per task: `abar2`, mean `Q_ii`, mean `a_i²`, mean and max of `|a_i|√Q_ii`,
  `E(0)`, `L_end` — all taken *after* the last projection of the task. §12
  adds the same norms *before* that projection and the per-step traces.

## 6. Checks from the plan

| check | where |
|---|---|
| 1 gauge | `max|Δ_i − Δ₀|` over active units and `|u_c^{after} − u_c^{before}|` after the last projection of each task, returned by `run_task`; expect ~1e-13 in float64 |
| 2 bounded norms | half-window means of the per-task scalars and of `S(0)`, `S(m)`; flag if halves differ by > 3 s.e.; reported in the checks table (§12.6), located on the drift map M1 and read off the norms slice S1 (§12.3) |
| 3 self-averaging | CV of `s_k` at fixed age from the running moments; fit the exponent vs N (the plan's N^{-1/2} is the frozen-readout prediction) |
| 4 Kaczmarz | `S(m)/S(0)` vs `κ²m/N` collapse across N for gated frozen, the curves slice S2 of that readout (§12.3) and a row of the checks table (for ungated frozen the variable is `m/N`, which needs ages up to ∝ N — the κ_min schedule of §5) |

## 7. Tests (`pytest`, run on the laptop before every push)

1. `test_x64`: default dtype is float64 after `import mup`.
2. `test_gradients`: (sgdW), (sgda) equal `pN·η·∇_W ℓ` and `(pN·η/D)·∇_a ℓ`
   from `jax.grad` of the literal loss, to 1e-12.
3. `test_projection`: `u_c` unchanged to 1e-13; `Δ_i = Δ₀` to 1e-13; stored
   heads keep every stored `u_k` unchanged.
4. `test_conservation`: population flow without projection drifts `Δ_i` by
   O(h²) per step (check by halving h); SGD drifts it by O(η²).
5. `test_vmap_vs_loop`: a batch of 5 runs (one per readout) equals five
   single runs with the same keys. Draws (teacher, gate, inputs) are
   compared exactly; states to 1e-10, not bit-for-bit, because XLA reduces
   the batched `(B, N, D)` matmul in a different order from the single
   matvec (one ulp per step).
6. `test_reproducible`: a 5-task run at N = 10 hashes to a stored value
   (regenerate the hash deliberately when the kernels change; compare the
   hash once on the Xeon).
8. `test_config`: grid expansion, task duration, bucketing and the age
   schedule; ungated runs get `T_burn`, `T_meas` and `ages` equal to those
   of the κ_min gated run at the same N.
10. `test_plotting`: the default figures (S1, S2, M1–M3, the checks
   table) and the per-cell F1–F4 render from a small synthetic record set
   without a `plotting.save` warning (§12.8).
7. `scripts/validate.py`: η-halving at fixed λ — the gap between SGD and the
   population curve should halve, consistent with the η² term in the plan;
   frozen-readout collapse; CV exponent.

## 8. Throughput and the Xeon plan

Per step per run: ~3 passes over `W` (N·D doubles = 320 kB at N = 100,
D = 400), i.e. ~1 MB of memory traffic. Measured on the laptop: ~17k
run-steps/s at N = 100 with `vmap` over B = 64. Costs at N = 100, κ = 1,
T = 20N/κ² = 2000 tasks:

- population, any δ₀: `n_steps = ⌈10λ⌉ ≤ 40` per task (§3.6) → ≤ 8·10⁴
  steps/run → seconds per run; the whole Stage-1 sweep is minutes of
  core-time per batch. (An earlier estimate of 680 steps/task assumed a
  different h; the Xeon is not needed for Stage 1 itself, only for the SGD
  corner and for `pop_contraction = 0.01` reruns.)
- SGD, λ = 4, δ₀ = 16: η = 0.05/17 ≈ 0.003, `n_steps = λ/(ηā²)·D ≈ 3·10⁴`
  per task → 6·10⁷ steps/run → hours per run per core.
  Run SGD only in the low-(λ, δ₀) corner and at two η (validation 7):
  `configs/sgd_corner.yaml` is N ∈ {10, 20, 50}, κ = 1, δ₀ = 0,
  λ ∈ {0.5, 1}, η ∈ {0.05, 0.025}, 4 seeds.
  **Open (2026-10-30):** this corner is at λ ≤ 1, so it does not test the
  population solver at the Stage-1 λ = 4. Adding λ = 4 at δ₀ = 0,
  N ≤ 20 costs 4× the steps of λ = 1 (`n_steps = λ/(ηā²)·D`); decide
  before X4.

Stage 1 (Table 1 of the plan, at λ = 4 only, decision 7): gated
conditions 5 N × 2 κ × 3 readouts (gated shared, gated frozen, task heads)
× 6 δ₀ × 1 λ × 8 seeds = 1440, plus ungated conditions 5 N × 2 readouts
(learned shared, frozen) × 6 δ₀ × 1 λ × 8 seeds = 480, i.e. 1920
population runs (a quarter of the 7680 of the earlier 4-λ grid). Ungated runs have `pN = N`
active units per task, so each step touches every row; their cost per step
is the same as a gated run's (the gate is a mask, decision 2). With ~90
workers that is well under a day. How the sweep is organized:

1. `run_sweep.py --dry-run` expands the YAML grid into runs, buckets them
   into `vmap` batches (§3.6) and writes `results/<name>/manifest.json`
   listing every batch and its status; nothing is computed.
2. `driver.py` (invoked by `run_sweep.py --workers K`) launches one
   `run_sweep.py --batch-id i` subprocess per pending batch from that
   manifest, with the thread-pinning environment of decision 5 and optional
   `numactl` binding. Each batch writes its own `batch_i.npz`; a batch whose
   file exists is skipped, so an interrupted sweep resumes by rerunning the
   same command.
3. `analyze.py results/<name>` merges the batch files into `summary.csv`,
   `curves.json` and `figures/`; it never reads raw state.

## 9. Running the code: what to run, where, in what order, and why

All commands are run from `muP_experiment/` with the venv active
(laptop: `source ../../sscl_env/bin/activate`; Xeon: the venv created in the
preamble). Results land in `results/<config name>/` and are gitignored, so
Xeon results come back by `rsync`, not by git.

### 9.1 Laptop (develop / test) — do these in order, every time the kernels change

| step | command | what it does | why this order |
|---|---|---|---|
| L1 | `pytest tests` | tests 1–6 of §7 (~15 s) | nothing else is trusted until the kernels, projection, batching and the stored hash pass |
| L2 | `python scripts/run_sweep.py --config configs/pilot.yaml` | in-process (`--workers 0`) pilot at N ∈ {10, 20}, κ = 1, all readouts, 2 seeds (~10 s) → `results/pilot/` | the smallest end-to-end exercise of the full protocol (burn-in, probes, moments, `.npz` schema) |
| L3 | `python scripts/analyze.py results/pilot` | `summary.csv`, `curves.json`, `figures/{slices,summary,checks}/` for the pilot (§12.3) | read the checks table, the maps M1 and the slices S1/S2 before spending Xeon time; a wrong sign, a non-stationary norm or a ✗ shows here |
| L4 | `python scripts/validate.py [--N-max 40 --seeds 4]` | validation 7: η-halving at fixed λ, Kaczmarz collapse for gated frozen, CV exponent → `results/validate/` | these are the physics checks that the equations, not just the code, are right; minutes on the laptop, so they run here, not on the Xeon |
| L5 | `python scripts/bench.py --N 20 100 --B 1 32 128` | run-steps/s vs batch size | optional on the laptop; gives a baseline to compare against the Xeon number in X1 |

L1–L3 are the loop during development (milestones 1–3). L4 is run once per
kernel change and before any Xeon sweep. Push only when L1 passes.

### 9.2 Xeon (full run) — once per sweep

| step | command | what it does | why this order |
|---|---|---|---|
| X0 | `git pull`; (from `code/`) `python3.12 -m venv sscl_env && . sscl_env/bin/activate && pip install -r SteadyStateCL/requirements.txt`; then from `muP_experiment/`: `pytest tests` | same code, same pinned stack, same tests | test 6's hash is compared here once; a mismatch is a reason to compare curves, not an error (§11) |
| X1 | `python scripts/bench.py --N 20 100 --B 1 32 128` with the decision-5 env (`XLA_FLAGS=... OMP_NUM_THREADS=1 ...`) | run-steps/s per pinned core vs B | chooses `batch_size` in the YAML and `--workers`; done before the sweep so the worker count is set from a measurement, not a guess. Matters mostly for X4 (SGD), since the population sweep is cheap (§8) |
| X2 | `python scripts/run_sweep.py --config configs/stage1.yaml --dry-run` | writes `results/stage1/manifest.json`, prints the batches and the skipped (infeasible δ₀) runs | confirms the grid (1920 runs at λ = 4), the batch count and the per-batch step counts before launching 90 processes |
| X3 | `python scripts/run_sweep.py --config configs/stage1.yaml --workers 90 --numa-nodes 4` (in `tmux`/`nohup`) | the Stage-1 population sweep; one log per batch in `results/stage1/logs/` | resumable: rerun the same command after an interruption and finished batches are skipped |
| X4 | `python scripts/run_sweep.py --config configs/sgd_corner.yaml --workers 90 --numa-nodes 4` | the SGD validation corner (N ≤ 50, λ ≤ 1, δ₀ = 0, two η) | after X3, because it is the expensive solver (§8) and only needed once the population sweep has shown where the interesting corner is |
| X5 | `python scripts/analyze.py results/stage1` and `... results/sgd_corner` | summaries and figures; check 1–4 output to stdout | can be run while X3 is still going (it only reads finished batches) to catch problems early |
| X6 | `rsync -a xeon:.../results/{stage1,sgd_corner} results/` to the laptop | bring back `.npz`, `summary.csv`, `curves.json`, figures | figure iteration (`analyze.py` is cheap) happens on the laptop |

A single batch can be run in the foreground for debugging with
`python scripts/run_sweep.py --config configs/stage1.yaml --batch-id 3`
(after X2 has written the manifest), and the sweep can be restricted to
one width with `--N 100`.

### 9.3 Which file does what when you run it

- `scripts/run_sweep.py` is the only entry point that computes runs. With
  no `--workers` it runs every batch in-process (laptop); with `--workers K`
  it hands the manifest to `mup/driver.py`, which runs `run_sweep.py
  --batch-id i` subprocesses; `--batch-id` is therefore also the way to run
  one batch by hand.
- `scripts/analyze.py` only reads `results/<name>/batch_*.npz`.
- `scripts/validate.py` and `scripts/bench.py` build their own small runs
  internally and do not touch `results/<sweep>/`.
- `mup/` is a library; nothing in it is run directly.

## 10. Milestones

1. `config.py`, `model.py`, tests 1–4; single-run `run_task` and
   `run_stream`; pilot at N ∈ {10, 20} on the laptop (§9.1 L1–L3). Look at
   the curves.
2. `vmap` batching, readout and gate masks, task heads; tests 5–6.
3. Observables, checks 1–3, `.npz` schema, `analyze.py` for a pilot;
   `validate.py` (L4).
4. `driver.py` + `bench.py`; X0–X2 on the Xeon with the pilot config
   (`--config configs/pilot.yaml --workers 4`) as the dry run.
5. X3–X6: Stage 1 population sweep, SGD validation corner, analysis.

## 11. JAX notes

- Enable x64 before anything else; `jnp.array(1.0)` made earlier stays
  float32.
- `jit` retraces on new static shapes and on Python scalars passed as
  static args: pass `n_steps`, N, D as `static_argnums`, everything else as
  arrays.
- `lax.fori_loop` with a static trip count unrolls nothing and compiles
  fast; use `unroll=8` or so only if `bench.py` shows a gain.
- `run_task` is `jit(vmap(f), donate_argnums=(0,))`: the batched state is
  donated so the probe and head buffers are updated in place rather than
  copied every task; don't reuse the old state object after the call.
- Keep `jax.random.fold_in` for task keys so tasks are independent of how
  runs are batched.
- Host transfers (`np.asarray`) once per task; never inside the step loop.
- CPU determinism: XLA CPU is deterministic for this op set; the only
  cross-machine difference to watch is fused-multiply-add usage by the
  compiler, which `test_reproducible` on both machines will reveal. If the
  hashes differ, that is not an error, just a reason to compare curves
  instead of hashes.

## 12. Figures: weight norms vs task, memory vs age, and the checks table

Purpose: see with our own eyes whether the norms are stationary across tasks
(check 2) and what a single task does to them step by step, for every
readout. Two figure families: **norms vs task** (one point per task, taken
at the end of training on that task, before the projection rescales it) and
**norms vs step within a task** (every task overlaid, plus a median).

### 12.0 Figure standard (`mup/plotting.py` is the only style file)

The figures are inspection figures: looked at on a screen, in a folder of
hundreds, often several side by side. They are PNG only, sized by panel
so that every panel is legible at 100 % zoom, and they all come from one
module so they look like one set. `mup/plotting.py` owns everything
below; no script sets `rcParams` or picks a colour itself.

**Size and resolution.** One panel is 2.6 in × 2.0 in (w × h) by
default; a figure is `nrows × ncols` panels plus margins, built by
`plotting.grid(nrows, ncols, sharex="col", sharey="row", panel=(2.6, 2.0))`
which calls `plt.subplots(..., layout="constrained")`. Figures wider
than 6 columns wrap, or split into two files, rather than shrink the
panels. `savefig.dpi = 150` (≈ 400 px per panel, crisp but a 5-column
figure stays under 1 MB). Save with `plotting.save(fig, path)` which
appends `.png`, never uses `bbox_inches="tight"` (it changes the size
from file to file), and closes the figure.

**Text.** `font.size 10`, axis labels 10, tick labels 9, legend 9, panel
titles 10, suptitle 11 (bold). Sans-serif default (DejaVu Sans; nothing
needs installing). Mathtext for symbols (`$Q_{ii}$`, `$|a_i|\sqrt{Q_{ii}}$`,
`$t\kappa^2/N$`), plain words otherwise. At most 4 major ticks per
linear axis (`MaxNLocator(4)`), scientific offset text turned off
(`axes.formatter.useoffset = False`); log axes use `LogLocator` with
`numticks=4` and plain `10^k` labels. An axis that carries ≤ 6 discrete
values (the sweep's N, a categorical cell axis) instead gets ticks at
exactly those values with plain labels, log or not. Parameter values
print through `plotting.fmt` (`:g`, so `δ₀ = 4`, `λ = 1`, never `4.0`);
a figure-level number (a colour limit, a fitted slope) prints to 2
significant figures, and small magnitudes get a scale in the axis label
(`β (10⁻⁴ per lifetime)`) rather than `0.00025` tick labels. Quantities
are named only through `plotting.QUANTITY_LABEL` (key → mathtext), never
by their schema key (`aQ_act_post`, `dm_half`); file stems keep the keys.
Captions wrap to the figure width.

**Axes.** No top/right spines. Light grid (`alpha 0.25, lw 0.5`) on the
major ticks only. Column titles `N = …` on the top row only; y-labels on
the left column only; x-labels on the bottom row only (what `sharex/
sharey` give). One `suptitle` per figure stating the cell in words
(`plotting.cell_title(fig, kappa, d0, lam, solver, eta)`, e.g.
`κ = 1, δ₀ = 4, λ = 1, population, η = 1`) so a figure identifies itself
without its file name. Symmetric limits around a reference (ratio = 1,
slope = 0) where the quantity has one; otherwise matplotlib's limits
with 5 % margins.

**Colour.** Readouts have fixed Okabe–Ito hues (`READOUT_COLOR`, bound
to the readout and never to rank): gated shared `#0072B2`, learned
shared `#56B4E9`, gated frozen `#D55E00`, frozen `#E69F00`, task heads
`#009E73`. Ungated conditions are also dashed (`READOUT_STYLE`) so
gating is visible without colour. Width N, when it is a trace rather
than a column, uses `N_SHADES` (one sequential blue, light → dark with
N); the legend then lists N. Where a marker stands in for a line
(points, dumbbells), ungated readouts use hollow markers, the
counterpart of dashing and of hatched boxes. Three maps only: `viridis`
for an ordered index (task number) or an unsigned magnitude in a heatmap
(E(0) in M1), `RdBu_r` centred at zero for signed quantities (β/se and
the log₂ lifetime ratio in M1), and the white → red flag map of the
checks table. Reference lines and error bars in `0.4` grey; burn-in
shading `0.92` grey (diagnostic figures only; F1/F2 shade nothing);
not-applicable slots hatched `0.85` grey; a failed check in the table
is `#CC0000` on a light red fill (§12.6). Sign is never a colour: the
direction of a change is read from the zero line. Nothing else
introduces a colour.

**Lines and layering (bottom → top).** (1) burn-in `axvspan` and
reference lines, `lw 0.8`; (2) per-task or per-seed overlays, `lw 0.6,
alpha 0.08` (tasks) or `alpha 0.5` (seeds, ≤ 8 lines); (3) bands,
`fill_between` with `alpha 0.2`, no edge; (4) the main trace per readout,
`lw 1.6` (on a per-task x-axis this is a block mean, §12.7 G10, never
the raw per-task value); (5) medians and seed-means, `lw 2.2`, black when they summarise
one readout's overlay, readout colour when several readouts share a
panel; (6) markers only where there is one point per cell (dumbbells,
m_half ticks), size 4–5. Half-boundary and other event lines dotted
black `lw 0.8`.

**Legend.** One per figure, as a single row above the panels and below
the caption, with its height reserved in the layout
(`plotting.fig_legend(fig, readouts, styles=None)`); never inside a
panel, where it covers data. No frame, handles in the fixed readout
order of §4. When line style means something other than gating in a
figure, the readout handles are drawn solid and `styles` adds one
handle per style meaning to the same row.
Overlays and bands are not listed: the caption (`fig.text` at 9 pt,
`0.3` grey) says what they are, e.g. "thin: individual tasks; black:
median over tasks; band: 10–90 %".

**Helpers in `plotting.py`.** `grid`, `save`, `cell_title`,
`plot_readout(ax, x, y, ro, **kw)` (colour + style
from the tables), `shade_burnin(ax, T_burn, T_meas)` (shading plus the
half-boundary line), `label_cols_N(axes, Ns)`, `overlay(ax, X, Y,
color)` (thin lines + median + 10–90 % band in one call),
`check_table(ax, marks, rows, cols)` (§12.6), and the §12.7 helpers
`fig_legend`, `fmt`, `QUANTITY_LABEL`, `na_slot`, `block_mean`,
`age_axis`. The diagnostic families (`--diag`, §12.3) keep `dumbbell`;
`drift_map` generalizes to `heat_map` for M1 (§12.8). The existing `rcParams.update` block in
`plotting.py` is replaced by the values above (`plotting.apply_style()`,
called at import).

**Figure-level checks** (`plotting.save` warns): any axis with > 6
ticks, any text smaller than 8 pt, a panel narrower than 2 in, a figure
wider than 6 panels, a figure without a suptitle, any text extending
past the figure edge, a legend inside an axes, and a panel whose
plotted values are all identical (that panel should be an n/a slot,
§12.7 G4).

### 12.1 What "the norm" is, and "before rescaling"

Record, per unit, `Q_i = |W_i|²/D` and `a_i²`; reduce to

| key | definition | why |
|---|---|---|
| `Q_act` | mean of `Q_i` over active units (`g_i = 1`) | the units the task trains; the quantity the projection pins |
| `Q_all` | mean of `Q_i` over all units | gated runs: inactive units are untouched, so `Q_all` lags `Q_act`; the gap is a gating fingerprint |
| `W_fro2` | `Σ_i Q_i` (= squared root-sum-of-squares of all entries of W, /D) | the one-number "weight norm" people expect |
| `a2_act`, `a2_all` | mean of `a_i²` over active / all units | readout norm; frozen readouts should be flat exactly |
| `aQ_act` | mean over active units of `|a_i|√Q_i` | invariant under the projection (A′B′ = AB), so this is the *only* norm that is the same before and after rescaling and the one to compare across readouts |
| `heads_fro2` | mean over stored heads of `|heads[k]|²` (task-heads readout only) | stored heads are divided by f every projection, so their norm drifts even though `a` is reset per task |
| `Q_q05`, `Q_q50`, `Q_q95`, `aQ_q05`, `aQ_q95` | quantiles of `Q_i` and `|a_i|√Q_i` over active units | the write-up commits to these: a mean can be stationary while a few units run away (the 5 %/95 % lines and `aQ_max` row of F1/F2, §12.3) |

**Before rescaling.** The projection of §3.4 runs after *every* step, not
once per task, so there is no single "end of training, before rescaling"
state: the pre-projection state at the last step differs from the
post-projection state by one step's drift, O(hη ā²) for population flow
and O(η) for SGD. Record both, and record what the projection actually did:

- `*_pre`: the keys above evaluated on the state after the last `step` and
  before the last `project` (the `uc_before` point in `run_task`, so no
  extra work).
- `*_post`: the existing post-projection values (rename the current
  `Q_mean` → `Q_act_post`, `a2_mean` → `a2_all_post`, keep the old names in
  `io.py` for one schema version).
- `logf_sum`: `Σ_steps mean_act log f` accumulated in the `fori_loop`
  carry, i.e. the total rescaling of `|W_i|` the projection applied during
  the task. This is the honest "how far would the norm have moved without
  rescaling" number, and `Q_act_pre/Q_act_post − 1 ≈ −2·logf(last step)`
  is a consistency check on it. Also `logf_max`, the largest single-step
  `|log f|` (gauge violations show up here before they show up in check 1).

For frozen readouts (`do_proj = 0`) `f = 1`, pre = post, and `Q` drifts
freely; those two conditions are the real test of bounded norms and the
reason this section exists.

Schema: all of the above join `SCALARS` in `protocol.py` as `(T, B)`
arrays in `batch_*.npz`; `SCHEMA_VERSION` bumps; `analyze.py` reads either
version. Cost: a dozen scalars per task, nothing.

### 12.2 Per-step traces (`configs/trace.yaml`, `results/<sweep>/traces/`)

Per-step values need `lax.scan` with a per-step output in place of the
`fori_loop`. Do **not** turn this on for the sweep: at N = 100 the
population sweep is `T × n_steps = 2000 × 40` floats per run per quantity,
×1920 Stage-1 runs ≈ 1.2 GB per quantity (5 GB on the earlier 4-λ
grid); SGD is 1000× worse. Instead:

- `run_task(..., trace=False)` keeps the `fori_loop`; `trace=True` switches
  to `scan` and returns `(n_steps // stride, n_quant)` per task, `stride`
  static from the config (1 for population, `n_steps // 200` for SGD so
  every task gives ≈ 200 points). Everything else is identical, so the
  traced run reproduces the untraced one bit-for-bit (add this as test 9).
- Quantities per step: `Q_act`, `Q_all`, `a2_act`, `aQ_act`, `log f`
  (mean over active units) and the loss — both pre- and post-projection
  for `Q_act` (two numbers), post only for the rest.
- `configs/trace.yaml`: N ∈ {10, 50, 100}, κ = 1 and √N, δ₀ ∈ {0, 4}, λ ∈
  {1, 4}, all five readouts, population solver, 2 seeds; plus the SGD
  corner at N = 20, η = 0.05, 2 seeds. Order 100 runs, laptop-sized.
- Storage: one `traces/run_<id>.npz` per run with a `(T, n_steps/stride,
  n_quant)` float32 array (float32 is fine for plotting; the scalars of
  §12.1 stay float64) and the run metadata; `manifest.json` lists them.
  A trace run also writes its ordinary `batch_*.npz`, so §12.1 figures can
  be made from the same sweep.

### 12.3 The figures: slices and summaries by default, per-cell families on request

*Revised 2026-10-30.* One figure per (κ, δ₀, λ, solver, η) cell, in
four families with `_lt` variants, is six files per cell: about 290 files
on the earlier 4-λ Stage-1 grid and still 72 at λ = 4 alone, too many
to read. The default output therefore has two kinds of figure.
**Slices** show whole trajectories (norms vs task, S(m)/E(m) vs age)
along one axis of the grid, with the varied parameter on the column
axis, so a whole slice is one file. **Summaries** reduce every condition
to a few numbers and show the whole grid at once. A summary is where a
problem is found; a slice is where it is read. The per-cell families
F1–F4 stay, drawn only for the cells named with `--cells`.

The questions, in order: (1) are the weight norms bounded, where do they
stop trending, and is there a steady state at every (readout, δ₀, N);
(2) how does the memory lifetime scale with N. Stage 1 fixes λ = 4
(decision 7), so the grid is (readout, κ, δ₀, N). The learning-per-task
question (the λ axis at one δ₀) is a later stage; its figures (a λ
slice, maps over (δ₀, λ)) are specified when that stage is planned.
Until then a sweep with several λ (the pilot, `trace.yaml`) gets one
set of slices and summaries per λ, with λ in the stem.

**Reading order.** The checks table (§12.6) → the maps M1 → lifetime vs
N (M2) and vs E(0) (M3) → the slices S1/S2 of any readout that a map or
a ✗ points to → per-cell F1–F4 with `--cells`, only if the slice does
not settle it.

| figure | one file per | panels | traces | Stage-1 files |
|---|---|---|---|---|
| **S1** norms slice `slices/norms_…` | readout, κ, λ, solver, η | rows = norm quantities, columns = δ₀ | N | 8 |
| **S2** curves slice `slices/curves_…` | readout, κ, λ, solver, η | rows = S/S(0), E; columns = δ₀ | N | 8 |
| **M1** maps `summary/maps_…` | κ, λ, solver, η | rows = 3 statistics, columns = readouts; each panel δ₀ × N | — | 2 |
| **M2** lifetime vs N `summary/mhalf_vs_N_…` | κ, λ, solver, η | columns = δ₀ | readouts | 2 |
| **M3** lifetime vs E(0) `summary/mhalf_vs_E0_…` | κ, solver, η | columns = N | readouts (one point per δ₀, λ) | 2 |
| checks table `checks/checks_…` (§12.6) | solver, η | — | — | 1 |

Stage 1 (3 gated readouts × 2 κ + 2 ungated readouts for the slices;
2 κ for the summaries; one solver) is 23 files. Ungated readouts have no
κ: their slices are one file each with no `k` in the stem (they run on
the κ_min schedule, §5); in M1–M3 they are repeated in every κ figure,
as in F1–F4.

#### S1. Norms slice (`slices/norms_{readout}_k{κ}_lam{λ}_{solver}_eta{η}`)

`plotting.grid(n_rows, n_δ₀, sharex="all", sharey="row")`. Rows = the
F1/F2 rows below (`Q_act_pre`, `a2_act_post`, `aQ_act_post`,
`aQ_max_post`, `logf_sum`, and `heads_fro2_post` for task heads), less
the rows that are constant by construction for this readout (§12.7 G4):
since the figure holds one readout, those rows are dropped, not drawn as
n/a slots, and the caption names them. Columns = δ₀ in sweep order,
titled `δ₀ = …`. Traces = N in `N_SHADES`, each drawn as in F2: the
block mean of §12.7 G10 with its ±1 s.e. band, the raw per-task seed
mean faint behind it, and the 5 % and 95 % lines. **x = t κ_s²/N
(lifetimes).** The half boundary is drawn per N in its shade, as in F2;
in lifetimes these lines coincide (15 lifetimes at `T_factor = 10`)
unless `T_min` binds. The suptitle names the readout with the cell
(`gated frozen, κ = 1, λ = 4, population, η = 1`).

Reading: within a column, flat after the dotted line = bounded and
steady; still bending at the line = burn-in too short (lengthen
`T_factor`); trending through the window = no steady state at that δ₀.
Within a column, N traces on top of each other = the burn-in and the
stationary level are fixed numbers of lifetimes (the scaling claim); a
fan = they are not. Across columns: where along δ₀ a norm starts to
trend.

#### S2. Curves slice (`slices/curves_{readout}_k{κ}_lam{λ}_{solver}_eta{η}`)

`plotting.grid(2, n_δ₀, sharex="all", sharey="row")`. Rows `S(m)/S(0)`
and `E(m)`; columns = δ₀; traces = N in `N_SHADES` with a ±1 s.e.m.
band over seeds. **x = age m in tasks** with age 0 plotted
(`age_axis`, §12.7 G13), the few-sample thinning of §12.7 G14, and a grey
reference at S/S(0) = 1/2 (where the half-life is read). The x-axis was
m κ_s²/N; it is plain m because the rescaled axis was confusing to read.
If the lifetime scales as N/κ_s², the N traces in a column are copies
shifted right in proportion to N; the collapse itself is check 4 in the
table.

#### M1. Maps (`summary/maps_k{κ}_lam{λ}_{solver}_eta{η}`)

Rows = three statistics; columns = readouts (fixed order, column title
in the readout's colour). Each panel is a heatmap with **N on the
x-axis** (the sweep's widths, discrete) and **δ₀ on the y-axis**
(ascending upwards). Every cell's value is printed at 8 pt; a missing
condition (all seeds infeasible) is hatched `0.85` grey. This is the
generalization of `drift_map` (§12.8). One colourbar per row on the
right (`extra_w`); enlarge the panel (`panel=(3.0, 2.4)`) if the
printed values do not fit.

1. **Lifetime in lifetimes.** Text: `m½ κ_s²/N` to 2 significant
   figures. Colour: log₂ of its ratio to the smallest-N value in the same
   (readout, δ₀) row, `RdBu_r` on fixed ±1 (a factor of 2). A row of
   white cells = the lifetime is ∝ N/κ_s², which is the scaling claim;
   red or blue = it grows faster or slower than that. Censored
   conditions are hatched with the text `> m_max`.
2. **Drift.** As `drift_map`: colour β/se(β) on fixed ±5, β printed as
   % change per lifetime, black outline where |β/se| > 3. β is the slope
   of log mean |aᵢ|√Qᵢᵢ (`aQ_act_post`) per lifetime over the
   measurement window, for every readout. That norm is
   projection-invariant (§12.5 item 2), and for the frozen readouts,
   where `a` is fixed, it moves with Qᵢᵢ. This row is the check-2
   boundary: where on (δ₀, N) the steady state is lost.
3. **Learning per task achieved.** Text and colour: E(0) (`viridis`,
   from 0 to the figure's maximum). At fixed λ the readout channel still
   changes E(0) across readouts and δ₀; this row shows how far apart
   the readouts are when they are compared at the same λ.

#### M2. Lifetime vs N (`summary/mhalf_vs_N_k{κ}_lam{λ}_{solver}_eta{η}`)

The plan's deliverable plot. One row; columns = δ₀ (≤ 6, `col_chunks`
beyond). Each panel shows `m½` (tasks) against N on log–log axes with
ticks at the sweep's N; traces = readouts in their colours and styles,
with bars of ±1 s.e.m. over seeds, ▼ at the top for censored
conditions, and a grey reference ∝ N through the panel's geometric
mean. It is the existing diagnostic `fig_mhalf_vs_N` with panels per
δ₀ instead of per λ.

#### M3. Lifetime vs learning achieved (`summary/mhalf_vs_E0_k{κ}_{solver}_eta{η}`)

The existing `fig_mhalf_vs_E0`, moved from `diag/mhalf/` to the
default: columns = N, one point per (readout, δ₀, λ) with s.e.m. bars
in both directions, hollow squares for ungated readouts. This is where
readouts are compared at matched E(0) (plan, Deliverable).

#### Per-cell families F1–F4 (`cells/`, only with `--cells`)

Unchanged in content from the previous revision: one file per (κ, δ₀,
λ, solver, η) cell, written under `cells/norms/` and `cells/curves/`,
only for the cells selected by `--cells` (`--cells all` draws every
cell, the previous default). They are built from two choices: **what
is on the y-axis** (norms vs task, or S(m)/E(m) vs age) and **what the
traces are** (the readouts at fixed N, to compare methods; or N at a
fixed readout, to see the scaling).

| | traces = readouts, columns = N | traces = N, columns = readouts |
|---|---|---|
| norms vs task | **F1** `cells/norms/norms_…` | **F2** `cells/norms/normsN_…` |
| S(m), E(m) vs age | **F3** `cells/curves/curves_…` | **F4** `cells/curves/curvesN_…` |

The `check2/`, `kaczmarz/` and `by_step/` diagnostic families are
produced only with `--all` (or `--diag <family>`) when a check in the
table fails and needs locating. The `mhalf` diagnostic family is
retired: its two figures are now M2 and M3.

**Common to F1 and F2 (norms vs task).** Rows = `Q_act_pre`,
`a2_act_post`, `aQ_act_post` (mean |aᵢ|√Qᵢᵢ), `aQ_max_post`,
`logf_sum` — the §12.1 quantities, S(0) left to F3/F4; `heads_fro2` as
a sixth row only when the task-heads readout is in the figure (§12.5
item 9). The §12.1 quantile scalars (5 % and 95 % of `Qᵢ` and
`|aᵢ|√Qᵢᵢ` over active units) are drawn as two thin lines in the trace
colour and style. **x = task index t over the whole stream, 0 … T−1**;
nothing is shaded: the figure is for seeing with one's own eyes where a
curve stops trending, so it must not presuppose where the burn-in ends.
The only marks are a dotted black line at the half boundary
`T_burn + T_meas/2` ("first half | second half", the split check 2
uses) and, at the top of the top row, a secondary axis in lifetimes
`t κ_s²/N` (κ_s = schedule κ, §5). Each trace is the block mean of
§12.7 G10 (blocks of half a lifetime, mean over block × seeds, ±1 s.e.
band) with the raw per-task seed mean faint behind it; individual seeds
are not drawn. A row that is constant by construction for a trace
(`logf_sum` for the two frozen readouts; `a2_act_post` for ungated
frozen) is left out of that panel and named in the caption (§12.7 G4).
Reading: a curve that flattens well before the dotted line and stays
flat after it = bounded, steady; one still bending at the dotted line =
burn-in too short for that cell (lengthen `T_factor`); monotone through
the stream = no steady state at that (δ₀, λ).

**F1. Norms vs task, traces = readouts**
(`norms/norms_k{κ}_d{δ₀}_lam{λ}_{solver}_eta{η}`). `plotting.grid(5,
len(Ns), sharex="col", sharey="row")`; columns = N; traces = the five
readouts in `READOUT_COLOR`/`READOUT_STYLE`, legend above the panels.
Every trace at one N shares one task axis and one half boundary (the
κ_s schedule of §5; see the note in §12.7).

**F2. Norms vs task, traces = N**
(`norms/normsN_k{κ}_d{δ₀}_lam{λ}_{solver}_eta{η}`). `plotting.grid(5,
len(readouts), sharey="row")`; columns = readouts (column title = the
readout name in its colour, dashed for ungated); traces = N in
`N_SHADES`, light → dark, legend lists N. With x = t the streams have
different lengths (T = 20N/κ_s²) and the half boundary is one dotted
line per N, in that N's shade; the columns share the x-range of the
largest N. A second file, `normsN_lt_…`, is the same figure with **x =
t κ_s²/N (lifetimes)**: the half boundary is then one line for all N
and the N traces lie on top of each other if the burn-in is a fixed
number of lifetimes — which is the scaling claim — and fan out if it is
not. Both files are written; the `_lt` one is the one to look at first.

**F3. S(m), E(m) vs age, traces = readouts**
(`curves/curves_k{κ}_d{δ₀}_lam{λ}_{solver}_eta{η}`): the §5 figure,
unchanged in content — rows S, E; columns N; traces readouts; band ±1
s.e.m. over seeds — with the age axis of §12.7 G13 (symlog, age 0
plotted) and the few-sample thinning of G14. No `m_half` ticks (the
number is in `summary.csv` and the checks table).

**F4. S(m), E(m) vs age, traces = N**
(`curves/curvesN_k{κ}_d{δ₀}_lam{λ}_{solver}_eta{η}`): rows S, E;
columns = readouts; traces = N in `N_SHADES`; x = age m in tasks. (The
second file `curvesN_lt_…`, against m κ_s²/N, is no longer made: every
curves figure uses x = m. The collapse across N is check 4 in the table.)

### 12.4 Figure and data organisation

Every figure is a `.png` (§12.0); the tree lists stems only. A stem names the cell it belongs to in the fixed order
`k{κ}_d{δ₀}_lam{λ}_{solver}_eta{η}`, dropping only the fields the figure
varies over. `{readout}` appears only when the figure is for a single
readout. `_lt` marks a lifetimes x-axis.

```
results/<sweep>/figures/
  slices/    norms_{readout}_k{κ}_lam{λ}_{solver}_eta{η}             (S1; no k for ungated readouts)
             curves_{readout}_k{κ}_lam{λ}_{solver}_eta{η}            (S2)
  summary/   maps_k{κ}_lam{λ}_{solver}_eta{η}                        (M1)
             mhalf_vs_N_k{κ}_lam{λ}_{solver}_eta{η}                  (M2)
             mhalf_vs_E0_k{κ}_{solver}_eta{η}                        (M3)
  checks/    checks_{solver}_eta{η}            the ✓/✗ table (§12.6)
             checks_{solver}_eta{η}.csv        every number behind a mark
  INDEX.md   written by analyze.py: one line per figure (family, cell, file)
  cells/     only with --cells: norms/ (F1, F2 + normsN_lt_…), curves/ (F3, F4)
  diag/      only with --all / --diag: kaczmarz/, by_step/, check2/ (the
             C2-A … C2-E and cv_exponent figures of the previous revision, kept
             in plot_check2.py / plot_norms.py for locating a failed check)
results/<sweep>/traces/   run_<id>.npz  manifest.json
results/validate/figures/ eta_halving  kaczmarz_collapse  cv_exponent   (validate.py; same conventions)
```

`analyze.py results/<name>` writes `summary.csv`, `curves.json`, the
slices, the summaries and the checks table; `plot_norms.py` draws S1
(and F1/F2 for `--cells`) and is called by `analyze.py`;
`plot_check2.py` computes the check statistics behind the table and M1's
drift row and, only under `--all`, the diagnostic figures. Figure data
(the arrays actually plotted) are written next to the figure as `.npz`
with the same stem when `--save-figdata` is given.

### 12.5 Pushback and suggestions (decide before implementing)

Items 3–5, 8 and 10 concern the per-step overlay (`by_step/`), which is
now a `--diag` family; they stand but are not on the default path.

1. **"Before rescaling" is not one number here.** Because the projection
   is per step, the pre-projection norm is one step away from the pinned
   value and will look flat for every projected readout. The quantity that
   answers "how do the weights want to move" is `logf_sum` (total
   rescaling per task), and the readouts where the norm really evolves are
   the two frozen ones. Headline row = `Q_act_pre` for frozen, `logf_sum`
   for projected, or show both and accept that one of them is boring.
   Alternative worth considering: a `project_every: task` option that
   projects once at the end of a task; then "before rescaling" is
   meaningful, but it changes the dynamics (the plan prescribes per-step
   projection), so it belongs in a validation config, not the sweep.
2. **Plot `aQ_act` next to everything.** It is projection-invariant, so it
   is the one norm that is comparable across the five readouts and across
   pre/post; a readout whose `Q` is flat but whose `aQ` drifts is doing
   its learning in the readout, and vice versa.
3. **Relative, not absolute, for the step overlay.** Across tasks the
   starting `Q` differs (especially frozen readouts, where it drifts), so
   overlaying absolute values smears the shape. Plot `Q(i) − Q(0)` or
   `Q(i)/Q(0)` per task so the within-task shape collapses and the median
   is sharp; keep the absolute version as a second row if the level matters.
4. **Colour tasks by index, not a single colour at low α.** A colormap
   from early to late measurement tasks (still at low α) shows whether the
   per-task shape is itself drifting, which a single-colour overlay and the
   median both hide. Costs nothing.
5. **Log x for SGD.** With `n_steps ≈ 3·10⁴` the interesting part is the
   first few hundred steps; use `symlog`/log x for SGD panels and linear
   for population (≤ 40 steps). Stride the stored trace logarithmically for
   SGD rather than uniformly for the same reason.
6. **Gated runs: active vs all.** The mean over all units of a gated run
   is mostly units that did nothing this task. Always plot `Q_act` and
   show `Q_all` as a dotted companion rather than mixing them.
7. **Per-unit distributions, not just means.** Adopted: the quantile
   scalars are in the §12.1 table and the write-up; F1 and F2 draw the
   5 % and 95 % lines.
8. **Figure count.** Family B as specified is 5 readouts × 2 κ × 2 δ₀ ×
   2 λ × 3 N ≈ 120 panels per solver. Keep the per-(readout, combination)
   files for inspection but add one summary figure per solver:
   readouts × N grid of the medians only (at δ₀ = 0, λ = 1), which is what
   will go in a write-up.
9. **Task-heads readout.** `a` is reset every task, so `a2_act` by step is
   the head's trajectory from its fresh draw, while the stored heads are
   being rescaled by 1/f behind it; `heads_fro2` vs task is the quantity
   that can blow up. Include it as a sixth row of F1/F2 for that readout only.
10. **Trace runs are a separate config, not a flag on the sweep.** This
    keeps `batch_*.npz` small and the sweep bit-identical with or without
    tracing (test 9). If per-step data for the full grid is ever wanted,
    store running moments per step (mean, M2 over tasks) inside the scan
    carry instead of the raw trace — `(n_steps, n_quant, 2)` per run,
    independent of T — and the median becomes a mean ± std.

### 12.6 The checks table (`checks/checks_{solver}_eta{η}.png`)

The checks of §6 and the stationarity statistics of the write-up's
`sec:check2` are reported as **one PNG table per (solver, η)**, not as
figures. The table answers "did it pass" at a glance; the numbers behind
every mark are in `checks_{solver}_eta{η}.csv` (one row per mark, with
the statistic, its threshold and the worst cell), and the diagnostic
figures of the previous revision are made on demand with `--diag` to
locate a ✗. The table is drawn by `plotting.check_table(ax, marks,
rows, cols)` and follows §12.0 (text ≥ 8 pt, one suptitle, nothing
else).

**Layout.** Columns = readout × N, grouped by readout (readout name
once per group in its colour; N under it); rows = the checks below.
Each cell holds a mark and, under it at 8 pt, the number it was judged
on: **✓** (black) passed; **✗** (`#CC0000`, on a light red fill) failed;
**~** (`0.4` grey) "within tolerance but worth a look" (one of the ≤ 48
cells flagged, or a slope on the wrong side by less than its s.e.);
**—** on hatched `0.85` grey = not applicable by construction (§12.7
G4). Nothing else is coloured.

**Rows and pass criteria** (thresholds are the plan's; set them once in
`plot_check2.py:CRITERIA` and print them in the caption):

| row | statistic (per readout, N; worst over (κ, δ₀, λ) cells) | ✓ | ~ | ✗ |
|---|---|---|---|---|
| 1 gauge: Δᵢ pinned | max over tasks of `max|Δᵢ − Δ₀|` | < 1e-10 | < 1e-8 | else |
| 1 gauge: u_c invariant | max over tasks of `|u_c^after − u_c^before|` | < 1e-10 | < 1e-8 | else |
| 2 bounded: mean |a|√Q | cells with |z| > 3 (`z` = (half1 − half0)/se) | 0 | 1 | ≥ 2 |
| 2 bounded: max |a|√Q | same | 0 | 1 | ≥ 2 |
| 2 bounded: Q_act (pre) | same | 0 | 1 | ≥ 2 |
| 2 bounded: a²_act | same (— for ungated frozen) | 0 | 1 | ≥ 2 |
| 2 bounded: Σ log f | same (— for the two frozen readouts) | 0 | 1 | ≥ 2 |
| 2 steady memory: S(0) | same | 0 | 1 | ≥ 2 |
| 2 steady memory: m½ | cells with `|Δm½| > 2 s.e.(m½ over seeds)`; censored cells counted separately | 0 | 1 | ≥ 2 or any censored |
| 2 drift: β | cells with `|β/se_β| > 3` for mean |a|√Q (and Q_act pre for the frozen readouts) | 0 | 1 | ≥ 2 |
| 3 self-averaging | slope γ of log CV(s, m = 1) vs log N over the sweep's N (one value per readout, so the same mark in every N column) | γ ≤ −0.4 | −0.4 < γ ≤ −0.2 | else |
| 4 collapse | max over m of the spread across N of `S(m)/S(0)` at matched `m κ_s²/N` (frozen readouts; — for the others) | < 0.05 | < 0.10 | else |
| feasibility | runs skipped for infeasible δ₀ (§3.5), printed as `k/n` | 0 | ≤ 10 % | else |

Row 2's `z` treats consecutive tasks as independent, so it is
over-sensitive to slow drift (§5); that is why one flagged cell is `~`
and not `✗`, and why the table prints the count rather than only the
mark. Check 3's γ needs ≥ 3 widths to mean anything; with two (the
pilot) the cell prints the two-point slope in grey and no mark. The
checks that need figures to interpret (where in (δ₀, N) a drift is;
whether a bend is burn-in or no steady state) are answered by the drift
map M1 and the norms slice S1, not by the table.

**Output.** `plot_check2.py` also prints the table as text to stdout
(the same marks, so a run log carries the verdict) and the three worst
cells per ✗. `check2_cells.csv` (per-cell statistics, columns as before)
is still written as the machine-readable export; `summary_*.npz` is
written only with `--save-figdata`.

**What the write-up cites.** The checks table replaces the table and
the five check-2 figures in `sec:check2`; the prose there must be
brought in step (what each row tests, the thresholds, and that M1 locates a
failure and S1/S2 are where it is read). Done 2026-10-30, together with
the Deliverable paragraph and the λ row of the parameter table.

### 12.7 Readability notes from the pilot figures

Reviewed on 2026-10-29: `results/pilot/figures/checks/` and
`results/pilot/figures/check2/` — population solver, η = 1, N ∈ {10,
20}, κ = 1, δ₀ ∈ {0, 4}, λ ∈ {1, 4}, 2 seeds. Most of those figures are
now diagnostics (§12.3), so only the findings that bear on F1–F4 and
the table are kept here; the changes to the figure standard are in
§12.0.

Not a plotting issue, but visible in these figures: §5's schedule κ
(`Run.kappa_sched`) was not in the code until 2026-10-30 (the pilot's
and the first Stage-1 ungated runs have T = 20). That is why the ungated
S(m) traces stop at age ≈ 20 while the gated ones run to ≈ 300, and why
the ungated norm traces have one point per lifetime. §5 is now in
`make_runs`; rerun
the pilot (§9.1 L2–L3) before judging those traces.

- **G1. Legend outside the panels.** The legend covered data in every
  figure that had one. `fig_legend` (§12.0 Legend) for F1–F4.
- **G2. Display labels, never schema keys.** `aQ_act_post`, `logf_sum`,
  `S0` appeared in titles and axes. `QUANTITY_LABEL` is the only source
  of names (§12.0 Text).
- **G3. Numbers.** `κ = 1.0`, `δ₀=0.0`, colorbar ticks `0.00025`.
  `fmt` and the scale-in-label rule (§12.0 Text).
- **G4. Constant by construction is n/a, not data.** Σ log f is
  identically 0 for the two frozen readouts and a²_act is constant for
  ungated frozen; the pilot drew them as flat lines, zero-height boxes
  and, divided by their own rounding noise, as a "drift". `NA_BY_CONSTRUCTION
  = {(key, readout): reason}` in `plot_check2.py` — (`logf_sum`,
  `frozen`/`gated_frozen`): "not projected, f = 1"; (`a2_act_post`,
  `frozen`): "a fixed, all units active"; (`Q_act_post`, the projected
  readouts): "pinned by the projection". No statistic is computed for
  these; F1/F2 leave the trace out and say so in the caption; the table
  shows —. `gated_frozen` keeps `a2_act` (its active set changes every
  task).
- **G10. Block means on per-task axes.** The task-to-task scatter (S(0)
  moved between ≈ 0 and 0.9 from one task to the next) was far larger
  than the drift the figure is meant to show, and five per-task traces
  tangled into one band. `block_mean(x, y, w)` with `w = max(1,
  round(N/(2κ_s²)))` tasks (half a lifetime; 5 tasks at N = 10, κ = 1):
  mean over the block and the seeds, ±1 s.e. band over the block's
  task × seed values, the raw per-task mean faint behind it. Blocks are
  cut at the half boundary so a block never mixes the two halves.
- **G11. Captions fit.** The check-3 caption ran off both sides.
  `caption` wraps to the figure width; `_check` warns about text outside
  the figure.
- **G12. Ticks at the sweep's N.** A log axis over N = 10, 20 labelled
  only "10¹". Discrete-value axes (§12.0 Text); applies to the γ fit
  behind the table only now.
- **G13. Age 0 on the age axis.** `ages[1:]` on a log axis dropped
  S(0). `age_axis(ax)`: `symlog`, `linthresh = 1`, `linscale = 0.3`,
  age 0 plotted (F3, F4).
- **G14. Few-sample ages do not set the scale.** In the pilot the
  task-heads E(m) climbed to ≈ 8 over the last few ages (half 1, κ = 1,
  δ₀ = 0, λ = 1) and set the E range to 0–8 while every other trace
  stayed below 2.5. Fewer tasks contribute at the oldest ages, but the
  climb may be real (stored heads are rescaled by 1/f every step, §12.5
  item 9), so it must stay visible without flattening the rest: ages
  where a trace's pooled count is below 25 % of its maximum are drawn
  thin (`lw 0.6`, α 0.5) and kept out of the autoscaling, out-of-range
  points get an edge marker, and the count at the last age goes in the
  caption (F3, F4).

Findings that applied only to the now-diagnostic figures (box plots
of four values, per-panel y-ranges, sign colours, flag marks, the
`m_half` tick bug in `Sm_halves`, the one-panel check-3 figure) are
recorded in the git history of this file; if a `--diag` figure is
brought back into regular use, fix those first.

#### 12.7.1 Order of work and acceptance (previous revision)

Implemented; kept for the record. The folder layout of item 6 and the
default figure set are superseded by §12.3–§12.4 and §12.8.


1. `plotting.py`: `fmt`, `QUANTITY_LABEL`, `fig_legend`, caption
   wrapping, discrete-value ticks, `age_axis`, `block_mean`,
   `na_slot`, `check_table`, and the `_check` warnings of §12.0; remove
   `UP`/`DOWN`, `boxes_by_readout`, `flag_bars` and the in-panel
   `readout_legend`.
2. `plot_norms.py`: F1 and F2 (+ `_lt`); drop the burn-in shading and
   the `norms_prepost` file (pre − post is one step of drift and is
   covered by `logf_sum`); `by_step/` behind `--diag`.
3. `analyze.py`: F3 with `age_axis`/`fig_legend`; F4 (+ `_lt`); the
   `mhalf/`, `kaczmarz/`, `cv_exponent` figures behind `--diag`.
4. `plot_check2.py`: `NA_BY_CONSTRUCTION`, `CRITERIA`, the table and
   its CSV and stdout form; C2-A … C2-E behind `--diag`.
5. Test 10 (`tests/test_plotting.py`, §7): render F1–F4 and the table
   from a small synthetic record set (two widths, two cells, gated and
   ungated readouts) and fail on any `plotting.save` warning.
6. `python scripts/analyze.py results/pilot` (§9.1 L3; after §5's
   `kappa_sched` if it has landed), then check by eye: `figures/` holds
   only `norms/`, `curves/`, `checks/`, `INDEX.md`; every legend above
   the panels; no shaded burn-in; the half boundary one dotted line per
   trace set; the `_lt` files of F2/F4 overlay their N traces on one
   axis; every table cell legible at 100 % zoom.
7. Keep the write-up's `sec:check2` prose in step (§12.6).

### 12.8 Order of work for the slice/summary revision (2026-10-30)

Implemented 2026-10-30 (items 1–5; test 10 passes, the pilot gives the 26
files of item 5). Item 6 is the Xeon run. The order was:

1. `plotting.py`: generalize `drift_map` to `heat_map(ax, colour,
   text, row_vals, col_vals, cmap, vmin, vmax, text_fmt, outline=None)`
   (NaN colour → hatched cell; `outline` a boolean mask for the black
   box). Keep `drift_map` as a thin wrapper so the `--diag` figures do
   not change.
2. `plot_norms.py`: `fig_S1(records, …)`, built from `fig_F2`'s
   per-readout column code with columns = δ₀ and x in lifetimes. It
   drops the rows in `NA_BY_CONSTRUCTION` for its readout and names them
   in the caption. `run()` draws S1 by default, and F1/F2 only for the
   records `analyze.py` passes from `--cells`, into `cells/norms/`.
3. `analyze.py`:
   - `fig_S2`, from `fig_curvesN` with columns = δ₀, S/S(0) in the top
     row and one readout per file;
   - `fig_maps` (M1), from `collect` (m½, censored, E(0)) and the drift
     β, se(β) of `aQ_act_post` from `plot_check2._grid_stats` (compute
     the statistics once and pass them to both the table and M1);
   - M2 = `fig_mhalf_vs_N` grouped by (κ, λ, solver, η) with panels =
     δ₀; M3 = `fig_mhalf_vs_E0`; both into `summary/`;
   - F3/F4 only for `--cells` (new value `all`), into `cells/curves/`;
   - retire the `mhalf` diag family; update `DIAG_FAMILIES` and the
     `--cells` and `--diag` help strings.
4. Test 10 (`tests/test_plotting.py`): render S1, S2, M1–M3 and the
   table from synthetic records with ≥ 2 δ₀, 2 N, one gated and one
   ungated readout, and F1–F4 through `--cells all`; fail on any
   `plotting.save` warning. Update any assertion that expects `norms/`
   or `curves/` at the top of `figures/`.
5. `python scripts/analyze.py results/pilot` (§9.1 L3). The pilot has
   λ ∈ {1, 4}, so expect one set per λ: 10 slice files and 2 maps/M2
   files per λ, one M3 and the table (26 files). Check by eye: `figures/`
   holds only `slices/`, `summary/`, `checks/` and `INDEX.md`; every M1
   value is legible at 100 % zoom; in S1 and S2 of the frozen readouts
   the N traces lie on top of each other in lifetimes.
6. Stage 1 on the Xeon at λ = 4 (§9.2 X2–X6): 1920 runs, 23 default
   figures.
