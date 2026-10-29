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
   1), fixed per run. Runs are grouped into batches by
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
    analyze.py         S(m), E(m) (columns = N, traces = readout), m_half vs N
                       at matched E(0), checks 1–4, figures (§12.4 for the layout)
    plot_norms.py      weight-norm figures of §12 from batch files and trace files
  configs/
    pilot.yaml  stage1.yaml  sgd_corner.yaml  trace.yaml  (trace: per-step norms, §12.2)
  tests/
    test_x64.py test_gradients.py test_projection.py test_conservation.py
    test_vmap_vs_loop.py test_reproducible.py test_config.py conftest.py
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
infeasible for many seeds (624 of the 5760 gated Stage-1 runs). Such runs
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
(N, δ₀, λ, seed). An ungated task-heads condition is the same flag flip
and can be added later if wanted. Probe evaluation uses `heads[k]` when
`use_heads` else the current `a`.

## 5. Probes and observables (`observe.py`)

- Every task is a probe (N ≤ 100): `P = T_burn + T_meas`, preallocated,
  with `T_burn = T_meas = max(T_min, round(T_factor·N/κ²))`, `T_factor =
  10`, `T_min = 4`.
- Age schedule (static): `ages = [0..20] ∪ geomspace(24, m_max, 24)`
  rounded to unique ints, `m_max = 4·10N/κ²` capped at `T−1`. For task `t`
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
- Figures for `S(m)` and `E(m)` (`analyze.py`): one figure per
  (κ, δ₀, λ, solver, η); two rows (`S`, `E`); **columns indexed by N**
  (each column is one width and hence one stream length T = 20N/κ²);
  **traces within a panel are the readout types** in the fixed colours of
  `plotting.py`, so the five readouts are compared directly at each N. The
  ungated conditions have no κ and are repeated in every κ figure.
- Per task: `abar2`, mean `Q_ii`, mean `a_i²`, mean and max of `|a_i|√Q_ii`,
  `E(0)`, `L_end` — all taken *after* the last projection of the task. §12
  adds the same norms *before* that projection and the per-step traces.

## 6. Checks from the plan

| check | where |
|---|---|
| 1 gauge | `max|Δ_i − Δ₀|` over active units and `|u_c^{after} − u_c^{before}|` after the last projection of each task, returned by `run_task`; expect ~1e-13 in float64 |
| 2 bounded norms | half-window means of the per-task scalars and of `S(0)`, `S(m)`; flag if halves differ by > 3 s.e. |
| 3 self-averaging | CV of `s_k` at fixed age from the running moments; fit the exponent vs N (the plan's N^{-1/2} is the frozen-readout prediction) |
| 4 Kaczmarz | `S(m)/S(0)` vs `κ²m/N` collapse across N for gated frozen, in `analyze.py` (for ungated frozen the variable is `m/N`) |

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
   schedule.
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

Stage 1 (Table 1 of the plan): gated conditions 5 N × 2 κ × 3 readouts
(gated shared, gated frozen, task heads) × 6 δ₀ × 4 λ × 8 seeds = 5760, plus
ungated conditions 5 N × 2 readouts (learned shared, frozen) × 6 δ₀ × 4 λ ×
8 seeds = 1920, i.e. 7680 population runs. Ungated runs have `pN = N`
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
| L3 | `python scripts/analyze.py results/pilot` | `summary.csv`, `curves.json`, `figures/` for the pilot | look at S(m), E(m) (columns = N, traces = readouts) and the check-1/2 output before spending Xeon time; a wrong sign, a non-stationary norm or a censored `m_half` shows here |
| L4 | `python scripts/validate.py [--N-max 40 --seeds 4]` | validation 7: η-halving at fixed λ, Kaczmarz collapse for gated frozen, CV exponent → `results/validate/` | these are the physics checks that the equations, not just the code, are right; minutes on the laptop, so they run here, not on the Xeon |
| L5 | `python scripts/bench.py --N 20 100 --B 1 32 128` | run-steps/s vs batch size | optional on the laptop; gives a baseline to compare against the Xeon number in X1 |

L1–L3 are the loop during development (milestones 1–3). L4 is run once per
kernel change and before any Xeon sweep. Push only when L1 passes.

### 9.2 Xeon (full run) — once per sweep

| step | command | what it does | why this order |
|---|---|---|---|
| X0 | `git pull`; (from `code/`) `python3.12 -m venv sscl_env && . sscl_env/bin/activate && pip install -r SteadyStateCL/requirements.txt`; then from `muP_experiment/`: `pytest tests` | same code, same pinned stack, same tests | test 6's hash is compared here once; a mismatch is a reason to compare curves, not an error (§11) |
| X1 | `python scripts/bench.py --N 20 100 --B 1 32 128` with the decision-5 env (`XLA_FLAGS=... OMP_NUM_THREADS=1 ...`) | run-steps/s per pinned core vs B | chooses `batch_size` in the YAML and `--workers`; done before the sweep so the worker count is set from a measurement, not a guess. Matters mostly for X4 (SGD), since the population sweep is cheap (§8) |
| X2 | `python scripts/run_sweep.py --config configs/stage1.yaml --dry-run` | writes `results/stage1/manifest.json`, prints the batches and the skipped (infeasible δ₀) runs | confirms the grid (7680 runs), the batch count and the per-batch step counts before launching 90 processes |
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

## 12. Weight-norm diagnostics (plots of how the norms evolve)

Purpose: see with our own eyes whether the norms are stationary across tasks
(check 2) and what a single task does to them step by step, for every
readout. Two figure families: **norms vs task** (one point per task, taken
at the end of training on that task, before the projection rescales it) and
**norms vs step within a task** (every task overlaid, plus a median).

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
×7680 runs ≈ 5 GB per quantity; SGD is 1000× worse. Instead:

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

### 12.3 The figures (`scripts/plot_norms.py`)

Conventions as in `plotting.py`: readout colours fixed, ungated dashed,
columns indexed by N; burn-in window shaded grey in every task-axis panel;
one figure = one `(solver, η)` and one of the cells below; file names
carry the cell exactly as `analyze.py` does.

**A. Norms vs task** (`figures/norms/by_task/`). One figure per
(κ, δ₀, λ, solver, η); rows = quantity (`Q_act_pre`, `a2_act_post`,
`aQ_act`, `logf_sum`), columns = N, traces = readouts, x = task index,
seeds as thin lines in the readout colour with the seed-mean on top. Plot
`Q_act_pre` (not post) as the headline row: post is pinned by construction
for the projected readouts and shows nothing. Add a second file,
`..._prepost.png`, with `Q_act_pre − Q_act_post` and `logf_sum` so the size
of the rescaling is visible on its own axis. Also, since §9.1 L3 is where
problems are caught, print the check-2 half-window comparison for every
key of §12.1 next to the figure.

**B. Norms vs step within a task** (`figures/norms/by_step/`). One figure
per (readout, solver, η, κ, δ₀, λ) — the "method/combination" — with
columns = N and rows = quantity (`Q_act`, `a2_act`, `aQ_act`, `log f`).
In each panel: every task of the measurement window as a line at α ≈ 0.05
in the readout colour, the across-task **median** as a heavy black line,
the 10–90 % band as a light fill, x = step within the task. Burn-in tasks
are left out of the overlay (or drawn in grey if wanted) so the median is a
steady-state statistic. Two seeds → two panels side by side or two
columns; do not pool seeds into one median without saying so.

### 12.4 Figure and data organisation

```
results/<sweep>/figures/
  curves/      curves_k{κ}_d{δ₀}_lam{λ}_{solver}_eta{η}.png        (S, E vs age; §5)
  mhalf/       mhalf_vs_N_*.png  mhalf_vs_E0_*.png
  kaczmarz/    kaczmarz_{readout}_*.png
  checks/      cv_exponent.png  (and validate.py output when run on a sweep)
  norms/
    by_task/   norms_k{κ}_d{δ₀}_lam{λ}_{solver}_eta{η}.png  and  *_prepost.png
    by_step/   steps_{readout}_N{N}_k{κ}_d{δ₀}_lam{λ}_{solver}_eta{η}.png
  INDEX.md     written by analyze.py: one line per figure with its cell, so the folder is searchable
results/<sweep>/traces/   run_<id>.npz  manifest.json
```

`analyze.py --fig-dir` keeps working but now creates the subfolders;
`plot_norms.py results/<sweep>` fills `norms/` and is called from
`analyze.py` unless `--no-norms`. `results/validate/` keeps its three
figures (they are not a sweep). Figure data (the arrays actually plotted)
are written next to the figure as `.npz` with the same stem when
`--save-figdata` is given, so a figure can be restyled without rerunning
the merge.

### 12.5 Pushback and suggestions (decide before implementing)

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
7. **Per-unit distributions, not just means.** Check 2 is about bounded
   norms; a mean can be flat while a few units run away. Store per task
   the 5/50/95 % quantiles of `Q_i` over active units (3 more scalars) and
   draw the 5–95 band in family A. Cheap, and it is the only way a
   runaway unit is seen.
8. **Figure count.** Family B as specified is 5 readouts × 2 κ × 2 δ₀ ×
   2 λ × 3 N ≈ 120 panels per solver. Keep the per-(readout, combination)
   files for inspection but add one summary figure per solver:
   readouts × N grid of the medians only (at δ₀ = 0, λ = 1), which is what
   will go in a write-up.
9. **Task-heads readout.** `a` is reset every task, so `a2_act` by step is
   the head's trajectory from its fresh draw, while the stored heads are
   being rescaled by 1/f behind it; `heads_fro2` vs task is the quantity
   that can blow up. Include it in family A for that readout only.
10. **Trace runs are a separate config, not a flag on the sweep.** This
    keeps `batch_*.npz` small and the sweep bit-identical with or without
    tracing (test 9). If per-step data for the full grid is ever wanted,
    store running moments per step (mean, M2 over tasks) inside the scan
    carry instead of the raw trace — `(n_steps, n_quant, 2)` per run,
    independent of T — and the median becomes a mean ± std.
