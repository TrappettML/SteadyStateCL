# Coding plan: μP continual-learning experiment (JAX, float64, CPU)

Implements `writing/derivations/muP_experiment/muP_experiment.tex` (the
"plan"; equation labels below refer to it). Pinned stack (requirements.txt):
Python 3.12, `jax==0.12.0.1` / `jaxlib==0.12.0.1` (CPU), `numpy==2.5.4`,
`scipy==1.18.1`, `matplotlib==3.11.2`.

Two machines:

| | develop / test | full run |
|---|---|---|
| hardware | Apple silicon laptop | 4-socket Xeon Gold 6348H, 96 cores / 192 threads, x86_64 |
| JAX backend | CPU | CPU |
| precision | float64 | float64 |
| role | correctness, pilot at N ≤ 20 | Stage-1 sweep |

Same code path on both; nothing is backend-specific. The venv on the laptop
is `code/sscl_env` (outside the repo); recreate it on the Xeon with
`python3.12 -m venv sscl_env && pip install -r requirements.txt`.

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
   three readouts can share a batch.
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
   λ/(η ā²)` from the measured ā² at the start of task 1, fixed per run. Runs
   are grouped into batches by their resulting step count (§3.6).

## 1. Repository layout (`muP_experiment/` in this repo)

```
muP_experiment/
  PLAN.md
  mup/
    __init__.py        enables x64; exports version
    config.py          Config dataclass, YAML loader, grid → list of runs → batches
    model.py           pure functions on a State pytree: init, sgd_step, pop_step,
                       project, forward, observables; all single-run, vmapped by caller
    protocol.py        run_task (jit), run_stream (Python loop over tasks), probes
    observe.py         s_k, ε_k, running moments, half-life, checks 1–4
    io.py              result schema (.npz per batch + JSON summary), resume
    driver.py          multiprocessing launcher with env/affinity for the Xeon
  scripts/
    run_sweep.py       --config configs/stage1.yaml --N 50 [--workers 90]
    bench.py           steps/s vs B at fixed N on this machine
    validate.py        η-halving, Kaczmarz collapse, CV exponent
    analyze.py         S(m), E(m), m_half vs N at matched E(0), figures
  configs/
    pilot.yaml  stage1.yaml
  tests/
    test_x64.py test_gradients.py test_projection.py test_conservation.py
    test_vmap_vs_loop.py test_reproducible.py
  results/             (gitignored) one subfolder per sweep
```

## 2. State (one run; `vmap` adds the batch axis)

```python
class State(NamedTuple):      # all float64 unless noted
    W:      (N, D)            # hidden rows
    a:      (N,)              # shared readout, or the current head
    g:      (N,)              # current gate, 0/1
    w:      (D,)              # current teacher
    heads:  (P, N)            # stored heads (zeros unless readout == heads)
    probes_w: (P, D);  probes_g: (P, N)
    key:    PRNGKey
    t:      int32             # task index
    # per-run constants, passed as a separate Params pytree:
    # Delta0, eta, h, pN (float), lam, n_steps, learn_a (0/1), do_proj (0/1), use_heads (0/1)
```
Derived per step: `key_vec = g*a`, `u = key_vec @ W / pN`, `r = u − w`,
`L = r·r/(2D)`. `pN = κ√N` is the nominal value from eq. (student), not the
realized gate count.

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
`config.py` refuses any δ₀ ≤ −min_i Q0_i (δ₀ > −1 in practice; the plan's
table uses δ₀ ≥ −0.75). Heads: a fresh head is drawn this way from the
current `Q` at the start of each task and stored into `heads[t]`.

### 3.6 Task duration (`config.py`, host side, before batching)
```
abar2       = mean_{g_i=1} a_i²  at the start of task 1   (≈ 1 + δ₀)
alpha_task  = lam / (eta * abar2)
SGD:        n_steps = round(alpha_task * D);     require eta*abar2 ≤ 0.05
population: h = min(h_max, 0.1/(eta*abar2));  n_steps = ceil(alpha_task / h); h = alpha_task/n_steps
```
`abar2` is computed from the deterministic part (1 + δ₀) so that `n_steps`
is known before any device work; the realized `abar2` of task 1 is recorded
and the realized λ reported. Runs are bucketed by `(N, solver, n_steps)`;
each bucket becomes one or more `vmap` batches.

### 3.7 `run_task` (jit, single run)
```
draw (w_t, g_t[, head_t]) from fold_in(key, t)
lax.fori_loop(0, n_steps, step_and_project, state)        # step = sgd or pop
store w_t, g_t[, head_t] at probes[t]
evaluate probes at the age schedule (§5) → (n_ages,) s and ε, plus validity mask
per-task scalars: abar2, mean Q, mean a², mean/max |a|√Q, L_end, gauge residuals
```
`run_stream` (Python) loops `t = 0 … T−1` calling the vmapped `run_task`,
accumulating the `(B, n_ages)` moments and per-task scalars on host with
`np.asarray` once per task (one device→host copy per task, not per step).

## 4. Readout conditions as masks

| readout | learn_a | do_proj | use_heads | notes |
|---|---|---|---|---|
| learned shared | 1 | 1 | 0 | |
| frozen shared | 0 | 0 | 0 | a = a⁽⁰⁾ forever; Q_ii unpinned |
| heads | 1 | 1 | 1 | `a` ← fresh head each task; stored heads rescaled by 1/f |

Probe evaluation uses `heads[k]` when `use_heads` else the current `a`.

## 5. Probes and observables (`observe.py`)

- Every task is a probe (N ≤ 100): `P = T_burn + T_meas`, preallocated.
- Age schedule (static): `ages = [0..20] ∪ geomspace(24, m_max, 24)`
  rounded to unique ints, `m_max = 4·10N/κ²` capped at `T−1`. For task `t`
  the probe indices are `k = t − ages` with validity `k ≥ 0`; invalid
  entries read probe 0 and are masked out of the sums.
- `u_k = (g_k*a_k) @ W / pN`, `s_k = u_k·w_k/|w_k|²`,
  `ε_k = |u_k−w_k|²/|w_k|²`, computed for all ages in one `(n_ages, N) @
  (N, D)` matmul.
- Accumulate `(Σs, Σs², Σε, Σε², count)` per `(run, age, half)` where
  `half ∈ {0,1}` splits the measurement window; burn-in tasks are not
  accumulated. Nothing per-sample is stored.
- `m_half`: first age with `S(m) ≤ S(0)/2`, linear interpolation between
  bracketing ages; `NaN` with `censored=True` if not reached.
- Per task: `abar2`, mean `Q_ii`, mean `a_i²`, mean and max of `|a_i|√Q_ii`,
  `E(0)`, `L_end`.

## 6. Checks from the plan

| check | where |
|---|---|
| 1 gauge | `max|Δ_i − Δ₀|` over active units and `|u_c^{after} − u_c^{before}|` after the last projection of each task, returned by `run_task`; expect ~1e-13 in float64 |
| 2 bounded norms | half-window means of the per-task scalars and of `S(0)`, `S(m)`; flag if halves differ by > 3 s.e. |
| 3 self-averaging | CV of `s_k` at fixed age from the running moments; fit the exponent vs N (the plan's N^{-1/2} is the frozen-readout prediction) |
| 4 Kaczmarz | `S(m)/S(0)` vs `κ²m/N` collapse across N for the frozen readout, in `analyze.py` |

## 7. Tests (`pytest`, run on the laptop before every push)

1. `test_x64`: default dtype is float64 after `import mup`.
2. `test_gradients`: (sgdW), (sgda) equal `pN·η·∇_W ℓ` and `(pN·η/D)·∇_a ℓ`
   from `jax.grad` of the literal loss, to 1e-12.
3. `test_projection`: `u_c` unchanged to 1e-13; `Δ_i = Δ₀` to 1e-13; stored
   heads keep every stored `u_k` unchanged.
4. `test_conservation`: population flow without projection drifts `Δ_i` by
   O(h²) per step (check by halving h); SGD drifts it by O(η²).
5. `test_vmap_vs_loop`: a batch of 3 runs (one per readout) equals three
   single runs with the same keys, bit-for-bit.
6. `test_reproducible`: a 5-task run at N = 10 hashes to a stored value
   (regenerate the hash deliberately when the kernels change; compare the
   hash once on the Xeon).
7. `scripts/validate.py`: η-halving at fixed λ — the gap between SGD and the
   population curve should halve, consistent with the η² term in the plan;
   frozen-readout collapse; CV exponent.

## 8. Throughput and the Xeon plan

Per SGD step per run: ~3 passes over `W` (N·D doubles = 320 kB at N = 100,
D = 400), i.e. ~1 MB of memory traffic → ~0.1–0.3 ms/step/run on one core
once jitted with B = 1; `vmap` over B mostly amortizes the loop overhead
rather than the bandwidth. Costs at N = 100, κ = 1, T = 20N/κ² = 2000 tasks:

- population, λ = 4, δ₀ = 16: `n_steps ≈ λā²/0.1 = 680` per task → 1.4·10⁶
  steps/run → minutes per run.
- SGD, same corner: η = 0.05/17 ≈ 0.003, `n_steps = λ/(ηā²)·D ≈ 3·10⁴`
  per task → 6·10⁷ steps/run → hours per run per core.
  Run SGD only in the low-(λ, δ₀) corner and at two η (validation 7).

Stage 1 (Table 1 of the plan): 5 N × 2 κ × 3 readouts × 6 δ₀ × 4 λ × 8 seeds
= 5760 population runs. With ~90 workers that is well under a day. Steps:

1. `scripts/bench.py` on the Xeon: steps/s vs B for N ∈ {20, 100}, with the
   thread-pinning env above, to pick B and worker count.
2. `driver.py` launches `run_sweep.py --batch-id i` subprocesses from a
   manifest (`results/<sweep>/manifest.json` listing batches and their
   status) so the sweep is resumable; each batch writes its own `.npz`.
3. `analyze.py` merges batch files; never reads raw state.

## 9. Milestones

1. `config.py`, `model.py`, tests 1–4; single-run `run_task` and
   `run_stream`; pilot at N ∈ {10, 20} on the laptop. Look at the curves.
2. `vmap` batching, readout masks, heads; tests 5–6.
3. Observables, checks 1–3, `.npz` schema, `analyze.py` for a pilot.
4. `driver.py` + `bench.py`; dry run on the Xeon with the pilot config.
5. Stage 1 population sweep; SGD validation corner; `validate.py` figures.

## 10. JAX notes

- Enable x64 before anything else; `jnp.array(1.0)` made earlier stays
  float32.
- `jit` retraces on new static shapes and on Python scalars passed as
  static args: pass `n_steps`, N, D as `static_argnums`, everything else as
  arrays.
- `lax.fori_loop` with a static trip count unrolls nothing and compiles
  fast; use `unroll=8` or so only if `bench.py` shows a gain.
- Keep `jax.random.fold_in` for task keys so tasks are independent of how
  runs are batched.
- Host transfers (`np.asarray`) once per task; never inside the step loop.
- CPU determinism: XLA CPU is deterministic for this op set; the only
  cross-machine difference to watch is fused-multiply-add usage by the
  compiler, which `test_reproducible` on both machines will reveal. If the
  hashes differ, that is not an error, just a reason to compare curves
  instead of hashes.
