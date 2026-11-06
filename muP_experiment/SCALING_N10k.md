# Scaling the weight-norm experiment to N = 10,000 — audit and plan (2026-11-05)

Scope: `scripts/run_weight_norms.py` and the NumPy kernels it ran on
(`mup/npmodel.py`); `run_Nscaling.py` shared the same kernels.  *(2026-11-05:
both scripts now run the JAX kernels of `mup/experiments.py` through the
multi-GPU pool `mup/pool.py`; the NumPy audit below describes the CPU path
they replaced, and `npmodel.py` remains as a test reference.)*  No code changes are made here except the start-to-end timing
line added to every `scripts/run_*.py`.  All measurements below are on the
laptop (Apple M2, one BLAS thread per process), population solver, Stage-1
settings (λ = 4 → `n_steps = 40`, D = 4N, T = T_burn + T_meas = 14N), κ = 1.

## 1. Where the time goes today

Per-task cost of one run (`npmodel.tasks` + the bookkeeping in `norms_run`),
and the full-run time it implies (T = 14N tasks):

| N | D | T | gated_shared | task_heads | frozen (ungated) |
|---|---|---|---|---|---|
| 100 | 400 | 1,400 | 1.0 ms → 1.4 s | 1.0 ms → 1.4 s | 0.19 ms → 0.3 s |
| 200 | 800 | 2,800 | 1.3 ms → 3.5 s | 1.2 ms → 3.4 s | 0.31 ms → 0.9 s |
| 400 | 1,600 | 5,600 | 3.1 ms → 17 s | 2.9 ms → 16 s | 1.3 ms → 7 s |
| 800 | 3,200 | 11,200 | 8.0 ms → 89 s | 7.8 ms → 87 s | 7.5 ms → 84 s |

The per-task cost grows like N² once W (N·D doubles = 32·N² bytes) leaves
the cache, and the task count grows like N, so a run costs ~N³ in the
large-N limit.  Breakdown of one task at N = 1600 (D = 6400, W = 82 MB):

| piece | gated_shared (n_act = 63) | learned_shared (ungated, n_act = 1600) |
|---|---|---|
| `Q_diag(W)` over all of W, called **twice** per task (`draw_readout` in `tasks()`, and `norms_run`) | 2 × 10.5 ms | 2 × 9.7 ms |
| `(g*a) @ W` over all N rows in `norms_run` (only n_act rows are nonzero) | 3.5 ms | 3.4 ms |
| `step_task` (the actual training: 40 steps on the active block) | 6 ms | 347 ms |
| **total** | **≈ 27 ms** | **≈ 380 ms** |

So for every gated readout about 80 % of a task is spent in full sweeps
over W that are not training at all; each sweep is O(N²) per task, i.e.
O(N³) per run.  `Q_diag` is also written as `(W*W).sum(-1)`, which
materialises an N×D temporary (another 82 MB here, 3.2 GB at N = 10k).

For the ungated learned readout the 40-step loop dominates: each step
reads the whole of W twice (two matrix–vector products) and writes it once
(`dger`), ~120 passes over W per task.

### Extrapolation to N = 10,000 (D = 40,000, T = 140,000 tasks, W = 3.2 GB)

One pass over W is ~0.3 s at laptop memory bandwidth; `Q_diag` with its
temporary is ~0.7 s and briefly doubles the memory footprint.

| readout | per task today | per run today |
|---|---|---|
| gated (shared / frozen / heads) | ≈ 1–2 s | ≈ 1.5–3 days |
| frozen, ungated (already uses the δv shortcut: 2 passes/task) | ≈ 0.7 s | ≈ 1 day |
| learned_shared, ungated (≈120 passes/task) | ≈ 13 s | ≈ 3 weeks |

And memory: 3.2 GB of W per run, plus the 3.2 GB `Q_diag` temporary, so
`--workers 8` would need ~50 GB; the 16 GB laptop can hold two or three
such runs at once.  The N = 10k cell alone is 3 κ × 3 gated readouts × 4
seeds = 36 gated runs + 8 ungated runs.

## 2. The plan, in order of payoff per effort

> **Status (2026-11-05, later the same day): A and the `einsum` form of `Q_diag`
> (F, first bullet of A.4) are implemented** in `mup/npmodel.py`,
> `scripts/run_weight_norms.py` and `scripts/run_Nscaling.py`; `step_task`
> now takes the tracked `Q` and returns `(fcum, u)`, `tasks()` yields
> `(t, W, a, g, w, fcum, Q, u)`, and `tests/test_npmodel.py` checks the
> tracked Q and u against the full matrix over a task stream for every
> readout.  Measured per task on the laptop, before → after:
>
> | N | gated_shared | task_heads | frozen (ungated) |
> |---|---|---|---|
> | 400 | 3.1 → 1.9 ms | 2.9 → 1.9 ms | 1.3 → 0.6 ms |
> | 800 | 8.0 → 2.4 ms | 7.8 → 2.4 ms | 7.5 → 2.4 ms |
> | 1600 | 27 → 4.9 ms | 5.2 ms | 12.4 ms |
>
> A gated run at N = 800 now takes 27 s instead of 89 s.  The remaining
> gated cost is the 40-step loop over the active block (item B); the
> ungated frozen path still does three passes per task (item C).  Peak
> memory per run is now W plus the active block.  Items B–G are unchanged.

### A. Stop touching all of W once per task  (small change, ~5× for gated runs)

All three O(N²)-per-task sweeps are avoidable; nothing in a task reads
inactive rows except these.

1. **Keep Q as run state.**  `step_task` already advances the row norms Q
   of the active block analytically; have `tasks()` own a persistent `Q`
   (N,) array and write the block's final Q back with `W[act]`, instead of
   recomputing `Q_diag(W)` over all of W at the start of each task.
   Inactive rows are untouched, so their Q cannot drift.  Rounding does
   accumulate on an active row across the tasks it takes part in; refresh
   the block's Q from the gathered block (`np.einsum('ij,ij->i', Wa, Wa)`,
   which reads the block the gather already copied, ~free) at the start of
   every task, which is what the code does today for the whole matrix.
2. **`draw_readout` only when it is used.**  `tasks()` draws a head every
   task to keep the RNG stream fixed; it needs Q, which is now the tracked
   array — no sweep.  (Keep the draw so seeds stay reproducible.)
3. **Norms from the tracked Q and from the active rows.**  In `norms_run`:
   `Q_all = Q.mean()`, `Q_act = Q[act].mean()`, and
   `u = a[act] @ W[act] / pN` (n_act × D, not N × D).  Better: have
   `step_task` return `u` — it computes `(aa*s) @ Wa` at the last step
   anyway.
4. **`Q_diag` without the temporary** wherever it survives:
   `np.einsum('ij,ij->i', W, W)` (≈3× faster, no N×D allocation).  The
   same applies to `init_run`.

After A, a gated task costs only `step_task`: ~0.35 s at N = 10k (40
steps × 3 passes over the 32 MB active block), ≈ 14 h per run.  The frozen
ungated run is unchanged (it was already the two-pass δv path).

### B. Do each task's 40 steps in the n_act-dimensional subspace  (medium change, ~10× more for gated runs)

Within a task the active block changes only by rank-1 updates along
D-vectors r_j (population solver) and the projection only rescales rows.
Everything the step loop needs is therefore expressible through the
n_act × n_act Gram matrix of the block at the start of the task:

- once per task: `G = Wa₀ Wa₀ᵀ` (`dsyrk`, n_act² D flops) and `b = Wa₀ w`
  — **one read** of the block;
- per step, with coef = aa ⊙ s (n_act): `Wa r = G coef / pN − b` and
  `|r|² = coefᵀ G coef / pN² − 2 coef·b / pN + |w|²`, which is all the
  step, the a-update and the Q-update use (the D-vector r is never
  formed); 40 steps of O(n_act²) work;
- once per task: the accumulated update is `Wa_T = M Wa₀ + v wᵀ` with M an
  n_act × n_act matrix built from the per-step coefficients — **one
  `dgemm` write** of the block, then scatter back.

At N = 10k with n_act ≈ 100: `dsyrk`/`dgemm` are 4·10⁸ flops each
(~10–20 ms single-core) and the block is read and written once (~5 ms), so
≈ 30–40 ms per task against 350 ms after A alone: **≈ 1.5 h per gated run
at N = 10k**.  The dynamics are identical in exact arithmetic; the test is
`tests/test_npmodel.py` against the JAX kernels at N = 10–50 (expect
agreement to ~1e-12, not bitwise).

SGD solver, same idea: draw all 40 inputs at once as X (D × 40), compute
`Wa₀ X` with one `dgemm` (one read), and use the Gram matrix `XᵀX` for the
cross terms between steps; the writes are again deferred to one `dgemm`.

Together with A this is the item that makes gated runs at N = 10k
routine; it also speeds up every current N (the block passes are the whole
cost of `run_Nscaling.py` and the gated part of `run_weight_norms.py`).

### C. Ungated frozen: never touch W at all  (small change, that readout becomes free)

With `learn_a = 0`, `do_proj = 0` and every unit active, W(t) = W₀ + a vᵀ
for one accumulated D-vector v over the whole run, so `u = u₀ + |a|² v/pN`
with `u₀ = a @ W₀ / pN` computed once; mean Q, mean a² and |u|² are closed
forms in `u₀·v`, `|v|²` and the initial Q.  O(D) per step, no pass over W
after the first.  `mhalf_run` (run_Nscaling) needs only `u` as well.  The
current code does the rank-1 write and the `a @ W` read once per task
(2 passes = 0.7 s at N = 10k, i.e. a day per run); this removes them.

### D. Ungated learned_shared: the hard case  (keep, but plan for it)

With the projection on, the row scaling s changes every step and so the
coefficient vector aa ⊙ s moves in all N components: the step needs
`W r` and `(aa s) @ W` freshly each step, and the Gram trick costs N² D per
task (4·10¹² flops) — no help.  I believe at least one full read of W per
step is unavoidable here.  What can be done:

- defer the writes: accumulate the 40 rank-1 updates and apply them with
  one `dgemm` at the end of the task (41 passes per task instead of ~120,
  ≈ 3×); fuse the two reads into one pass with a hand-rolled kernel only
  if a second 2× matters;
- give that readout its own BLAS thread budget: it is memory-bound, so
  4–8 threads per process on the Xeon buy nearly linear speed-up; the
  pool should launch these jobs with `OPENBLAS_NUM_THREADS=k` and count
  them as k workers;
- expect ≈ 4–5 s per task single-threaded after deferral → ~7 days per run,
  or ~1–2 days with 4–8 threads.  Either run the learned ungated readout at
  N = 10k with 1–2 seeds as a dedicated job, or stop that readout at
  N ≈ 2,000–3,000 and say so in the figure (the ungated readouts are the
  κ = √N reference points, and `run_Nscaling` already treats them as such).
  Reducing D_factor would change the setting of the experiment and is not
  proposed here.

### E. Memory, scheduling and resumability  (needed before any N = 10k job)

- **Workers by RAM, not cores.**  Per run: W = 32·N² bytes (3.2 GB at
  N = 10k) plus the active block and the output arrays; after A there is no
  N×D temporary.  `physical_cores()` should become
  `min(cores, floor(0.8 · RAM / (1.3 · 32 N_max²)))` — 3 on the 16 GB
  laptop at N = 10k; check the Xeon's RAM before planning 44 concurrent
  runs there (≈ 150 GB).
- **Checkpoints.**  A 140k-task run is hours; `norms_run` should write
  (rng state, W, a, Q, the per-task arrays, t) to the results directory
  every ~30 min and resume from it; `collect` skips conditions whose
  per-run file already exists, so a crashed or interrupted sweep restarts
  where it stopped (the JAX sweep already works this way via its manifest).
- **Per-run output files** (one `.npz` per (N, κ, readout, seed)) instead of
  one dict at the end, so N = 10k seeds that finish early are usable and a
  partial sweep still plots.
- **Largest-N-first ordering** is already there; with RAM-limited workers
  add "do not start a second N = 10k job if it does not fit".

### F. Smaller things

- `draw_readout`: `rng.choice([-1, 1], size=N)` → `rng.integers(0, 2, N)*2-1`
  (minor, O(N)).
- Per-task Python overhead is ~50–100 µs; at 140k tasks that is ~10 s per
  run and can be ignored.
- Figures: `fig_vs_task` draws every task; at 140k tasks per trace bin or
  stride to ~2,000 points per trace for the plot (keep the full arrays in
  the `.npz`).  Per-run curves are 5 keys × 140k × 8 B ≈ 6 MB, fine.
- The `norms_….npz` written at the end: use `float32` for the curves if
  the file size matters (250 MB for the N = 10k cell in float64).

### G. Not this script, but worth knowing: the JAX sweep cannot reach N = 10k as written

> **Done (2026-11-05):** the probes' teachers and gates are now regenerated
> from `fold_in(key, k)` in `observe.evaluate_probes` (`model.probe_task`);
> only the heads (P × N) are stored.  Per run at N = 10k that is 11 GB of
> heads instead of 56 GB of probes, bit-identical results.

`protocol.run_task` stores every task's teacher (`probes_w`, P × D) and
gate (`probes_g`, P × N) for the whole stream: at N = 10k that is
140k × 40k × 8 B ≈ 45 GB of teachers per run, before the batch axis.  If
`run_sweep.py` ever has to go there: keep a ring buffer of the last
m_max + 1 probes, or regenerate a probe's teacher and gate from
`fold_in(key, k)` when it is evaluated (the keys are deterministic), and
drop the stored `heads` for the same reason.

## 3. Expected cost at N = 10k after A–C (+ D mitigated)

| readout group | runs in the N = 10k cell | per run | total core-time |
|---|---|---|---|
| gated (3 readouts × 3 κ × 4 seeds) | 36 | ≈ 1.5 h | ≈ 55 h |
| frozen, ungated | 4 | minutes | — |
| learned_shared, ungated | 4 | 1–7 days depending on threads | the long pole; run separately or cap N |

On the laptop (RAM allows 3 concurrent N = 10k runs) the gated cell is
about a day; on the Xeon it is a couple of hours if the memory is there.
The smaller N in the same sweep (≤ 2,000) become negligible after A and B.

## 4. Suggested order of work

1. A + F (an afternoon; gives 5× at every N and removes the memory spike).
2. E (RAM-aware workers, per-run files, checkpoints) — required before a
   multi-day run is launched.
3. B (the subspace step; a day with the test against the JAX kernels).
4. C (the closed-form ungated frozen path).
5. Decide D: dedicated multi-threaded jobs for ungated learned, or cap its N.
