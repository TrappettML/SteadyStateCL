# muP_experiment

Implementation of `PLAN.md` (JAX, float64, CPU).  Run everything from this
directory with the project venv active (`code/sscl_env`).

```bash
pytest tests                                              # 19 tests, ~15 s
python scripts/run_sweep.py --config configs/pilot.yaml   # laptop pilot, ~10 s
python scripts/analyze.py results/pilot                   # summary.csv, curves.json, figures/
python scripts/validate.py                                # η-halving, Kaczmarz collapse, CV exponent
python scripts/bench.py --N 20 100 --B 1 32 128           # run-steps/s vs batch size
```

On the Xeon:

```bash
python scripts/run_sweep.py --config configs/stage1.yaml --dry-run      # writes results/stage1/manifest.json
python scripts/run_sweep.py --config configs/stage1.yaml --workers 90 --numa-nodes 4
python scripts/analyze.py results/stage1
```

`--workers` launches one thread-pinned subprocess per batch (`mup/driver.py`);
finished batches are skipped on rerun, so an interrupted sweep resumes.
`--batch-id i` runs a single batch in-process (what the driver calls).

## Layout

| path | what |
|---|---|
| `mup/model.py` | single-run kernels on a `State` pytree: `init_state`, `sgd_step`, `pop_step`, `project`, observables |
| `mup/protocol.py` | `run_task` (jit; `vmap`ed over a batch with the state donated) and `run_stream` (Python loop over tasks) |
| `mup/observe.py` | age schedule, probe evaluation, running moments, `half_life`, checks 1–3 |
| `mup/config.py` | `Config` (YAML), `Run`, task duration (§3.6), grid → runs → batches |
| `mup/io.py` | `.npz` per batch + JSON manifest, resume |
| `mup/driver.py` | subprocess pool with thread pinning and optional `numactl` |
| `mup/plotting.py` | the only style file (PLAN.md §12.0): panel-sized PNG grids, fixed readout colours, `grid`/`save`/`overlay`/`dumbbell`/`drift_map` helpers and figure-level checks |
| `mup/results.py` | per-run records from batch files (`load_runs`), grouping, half-window and drift statistics for the figure scripts |
| `scripts/plot_norms.py` | weight-norm figures (§12.3): norms vs task from batch files, norms vs step from `traces/` |
| `scripts/plot_check2.py` | steady-state figures (C2-A … C2-E) and `check2_cells.csv` (§12.6) |
| `configs/` | `pilot.yaml`, `stage1.yaml`, `sgd_corner.yaml`, `trace.yaml`, `trace_sgd.yaml` (per-step traces, §12.2) |

## Readouts (PLAN.md §4)

Config keys `gated_shared`, `learned_shared`, `gated_frozen`, `frozen`,
`task_heads`.  Gating is a per-run flag: a gated run draws its gate with
density κ/√N, an ungated run has every unit active (pN = N) and no κ, so the
sweep lists the ungated conditions once per (N, δ₀, λ, η, seed) with
`kappa_label = "sqrtN"`.  Figures draw ungated conditions dashed.
| `tests/` | tests 1–6 of the plan plus config/schedule tests; `reference_hash.txt` is test 6's stored hash |

## Result schema (`results/<name>/batch_*.npz`, schema 2)

`ages` (n_ages,); running moments `sum_s`, `sum_s2`, `sum_e`, `sum_e2`, `count`
(B, n_ages, 2 halves); per-task scalars (T, B), all listed in
`protocol.SCALARS`: `abar2`, `n_active`, `L_end`, `E0`, `S0`, `gauge_delta`,
`gauge_uc`, `logf_sum`, `logf_max`, the §12.1 norms `Q_act`, `Q_all`,
`W_fro2`, `a2_act`, `a2_all`, `aQ_act`, `aQ_all`, `aQ_max`, `heads_fro2` each
as `_pre` (after the last step, before the last projection) and `_post`,
the active-unit quantiles `Q_q05`, `Q_q50`, `Q_q95`, `aQ_q05`, `aQ_q95`, and
the schema-1 names `Q_mean`, `a2_mean`, `aQ_mean`, `aQ_max`; `alpha_task`
(B,); `meta_json` with every run's parameters.  Active-unit means are NaN
for a task whose gate drew no unit.  Trace runs (`trace: true`) also write
`traces/run_<batch>_<j>.npz` with a `(T, n_trace, 7)` float32 array
(`protocol.TRACE_QUANT`) and `traces/manifest.json`.

Figures go to `figures/{curves,mhalf,kaczmarz,checks,norms/by_task,norms/by_step,check2}/`
as `.png` with `INDEX.md`; `--save-figdata` writes the plotted arrays next
to each figure.  `analyze.py`
pools seeds into `summary.csv` (one row per condition: `S0`, `E0`, `m_half`,
`censored`, realized λ, gauge maxima, check-2 flag counts) and `curves.json`.

## Notes

- Ungated runs get their own `vmap` batches (κ = √N sets a shorter stream,
  T = 2·max(T_min, 10) tasks).  Test fixtures that want all five readouts in
  one batch use `conftest.one_batch`, which gives the ungated runs the gated
  runs' κ and T (κ does not enter an ungated run's dynamics).
- The stored-head rescaling by 1/f (eq. proj) is accumulated over the steps
  of a task and applied to the (P, N) head buffer once per task
  (`run_task`); `model.project` still does it per step for tests and
  single-step use.  Both agree to rounding.
- Under the driver, a `--batch-id` subprocess never writes the manifest:
  the driver owns it (status updates) and the subprocess only writes its
  batch file.  The manifest records the `--N` filter it was made with.
- `plotting.py` is the only style file (PLAN.md §12.0): PNG only, panels of
  2.6 in × 2.0 in, DejaVu Sans at 10/9 pt, ≤ 4 ticks per axis; `save`
  warns about panels narrower than 2 in, more than 6 columns, more than 6
  ticks, text below 8 pt or a missing suptitle.
- Batch files of the first code version (readouts named `learned`, `frozen`,
  `heads`, no `gated` key) load with the names mapped to today's
  (`config.LEGACY_READOUTS`; a legacy `frozen` at κ = √N is today's
  ungated `frozen`).
- Test 9 (`tests/test_trace.py`) checks that a traced run (`lax.scan`)
  reproduces the untraced one (`fori_loop`) bit-for-bit.
- Test 5 compares floats to 1e-10, not bit-for-bit (XLA reduction order,
  PLAN.md §7).  Test 6's hash changes whenever the kernels or the readout
  set change; delete `tests/reference_hash.txt` and rerun to regenerate.
- The population Euler bias at `pop_contraction: 0.1` and the infeasible
  δ₀ = −0.75 runs at N ≤ 20 are documented in PLAN.md §3.5–3.6; `--dry-run`
  reports the skipped runs per batch.
