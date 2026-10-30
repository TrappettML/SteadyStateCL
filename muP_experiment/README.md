# muP_experiment

Implementation of `PLAN.md` (JAX, float64, CPU).  Run everything from this
directory with the project venv active (`code/sscl_env`).

```bash
pytest tests                                              # 35 tests, 1 to 2 min: the plotting tests render every figure family
python scripts/run_sweep.py --config configs/pilot.yaml   # laptop pilot, ~10 s
python scripts/analyze.py results/pilot                   # summary.csv, curves.json, figures/{slices,summary,checks}/ (--cells for per-cell figures, --all for diagnostics)
python scripts/validate.py                                # η-halving, Kaczmarz collapse, CV exponent
python scripts/bench.py --N 20 100 --B 1 32 128           # run-steps/s vs batch size
```

On the Xeon:

```bash
python scripts/run_sweep.py --config configs/stage1.yaml --dry-run      # writes results/stage1/manifest.json (2448 runs, λ = 4 only)
python scripts/run_sweep.py --config configs/stage1.yaml --workers 90 --numa-nodes 4
python scripts/analyze.py results/stage1
```

## Run flags

`scripts/run_sweep.py` (the sweep):

| flag | meaning |
|---|---|
| `--config PATH` | YAML config to run (required). Results go to `results/<name>/`, `name` from the config. |
| `--out DIR` | Results directory instead of `results/<name>/`. |
| `--N 5 10 …` | Restrict the sweep to these widths; recorded in the manifest so subprocesses expand the same batches. |
| `--workers K` | Launch one thread-pinned subprocess per batch, `K` at a time (`mup/driver.py`). `0` (default) runs batches in-process, one after another. |
| `--numa-nodes M` | With `--workers`, bind workers round-robin to `M` NUMA nodes via `numactl` (`0`: no binding). |
| `--threads T` | XLA/BLAS threads per worker (default 1, PLAN.md decision 5). |
| `--batch-id i` | Run exactly batch `i` in-process from the existing manifest without touching it (what the driver's subprocesses call). |
| `--dry-run` | Only write `manifest.json` and print every batch with its infeasible-run count; nothing runs or is deleted. |
| `--fresh` | Delete the sweep's `batch_*.npz`, `traces/`, `logs/`, `figures/`, `summary.csv` and `curves.json`, then rerun every batch. Irreversible. |
| `--log-every n` | Print progress every `n` tasks within a batch (`0`: only the per-batch line). |

Without `--fresh` a rerun resumes: a batch is skipped when its file exists
with the current result schema.  A batch file written under an older schema
(`io.SCHEMA_VERSION`, stored in each file's `meta_json`) counts as stale and
is rerun, since it lacks the newer scalars, and so is one whose stream length
differs from the manifest's; `load_runs` warns about older-schema
files, the checks table prints "not recorded" and the norms figures hatch the
panel instead of leaving it blank.

`scripts/analyze.py results/<sweep>` (summary, curves and figures):

| flag | meaning |
|---|---|
| `--fig-dir DIR` | Write figures there instead of `results/<sweep>/figures/`. |
| `--no-figures` | Only `summary.csv`, `curves.json` and the printed summary. |
| `--save-figdata` | Write the plotted arrays next to each figure as `.npz`. |
| `--cells d0=0,lam=1` | Also draw the per-cell families F1–F4 for these cells, under `figures/cells/`; values may be `a/b` lists; `all` draws every cell. |
| `--all` | Also draw every diagnostic family under `figures/diag/`. |
| `--diag kaczmarz by_step check2` | Draw only these diagnostic families. |
| `--levels` | With the check2 diagnostics, also the absolute half-window levels (`halves_*`). |

`scripts/plot_norms.py` and `scripts/plot_check2.py` take `out_dir`,
`--fig-dir`, `--save-figdata` and `--diag` with the same meanings (`--levels`
for `plot_check2.py`); `analyze.py` calls both.

`scripts/validate.py`: `--out DIR` (default `results/validate`), `--seeds n`
(default 4), `--N-max` (default 40), `--parts eta kaczmarz cv` (which of the
three checks to run).  `scripts/bench.py`: `--N`, `--B` (batch sizes),
`--solver`, `--eta`, `--tasks` (tasks per timed run).

## Layout

| path | what |
|---|---|
| `mup/model.py` | single-run kernels on a `State` pytree: `init_state`, `sgd_step`, `pop_step`, `project`, observables |
| `mup/protocol.py` | `run_task` (jit; `vmap`ed over a batch with the state donated) and `run_stream` (Python loop over tasks) |
| `mup/observe.py` | age schedule, probe evaluation, running moments, `half_life`, checks 1–3 |
| `mup/config.py` | `Config` (YAML), `Run`, task duration (§3.6), grid → runs → batches |
| `mup/io.py` | `.npz` per batch + JSON manifest, resume |
| `mup/driver.py` | subprocess pool with thread pinning and optional `numactl` |
| `mup/plotting.py` | the only style file (PLAN.md §12.0, §12.7): panel-sized PNG grids, fixed readout colours, `QUANTITY_LABEL`, the figure legend row, `block_mean`/`dist_by_readout`/`flag_table`/`heat_map`/`kappa_shades` helpers and figure-level checks |
| `mup/results.py` | per-run records from batch files (`load_runs`), grouping, half-window, drift, half-life (`mhalf_by_condition`) and level (`level_stats`) statistics for the figure scripts |
| `scripts/plot_norms.py` | norms vs task from batch files: the slices S1 and M8 by default, per-cell F1/F2 for `--cells` (PLAN.md §12.3); the per-step family from `traces/` only with `--diag` |
| `scripts/plot_check2.py` | check statistics → the ✓/✗ checks table, its CSV and `check2_cells.csv` (§12.6); the C2-A … C2-E figures only with `--diag` |
| `configs/` | `pilot.yaml`, `stage1.yaml`, `sgd_corner.yaml`, `trace.yaml`, `trace_sgd.yaml` (per-step traces, §12.2) |

## Readouts (PLAN.md §4)

Config keys `gated_shared`, `learned_shared`, `gated_frozen`, `frozen`,
`task_heads`.  Gating is a per-run flag: a gated run draws its gate with
density κ/√N, an ungated run has every unit active (pN = N) and no κ, so the
sweep lists the ungated conditions once per (N, δ₀, λ, η, seed) with
`kappa_label = "sqrtN"`.  Figures draw ungated conditions dashed.
| `tests/` | tests 1–6 and 9–10 of the plan plus config/schedule tests; `reference_hash.txt` is test 6's stored hash |

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

Figures go to `figures/` as `.png` with `INDEX.md` (PLAN.md §12.3–§12.4).
By default: `slices/` holds one norms slice (S1) and one curves slice (S2)
per readout and κ, with columns = δ₀ and traces = N (x in lifetimes for
the norms, age m in tasks for the curves);
`summary/` holds the maps M1 (lifetime m½κ_s²/N, drift of mean |a|√Q and
E(0) over δ₀ × N, one panel per readout), M2 (m½ vs N), M3 (m½ vs E(0))
and the scaling figures of PLAN.md §12.3 M4–M8, each on an axis where
passing is flat: M4 `mhalf_per_N_…` (m½/N vs N, traces = κ, grey ln 2),
M5 `kappa_exponent_…` (β(N) = log[m½(κ_min)/m½(κ)]/log(κ/κ_min), grey 2
and 0; a second row of m½ vs κ with the fitted slope when the sweep has
≥ 3 κ), M6 `collapse_…` (S(m)/S(0) on m/N, traces = N, e^(−m/N) in the
task-heads column), M7 `level_vs_N_{readout}_…` and `level_vs_pN_…`
(second-half level of mean |a|√Q, its 95 % quantile and max vs N and vs
pN = κ√N, traces = κ; a Q_act row for the frozen readouts).  M8
`slices/norms_lt2_{readout}_…` is S1 with x = t/m½ (the condition's own
half-life) and y divided by the second-half level.  `checks/` holds one
`checks_{solver}_eta{η}` table per solver with `checks_….csv` behind it;
its last rows are check 4′ (change of m½/N between the two largest N,
with m½/N and the κ exponent at N_max printed) and the level row (slope of
mean |a|√Q per doubling of N, ✓ within 2 s.e. of 0).  A sweep with several
λ gets one set per λ.  Stage 1 (λ = 4, κ ∈ {1, 2} plus κ = 4 at N = 100)
is 54 files.  `--cells d0=0,lam=1` (or `--cells all`)
adds the per-cell families F1 `norms_…`, F2 `normsN_…`/`normsN_lt_…`, F3
`curves_…`, F4 `curvesN_…` under `figures/cells/`; `--all`
or `--diag kaczmarz by_step check2` adds the diagnostic families under
`figures/diag/`; `--save-figdata` writes the plotted arrays next to each
figure.  The slice/summary layout is specified in PLAN.md §12.3 and its
implementation order in §12.8.  `analyze.py`
pools seeds into `summary.csv` (one row per condition: `S0`, `E0`, `m_half`,
`censored`, realized λ, gauge maxima, check-2 flag counts) and `curves.json`.

## Notes

- Stream length: `T_per_N: c` in a config sets `T_burn = T_meas = c·N` for
  every κ (Stage 1: c = 7, ≈ 10 measured half-lives at m½ ≈ 0.69 N, PLAN.md
  §12.3 M4–M8), so M8 compares conditions at the same number of half-lives;
  without it `T_factor` gives the older `T_factor·N/κ_s²` form.  `extra:
  [{N: 100, kappa: 4.0}]` adds gated cells off the N × κ product (an N not
  in `N` gets no ungated runs).  Changing either makes every batch of the
  sweep stale by length, so Stage 1 reruns in full after this change.
- Ungated runs get their own `vmap` batches (their κ = √N never enters the
  dynamics) but take their stream length and age schedule from the schedule κ
  κ_s = the sweep's smallest gated κ (PLAN.md §5; `Run.kappa_sched`), so at
  every N they run as long as the κ_min gated runs (T = 20N/κ_min² without
  `T_per_N`).  A batch
  file whose T disagrees with the manifest counts as stale and is rerun on
  resume (`run_sweep.py` without `--fresh`), which is how a Stage-1 sweep run
  before this change picks up only its 480 ungated runs.  Test fixtures that
  want all five readouts in one batch use `conftest.one_batch`, which gives
  the ungated runs the gated runs' κ and T.
- The stored-head rescaling by 1/f (eq. proj) is accumulated over the steps
  of a task and applied to the (P, N) head buffer once per task
  (`run_task`); `model.project` still does it per step for tests and
  single-step use.  Both agree to rounding.
- Under the driver, a `--batch-id` subprocess never writes the manifest:
  the driver owns it (status updates) and the subprocess only writes its
  batch file.  The manifest records the `--N` filter it was made with.
- `plotting.py` is the only style file (PLAN.md §12.0, §12.7): PNG only,
  panels of 2.6 in × 2.0 in, DejaVu Sans at 10/9 pt, ≤ 4 ticks per axis,
  one legend row above the panels, quantities named only through
  `QUANTITY_LABEL`; `save` warns about panels narrower than 2 in, more than
  6 columns, more than 6 ticks, text below 8 pt or past the figure edge, a
  missing suptitle, a legend inside an axes, a box from < 10 values or an
  all-constant panel.  `tests/test_plotting.py` renders every family from
  synthetic records and fails on any such warning.
- The checks table (PLAN.md §12.6): thresholds live in
  `plot_check2.CRITERIA` and are printed in the caption; pairs that are
  constant by construction (`plot_check2.NA_BY_CONSTRUCTION`, e.g. Σ log f
  for the frozen readouts) get no statistics and a — cell; check 3's γ is
  printed without a mark when the sweep has fewer than three widths; the
  table is also printed to stdout with the worst cell per ✗.
- Batch files of the first code version (readouts named `learned`, `frozen`,
  `heads`, no `gated` key) load with the names mapped to today's
  (`config.LEGACY_READOUTS`; a legacy `frozen` at κ = √N is today's
  ungated `frozen`).
- The projection (`model.project_rows`) computes the larger of a′², Q′ from
  the closed form and the other by division, so Δᵢ is pinned to rounding for
  either sign of Δ₀.  Before 2026-10-30 it always took a′² from the formula,
  which for Δ₀ < 0 and aᵢ → 0 (learned shared at δ₀ < 0) left check-1
  residuals up to 0.06 at N = 100; Stage-1 ungated batches from before that
  date are stale for this reason too.
- Test 9 (`tests/test_trace.py`) checks that a traced run (`lax.scan`)
  reproduces the untraced one (`fori_loop`) bit-for-bit.
- Test 5 compares floats to 1e-10, not bit-for-bit (XLA reduction order,
  PLAN.md §7).  Test 6's hash changes whenever the kernels or the readout
  set change; delete `tests/reference_hash.txt` and rerun to regenerate.
- The population Euler bias at `pop_contraction: 0.1` and the infeasible
  δ₀ = −0.75 runs at N ≤ 20 are documented in PLAN.md §3.5–3.6; `--dry-run`
  reports the skipped runs per batch.
