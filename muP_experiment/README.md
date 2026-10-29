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
| `mup/plotting.py` | figure conventions (one fixed colour per readout) |
| `configs/` | `pilot.yaml`, `stage1.yaml`, `sgd_corner.yaml` |

## Readouts (PLAN.md §4)

Config keys `gated_shared`, `learned_shared`, `gated_frozen`, `frozen`,
`task_heads`.  Gating is a per-run flag: a gated run draws its gate with
density κ/√N, an ungated run has every unit active (pN = N) and no κ, so the
sweep lists the ungated conditions once per (N, δ₀, λ, η, seed) with
`kappa_label = "sqrtN"`.  Figures draw ungated conditions dashed.
| `tests/` | tests 1–6 of the plan plus config/schedule tests; `reference_hash.txt` is test 6's stored hash |

## Result schema (`results/<name>/batch_*.npz`)

`ages` (n_ages,); running moments `sum_s`, `sum_s2`, `sum_e`, `sum_e2`, `count`
(B, n_ages, 2 halves); per-task scalars (T, B): `abar2`, `n_active`, `Q_mean`,
`a2_mean`, `aQ_mean`, `aQ_max`, `L_end`, `E0`, `gauge_delta`, `gauge_uc`;
`alpha_task` (B,); `meta_json` with every run's parameters.  `analyze.py`
pools seeds into `summary.csv` (one row per condition: `S0`, `E0`, `m_half`,
`censored`, realized λ, gauge maxima, check-2 flag counts) and `curves.json`.

## Notes

- Ungated runs get their own `vmap` batches (κ = √N sets a shorter stream,
  T = 2·max(T_min, 10) tasks).  Test fixtures that want all five readouts in
  one batch use `conftest.one_batch`, which gives the ungated runs the gated
  runs' κ and T (κ does not enter an ungated run's dynamics).
- Test 5 compares floats to 1e-10, not bit-for-bit (XLA reduction order,
  PLAN.md §7).  Test 6's hash changes whenever the kernels or the readout
  set change; delete `tests/reference_hash.txt` and rerun to regenerate.
- The population Euler bias at `pop_contraction: 0.1` and the infeasible
  δ₀ = −0.75 runs at N ≤ 20 are documented in PLAN.md §3.5–3.6; `--dry-run`
  reports the skipped runs per batch.
