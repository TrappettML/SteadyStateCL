# muP_experiment

A two-layer linear student in mean-field (μP) scaling, trained on an
unbounded stream of teacher–student tasks with context-dependent gating,
for five readout conditions.  JAX, float64, CPU or GPU.  The equations are
in `writing/Derivations/muP_experiment/muP_experiment.tex`; the maintenance
notes for the code are in `PLAN.md`.

Five scripts make the figures of the write-up.  Each builds its own runs
from `configs/stage1.yaml` (population solver, λ = 4, D = 4N, 40 steps per
task, T_burn = T_meas = 7N tasks), trains them, and writes PNG figures with
the plotted arrays beside them as `.npz`.  Nothing is cached between runs:
a script recomputes everything it draws.

| script | question | writes to |
|---|---|---|
| `scripts/run_gauge_comp.py` | does the projection pin the imbalance Δᵢ, and how far does Δᵢ drift without it | `results/gauge_comp/` |
| `scripts/run_szero.py` | how fast is the FIRST task forgotten from a fresh start: S₀(t) vs t | `results/szero/` |
| `scripts/run_stask.py` | how fast is a task forgotten at steady state: S_T(τ) vs τ | `results/stask/` |
| `scripts/run_Nscaling.py` | how the steady-state half-life m½ scales with N and κ, against N ln 2 | `results/Nscaling/` |
| `scripts/run_weight_norms.py` | are the weight norms bounded, and how do they scale with N and κ | `results/weight_norms/` |
| `scripts/run_imbalance_comp.py` | how the steady-state norms and the half-life depend on δ₀, across N, κ and readouts | `results/imbalance_results/` |

## Setup

Run everything from this directory.

Laptop and Xeon (CPU): Python 3.12 with the pinned stack of
`../requirements.txt` in the venv `code/sscl_env` (one level above the
repo):

```bash
source ../../sscl_env/bin/activate
pytest tests                       # ~2 min; test_experiments.py alone covers the kernels these scripts use
```

GPU node (cyclops, 4× V100 32 GB): `source ~/opt/jax-env.sh` (the
environment of `../../jax-cuda-power9.md`, jax 0.4.28, numpy 1.26 — do not
`pip install -r requirements.txt` there).  `python -m pytest tests/test_gpu.py -v`
checks the cards.

## Running

Every script takes the same grid and placement flags.  With no placement
flag it runs in this process; `--gpus` alone is the whole GPU placement.

```bash
# laptop, in-process, small
python scripts/run_szero.py --N 20 --seeds 2
python scripts/run_weight_norms.py --N 5 10 20 --kappa 1 2 --seeds 2

# GPU node, the write-up grids
python scripts/run_gauge_comp.py --N 100 --kappa 1 --gpus
python scripts/run_szero.py --N 100 200 --kappa 1 2 --seeds 8 --tasks 2800 --gpus   # several N: fix the stream with --tasks
python scripts/run_stask.py --N 100 200 --kappa 1 2 --seeds 8 --T 2800 --gpus        # several N: fix T with --T
python scripts/run_Nscaling.py --N 5 10 20 50 100 200 --kappa 1 2 4 --seeds 8 --gpus
python scripts/run_weight_norms.py --N 5 10 20 50 100 200 --kappa 1 2 4 --seeds 8 --gpus
python scripts/run_imbalance_comp.py --N 5 10 20 50 100 200 --kappa 1 2 4 --delta0 0 1 2 4 --seeds 8 --gpus

# Xeon, CPU processes (keep --workers moderate: many XLA compiles at once stall the node)
python scripts/run_Nscaling.py --N 5 10 20 50 100 --kappa 1 2 4 --seeds 8 --workers 16
```

Each script prints a banner with the grid and the placement, one progress
line per job at every tenth of its tasks, a summary line per condition, the
figure paths, and the total wall time.

### Grid flags (all scripts)

| flag | meaning |
|---|---|
| `--config PATH` | YAML supplying solver, λ, step-size and stream-length rules (default `configs/stage1.yaml`). The script overrides the grid keys below. |
| `--N 5 10 …` | widths (D = 4N). `run_gauge_comp`, `run_szero`, `run_stask` default to `100`; `run_Nscaling`, `run_weight_norms` to `5 10 20 50 100`. |
| `--kappa 1 2 …` | gate scales of the gated readouts, as typed (`1` and `1.0` are different labels; use one spelling). The ungated readouts have no κ and are run once. Default `1.0` (cell scripts) or `1 2 4` (scaling scripts). |
| `--delta0 …` | dimensionless imbalance δ₀ = ηΔ₀ (> −1); a list for the cell scripts, one value for the scaling scripts. Default 0. |
| `--lam λ` | learning per task; default the config's first λ (4). |
| `--seeds n` | number of seeds, 0 … n − 1 (`run_szero`, `run_stask`, `run_Nscaling`, `run_weight_norms`: default 4; `run_gauge_comp`: default 1, seed mean ± s.e. in one figure per cell). |
| `--readout …` | subset of `gated_shared learned_shared gated_frozen frozen task_heads` (default all). |
| `--out DIR` | results directory (default `results/<script name>/`); figures go to `DIR/figures/`. |

### Placement flags (all scripts, `mup/pool.py`)

| flag | meaning |
|---|---|
| `--gpus [ids]` | run on the GPUs: no ids (or `auto`) for every card `nvidia-smi` lists (else four), or a list such as `--gpus 0 1`. Processes per card (≤ 3), chunk size and how each group of runs is split are planned from the runs' sizes and the cards' memory; the banner prints the plan. |
| `--workers K` | without `--gpus`: K thread-pinned CPU processes (`≤ 1`, the default, runs in this process). |
| `--per-gpu k` | override: processes per card. |
| `--chunk c` | override: tasks scanned per device call (planned 64/32/16 on a GPU, 1 on the CPU). |
| `--split k` | override: pieces every group of runs is cut into. |

The work of a script is a list of independent jobs (a cell, or one width's
runs as one `vmap` batch); the largest jobs are pinned one per worker, the
rest claimed largest first, so idle cards in the tail are expected.  These
kernels are launch-latency bound at N ≤ 200: a GPU is not much faster than a
core per run, the gain is the batching and the four cards.

## The scripts

### `run_gauge_comp.py` — Δᵢ per step with and without the projection

```
python scripts/run_gauge_comp.py [--N 100] [--kappa 1] [--delta0 0] [--lam 4] [--seeds 1] [--tasks T] [--gpus]
```

Every readout of a cell is trained twice from the same initial state and
task stream, once with the projection forced off and once on, and
Δᵢ = (aᵢ² − Qᵢᵢ)/η is recorded after every step (mean, min, max over the
active units, and max |Δᵢ − Δ₀|).  `--tasks` shortens the stream (default
T_burn + T_meas = 14N tasks × 40 steps).

Output per cell (N, κ, δ₀, λ):
`figures/gauge_comp_N{N}_k{κ}_d0{δ₀}_lam{λ}.png` — two rows
(no projection / projection), x = training step (log), y = the seed mean of
the active-unit mean of Δᵢ with a ±1 s.e. band over seeds, one trace per
readout, grey line at Δ₀ (the seed means of the per-run min and max over
active units are in the .npz) — and the
per-step arrays in the `.npz` of the same stem.  The console prints the
final mean Δ and max |Δ − Δ₀| per readout and row.  Note the "projection"
row projects the frozen readouts too, which the sweep's readout definition
does not: that row tests the projection, not a readout condition.

### `run_szero.py` — first-teacher signal against task

```
python scripts/run_szero.py [--N 100 200] [--kappa 1 2] [--delta0 0] [--lam 4] [--seeds 4] [--tasks T] [--gpus]
```

From a fresh initialisation, after every task t the signal of the FIRST
teacher is read: S₀(t) = u_t·w₀*/|w₀*|² with u_t = (g₀ ⊙ a) W_t / pN, the
gate of task 0 and its stored head (task heads) or the current shared
readout.  Default stream T_burn + T_meas = 14N tasks; since every cell must
run the same number of tasks, a call with several `--N` needs `--tasks`
(`run_stask`: `--T`).

Output per cell (N, κ, δ₀, λ): `figures/szero_N{N}_k{κ}_d0{δ₀}_lam{λ}.png`
— x = t + 1 (log), y = S₀(t)/S₀(0), one trace per readout (seed mean,
±1 s.e. band), grey lines at 0 and ½, the half-life m½ (first crossing of
½, interpolated) per readout in the caption — and the `.npz` with the
per-seed arrays `S0_{readout}` (T, n_seeds) and the normalised means.  The
console prints S₀(0), m½ and S₀(T−1) per readout.

### `run_stask.py` — steady-state teacher signal against age

```
python scripts/run_stask.py [--N 100] [--kappa 1] [--delta0 0] [--lam 4] [--seeds 4] [--T 1400] [--tau 1000] [--gpus]
```

The companion of `run_szero.py` with the probe taken at steady state:
train T tasks (default T_burn + T_meas), then follow the teacher of task T
for τ = 0 … `--tau` more tasks, S_T(τ) = u_{T+τ}·w_T*/|w_T*|².

Output per cell: `figures/stask_N{N}_k{κ}_d0{δ₀}_lam{λ}.png` — x = τ + 1
(log), y = S_T(τ)/S_T(0), same layout and caption as `run_szero` — and the
`.npz` with `ST_{readout}` (τ + 1, n_seeds).  Console: S_T(0), m½,
S_T(τ_max) per readout.

### `run_Nscaling.py` — steady-state half-life against width

```
python scripts/run_Nscaling.py [--N 5 10 20 50 100] [--kappa 1 2 4] [--delta0 0] [--lam 4] [--seeds 4] [--theory nln2|nk2] [--gpus]
```

For every (N, κ, readout, seed): train T = T_burn + T_meas tasks, then T
more while the teacher of task T is followed; the half-life m½ is the
first m with S_T(m)/S_T(0) ≤ ½ (interpolated), computed per seed.  A seed
that never crosses within T tasks is censored (NaN, counted in the caption).

Output, in `figures/`:
- `mhalf_vs_N_k{κ₁}_d0{δ₀}_lam{λ}.png` — x = N (log), y = m½ (log), one
  trace per readout at the FIRST κ listed (seed mean, ±1 s.e. bars); the
  ungated readouts have no κ and appear here only; dotted grey N ln 2.
- `mhalf_theory_{nln2|nk2}_d0{δ₀}_lam{λ}.png` — one panel per gated
  readout, traces = every κ in shades of the readout's colour (lightest =
  smallest κ), solid = simulation, dotted = the equation: m½ = N ln 2
  (`--theory nln2`, default) or N ln 2 / κ² (`nk2`).
- `mhalf_d0{δ₀}_lam{λ}.npz` — the per-seed m½ of every condition, keys
  `N{N}_k{κ}_{readout}`.

Console: m½ seed mean ± s.e. and the censored count per condition.

### `run_weight_norms.py` — weight norms against task and against width

```
python scripts/run_weight_norms.py [--N 5 10 20 50 100] [--kappa 1 2 4] [--delta0 0] [--lam 4] [--seeds 4] [--gpus]
```

For every (N, κ, readout, seed) the stream of T = T_burn + T_meas tasks is
trained and after every task the following are recorded: the hidden-layer
norm mean_i Q_ii = mean_i |W_i|²/D and the output-layer norm mean_i a_i²,
each over all N units (`Q_all`, `a2_all`) and over the task's active units
(`Q_act`, `a2_act`), and the effective weight norm |u_t|²/D of the task
just trained (`u2`), u_t = (g ⊙ a) W / pN.  "Steady state" is the mean over
the measurement window t ≥ T_burn.

Output, in `figures/`:
- `norms_vs_task_k{κ₁}_d0{δ₀}_lam{λ}.png` — rows = mean Q_ii, mean a_i²,
  |u_t|²/D, and the scaling collapse Q̄/pN (frozen readouts) or Q̄²/pN
  (learned readouts); columns = N; traces = readouts at the first κ
  listed; x = task; seed mean with ±1 s.e. band; burn-in shaded.
- `norms_vs_N_d0{δ₀}_lam{λ}.png` — the same four rows; columns = κ;
  x = N (log); y = steady-state value (seed mean ± 1 s.e. bars); traces =
  readouts; the ungated readouts repeat in every column.  A flat bottom
  row confirms the mechanism (pN = κ√N, or N when ungated).
- `norms_d0{δ₀}_lam{λ}.npz` — every per-run curve, keys
  `N{N}_k{κ}_{readout}_{quantity}` with value (n_seeds, T).

Console: steady mean Q, mean a², |u|²/D and the infeasible count per
condition.

### `run_imbalance_comp.py` — norms and half-life across δ₀

```
python scripts/run_imbalance_comp.py [--N 5 10 20 50 100] [--kappa 1 2 4] [--delta0 0 1 2 4] [--lam 4] [--seeds 4] [--gpus]
```

The grid of `run_weight_norms.py` and `run_Nscaling.py` at every `--delta0`:
per (N, κ, readout, δ₀, seed) the norms after each task (steady state =
mean over t ≥ T_burn) and the steady-state half-life m½ of the probe stored
after T tasks.  Two job pools run, the norms first, then the half-life.

Output, under `results/imbalance_results/`:
- `weight_norm_by_delta/norms_vs_N_{readout}.png` — one file per readout;
  the four rows of `norms_vs_N` (mean Q_ii, mean a_i², |u_t|²/D, the
  collapse Q̄/pN or Q̄²/pN), columns = κ, x = N, traces = δ₀ in shades of
  the readout's colour (lightest = smallest).
- `weight_norm_by_model/norms_vs_N_d0{δ₀}.png` — the roles flipped: one
  file per δ₀, traces = readouts (the `run_weight_norms` figure at that δ₀).
- `halflife_by_delta0.png` — rows = readouts, columns = κ, x = N, y = m½
  (seed mean ± 1 s.e. over the uncensored seeds), traces = δ₀; dotted grey
  N ln 2.
- `halflife_by_model.png` — rows = δ₀, columns = κ, traces = readouts.
- `imbalance_comp.npz` — every per-run curve and half-life,
  `N{N}_k{κ}_{readout}_d0{δ₀}_{key}` with value (n_seeds, T), and
  `_mhalf` (n_seeds,).

Console: per (N, κ, readout, δ₀) the steady means, m½ ± s.e. and the
censored-or-infeasible count.

## Readouts

| key | gated | learns a | projection | heads | what it is |
|---|---|---|---|---|---|
| `gated_shared` | ✓ | ✓ | ✓ | – | one readout a learned every task; gates drawn with density κ/√N |
| `learned_shared` | – | ✓ | ✓ | – | the same with every unit active (pN = N) |
| `gated_frozen` | ✓ | – | – | – | a = a⁽⁰⁾ forever, row norms unpinned; gated |
| `frozen` | – | – | – | – | the same, every unit active |
| `task_heads` | ✓ | ✓ | ✓ | ✓ | a fresh head per task, stored heads rescaled so every stored u_k is unchanged; gated |

Figures draw ungated readouts dashed.  A run whose initial readout would be
imaginary (δ₀ ≤ −min_i Q0_i after the row redraws) is infeasible: it is
dropped from its batch, comes back as NaN, and is reported on the console
and in the caption.

## Layout

| path | what |
|---|---|
| `scripts/run_*.py` (the five above) | each: build the runs, group them into jobs, run the jobs through `mup.pool`, draw |
| `mup/experiments.py` | the batched kernels and job functions of the scripts: one task without observables, the per-task/per-step observables evaluated inside a `scan`, NaN columns for infeasible runs |
| `mup/pool.py` | the jobs over GPUs, CPU processes or in-process; `--gpus` planning |
| `mup/config.py` | `Config` (YAML), `Run`, readout table, task duration, grid → runs |
| `mup/model.py` | single-run kernels on a `State` pytree: init, SGD and population steps, the projection, the active-row block loop |
| `mup/protocol.py` | the sweep kernel the experiments are checked against; batching and feasibility helpers |
| `mup/device.py`, `mup/driver.py` | backend defaults (chunk size); worker environments (thread pinning, `CUDA_VISIBLE_DEVICES`) |
| `mup/plotting.py` | the only style file: readout colours, panel grid, legend row, caption, `save` with its legibility checks |
| `configs/stage1.yaml` | the settings every script starts from |
| `tests/` | `test_experiments.py` (kernels vs the sweep kernel, the pool, the planner), `test_block.py`, `test_gpu.py`, and the kernel tests |

Other scripts in `scripts/` (the sweep `run_sweep.py`, `analyze.py`,
`plot_*.py`, `validate.py`, `bench.py`, `run_sigma0.py`, `gpu_check.py`) and
the modules they use (`observe.py`, `io.py`, `results.py`, `npmodel.py`)
are not part of the current write-up and are not documented here.
