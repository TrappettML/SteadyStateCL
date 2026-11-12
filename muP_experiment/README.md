# muP_experiment

A two-layer linear student in mean-field (μP) scaling, trained on an
unbounded stream of teacher–student tasks with context-dependent gating,
for five readout conditions.  JAX, float64, CPU or GPU.  The equations are
in `writing/Derivations/muP_experiment/muP_experiment.tex`; the maintenance
notes for the code are in `PLAN.md`.

Five scripts make the figures of the write-up.  Each builds its own runs
from `configs/stage1.yaml` (population solver, λ = 4, D = 4N, 40 steps per
task, clock (ii), T_burn = T_meas = 7N tasks), trains them, and writes PNG
figures with the plotted arrays beside them as `.npz`.  Nothing is cached
between runs: a script recomputes everything it draws.

**What the code takes from the write-up** (audited 2026-11-11; each item
names the equation label in the `.tex`):

- **λ is the hidden-layer learning per task.**  A task removes θ = 1 − e^(−λ)
  of its residual through the hidden layer (eq. `lam`).  The readout's
  exponent λᵃ_c = Λ‖M_c‖ is not a setting: it is recorded per task from the
  largest eigenvalue of the active block of Q, with its bounds eq. `lama`
  (`run_weight_norms.py`, figure `exponents_vs_N`; scalars `lam_a`, `qmax_c`).
- **Task clock (ii) by default** (`clock: realized`, eq. `clock_real`): every
  task lasts Λ_c = λ/ā²_c with the key norm ā²_c = Σᵢ gᵢaᵢ²/pN it starts with,
  so the step is h_c = λ/(η n ā²_c) with n = ⌈λ/0.1⌉ unchanged.  Clock (i)
  (`--clock nominal`, eq. `clock_nom`: Λ = λ/(1 + δ₀) for every task) is what
  every run before 2026-11-11 used and still reproduces bit for bit
  (`tests/test_reproducible.py`).  SGD always runs clock (i).  Clock (iii),
  the stopping rule, is not implemented.
- **Half-life = root of the mean curve** (`--stat mean`, eq. `Sigma`):
  Σ(m) = ⟨S(m)⟩/⟨S(0)⟩ over seeds, m½ its first crossing of ½ (linear
  interpolation), jackknife s.e.  `--stat seed` gives the mean of per-seed
  half-lives, a different (smaller, noisier) statistic; both go to the `.npz`.
- **Equations drawn, not N ln 2** (`mup/theory.py`): eq. `S_heads`,
  `S_frozen`, `S_learned` with p = κ/√N, ν = 1 − (1 − p)^N, θ = 1 − e^(−λ);
  the key drift of `S_learned` comes from the closed-form steady-state norms
  eq. `abarss`, with no measured input.  The weight-norm figures overlay
  eq. `abarss` / `Qss`.
- **Checks made when the runs are built** (`config.make_runs`, notes on
  stdout): the sufficient contraction condition eq. `popstab`,
  hη(n_c/pN)(δ₀ + 2Q̄_c) < 2, at the estimated steady state; and the burn-in
  in half-lives of the slowest readout, T_burn θν/(N ln 2) (7θν/ln 2 ≈ 10 at
  T_burn = 7N).  The value each task realizes is the scalar `popstab`.
- **Initialization**: aᵢ² = δ₀ + Qᵢᵢ needs δ₀ > −Qᵢᵢ ≈ −σ₀².  `make_runs`
  refuses δ₀ ≤ −σ₀²; rows of W⁽⁰⁾ with δ₀ + Qᵢᵢ < 0 are redrawn, and a unit
  still below zero is marked infeasible (NaN, the run is dropped), never
  passed through a square root unchecked (`model.readout_infeasible`).

| script | question | writes to |
|---|---|---|
| `scripts/run_gauge_comp.py` | does the projection pin the imbalance Δᵢ, and how far does Δᵢ drift without it | `results/gauge_comp/` |
| `scripts/run_szero.py` | how fast is the FIRST task forgotten from a fresh start: S₀(t) vs t | `results/szero/` |
| `scripts/run_stask.py` | how fast is a task forgotten at steady state: S_T(τ) vs τ | `results/stask/` |
| `scripts/run_Nscaling.py` | how the steady-state half-life m½ scales with N and κ, against the half-life of each readout's equation (S_heads, S_frozen, S_learned) | `results/Nscaling/` |
| `scripts/run_weight_norms.py` | are the weight norms bounded, how do they scale with N and κ against the closed form eq. abarss, and what the hidden-layer and readout exponents and the step's stability bound are per task | `results/weight_norms/` |
| `scripts/run_imbalance_comp.py` | how the steady-state norms and the half-life depend on δ₀, across N, κ and readouts | `results/imbalance_results/` |
| `scripts/run_halflife_lam.py` | how the half-life depends on the hidden-layer learning per task λ, for the first memory and for a steady-state one | `results/halflife_lam/` |
| `scripts/run_lr_comp.py` | how the half-life depends on both rate parameters, the hidden-layer learning per task λ and the imbalance δ₀ (the rate ratio η_a/η_W) (heat maps per readout), for the first memory and for a steady-state one | `results/lr_comp/` |
| `scripts/run_halflife_kappa.py` | how the half-life depends on the gate scale κ, from κ = 1 to √N, for the first memory and for a steady-state one | `results/halflife_kappa/` |
| `scripts/run_synapse_comp.py` | does gating synapses instead of hidden units change the half-life: m½ vs N, both gatings | `results/synapse_comp/` |
| `scripts/run_training_verify.py` | is every task learned: the per-step loss at the start, middle and end of the stream | `results/training_verify/` |

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
python scripts/run_training_verify.py --seeds 8 --gpus                 # the four default cells
python scripts/run_halflife_kappa.py --N 1000 --seeds 8 --gpus          # 6 κ from 1 to √N; add --synaptic, --logx
python scripts/run_synapse_comp.py --synapse --N 5 10 20 50 100 200 --kappa 1 2 4 --seeds 8 --gpus
python scripts/run_lr_comp.py --N 100 --kappa 1 --seeds 8 --gpus       # λ × δ₀ heat maps; unsafe cells left blank
python scripts/run_halflife_lam.py --N 100 --kappa 1 --seeds 8 --gpus  # λ = 0.5 … 20

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
| `--delta0 …` | dimensionless imbalance δ₀ = ηΔ₀ = aᵢ² − Qᵢᵢ (> −1), the rate ratio η_a/η_W in equal-rate variables; a list for the cell scripts, one value for the scaling scripts. Default 0. |
| `--lam λ` | hidden-layer learning per task (the frozen-readout exponent); default the config's first λ (4). |
| `--clock realized\|nominal` | task clock: `realized` = (ii), Λ_c = λ/ā²_c per task (default, from the config); `nominal` = (i), Λ = λ/(1 + δ₀). |
| `--stat mean\|seed` | half-life statistic (the half-life scripts): `mean` = first crossing of ½ by the seed-mean curve, the write-up's m½, jackknife s.e. (default); `seed` = mean of per-seed half-lives. |
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
— x = t + 1 (log), y = Σ(t) = ⟨S₀(t)⟩/⟨S₀(0)⟩ over seeds (the write-up's
mean normalized signal; `--stat seed`: the seed mean of each seed's own
ratio), one trace per readout with a ±1 s.e. band, dotted in the readout's
colour the equations that hold from a fresh start (S_heads, S_frozen),
grey lines at 0 and ½, the half-life m½ (first crossing of ½ by the mean
curve, interpolated, ± jackknife s.e.) per readout in the caption — and
the `.npz` with the per-seed arrays `S0_{readout}` (T, n_seeds), the
normalised curve, `theory_{readout}`, `mhalf_{readout}` and
`mhalf_seed_{readout}`.  The console prints S₀(0) beside θ, m½, S₀(T−1)
and the equation's half-life per readout.

### `run_stask.py` — steady-state teacher signal against age

```
python scripts/run_stask.py [--N 100] [--kappa 1] [--delta0 0] [--lam 4] [--seeds 4] [--T 1400] [--tau 1000] [--gpus]
```

The companion of `run_szero.py` with the probe taken at steady state:
train T tasks (default T_burn + T_meas), then follow the teacher of task T
for τ = 0 … `--tau` more tasks, S_T(τ) = u_{T+τ}·w_T*/|w_T*|².

Output per cell: `figures/stask_N{N}_k{κ}_d0{δ₀}_lam{λ}.png` — x = τ + 1
(log), y = Σ(τ) = ⟨S_T(τ)⟩/⟨S_T(0)⟩, same layout and caption as `run_szero`,
with every readout's equation dotted: S_heads, S_frozen, and S_learned
drawn from the closed-form steady-state norms eq. abarss (no measured
input) — and the `.npz` with `ST_{readout}` (τ + 1, n_seeds).  Console:
S_T(0), m½, S_T(τ_max) and the equation's half-life per readout.

### `run_Nscaling.py` — steady-state half-life against width

```
python scripts/run_Nscaling.py [--N 5 10 20 50 100] [--kappa 1 2 4] [--delta0 0] [--lam 4] [--seeds 4] [--stat mean|seed] [--theory eq|nln2|nk2] [--clock realized|nominal] [--gpus]
```

For every (N, κ, readout, seed): train T = T_burn + T_meas tasks, then T
more while the teacher of task T is followed.  The half-life m½ is the
first m at which the mean curve Σ(m) = ⟨S_T(m)⟩/⟨S_T(0)⟩ over seeds falls
to ½ (interpolated), with a jackknife s.e. — the root the write-up's
equations predict (`--stat mean`).  `--stat seed` averages each seed's own
half-life instead; a curve that never crosses within T tasks is censored
(NaN, counted in the caption).

Output, in `figures/`:
- `mhalf_vs_N_k{κ₁}_d0{δ₀}_lam{λ}.png` — x = N (log), y = m½ (log), one
  trace per readout at the FIRST κ listed (±1 s.e. bars); the
  ungated readouts have no κ and appear here only; dotted grey N ln 2.
- `mhalf_theory_{eq|nln2|nk2}_d0{δ₀}_lam{λ}.png` — one panel per gated
  readout, traces = every κ in shades of the readout's colour (lightest =
  smallest κ), solid = simulation, dotted = the equation: with
  `--theory eq` (default) the half-life of the readout's own curve, eq.
  S_heads, S_frozen or S_learned at the run's (N, κ, λ, δ₀)
  (`theory.mhalf`); `nln2` draws N ln 2, `nk2` N ln 2 / κ².
- `mhalf_d0{δ₀}_lam{λ}.npz` — keys `N{N}_k{κ}_{readout}` (the statistic
  drawn: an (n_seeds,) array whose mean is m½ and whose s.e. is its error),
  `seed_N{N}_k{κ}_{readout}` (per-seed half-lives) and
  `theory_N{N}_k{κ}_{readout}`.

Console: m½ ± s.e., the per-seed mean and the equation's value per condition.

### `run_weight_norms.py` — weight norms against task and against width

```
python scripts/run_weight_norms.py [--N 5 10 20 50 100] [--kappa 1 2 4] [--delta0 0] [--lam 4] [--seeds 4] [--clock realized|nominal] [--gpus]
```

For every (N, κ, readout, seed) the stream of T = T_burn + T_meas tasks is
trained and after every task the following are recorded: the hidden-layer
norm mean_i Q_ii = mean_i |W_i|²/D and the output-layer norm mean_i a_i²,
each over all N units (`Q_all`, `a2_all`) and over the task's active units
(`Q_act`, `a2_act`), and the effective weight norm |u_t|²/D of the task
just trained (`u2`), u_t = (g ⊙ a) W / pN.  Before every task the state it
starts with gives the inputs of the two exponents (`n_c`, `abar_c`,
`Qbar_c`, `qmax_c` the largest eigenvalue of the active block of Q, `h_c`
the task's step).  "Steady state" is the mean over the measurement window
t ≥ T_burn.

Output, in `figures/`:
- `norms_vs_task_k{κ₁}_d0{δ₀}_lam{λ}.png` — rows = mean Q_ii, mean a_i²,
  |u_t|²/D, and the scaling collapse Q̄/pN (frozen readouts) or Q̄²/pN
  (learned readouts); columns = N; traces = readouts at the first κ
  listed; x = task; seed mean with ±1 s.e. band; burn-in shaded.
- `norms_vs_N_d0{δ₀}_lam{λ}.png` — the same four rows; columns = κ;
  x = N (log); y = steady-state value (seed mean ± 1 s.e. bars); traces =
  readouts; the ungated readouts repeat in every column.  A flat bottom
  row confirms the mechanism (pN = κ√N, or N when ungated).  Dotted in a
  gated readout's colour, in both figures: the closed-form steady state,
  eq. abarss (projected readouts) and eq. Qss (frozen), which the recorded
  norms check.
- `exponents_vs_N_d0{δ₀}_lam{λ}.png` — rows = the hidden-layer exponent
  λ_c = Λ_c ā²_c each task realizes (λ under clock (ii); eq. lamreal under
  clock (i)), the readout exponent λᵃ_c = Λ_c‖M_c‖ with the band of its
  bounds eq. lama, and the left side of the contraction condition
  eq. popstab against its limit 2; columns = κ; x = N.
- `norms_d0{δ₀}_lam{λ}.npz` — every per-run curve, keys
  `N{N}_k{κ}_{readout}_{quantity}` with value (n_seeds, T).

Console: steady mean Q, mean a², |u|²/D and the infeasible count per
condition, then λ_c, λᵃ_c with its bounds and the number of tasks outside
them (none should be), the stability bound (mean and maximum) and the
closed-form Q̄ and ā².

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
  (± 1 s.e.; `--stat`), traces = δ₀; dotted in a trace's colour the
  half-life of the readout's equation at that δ₀, dotted grey N ln 2.
- `halflife_by_model.png` — rows = δ₀, columns = κ, traces = readouts.
- `imbalance_comp.npz` — every per-run curve and half-life,
  `N{N}_k{κ}_{readout}_d0{δ₀}_{key}` with value (n_seeds, T), `_mhalf`
  (n_seeds,) the statistic drawn and `_mhalf_seed` the per-seed half-lives.

Console: per (N, κ, readout, δ₀) the steady means, m½ ± s.e. and the
censored-or-infeasible count.

### `run_halflife_lam.py` — half-life against the hidden-layer learning per task

```
python scripts/run_halflife_lam.py [--N 100] [--kappa 1] [--delta0 0] [--lam 0.5 1 2 … 20] [--seeds 4] [--T T] [--tau τ] [--gpus]
python scripts/run_halflife_lam.py --N 20 --lam 1 4 8 --seeds 2        # laptop
```

One cell (a single N, κ, δ₀) at every `--lam` (a list here; default
0.5, 1, 2, 3, 4, 6, 8, 10, 12, 16, 20; λ = 0 takes no step and is skipped).
Every run is one stream of T + τ + 1 tasks
(`experiments.first_and_steady_signal`): the first teacher is followed for
the first T tasks (as `run_szero`), then the teacher of task T for τ more
(as `run_stask`).  T defaults to T_burn + T_meas, τ to T.  n_steps = 10 λ
is static in the kernels, so each λ is its own set of jobs.

Output: `figures/mhalf_vs_lam_N{N}_k{κ}_d0{δ₀}.png` — two rows sharing
both axes (first memory / steady state), x = λ (linear, from 0), y = m½
(log; ± 1 s.e., `--stat`; censored values counted in the caption), one
trace per readout, dotted in its colour the half-life of its equation
over λ (∝ 1/θ), dotted grey N ln 2 — and the `.npz` with the half-lives
`mhalf_{first|steady}_{readout}_lam{λ}` (per-seed ones under `…_seed`).
Console: both half-lives and the censored counts per (λ, readout).

### `run_lr_comp.py` — half-life over λ and δ₀

```
python scripts/run_lr_comp.py [--N 100] [--kappa 1] [--lam 0.5 1 2 4 8 16] [--delta0 -0.75 -0.5 0 1 4 16] [--seeds 4] [--T T] [--tau τ] [--force] [--gpus]
python scripts/run_lr_comp.py --N 20 --lam 0.5 2 8 --delta0 -0.75 0 4 --seeds 3 --workers 6     # laptop, seconds
```

The two rows of `run_halflife_lam.py` on the grid `--lam` × `--delta0` (both
lists) at one N and κ; the two rates η_a, η_W are the pair (λ, δ₀).  Each
run is one stream of T + τ + 1 tasks (`experiments.first_and_steady_signal`);
every λ is its own set of jobs with all δ₀ and seeds batched together.

A (λ, δ₀) cell is run for a readout only when it is safe by the write-up:
λ > 0, δ₀ > −1, and the population step contracts in every direction,
h η (δ₀ + 2 Q̄) < 2 (eq. popstab at n_c = pN) with Q̄ the steady-state
estimate of eq. (abarss); the frozen readouts need only h η (δ₀ + σ₀²) < 2,
which always holds.  The bound is sufficient and conservative (it flags
δ₀ = −0.75 for the projected readouts and δ₀ ≤ 0 at λ ≥ 4 for the ungated
learned readout at N = 100).  Unsafe cells are skipped, listed on the
console with their bound, and left blank; `--force` runs the ones that fail
the step bound anyway.

Output: `figures/mhalf_lr_N{N}_k{κ}.png` — rows = first memory / steady
state, a column per readout, each panel a heat map with x = δ₀, y = λ,
colour and printed number = m½ (`--stat`).  One colour bar per row
(viridis), limits from that row's data, logarithmic when the row spans
more than a factor 8.  Blank = unsafe, not run; hatched = run but no seed
reached half.  The `.npz` holds the half-lives
`mhalf_{first|steady}_{readout}_lam{λ}_d0{δ₀}`, the cell means, `stab_bound`
and `skipped`.

### `run_halflife_kappa.py` — half-life against the gate scale

```
python scripts/run_halflife_kappa.py [--synaptic] [--logx] [--N 1000] [--n-kappa 6] [--kappa …] [--delta0 0] [--lam 4] [--seeds 4] [--T T] [--tau τ] [--gpus]
python scripts/run_halflife_kappa.py --synaptic --N 50 --seeds 4 --workers 4       # laptop
```

`run_halflife_lam.py` with κ on the x axis: one cell (a single N, δ₀, λ)
at `--n-kappa` values of κ from 1 (the smallest the model allows) to √N,
evenly spaced on a linear x axis, or with `--logx` evenly spaced in log κ
on a log x axis; `--kappa` lists the values instead.  Each run is one
stream of T + τ + 1 tasks.  T defaults to T_burn + T_meas = 14 N and τ to
T — 28 001 tasks per run at N = 1000; `--T` / `--tau` shorten it.

`--synaptic` adds every gated readout with the gate on the synapse
(`mup/synapse.py`, as `run_synapse_comp.py`: same initial weights,
teachers, heads and task duration as the unit-gated twin), dash-dotted
with open squares.  A synaptic step touches all N·D synapses at every κ,
so at N = 1000 these runs dominate the cost.

Output: `figures/mhalf_vs_kappa_N{N}_d0{δ₀}_lam{λ}.png` — two rows sharing
both axes (first memory / steady state), x = κ, y = m½ (log; ± 1 s.e.,
`--stat`), one trace per gated readout with the half-life of its equation
dotted in its colour, the ungated readouts (no κ) as dashed horizontal
lines, dotted grey N ln 2 — and the `.npz` with the half-lives.  Console: both
half-lives and the censored counts per (κ, readout), unit gating first.

### `run_synapse_comp.py` — unit gating against synaptic gating

```
python scripts/run_synapse_comp.py [--synapse] [--N 5 10 20 50 100] [--kappa 1 2 4] [--delta0 0] [--lam 4] [--seeds 4] [--stat mean|seed] [--theory eq|nln2|nk2] [--clock realized|nominal] [--gpus]
python scripts/run_synapse_comp.py --synapse --N 10 20 --kappa 1 --seeds 2     # laptop
```

The experiment and flags of `run_Nscaling.py`.  `--synapse` also runs
every gated readout with the gate on the synapse (`mup/synapse.py`: a
mask of i.i.d. indicators on the N × D weights at the density κ/√N of the
unit gate, the projection on all N units) and adds it to both figures as
a trace.  A synaptic run has the initial weights, teachers, fresh heads
and n_steps of its unit-gated twin.  Under the default clock (ii) each
gating runs its own realized duration λ/ā²_c (the synaptic one at the mean
rate over the input coordinates), as in the write-up's experiment: S(0) = θ̄
(eq. syn_theta) and the heads' half-lives differ by θν/θ̄ (eq. syn_heads,
1.017 at N = 100, κ = 1).  With `--clock nominal` both run the fixed
duration λ/(1 + δ₀), the pair differs in the mask alone, and the mean
signals are equal at every age (eq. syn_Sigma).  Without the flag the
script is `run_Nscaling.py`.  Population solver only.

Output, in `figures/`: `mhalf_vs_N_k{κ₁}_…` and `mhalf_theory_…` as
`run_Nscaling.py`, the synaptic runs dash-dotted with open squares in the
colour of their unit-gated trace, and `mhalf_….npz`, keys
`N{N}_k{κ}_{readout}` (unit) and `syn_N{N}_k{κ}_{readout}` (the statistic
drawn), `seed_…` (per-seed half-lives).
Console: every condition, then the two gatings side by side with the
ratio synaptic / unit.

### `run_training_verify.py` — the loss of the task being trained

```
python scripts/run_training_verify.py [--cell N,kappa,delta0 ...] [--seeds 4] [--lam 4] [--gpus]
```

For every cell (default `100,2,1`, `50,1,0`, `20,1,4`, `200,2,1`) every
readout × seed trains through the
Stage-1 stream, and the population loss of the task being trained is
recorded before its first step and after each step for six tasks: the
first two, the middle two (T/2 − 1, T/2) and the last two.  The rest of
the stream trains untraced (`experiments.loss_curves`).

Output, in `results/training_verify/`: `loss_N{N}_k{κ}_d0{δ₀}.png` — three
panels on a broken x axis (first two | middle two | last two tasks), y =
loss (log), one trace per readout drawn continuously across the task
switch (faint dotted line), seed mean ± 1 s.e. band; the loss jumps when a
new task starts and decays as it is learned.  `loss_….npz` holds the
curves, `{readout}` (n_seeds, 6, n_steps + 1), with `tasks`.

Console: per readout the loss at the start and end of the first and the
last traced task.

### `run_loss_slope.py` — area under the current task's loss curve against task

```
python scripts/run_loss_slope.py [--N 100] [--kappa 1 2] [--delta0 0] [--lam 4] [--seeds 4] [--tasks T] [--gpus]
```

For every task of the stream the population loss of the task being
trained is recorded before its first step and after each step, and the
area between that curve and L = 0 is taken from the start of training on
the task to its end (trapezoid rule, loss × steps;
`experiments.loss_slopes`).  The loss is non-negative, so the area is
positive, and it is smaller the lower the loss stays during the task.

Output, in `results/loss_slope/figures/`: `loss_slope_N{N}_d0{δ₀}_lam{λ}.png`
— one column per κ, x = task (log), y = the area, seed mean ± 1 s.e. band,
one trace per readout; the ungated readouts are drawn in every column.
`loss_slope_….npz` holds `area_k{κ}_{readout}`, `slope_…` (the
least-squares slope of the curve per step), `loss_start_…` and
`loss_end_…`, each (T, n_seeds), with `task`.

Console: per readout the area of task 0 and its mean over the second half
of the stream, beside n·L_start and the fitted slope.

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
| `scripts/run_*.py` (the scripts above) | each: build the runs, group them into jobs, run the jobs through `mup.pool`, draw |
| `mup/theory.py` | the write-up's closed forms, one function per equation label: Σ(m) of each readout and its half-life, the steady-state norms eq. abarss/Qss, the key drift, the readout-exponent bounds, eq. popstab, the burn-in in half-lives, the synaptic θ̄ |
| `mup/halflife.py` | half-life statistics: the mean curve's m½ with its jackknife (the write-up's), the per-seed mean |
| `mup/synapse.py` | synaptic gating: the masked task, probe and job function of `run_synapse_comp.py` |
| `mup/experiments.py` | the batched kernels and job functions of the scripts: one task without observables, the per-task/per-step observables evaluated inside a `scan`, NaN columns for infeasible runs |
| `mup/pool.py` | the jobs over GPUs, CPU processes or in-process; `--gpus` planning |
| `mup/config.py` | `Config` (YAML), `Run`, readout table, task duration and clock, grid → runs, the contraction and burn-in checks |
| `mup/model.py` | single-run kernels on a `State` pytree: init, SGD and population steps, the projection, the active-row block loop, the task clock (`task_h`) and the active-block spectrum (`active_qmax`) |
| `mup/protocol.py` | the sweep kernel the experiments are checked against; batching and feasibility helpers |
| `mup/device.py`, `mup/driver.py` | backend defaults (chunk size); worker environments (thread pinning, `CUDA_VISIBLE_DEVICES`) |
| `mup/plotting.py` | the only style file: readout colours, panel grid, legend row, caption, `save` with its legibility checks |
| `configs/stage1.yaml` | the settings every script starts from |
| `tests/` | `test_theory.py` (the closed forms against the numbers in the write-up; the clock, the readout exponent and its bounds, the half-life statistics), `test_experiments.py` (kernels vs the sweep kernel, the pool, the planner), `test_block.py`, `test_gpu.py`, and the kernel tests |

Old code lives in `old/`: the scripts of the earlier, sweep-based version
of the experiments (the sweep `run_sweep.py`, `analyze.py`, `plot_*.py`,
`validate.py`, `bench.py`, `run_sigma0.py`, `gpu_check.py`), run as
`python old/<script>.py`.  They and the `mup/` modules only they use
(`results.py`, `npmodel.py`, and most of `observe.py` and `io.py`) are not
part of the current write-up and are not documented here.
