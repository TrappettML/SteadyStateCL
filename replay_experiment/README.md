# replay_experiment

The μP student of `muP_experiment` with every unit active and both layers
learned (readout `learned_shared`: projection on, no gating, no task
heads), trained by SGD on the same unbounded stream of teacher–student
tasks, with a **buffer of stored samples that is replayed**.  JAX,
float64, CPU or GPU.  The kernel is `../sscl_core/replay.py`; there is no
write-up for this experiment yet, so its docstring is the specification.

The scripts and figures follow `muP_experiment`, with the axes swapped:

| μP experiment | here |
|---|---|
| readout (the traces, one colour each) | **replay method**: `none`, `fifo`, `random`, `priority` |
| κ (columns, shades, or one figure each) | **replay size C**, the buffer's capacity in samples (`--capacity`) |

Replay size is the only replay hyper-parameter the scripts sweep.  The
network is fixed, so there is no `--readout` and no `--kappa`.

## The model

A buffer holds C samples (x, y), y = w_k·x/√D the output of the teacher
the sample came from.  A replay run differs from the μP run in two places:

**Replay** (`--mode`, default `sleep`).  `sleep`: the task trains exactly
as in the μP experiment (n steps of B fresh samples), then n_sleep = ρn
more steps each sum the updates of B samples drawn uniformly from the
buffer.  `interleaved`: every one of the task's n steps sums the updates
of its B fresh samples and of ρB buffer samples.  ρ = `replay_ratio`
(default 1), so both modes replay ρnB samples per task at the same η.
The row projection follows every step, as in the μP run.

**Write** (the method), after the task and its sleep: m fresh samples of
the task just trained are stored (`write`, default 0.125D).  Empty slots
are filled first; in a full buffer the new samples overwrite

| method | the m slots that are overwritten |
|---|---|
| `fifo` | the oldest (the buffer holds the last C/m tasks) |
| `random` | m chosen at random (a sample survives C/m tasks on average) |
| `priority` | the m with the smallest priority, the norm of the gradient of the sample's loss ½e² with respect to (W, a) at the current weights, \|e\|√(\|a\|²\|x\|²/D + \|Wx\|²/D)/N, recomputed for the whole buffer at every write: well-fitted samples go, samples with a large gradient stay |
| `none` | no buffer, no replay: the μP run of `learned_shared` (checked to rounding by `tests/test_replay.py`) |

Choices this leaves open, all in `../sscl_core/replay.py`:

- The buffer is replayed **before** it is written, so in both modes a task
  trains against samples of earlier tasks only; the modes differ in when
  the replay happens, not in what is replayed.
- Replay draws **uniformly** from the buffer under every method; the
  priority decides what is kept, not what is replayed.
- New samples are **always written** (a new sample is never refused for
  having a low priority), and priorities are **refreshed** at every write
  rather than kept from the time a sample was stored.
- The stored samples are m fresh draws of the task, not a subset of the
  inputs its SGD steps saw (the same distribution; online SGD sees each
  input once).
- Methods and capacities at one seed share the run key: the same initial
  weights, teachers and fresh inputs.

A size is a number of samples or a multiple of the input dimension D
(`0.5D`, `4D`); D samples determine one teacher.  The defaults are one
small and one large buffer, C = 0.5D and 4D (below D the buffered samples
can all be fitted at once, above D they cannot), with m = 0.125D: 4 and
32 tasks' worth.

## Scripts

| script | question | writes to |
|---|---|---|
| `scripts/run_training_verify.py` | is every task still learned under replay: the per-step loss of the task being trained (and through its sleep) at the start, middle and end of the stream | `results/training_verify/` |
| `scripts/run_szero.py` | how fast is the FIRST task forgotten from a fresh start and an empty buffer: S₀(t) vs t | `results/szero/` |
| `scripts/run_stask.py` | how fast is a task forgotten at steady state: S_T(τ) vs τ | `results/stask/` |
| `scripts/run_Nscaling.py` | how the steady-state half-life m½ scales with N, per method and replay size | `results/Nscaling/` |
| `scripts/run_loss_slope.py` | does replay make a new task harder to learn: the area under the current task's loss curve against task | `results/loss_slope/` |
| `scripts/run_weight_norms.py` | are the weight norms bounded, what does replay cost the task just trained, and how old are the samples each method keeps | `results/weight_norms/` |
| `scripts/run_inter_vs_sleep.py` | interleaved against sleep replay: the steady-state half-life per replay method, as box plots over seeds | `results/inter_vs_sleep/` |

Each script's docstring (`--help`) lists its figures and `.npz` keys.
Every figure is one trace (or box) per replay method — no replay dashed
grey, FIFO blue, random replace orange, priority replace green — and
either one figure or one column per replay size.  `none` has no buffer,
so it is trained once and drawn in every capacity's cell.  No equation is
overlaid: the write-up has none for replay.

## Running

Run from this directory, in the environment of `../README.md`.

```bash
# laptop, small
python scripts/run_stask.py --N 20 --seeds 4 --T 60 --tau 30 --workers 3
python scripts/run_inter_vs_sleep.py --N 20 --seeds 12 --T 60 --tau 30 --workers 4

# GPU node, the config's grid
python scripts/run_training_verify.py --seeds 8 --gpus
python scripts/run_szero.py --N 100 --seeds 8 --gpus
python scripts/run_stask.py --N 100 --seeds 8 --gpus
python scripts/run_Nscaling.py --N 10 20 50 100 --seeds 8 --gpus
python scripts/run_loss_slope.py --N 100 --seeds 8 --gpus
python scripts/run_weight_norms.py --N 10 20 50 100 --seeds 8 --gpus
python scripts/run_inter_vs_sleep.py --N 100 --seeds 12 --gpus
./run_all.sh                              # all of the above, logged
```

`pytest` (about a minute) runs `tests/test_replay.py`; the tests of the
shared package are `../muP_experiment/tests`.

### Grid flags (all scripts)

| flag | meaning |
|---|---|
| `--config PATH` | YAML supplying the solver, λ, step-size and stream-length rules and the replay defaults (default `configs/replay.yaml`: SGD, B = 16, λ = 4, D = 4N, T_burn = T_meas = 7N). |
| `--N 20 50 …` | widths. |
| `--capacity 0.5D 4D …` | replay sizes C, as typed (the label in titles and file names). |
| `--method …` | subset of `none fifo random priority` (default all). |
| `--mode sleep\|interleaved` | when the buffer is replayed (default `sleep`; `run_inter_vs_sleep.py` runs both). |
| `--replay-ratio ρ` | replayed samples per fresh sample (default 1). |
| `--write m` | samples of each task stored after it, as `--capacity` (default `0.125D`). |
| `--delta0`, `--lam`, `--seeds`, `--sgd_batch`, `--out` | as in `muP_experiment`. |
| `--T`, `--tau`, `--tasks` | stream lengths, per script (`--help`); in units of N where a script takes several widths (`run_Nscaling.py`, `run_weight_norms.py`). |

The placement flags (`--gpus`, `--workers`, `--per-gpu`, `--chunk`,
`--split`) are those of `../sscl_core/pool.py`: every method and seed of
one (N, C, mode) is one `vmap` batch with the method a per-run constant,
and the batches are the jobs spread over the cards.  The buffer adds
8·C·(D + 3) bytes per run to the planner's memory estimate.

## Layout

| path | what |
|---|---|
| `../sscl_core/replay.py` | the buffer, the three write rules, the two replay schedules, the per-task observables and the job functions (`signal_after_tasks`, `norms_over_tasks`, `loss_curves`, `loss_slopes`) |
| `scripts/common.py` | the config, the grid flags, runs → cells and jobs, the method colours |
| `scripts/run_*.py` | each: build the runs, run the jobs through `sscl_core.pool`, draw |
| `configs/replay.yaml` | the settings every script starts from |
| `run_all.sh` | every script with every parameter written out |
| `tests/test_replay.py` | the kernel (no replay = the μP run, what each write rule overwrites, the priority against autodiff, chunking) and every script end to end on a tiny grid |
