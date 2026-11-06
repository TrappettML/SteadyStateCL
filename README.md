# SteadyStateCL

Analytic and empirical study of the steady-state behaviour of continual
learning algorithms: how long a memory lasts when a network is trained on an
unbounded stream of tasks without weight decay.

## Contents

| path | what |
|---|---|
| `muP_experiment/` | First experiment: a two-layer linear student in mean-field (μP) scaling, trained on a stream of teacher–student tasks with context-dependent gating, for three readout conditions. `muP_experiment/PLAN.md` is the implementation plan; the mathematical specification lives in the (separate) writing repository as `writing/derivations/muP_experiment/muP_experiment.tex`. |
| `requirements.txt` | Pinned Python dependencies (JAX on CPU, float64). |
| `../../jax-cuda-power9.md` | Build notes for JAX with CUDA on the POWER9 node (cyclops, 4× V100 32 GB). |

## Setup

Python 3.12. JAX runs in double precision on every machine: on the CPU on
the laptop and the Xeon, on the 4 GPUs of the POWER9 node (see below).

```bash
python3.12 -m venv sscl_env
source sscl_env/bin/activate
pip install -r requirements.txt
```

Keep the environment outside the repo or rely on the `.gitignore` entry for
`sscl_env/`.

## Running

Develop and test on the laptop:

```bash
cd muP_experiment
pytest tests
python scripts/run_sweep.py --config configs/pilot.yaml --N 20
```

Full sweeps run on the 96-core Xeon node through `scripts/run_sweep.py`
driven by `mup/driver.py` (one process per batch, thread-pinned; see
`PLAN.md` §8). Results go to `muP_experiment/results/` (ignored by git);
`scripts/analyze.py` produces the figures.

### GPU node (cyclops, 4× V100 32 GB, float64)

The JAX environment is the one of `../../jax-cuda-power9.md`
(`source ~/opt/jax-env.sh`; jax 0.4.28 built for the node). The sweep
code is unchanged between CPU and GPU; what changes is how it is driven:

```bash
cd muP_experiment
python scripts/gpu_check.py                      # devices, x64, fp64 matmul, kernel chunk per GPU vs CPU
python scripts/gpu_check.py --bench --N 100 200 --B 64 256 1024 --chunk 1 16   # all 4 GPUs at once
python scripts/run_sweep.py --config configs/stage1.yaml --dry-run --batch-size 256
python scripts/run_sweep.py --config configs/stage1.yaml --gpus auto --batch-size 256
```

- `--gpus auto` (or `--gpus 0 1 2 3`) starts one batch process per GPU
  (`--per-gpu 2` for two per card, sharing its memory); each process sees
  one device through `CUDA_VISIBLE_DEVICES`, so no collectives are needed
  and the manifest keeps the sweep resumable exactly as on the Xeon.
  Batches are launched largest first.
- A GPU wants large `vmap` batches: `--batch-size 256` (or more) puts a
  whole `(N, κ)` cell of Stage 1 in one batch. The batch size is recorded
  in the manifest, so keep it fixed for a results directory (or use
  `--out` / `--fresh`). `--dry-run` prints the device memory of each
  batch's state.
- Inside a process the task loop scans `--chunk` tasks (default 16 on a
  GPU, 1 on the CPU, env `MUP_CHUNK`) per jitted call with one
  device-to-host copy per chunk instead of ~40 per task; `chunk = 1` is the
  CPU loop that `tests/reference_hash.txt` was made with, and
  `tests/test_chunk.py` checks the two agree. Pick `--batch-size` and
  `--chunk` from the `--bench` table (run-steps/s).
- Device memory per run is `W` (N × D) plus the stored heads (T × N);
  probe teachers and gates are regenerated from the run key, not stored.
- The GPU build pins numpy 1.26 and jax 0.4.28: `requirements.txt` is for
  the laptop and the Xeon only; do not `pip install -r` it there.
