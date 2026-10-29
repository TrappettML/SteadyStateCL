# SteadyStateCL

Analytic and empirical study of the steady-state behaviour of continual
learning algorithms: how long a memory lasts when a network is trained on an
unbounded stream of tasks without weight decay.

## Contents

| path | what |
|---|---|
| `muP_experiment/` | First experiment: a two-layer linear student in mean-field (μP) scaling, trained on a stream of teacher–student tasks with context-dependent gating, for three readout conditions. `muP_experiment/PLAN.md` is the implementation plan; the mathematical specification lives in the (separate) writing repository as `writing/derivations/muP_experiment/muP_experiment.tex`. |
| `requirements.txt` | Pinned Python dependencies (JAX on CPU, float64). |

## Setup

Python 3.12. JAX runs on the CPU in double precision on every machine.

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
