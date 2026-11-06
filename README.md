# SteadyStateCL

Analytic and empirical study of the steady-state behaviour of continual
learning algorithms: how long a memory lasts when a network is trained on an
unbounded stream of tasks without weight decay.

## Contents

| path | what |
|---|---|
| `muP_experiment/` | First experiment: a two-layer linear student in mean-field (μP) scaling, trained on a stream of teacher–student tasks with context-dependent gating, for five readout conditions. `muP_experiment/README.md` says how to run the experiment scripts and what each produces; `muP_experiment/PLAN.md` holds the maintenance notes for the code. The mathematical specification is in the (separate) writing folder as `writing/Derivations/muP_experiment/muP_experiment.tex`. |
| `requirements.txt` | Pinned Python dependencies (JAX on CPU, float64) for the laptop and the Xeon. |
| `../../jax-cuda-power9.md` | Build notes for JAX with CUDA on the POWER9 node (cyclops, 4× V100 32 GB). |

## Setup

Python 3.12. JAX runs in double precision on every machine: on the CPU on
the laptop and the Xeon, on the 4 GPUs of the POWER9 node.

```bash
python3.12 -m venv sscl_env
source sscl_env/bin/activate
pip install -r requirements.txt
```

Keep the environment outside the repo or rely on the `.gitignore` entry for
`sscl_env/`.  On the GPU node use the environment of `../../jax-cuda-power9.md`
(`source ~/opt/jax-env.sh`; jax 0.4.28, numpy 1.26) and do not
`pip install -r requirements.txt` there.

## Running

```bash
cd muP_experiment
pytest tests
python scripts/run_szero.py --N 20 --seeds 2          # a laptop-sized example
```

The five experiment scripts (`scripts/run_gauge_comp.py`, `run_szero.py`,
`run_stask.py`, `run_Nscaling.py`, `run_weight_norms.py`) take the same grid
and placement flags; `--gpus` alone places a run on the GPU node, `--workers K`
on CPU processes, nothing runs in-process.  See `muP_experiment/README.md`.
Results go to `muP_experiment/results/` (ignored by git).
