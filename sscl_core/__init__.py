"""Shared core of the SteadyStateCL experiments: the μP continual-learning model (JAX, float64) and its tooling.

Kernels (``model``, ``protocol``, ``experiments``, ``synapse``, ``nested``, ``npmodel``), the write-up's closed
forms (``theory``, ``halflife``), grids and runs (``config``), job placement on GPUs / CPU workers (``pool``,
``driver``, ``device``), results and figures (``io``, ``results``, ``plotting``).  Every experiment folder
(``muP_experiment/``, …) holds only its ``scripts/``, ``configs/``, ``tests/`` and ``results/`` and imports this
package through its ``scripts/_path.py``.  Until 2026-11-13 this package was ``muP_experiment/mup``.

Every module that uses JAX imports ``sscl_core._x64`` first, which enables 64-bit
floats *before* any array is created.  The package itself does not import
JAX, so the NumPy-only scripts (``scripts/run_Nscaling.py``) and the config
and plotting modules load without it.
"""
__version__ = "0.1.0"
