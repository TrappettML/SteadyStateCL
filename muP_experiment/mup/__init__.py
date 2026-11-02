"""μP continual-learning experiment (JAX, float64, CPU).

Every module that uses JAX imports ``mup._x64`` first, which enables 64-bit
floats *before* any array is created.  The package itself does not import
JAX, so the NumPy-only scripts (``scripts/run_Nscaling.py``) and the config
and plotting modules load without it.
"""
__version__ = "0.1.0"
