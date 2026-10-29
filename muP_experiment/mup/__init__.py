"""μP continual-learning experiment (JAX, float64, CPU).

Importing this package enables 64-bit floats in JAX *before* any array is
created; every module in the package imports ``mup`` first for that reason.
"""
import jax

jax.config.update("jax_enable_x64", True)

__version__ = "0.1.0"
