"""Enable 64-bit floats in JAX; imported first by every JAX module of ``mup``."""
import jax

jax.config.update("jax_enable_x64", True)
