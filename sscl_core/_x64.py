"""Enable 64-bit floats in JAX; imported first by every JAX module of ``sscl_core``."""
import jax

jax.config.update("jax_enable_x64", True)
