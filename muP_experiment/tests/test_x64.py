import jax.numpy as jnp


def test_default_dtype_is_float64():
    import mup  # noqa: F401
    assert jnp.zeros(1).dtype == jnp.float64
    assert jnp.array(1.0).dtype == jnp.float64
