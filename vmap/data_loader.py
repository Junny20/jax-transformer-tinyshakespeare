import numpy as np
import jax
import jax.numpy as jnp


def get_batch(data: np.ndarray, batch_size: int, ctx_len: int, key):
    """Sample a random batch of contiguous sequences from data.

    Returns (x, y, new_key) where y = x shifted right by one token.
    Shapes: x, y → (batch_size, ctx_len).
    """
    n = len(data) - ctx_len
    key, subkey = jax.random.split(key)
    ix = np.array(jax.random.randint(subkey, (batch_size,), 0, n))
    x = jnp.array(np.stack([data[i : i + ctx_len] for i in ix]))
    y = jnp.array(np.stack([data[i + 1 : i + ctx_len + 1] for i in ix]))
    return x, y, key
