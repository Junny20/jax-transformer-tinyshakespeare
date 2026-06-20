import jax
import jax.numpy as jnp


def init_params(key, cfg: dict) -> dict:
    n_layers   = cfg['n_layers']
    d_model    = cfg['d_model']
    d_ff       = cfg['d_ff']
    ctx_len    = cfg['ctx_len']
    vocab_size = cfg['vocab_size']
    proj_std   = 0.02 / (2 * n_layers) ** 0.5

    def randn(k, *shape, std=0.02):
        return jax.random.normal(k, shape) * std

    def make_block(k):
        ks = jax.random.split(k, 6)
        return {
            'ln1': {
                'gamma': jnp.ones((d_model,)),
                'beta':  jnp.zeros((d_model,)),
            },
            'attn': {
                'wq': randn(ks[0], d_model, d_model),
                'bq': jnp.zeros((d_model,)),
                'wk': randn(ks[1], d_model, d_model),
                'bk': jnp.zeros((d_model,)),
                'wv': randn(ks[2], d_model, d_model),
                'bv': jnp.zeros((d_model,)),
                'wo': randn(ks[3], d_model, d_model, std=proj_std),
                'bo': jnp.zeros((d_model,)),
            },
            'ln2': {
                'gamma': jnp.ones((d_model,)),
                'beta':  jnp.zeros((d_model,)),
            },
            'mlp': {
                'w1': randn(ks[4], d_model, d_ff),
                'b1': jnp.zeros((d_ff,)),
                'w2': randn(ks[5], d_ff, d_model, std=proj_std),
                'b2': jnp.zeros((d_model,)),
            },
        }

    key, k_tok, k_pos, k_head = jax.random.split(key, 4)
    block_keys = jax.random.split(key, n_layers)

    return {
        'embed': {
            'token':    randn(k_tok, vocab_size, d_model),
            'position': randn(k_pos, ctx_len,    d_model),
        },
        'blocks': [make_block(block_keys[i]) for i in range(n_layers)],
        'ln_f': {
            'gamma': jnp.ones((d_model,)),
            'beta':  jnp.zeros((d_model,)),
        },
        'lm_head': randn(k_head, d_model, vocab_size),
    }
