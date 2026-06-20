import jax
import jax.numpy as jnp


def layer_norm(x, gamma, beta, eps: float = 1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return gamma * (x - mean) / jnp.sqrt(var + eps) + beta


def gelu(x):
    return 0.5 * x * (1.0 + jnp.tanh(jnp.sqrt(2.0 / jnp.pi) * (x + 0.044715 * x ** 3)))


def _dropout(x, key, rate: float):
    mask = jax.random.bernoulli(key, 1.0 - rate, x.shape)
    return jnp.where(mask, x / (1.0 - rate), 0.0)


def mha_forward(x, p, n_heads: int, key, drop_rate: float, training: bool):
    B, T, d = x.shape
    d_head = d // n_heads

    q = x @ p['wq'] + p['bq']
    k = x @ p['wk'] + p['bk']
    v = x @ p['wv'] + p['bv']

    q = q.reshape(B, T, n_heads, d_head).transpose(0, 2, 1, 3)
    k = k.reshape(B, T, n_heads, d_head).transpose(0, 2, 1, 3)
    v = v.reshape(B, T, n_heads, d_head).transpose(0, 2, 1, 3)

    scores = (q @ k.transpose(0, 1, 3, 2)) * (d_head ** -0.5)
    causal_mask = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))
    scores = jnp.where(causal_mask, scores, -1e9)
    weights = jax.nn.softmax(scores, axis=-1)

    if training and drop_rate > 0.0:
        weights = _dropout(weights, key, drop_rate)

    out = (weights @ v).transpose(0, 2, 1, 3).reshape(B, T, d)
    return out @ p['wo'] + p['bo']


def mlp_forward(x, p, key, drop_rate: float, training: bool):
    h = gelu(x @ p['w1'] + p['b1'])
    if training and drop_rate > 0.0:
        h = _dropout(h, key, drop_rate)
    return h @ p['w2'] + p['b2']


def decoder_block(x, p, n_heads: int, keys, drop_rate: float, training: bool):
    attn_out = mha_forward(
        layer_norm(x, p['ln1']['gamma'], p['ln1']['beta']),
        p['attn'], n_heads,
        keys[0] if keys is not None else None,
        drop_rate, training,
    )
    if training and drop_rate > 0.0:
        attn_out = _dropout(attn_out, keys[1], drop_rate)
    x = x + attn_out

    mlp_out = mlp_forward(
        layer_norm(x, p['ln2']['gamma'], p['ln2']['beta']),
        p['mlp'],
        keys[2] if keys is not None else None,
        drop_rate, training,
    )
    if training and drop_rate > 0.0:
        mlp_out = _dropout(mlp_out, keys[3], drop_rate)
    x = x + mlp_out

    return x


def transformer_forward(params, x, n_heads: int, n_layers: int, drop_rate: float, key, training: bool):
    B, T = x.shape

    if training and drop_rate > 0.0:
        all_keys = jax.random.split(key, 1 + n_layers * 4)
        embed_key = all_keys[0]
        block_keys = [all_keys[1 + i * 4 : 1 + (i + 1) * 4] for i in range(n_layers)]
    else:
        embed_key = None
        block_keys = [None] * n_layers

    h = params['embed']['token'][x] + params['embed']['position'][:T]
    if training and drop_rate > 0.0:
        h = _dropout(h, embed_key, drop_rate)

    for i in range(n_layers):
        h = decoder_block(h, params['blocks'][i], n_heads, block_keys[i], drop_rate, training)

    h = layer_norm(h, params['ln_f']['gamma'], params['ln_f']['beta'])
    return h @ params['lm_head']
