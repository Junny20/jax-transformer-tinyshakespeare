import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
import jax.numpy as jnp

from src.model import transformer_forward


def _top_k_logits(logits, k: int):
    if k <= 0:
        return logits
    threshold = jnp.sort(logits)[-k]
    return jnp.where(logits >= threshold, logits, -1e9)


def generate(params, prompt_ids, cfg: dict, max_new_tokens: int, temperature: float, top_k: int, key):
    n_heads  = cfg['n_heads']
    n_layers = cfg['n_layers']
    ctx_len  = cfg['ctx_len']

    ids = list(map(int, prompt_ids))

    for _ in range(max_new_tokens):
        ctx = ids[-ctx_len:]
        if len(ctx) < ctx_len:
            ctx = [0] * (ctx_len - len(ctx)) + ctx
        x = jnp.array(ctx, dtype=jnp.int32)[None, :]

        key, fwd_key, sample_key = jax.random.split(key, 3)
        logits = transformer_forward(
            params, x, n_heads, n_layers, 0.0, fwd_key, training=False,
        )

        pos = min(len(ids), ctx_len) - 1
        next_logits = logits[0, pos, :] / temperature
        next_logits = _top_k_logits(next_logits, top_k)

        next_id = int(jax.random.categorical(sample_key, next_logits))
        ids.append(next_id)

    return ids


if __name__ == '__main__':
    import pickle
    from vmap.tokenizer import encode, decode

    model_path = os.path.join(os.path.dirname(__file__), '..', 'params', 'model.pkl')
    with open(model_path, 'rb') as f:
        saved = pickle.load(f)

    params = saved['params']
    cfg    = saved['cfg']
    stoi   = saved['stoi']
    itos   = saved['itos']

    prompt = "HAMLET:\n"
    prompt_ids = encode(prompt, stoi)

    print(f"Generating from prompt: {repr(prompt)}\n{'─'*60}")
    generated = generate(
        params, prompt_ids, cfg,
        max_new_tokens=500, temperature=0.8, top_k=40,
        key=jax.random.PRNGKey(0),
    )
    print(decode(generated, itos))
