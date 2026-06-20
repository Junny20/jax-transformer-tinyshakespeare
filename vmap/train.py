import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import jax
import jax.numpy as jnp
import optax
import numpy as np

from src.model import transformer_forward
from init.attention import init_params
from vmap.tokenizer import build_vocab, encode
from vmap.data_loader import get_batch


def cross_entropy_loss(logits, targets):
    B, T, V = logits.shape
    return optax.softmax_cross_entropy_with_integer_labels(
        logits.reshape(B * T, V), targets.reshape(B * T)
    ).mean()


def make_train_step(cfg: dict, optimizer):
    n_heads   = cfg['n_heads']
    n_layers  = cfg['n_layers']
    drop_rate = cfg['dropout']

    @jax.jit
    def train_step(params, opt_state, x, y, key):
        def loss_fn(p):
            logits = transformer_forward(p, x, n_heads, n_layers, drop_rate, key, training=True)
            return cross_entropy_loss(logits, y)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    return train_step


def make_eval_step(cfg: dict):
    n_heads    = cfg['n_heads']
    n_layers   = cfg['n_layers']
    _dummy_key = jax.random.PRNGKey(0)

    @jax.jit
    def eval_step(params, x, y):
        logits = transformer_forward(params, x, n_heads, n_layers, 0.0, _dummy_key, training=False)
        return cross_entropy_loss(logits, y)

    return eval_step


def benchmark_jit_vs_eager(cfg: dict, optimizer, params, opt_state, x, y, n_trials: int = 20):
    n_heads   = cfg['n_heads']
    n_layers  = cfg['n_layers']
    drop_rate = cfg['dropout']
    train_step = make_train_step(cfg, optimizer)

    key = jax.random.PRNGKey(99)
    p, s, _ = train_step(params, opt_state, x, y, key)
    jax.block_until_ready(p)

    t0 = time.perf_counter()
    for _ in range(n_trials):
        key, k = jax.random.split(key)
        p, s, loss = train_step(p, s, x, y, k)
    jax.block_until_ready(loss)
    jit_ms = (time.perf_counter() - t0) / n_trials * 1000

    def eager_step(p, s, x, y, key):
        def loss_fn(params):
            logits = transformer_forward(params, x, n_heads, n_layers, drop_rate, key, training=True)
            return cross_entropy_loss(logits, y)
        loss, grads = jax.value_and_grad(loss_fn)(p)
        updates, new_s = optimizer.update(grads, s, p)
        return optax.apply_updates(p, updates), new_s, loss

    p2, s2, _ = eager_step(params, opt_state, x, y, jax.random.PRNGKey(0))
    jax.block_until_ready(p2)

    t0 = time.perf_counter()
    for _ in range(n_trials):
        key, k = jax.random.split(key)
        p2, s2, loss2 = eager_step(p2, s2, x, y, k)
    jax.block_until_ready(loss2)
    eager_ms = (time.perf_counter() - t0) / n_trials * 1000

    return jit_ms, eager_ms, eager_ms / jit_ms


def train(cfg: dict, data_train: np.ndarray, data_val: np.ndarray):
    batch_size   = cfg['batch_size']
    ctx_len      = cfg['ctx_len']
    max_steps    = cfg['max_steps']
    warmup_steps = cfg['warmup_steps']
    lr           = cfg['lr']
    weight_decay = cfg['weight_decay']
    val_interval = cfg['val_interval']
    val_batches  = cfg['val_batches']

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=lr,
        warmup_steps=warmup_steps,
        decay_steps=max_steps,
        end_value=lr * 0.1,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule, weight_decay=weight_decay),
    )

    key = jax.random.PRNGKey(42)
    key, init_key = jax.random.split(key)
    params    = init_params(init_key, cfg)
    opt_state = optimizer.init(params)

    n_params = sum(a.size for a in jax.tree_util.tree_leaves(params))
    print(f"Model: {n_params:,} parameters")
    print(f"Device: {jax.devices()[0]}")
    print(f"Training: {max_steps} steps | batch={batch_size} | ctx={ctx_len}")

    train_step = make_train_step(cfg, optimizer)
    eval_step  = make_eval_step(cfg)

    train_losses: list[float]            = []
    val_losses:   list[tuple[int, float]] = []

    for step in range(max_steps):
        key, bk, sk = jax.random.split(key, 3)
        x, y, _ = get_batch(data_train, batch_size, ctx_len, bk)

        t0 = time.perf_counter()
        params, opt_state, loss = train_step(params, opt_state, x, y, sk)
        loss.block_until_ready()
        ms = (time.perf_counter() - t0) * 1000

        train_losses.append(float(loss))

        if step % val_interval == 0 or step == max_steps - 1:
            val_loss = 0.0
            for _ in range(val_batches):
                key, vk = jax.random.split(key)
                vx, vy, _ = get_batch(data_val, batch_size, ctx_len, vk)
                val_loss += float(eval_step(params, vx, vy))
            val_loss /= val_batches
            val_losses.append((step, val_loss))
            print(f"step {step:5d} | train {float(loss):.4f} | val {val_loss:.4f} | {ms:.1f} ms/step")

    return params, optimizer, opt_state, train_losses, val_losses


if __name__ == '__main__':
    import yaml
    import pickle
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cfg_path = os.path.join(os.path.dirname(__file__), '..', 'params', 'configs', 'base.yaml')
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'tinyshakespeare.txt')
    with open(data_path) as f:
        text = f.read()

    stoi, itos, vocab_size = build_vocab(text)
    cfg['vocab_size'] = vocab_size

    data = encode(text, stoi)
    n = int(0.9 * len(data))
    data_train, data_val = data[:n], data[n:]

    params, optimizer, opt_state, train_losses, val_losses = train(cfg, data_train, data_val)

    save_path = os.path.join(os.path.dirname(__file__), '..', 'params', 'model.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump({'params': params, 'cfg': cfg, 'stoi': stoi, 'itos': itos}, f)
    print(f"\nModel saved → {save_path}")

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=cfg['lr'],
        warmup_steps=cfg['warmup_steps'], decay_steps=cfg['max_steps'],
        end_value=cfg['lr'] * 0.1,
    )
    bench_opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(learning_rate=schedule))
    bk = jax.random.PRNGKey(0)
    x_bench, y_bench, _ = get_batch(data_train, cfg['batch_size'], cfg['ctx_len'], bk)
    init_p = params
    init_s = bench_opt.init(init_p)
    jit_ms, eager_ms, speedup = benchmark_jit_vs_eager(cfg, bench_opt, init_p, init_s, x_bench, y_bench)
    print(f"\nJIT step:   {jit_ms:.1f} ms")
    print(f"Eager step: {eager_ms:.1f} ms")
    print(f"Speedup:    {speedup:.2f}×")

    curve_path = os.path.join(os.path.dirname(__file__), '..', 'params', 'training_curve.png')
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(train_losses, alpha=0.4, label='train (per step)')
    val_steps = [s for s, _ in val_losses]
    val_vals  = [v for _, v in val_losses]
    ax.plot(val_steps, val_vals, 'o-', label='val', linewidth=2)
    ax.set_xlabel('step')
    ax.set_ylabel('cross-entropy loss')
    ax.set_title('TinyShakespeare — GPT-style JAX transformer')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(curve_path, dpi=150)
    print(f"Training curve saved → {curve_path}")
