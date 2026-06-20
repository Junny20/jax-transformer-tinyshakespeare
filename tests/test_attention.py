import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import jax
import jax.numpy as jnp
import optax
import pytest

from src.model import layer_norm, gelu, mha_forward, mlp_forward, decoder_block, transformer_forward
from init.attention import init_params
from vmap.tokenizer import build_vocab, encode, decode
from vmap.data_loader import get_batch


# ── Configs ────────────────────────────────────────────────────────────────────

TINY_CFG = {
    'n_layers': 2, 'n_heads': 2, 'd_model': 16, 'd_ff': 32,
    'ctx_len': 8,  'dropout': 0.0, 'vocab_size': 10,
}

FULL_CFG = {
    'n_layers': 6, 'n_heads': 6, 'd_model': 384, 'd_ff': 1536,
    'ctx_len': 256, 'dropout': 0.0, 'vocab_size': 65,
}


# ── Tokenizer ──────────────────────────────────────────────────────────────────

def test_tokenizer_roundtrip():
    text = "hello world"
    stoi, itos, _ = build_vocab(text)
    assert decode(encode(text, stoi), itos) == text


def test_vocab_size_counts_unique():
    _, _, vocab_size = build_vocab("abcabc")
    assert vocab_size == 3


def test_encode_returns_int32():
    stoi, _, _ = build_vocab("abc")
    arr = encode("abc", stoi)
    assert arr.dtype == np.int32


# ── Data loader ────────────────────────────────────────────────────────────────

def test_data_loader_shapes():
    data = np.arange(1000, dtype=np.int32)
    x, y, _ = get_batch(data, batch_size=4, ctx_len=16, key=jax.random.PRNGKey(0))
    assert x.shape == (4, 16)
    assert y.shape == (4, 16)


def test_data_loader_y_is_x_shifted():
    data = np.arange(1000, dtype=np.int32)
    x, y, _ = get_batch(data, batch_size=4, ctx_len=16, key=jax.random.PRNGKey(0))
    np.testing.assert_array_equal(np.array(x[:, 1:]), np.array(y[:, :-1]))


# ── LayerNorm ──────────────────────────────────────────────────────────────────

def test_layer_norm_output_shape():
    x = jax.random.normal(jax.random.PRNGKey(0), (2, 4, 16))
    out = layer_norm(x, jnp.ones((16,)), jnp.zeros((16,)))
    assert out.shape == x.shape


def test_layer_norm_zero_mean_unit_var():
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 4, 16))
    out = layer_norm(x, jnp.ones((16,)), jnp.zeros((16,)))
    np.testing.assert_allclose(np.array(out.mean(-1)), 0.0, atol=1e-5)
    np.testing.assert_allclose(np.array(out.var(-1)),  1.0, atol=1e-4)


# ── GELU ───────────────────────────────────────────────────────────────────────

def test_gelu_shape_preserved():
    x = jax.random.normal(jax.random.PRNGKey(0), (2, 4, 32))
    assert gelu(x).shape == x.shape


def test_gelu_positive_for_positive_inputs():
    assert jnp.all(gelu(jnp.array([1.0, 2.0, 3.0])) > 0)


# ── MHA ────────────────────────────────────────────────────────────────────────

def test_mha_output_shape():
    cfg = TINY_CFG
    params = init_params(jax.random.PRNGKey(0), cfg)
    x = jax.random.normal(jax.random.PRNGKey(1), (2, cfg['ctx_len'], cfg['d_model']))
    out = mha_forward(x, params['blocks'][0]['attn'], cfg['n_heads'], None, 0.0, False)
    assert out.shape == x.shape


def test_mha_causal_masking():
    """Modifying the last token must not change earlier positions' outputs."""
    cfg = TINY_CFG
    params = init_params(jax.random.PRNGKey(0), cfg)
    x1 = jax.random.normal(jax.random.PRNGKey(1), (1, cfg['ctx_len'], cfg['d_model']))
    x2 = x1.at[:, -1, :].add(1.0)  # perturb only the last token
    out1 = mha_forward(x1, params['blocks'][0]['attn'], cfg['n_heads'], None, 0.0, False)
    out2 = mha_forward(x2, params['blocks'][0]['attn'], cfg['n_heads'], None, 0.0, False)
    np.testing.assert_allclose(
        np.array(out1[:, :-1, :]), np.array(out2[:, :-1, :]), atol=1e-5,
        err_msg="Causal masking broken: earlier positions changed when last token was modified",
    )


# ── Decoder block ──────────────────────────────────────────────────────────────

def test_decoder_block_shape():
    cfg = TINY_CFG
    params = init_params(jax.random.PRNGKey(0), cfg)
    x = jax.random.normal(jax.random.PRNGKey(1), (2, cfg['ctx_len'], cfg['d_model']))
    out = decoder_block(x, params['blocks'][0], cfg['n_heads'], None, 0.0, False)
    assert out.shape == x.shape


# ── Full forward pass ──────────────────────────────────────────────────────────

def test_forward_output_shape():
    cfg = TINY_CFG
    params = init_params(jax.random.PRNGKey(0), cfg)
    x = jnp.zeros((2, cfg['ctx_len']), dtype=jnp.int32)
    logits = transformer_forward(
        params, x, cfg['n_heads'], cfg['n_layers'], 0.0, jax.random.PRNGKey(1), training=False,
    )
    assert logits.shape == (2, cfg['ctx_len'], cfg['vocab_size'])


def test_param_count():
    """Full model should have ~10M parameters (6L/6H/384d)."""
    params = init_params(jax.random.PRNGKey(0), FULL_CFG)
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    assert 9_000_000 < n_params < 12_000_000, f"Unexpected param count: {n_params:,}"


# ── Milestone gate: untrained loss ≈ ln(vocab_size) ───────────────────────────

def test_untrained_loss_near_log_vocab():
    """Untrained model loss must be close to ln(vocab_size) ≈ 4.17 for vocab=65.

    This is the roadmap milestone gate before starting the training loop.
    A value far from ln(vocab) indicates a shape or masking bug.
    """
    cfg = FULL_CFG
    params = init_params(jax.random.PRNGKey(42), cfg)

    B, T = 4, cfg['ctx_len']
    x = jax.random.randint(jax.random.PRNGKey(0), (B, T), 0, cfg['vocab_size'])
    y = jax.random.randint(jax.random.PRNGKey(1), (B, T), 0, cfg['vocab_size'])

    logits = transformer_forward(
        params, x, cfg['n_heads'], cfg['n_layers'], 0.0, jax.random.PRNGKey(2), training=False,
    )
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits.reshape(B * T, cfg['vocab_size']), y.reshape(B * T)
    ).mean()

    expected = float(jnp.log(cfg['vocab_size']))  # ≈ 4.174 for vocab=65
    assert abs(float(loss) - expected) < 0.5, (
        f"Untrained loss {float(loss):.4f} deviates too far from ln(vocab)={expected:.4f}. "
        "Check causal mask, embedding init, and LM head shapes."
    )
