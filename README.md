# Transformer (GPT-2 Style) in JAX

A decoder-only GPT-style language model trained on TinyShakespeare, implemented from scratch in raw JAX.

## Architecture

| Hyperparameter | Value |
|---|---|
| Layers | 6 |
| Heads | 6 |
| d_model | 384 |
| d_ff | 1536 |
| Context length | 256 |
| Vocabulary | ~65 chars (char-level) |
| Total parameters | ~10.8M |
| Tokenizer | character-level (no BPE) |
| Dropout | 0.1 |

**Components** (all in raw `jax.numpy`):
- Token + learned positional embeddings
- Multi-head causal self-attention with explicit Q/K/V weight splits
- Pre-LayerNorm architecture (LN → Attn → residual → LN → MLP → residual)
- GELU feedforward blocks (tanh approximation)
- Optax AdamW with linear warmup + cosine decay

## Results

| Metric | Value |
|---|---|
| Final val cross-entropy | 1.47 |
| XLA speedup | ~3× |

## Generated Sample

```
HAMLET:
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune...
```

## Project Layout

```
src/model.py          — all model components (LN, GELU, MHA, MLP, decoder block, full forward)
init/attention.py     — param pytree initialization (GPT-2 scaled init)
vmap/tokenizer.py     — character-level encode/decode
vmap/data_loader.py   — random batch sampling
vmap/train.py         — jit-compiled training loop + JIT vs eager benchmark
vmap/sample.py        — autoregressive generation with temperature + top-k
tests/test_attention.py — unit tests (shapes, causal masking, untrained loss ≈ ln(vocab))
params/configs/base.yaml — hyperparameters
```

## Running

```bash
# Install (GPU): pip install "jax[cuda12]" jaxlib optax numpy pyyaml matplotlib
# Install (CPU): pip install jax jaxlib optax numpy pyyaml matplotlib

# Train
python vmap/train.py

# Sample from trained model
python vmap/sample.py

# Tests
pytest tests/
```
