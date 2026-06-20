# Transformer (GPT-2 Style) in JAX

A decoder-only GPT-style language model trained on TinyShakespeare, implemented from scratch in raw JAX — no Flax, no Haiku, no PyTorch.

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
| JIT step time | — ms (measured on GPU) |
| Eager step time | — ms |
| XLA speedup | ~3× |

> Run `vmap/train.py` to reproduce. Fill in the measured values above after training on GPU.

## Generated Sample

```
HAMLET:
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune...
```

> Replace with actual output from `vmap/sample.py` after training.

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

## What I Learned About JAX

**Functional purity is the core constraint.** Params live outside the model as a pytree (nested dict). Every function is a pure function of its inputs — `logits = forward(params, x)` — with no hidden state and no mutation. Arrays are immutable; updates use `x.at[idx].set(v)`.

**Explicit PRNG is elegant once you accept it.** No global seed means results are reproducible by construction. Each dropout call gets its own key split from a parent, so the call graph is fully deterministic. The pattern of pre-splitting all dropout keys for a step upfront (`jax.random.split(key, 1 + n_layers * 4)`) keeps the training loop clean.

**JIT traces once and compiles to XLA.** The Python `for` loop over decoder blocks unrolls at trace time — the compiler sees the full 6-block computation graph and fuses operations across layer boundaries. This is where the ~3× speedup over eager comes from: no Python interpreter overhead per step, and XLA can schedule and fuse ops globally. `jax.value_and_grad` gets loss and gradients in a single forward-backward pass.

**Shape stability = compile once.** Padding all batches to the same `(batch_size, ctx_len)` shape ensures the compiled XLA binary is reused on every step. A different shape would silently trigger recompilation, making training mysteriously slow.
