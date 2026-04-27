# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A hands-on Transformer learning repo implementing core architectures from scratch in PyTorch, with extensive Chinese comments explaining every tensor shape and design decision. This is an educational codebase, not a library.

## Running the Code

```bash
# Activate the virtual environment
source myenv/bin/activate

# Train & test encoder-decoder model (sequence sorting task)
python train.py

# Train & test decoder-only (GPT-style) model (same sorting task)
python decoder_only_train.py

# Run standalone demos
python transformer.py        # Basic transformer forward pass demo
python kv_cache.py           # KV cache mechanism demo (prefill + decode)
python gqa_cuda_simulate.py  # GQA Python simulation vs PyTorch reference

# Compile & run CUDA kernel (requires nvcc)
nvcc -o gqa_kernel gqa_cuda_kernel.cu && ./gqa_kernel
```

No test framework is used. Each script validates itself via accuracy checks or shape assertions when run directly.

## Architecture

Two parallel implementations of the same sorting task (sort random digit sequences), demonstrating architectural differences:

### Encoder-Decoder (`model.py` + `train.py`)
- `PositionalEncoding` (sinusoidal), `MultiHeadAttention`, `GroupedQueryAttention` (GQA), `FeedForward`
- `EncoderLayer` (self-attn), `DecoderLayer` (self-attn + cross-attn + FFN)
- `Transformer` class with `encode()`, `decode()`, `build_cache()`, `decode_one_step()`
- `train.py`: data generation, training loop, inference (with and without KV cache)

### Decoder-Only / GPT-style (`decoder_only_model.py` + `decoder_only_train.py`)
- Reuses same primitives but uses `GPTLayer` (self-attn + causal mask only, no cross-attention)
- `DecoderOnlyTransformer` with `build_cache()`, `decode_one_step()`
- Data format: prompt + SEP + sorted answer concatenated as one sequence
- Loss computed only on post-SEP tokens (prompt portion masked out)

### Supplementary Files
- `transformer.py`: Standalone simplified Transformer with inline test (d_model=512, separate src/tgt vocab)
- `kv_cache.py`: Minimal class demonstrating prefill/decode cache mechanics without full model
- `gqa_cuda_simulate.py`: Python simulation of GQA CUDA kernel logic, verified against PyTorch reference
- `gqa_cuda_kernel.cu`: Complete CUDA kernel for GQA decode attention with two-pass softmax, warp/block reductions, shared memory

## Key Design Patterns

- Vocab: PAD=0, BOS/EOS=1-2 (encoder-decoder) or EOS=1/SEP=2 (decoder-only), digits 0-9 mapped to token IDs 3-12
- `forward_cached()` methods on attention and layer classes handle KV cache inference; `update_cache` flag distinguishes self-attention (True) from cross-attention (False)
- GQA stores cache in compact `n_kv_head` format and uses `_repeat_kv()` (expand, not copy) only at compute time
- All comments are in Chinese with detailed tensor shape annotations at each step

## Environment

- Python 3.10 virtualenv at `myenv/`
- PyTorch 2.5.1+cu121, torchvision, torchaudio
- CUDA 12.1 (for GPU training and CUDA kernel)
