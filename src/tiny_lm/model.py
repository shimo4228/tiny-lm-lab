"""A GPT-style decoder-only transformer written from primitives.

Learning-first implementation: Embedding / Linear / LayerNorm / causal
self-attention / MLP are all hand-written from mlx.core tensor ops.
`mlx.nn.Module` is used only for parameter registration, and autograd /
AdamW come from mlx — the forward math is the point here.

Shapes follow the conventional (B, T, C): batch, time (sequence), channels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int
    ctx_len: int
    d_model: int
    n_heads: int
    n_layers: int

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )


class Linear(nn.Module):
    """y = x @ W^T + b, initialized N(0, 0.02) like GPT-2."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.weight = 0.02 * mx.random.normal((out_features, in_features))
        if bias:
            self.bias = mx.zeros((out_features,))

    def __call__(self, x: mx.array) -> mx.array:
        y = x @ self.weight.T
        if "bias" in self:
            y = y + self.bias
        return y


class Embedding(nn.Module):
    """Token-id -> vector lookup table."""

    def __init__(self, num_embeddings: int, dim: int) -> None:
        super().__init__()
        self.weight = 0.02 * mx.random.normal((num_embeddings, dim))

    def __call__(self, ids: mx.array) -> mx.array:
        return self.weight[ids]


class LayerNorm(nn.Module):
    """Normalize the channel dimension to zero mean / unit variance, then scale+shift."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.gamma = mx.ones((dim,))
        self.beta = mx.zeros((dim,))
        self._eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        mean = x.mean(axis=-1, keepdims=True)
        var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
        return self.gamma * (x - mean) * mx.rsqrt(var + self._eps) + self.beta


class CausalSelfAttention(nn.Module):
    """Multi-head scaled dot-product attention with a causal (lower-triangular) mask."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.qkv = Linear(config.d_model, 3 * config.d_model)
        self.proj = Linear(config.d_model, config.d_model)
        # leading underscore keeps this out of parameters() — a constant, not a weight
        self._causal = mx.tril(mx.ones((config.ctx_len, config.ctx_len)), k=0)

    def __call__(self, x: mx.array) -> mx.array:
        out, _ = self.forward_with_weights(x)
        return out

    def forward_with_weights(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """Return (output (B,T,C), attention weights (B,H,T,T))."""
        B, T, C = x.shape
        qkv = self.qkv(x)  # (B, T, 3C)
        q, k, v = qkv[..., :C], qkv[..., C : 2 * C], qkv[..., 2 * C :]
        # (B, T, C) -> (B, H, T, head_dim)
        q = q.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

        # scaled dot-product: (B, H, T, T)
        scores = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(self.head_dim)
        # causal mask: position t may attend to positions <= t only
        causal = self._causal[:T, :T]
        scores = mx.where(causal == 1, scores, mx.array(-mx.inf))
        weights = mx.softmax(scores, axis=-1)

        out = weights @ v  # (B, H, T, head_dim)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.proj(out), weights


class MLP(nn.Module):
    """Position-wise feed-forward: expand 4x, GELU, project back."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.fc = Linear(config.d_model, 4 * config.d_model)
        self.proj = Linear(4 * config.d_model, config.d_model)

    def __call__(self, x: mx.array) -> mx.array:
        return self.proj(nn.gelu(self.fc(x)))


class Block(nn.Module):
    """Pre-norm transformer block: x + attn(ln(x)), then x + mlp(ln(x))."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln1 = LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2 = LayerNorm(config.d_model)
        self.mlp = MLP(config)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.tok_emb = Embedding(config.vocab_size, config.d_model)
        self.pos_emb = Embedding(config.ctx_len, config.d_model)
        self.blocks = [Block(config) for _ in range(config.n_layers)]
        self.ln_f = LayerNorm(config.d_model)
        self.lm_head = Linear(config.d_model, config.vocab_size, bias=False)
        # GPT-2 residual scaling: each block adds two residual contributions,
        # so scale their output projections by 1/sqrt(2*n_layers) to keep the
        # residual-stream variance from growing with depth
        scale = 1.0 / math.sqrt(2 * config.n_layers)
        for block in self.blocks:
            block.attn.proj.weight = block.attn.proj.weight * scale
            block.mlp.proj.weight = block.mlp.proj.weight * scale

    def _embed(self, ids: mx.array) -> mx.array:
        T = ids.shape[1]
        if self.config.ctx_len < T:
            raise ValueError(f"sequence length {T} exceeds ctx_len {self.config.ctx_len}")
        return self.tok_emb(ids) + self.pos_emb(mx.arange(T))

    def __call__(self, ids: mx.array) -> mx.array:
        """ids (B, T) -> logits (B, T, vocab_size)."""
        x = self._embed(ids)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.ln_f(x))

    def attention_maps(self, ids: mx.array) -> list[mx.array]:
        """Per-layer attention weights for the first batch element: list of (H, T, T)."""
        x = self._embed(ids)
        maps: list[mx.array] = []
        for block in self.blocks:
            attn_out, weights = block.attn.forward_with_weights(block.ln1(x))
            maps.append(weights[0])
            x = x + attn_out
            x = x + block.mlp(block.ln2(x))
        return maps

    def num_params(self) -> int:
        flat = tree_flatten(self.parameters())
        pairs = flat.items() if isinstance(flat, dict) else flat
        return sum(int(p.size) for _, p in pairs)


def cross_entropy_loss(logits: mx.array, targets: mx.array) -> mx.array:
    """Mean negative log-likelihood, computed from logsumexp (no nn.losses)."""
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    picked = mx.take_along_axis(log_probs, targets[..., None], axis=-1)
    return -picked.mean()
