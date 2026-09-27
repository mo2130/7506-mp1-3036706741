"""Student model: a small GPT with RMSNorm, rotary positions (RoPE) and SwiGLU.

The course harness only requires two interfaces:

    build_model(config) -> model whose ``.context`` is 256
    model(ids)                  -> unnormalized logits, shape [batch, time, vocab]
    model.predict_log_probs(ids) -> log-probabilities, shape [batch, time, vocab]

Configuration (see ``configs/student.json``):

    vocab, width, heads, depth, context : same meaning as the baseline config
    norm      : "rms" (default) | "layer"   -> ablation switch for normalization
    pos       : "rope" (default) | "learned" -> ablation switch for positions
    mlp       : "swiglu" (default) | "gelu"  -> ablation switch for the MLP block
    dropout   : float, default 0.0
    expansion : SwiGLU hidden multiplier, default 8/3 (rounded to a multiple of 8)

Every variant keeps the same parameter count to within a few percent, so the
ablations compare mechanisms rather than model size.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    """LayerNorm without mean-centering or bias: cheaper, and it works as well."""

    def __init__(self, width, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


class Rotary(nn.Module):
    """Rotary position embedding: rotates query/key pairs by an angle that depends on
    the absolute position, so the model gets position information without learned
    position vectors, and attention depends only on relative distances."""

    def __init__(self, head_dim, context, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(context).float()
        angles = torch.outer(positions, inv_freq)  # [context, head_dim // 2]
        self.register_buffer('cos', angles.cos(), persistent=False)
        self.register_buffer('sin', angles.sin(), persistent=False)

    def forward(self, x):
        # x: [batch, heads, time, head_dim]
        length = x.shape[-2]
        cos = self.cos[:length][None, None]
        sin = self.sin[:length][None, None]
        even, odd = x[..., 0::2], x[..., 1::2]
        rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
        return rotated.flatten(-2)


class Block(nn.Module):
    def __init__(self, width, heads, norm, mlp, expansion, dropout):
        super().__init__()
        self.heads = heads
        self.norm1 = norm(width)
        self.norm2 = norm(width)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.proj = nn.Linear(width, width, bias=False)
        if mlp == 'swiglu':
            hidden = int(round(expansion * width / 8)) * 8
            self.gate = nn.Linear(width, hidden, bias=False)
            self.up = nn.Linear(width, hidden, bias=False)
            self.down = nn.Linear(hidden, width, bias=False)
        else:  # plain GELU MLP, the baseline block
            self.up = nn.Linear(width, 4 * width)
            self.act = nn.GELU()
            self.down = nn.Linear(4 * width, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, rope):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(
            batch, length, 3, self.heads, width // self.heads).permute(2, 0, 3, 1, 4)
        if rope is not None:
            q, k = rope(q), rope(k)
        # Causal: position t attends to positions <= t only.
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.dropout(self.proj(attended.transpose(1, 2).reshape(batch, length, width)))
        hidden = self.norm2(x)
        if hasattr(self, 'gate'):
            hidden = F.silu(self.gate(hidden)) * self.up(hidden)
        else:
            hidden = self.act(self.up(hidden))
        return x + self.dropout(self.down(hidden))


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.context = config['context']
        width = config['width']
        depth = config['depth']
        norm_name = config.get('norm', 'rms')
        mlp_name = config.get('mlp', 'swiglu')
        position = config.get('pos', 'rope')
        dropout = float(config.get('dropout', 0.0))
        expansion = float(config.get('expansion', 8 / 3))
        norm = (lambda w: RMSNorm(w)) if norm_name == 'rms' else (lambda w: nn.LayerNorm(w))

        self.token = nn.Embedding(config['vocab'], width)
        self.pos = nn.Embedding(self.context, width) if position == 'learned' else None
        self.rope = Rotary(width // config['heads'], self.context) if position == 'rope' else None
        self.blocks = nn.ModuleList(
            [Block(width, config['heads'], norm, mlp_name, expansion, dropout) for _ in range(depth)])
        self.norm = norm(width)
        self.head = nn.Linear(width, config['vocab'], bias=False)
        self.apply(self.initialize)
        # Shrink the residual projections so that the deep residual stream starts small.
        for block in self.blocks:
            nn.init.normal_(block.proj.weight, std=.02 / math.sqrt(2 * depth))
            nn.init.normal_(block.down.weight, std=.02 / math.sqrt(2 * depth))
        # Weight tying: the output head reuses the token embedding matrix.
        self.head.weight = self.token.weight

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=.02)
            if getattr(module, 'bias', None) is not None:
                nn.init.zeros_(module.bias)

    def features(self, ids):
        x = self.token(ids)
        if self.pos is not None:
            x = x + self.pos(torch.arange(ids.shape[1], device=ids.device))
        for block in self.blocks:
            x = block(x, self.rope)
        return self.norm(x)

    def forward(self, ids):
        """Training interface: unnormalized next-token logits [batch, time, vocab]."""
        return self.head(self.features(ids))

    def predict_log_probs(self, ids):
        """Evaluation interface: normalized log probabilities, strictly causal.

        No state is carried between calls, so independent evaluation windows
        cannot influence each other.
        """
        return F.log_softmax(self(ids).float(), dim=-1)


def build_model(config):
    return GPT(config)
