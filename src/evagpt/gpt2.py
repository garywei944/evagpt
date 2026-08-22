import attrs
import torch
from jaxtyping import Float
from torch import Tensor, nn
from torch.nn import functional as F


@attrs.define(frozen=True)
class GPT2Config:
    block_size: int = 1024
    vacab_size: int = 50304
    n_layers: int = 12
    n_heads: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True


class CausalSelfAttengion(nn.Module):
    bias: Float[Tensor, "1 1 T T"]

    def __init__(self, *, config: GPT2Config):
        self.config = config

        self.c_attn = nn.Linear(config.n_embd, config.n_embd * 3, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.register_buffer("bias", torch.tril(torch.ones(1, 1, config.block_size, config.block_size)))

    def forward(self, x: Float[Tensor, "B T C"]) -> Float[Tensor, "B T C"]:
        B, T, C = x.shape
        nh, hs = self.config.n_heads, C // self.config.n_heads

        # QKV are not contiguous
        q, k, v = self.c_attn(x).chunk(3, dim=-1)
        q = q.view(B, T, nh, hs).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, nh, hs).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, nh, hs).transpose(1, 2)  # (B, nh, T, hs)

        # attn is contiguous
        attn = torch.einsum("...ij,...kj->...ik", q, k) / hs**0.5
        attn = attn.masked_fill(self.bias == 0, -torch.inf)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        y = attn @ v  # (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.resid_dropout(self.c_proj(attn))

        return y
