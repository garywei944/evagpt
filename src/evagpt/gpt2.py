import attrs
import torch
from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.nn import functional as F


@attrs.define(frozen=True)
class GPT2Config:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layers: int = 12
    n_heads: int = 12
    n_embd: int = 768

    ffn_bias: bool = False
    attn_bias: bool = False
    dropout: float = 0.0

    swiglu_limit: float = 10.0


class CausalSelfAttention(nn.Module):
    bias: Float[Tensor, "1 1 T T"]

    def __init__(self, *, config: GPT2Config):
        super().__init__()
        self.config = config

        if config.n_embd % config.n_heads != 0:
            raise ValueError("config.n_embd must be dividable by config.n_heads")

        self.c_attn = nn.Linear(config.n_embd, config.n_embd * 3, bias=config.attn_bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.attn_bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.register_buffer("bias", torch.tril(torch.ones(1, 1, config.block_size, config.block_size)))

    def forward(self, x: Float[Tensor, "B T C"]) -> Float[Tensor, "B T C"]:
        B, T, C = x.shape
        nh, hs = self.config.n_heads, C // self.config.n_heads

        # TODO: flash attention

        # QKV are not contiguous
        q, k, v = self.c_attn(x).chunk(3, dim=-1)
        q = q.view(B, T, nh, hs).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, nh, hs).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, nh, hs).transpose(1, 2)  # (B, nh, T, hs)

        # attn is contiguous
        attn = torch.einsum("...ij,...kj->...ik", q, k) / hs**0.5
        causal_mask = self.bias[:, :, :T, :T] == 0
        attn = attn.masked_fill(causal_mask, -torch.inf)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        y = attn @ v  # (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.resid_dropout(self.c_proj(y))

        return y


class SwiGLU(nn.Module):
    def __init__(self, *, config: GPT2Config):
        super().__init__()
        self.config = config

        self.gate_proj = nn.Linear(config.n_embd, config.n_embd * 4, bias=config.ffn_bias)
        self.up_proj = nn.Linear(config.n_embd, config.n_embd * 4, bias=config.ffn_bias)
        self.down_proj = nn.Linear(config.n_embd * 4, config.n_embd, bias=config.ffn_bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Float[Tensor, "B T C"]) -> Float[Tensor, "B T C"]:
        dtype = x.dtype
        # follows deepseek v4, convert to float32
        gate = self.gate_proj(x).float()
        up = self.up_proj(x).float()

        if self.config.swiglu_limit > 0:
            gate = torch.clamp(gate, max=self.config.swiglu_limit)
            up = torch.clamp(up, min=-self.config.swiglu_limit, max=self.config.swiglu_limit)

        out = up * F.silu(gate)
        out = self.down_proj(out.to(dtype=dtype))
        return self.dropout(out)


class Block(nn.Module):
    def __init__(self, *, config: GPT2Config):
        super().__init__()
        self.config = config

        self.attn = CausalSelfAttention(config=config)
        self.attn_norm = nn.RMSNorm([config.n_embd])
        self.mlp = SwiGLU(config=config)
        self.mlp_norm = nn.RMSNorm([config.n_embd])

    def forward(self, x: Float[Tensor, "B T C"]) -> Float[Tensor, "B T C"]:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))

        return x


class GPT2(nn.Module):
    def __init__(self, *, config: GPT2Config):
        super().__init__()
        self.config = config

        # transformer
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.h = nn.ModuleList([Block(config=config) for _ in range(config.n_layers)])
        self.rms_norm = nn.RMSNorm([config.n_embd])

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self, input_ids: Int[Tensor, "B T"], targets: Int[Tensor, "B T"] | None = None
    ) -> tuple[Float[Tensor, "B T V"], Float[Tensor, ""] | None]:
        _, T = input_ids.shape

        if T > self.config.block_size:
            raise RuntimeError("sequence length greater than the maximum context window")

        pos = torch.arange(T, dtype=torch.long, device=input_ids.device)
        tok_emb = self.wte(input_ids)
        pos_emb = self.wpe(pos)
        x = self.dropout(tok_emb + pos_emb)

        for block in self.h:
            x = block(x)
        x = self.rms_norm(x)  # (B, T, C)

        logits = self.lm_head(x)  # (B, T, V)

        if targets is None:
            return logits, None

        loss = F.cross_entropy(logits.reshape(-1, self.config.vocab_size), targets.reshape(-1))

        return logits, loss
