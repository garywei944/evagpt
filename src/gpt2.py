import torch
from torch import nn
import torch.nn.functional as F

from dataclasses import dataclass
import math


__all__ = ["GPT2Config", "MLP", "CausalSelfAttention", "Block", "GPT2"]


@dataclass
class GPT2Config:
    block_size: int = 1024
    vocab_size: int = 50257  # GPT-2 vocab size
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1
    bias: bool = True


class MLP(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_embd * 4, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd * 4, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)

        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        self.c_attn = nn.Linear(self.n_embd, self.n_embd * 3, bias=config.bias)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=config.bias)
        # self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)

        # self.causal_mask = ~torch.tril(
        #     torch.ones(self.config.block_size, self.config.block_size)
        # ).bool()

    def forward(self, x):
        B, T, C = x.shape
        nh, hs = self.n_head, C // self.n_head

        q, k, v = self.c_attn(x).chunk(3, dim=-1)
        q = q.view(B, T, nh, hs).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, nh, hs).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, nh, hs).transpose(1, 2)  # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0, is_causal=True
        )  # (B, nh, T, hs)
        y = y.transpose(1, 2).reshape(B, T, C)

        return self.resid_dropout(self.c_proj(y))


class Block(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT2(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd, bias=config.bias),
            )
        )

        # initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if module._get_name() == "c_proj":
                nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=0.02 / math.sqrt(2 * self.config.n_layer),
                )
            else:
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, input_ids, attention_mask=None, labels=None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = input_ids.shape
        device = input_ids.device

        tok_emb = self.transformer.wte(input_ids)
        pos_emb = self.transformer.wpe(torch.arange(T, dtype=torch.long, device=device))
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if labels is not None:
            x = x @ self.transformer.wte.weight.T
            shift_logits = x[..., :-1, :]
            shift_labels = labels[..., 1:]
            loss = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
            )
            return x, loss
        x = F.linear(x[:, [-1], :], self.transformer.wte.weight)
        return x, None


if __name__ == "__main__":
    config = GPT2Config(
        block_size=1024,
        vocab_size=50257,  # GPT-2 vocab size
        n_layer=12,
        n_head=12,
        n_embd=768,
        dropout=0.1,
        bias=True,
    )
    model = GPT2(config)
    input_ids = torch.randint(
        0, config.vocab_size, (2, 1024)
    )  # Batch size of 2, sequence length of 20
    logits, loss = model(input_ids, labels=input_ids)
    print(logits.shape, loss)
    print(loss)
