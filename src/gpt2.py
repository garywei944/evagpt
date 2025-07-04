import torch
from torch import nn
import torch.nn.functional as F

from dataclasses import dataclass
import math


__all__ = ["GPT2Config", "GPT2"]

CONFIG_ARGS = {
    "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
    "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
    "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
    "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
}


@dataclass
class GPT2Config:
    block_size: int = 1024
    vocab_size: int = 50257  # GPT-2 vocab size
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1
    bias: bool = True

    @classmethod
    def from_pretrained(cls, model_name: str):
        assert model_name in CONFIG_ARGS, f"Unknown model name: {model_name}"
        config = CONFIG_ARGS[model_name]
        return cls(**config)


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
        # apply special scaled init to the residual projections, per GPT-2 paper
        std = 0.02 / math.sqrt(2 * config.n_layer)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.trunc_normal_(p, mean=0.0, std=std, a=-2 * std, b=2 * std)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02, a=-0.04, b=0.04)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02, a=-0.04, b=0.04)
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()

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

        # x = x @ self.transformer.wte.weight.T
        x = F.linear(x, self.transformer.wte.weight, bias=None)
        if labels is not None:
            shift_logits = x[..., :-1, :]
            shift_labels = labels[..., 1:]
            loss = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
            )
            return x, loss
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

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.shape}")
