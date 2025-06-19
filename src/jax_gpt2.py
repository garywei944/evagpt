from dataclasses import dataclass
import jax.numpy as jnp
import jax.random as jrandom
from flax import linen as nn
from flax.core.frozen_dict import freeze, unfreeze, FrozenDict
from jaxtyping import Array, Float

__all__ = ["GPT2Config", "MLP", "CausalSelfAttention", "Block", "GPT"]


@dataclass
class GPT2Config:
    block_size: int = 1024
    vocab_size: int = 50257  # GPT-2 vocab size
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class MLP(nn.Module):
    config: GPT2Config

    def setup(self):
        self.c_fc = nn.Dense(self.config.n_embd * 4)
        self.c_proj = nn.Dense(self.config.n_embd)

    def __call__(self, x: Float[Array, "B T C"]) -> Float[Array, "B T C"]:
        x = self.c_fc(x)
        x = nn.gelu(x)
        x = self.c_proj(x)
        return x


class CausalSelfAttention(nn.Module):
    config: GPT2Config

    def setup(self):
        assert self.config.n_embd % self.config.n_head == 0

        self.n_head = self.config.n_head
        self.n_embd = self.config.n_embd

        # q, k, v for all heads
        self.c_attn = nn.Dense(self.n_embd * 3)
        # output projection
        self.c_proj = nn.Dense(self.n_embd)

        self.causal_mask = ~jnp.tril(
            jnp.ones((self.config.block_size, self.config.block_size), dtype=jnp.bool)
        )

    def __call__(self, x: Float[Array, "B T C"]) -> Float[Array, "B T C"]:
        B, T, C = x.shape
        hs = C // self.n_head

        qkv = self.c_attn(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q = q.reshape(B, T, self.n_head, hs)  # (B, T, nh, hs)
        k = k.reshape(B, T, self.n_head, hs)  # (B, T, nh, hs)
        v = v.reshape(B, T, self.n_head, hs)  # (B, T, nh, hs)

        att = jnp.einsum("bihc,bjhc->bhij", q, k) / jnp.sqrt(hs)
        att = jnp.where(self.causal_mask[:T, :T], -jnp.inf, att)
        att = nn.softmax(att, axis=-1)
        y = jnp.einsum("bhij,bjhc->bihc", att, v).reshape(B, T, C)
        y = self.c_proj(y)

        return y


class Block(nn.Module):
    config: GPT2Config

    def setup(self):
        self.ln_1 = nn.LayerNorm()
        self.attn = CausalSelfAttention(self.config)
        self.ln_2 = nn.LayerNorm()
        self.mlp = MLP(self.config)

    def __call__(self, x: Float[Array, "B T C"]) -> Float[Array, "B T C"]:
        x += self.attn(self.ln_1(x))
        x += self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    config: GPT2Config

    def setup(self):
        self.wte = nn.Embed(self.config.vocab_size, self.config.n_embd)
        self.wpe = nn.Embed(self.config.block_size, self.config.n_embd)

        self.h = [Block(self.config) for _ in range(self.config.n_layer)]
        self.ln_f = nn.LayerNorm()
        # self.lm_head = nn.Dense(self.config.vocab_size, use_bias=False)

    def __call__(
        self, input_ids: Float[Array, "B T"], *args, **kwargs
    ) -> Float[Array, "B T V"]:
        *_, T = input_ids.shape

        assert T <= self.config.block_size, "Input sequence length exceeds block size"

        tok_emb = self.wte(input_ids)
        pos = jnp.arange(T)
        pos_emb = self.wpe(pos)
        x = tok_emb + pos_emb

        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        # logits = self.lm_head(x)

        # implement lm_head from self.wte embeddings
        logits = jnp.einsum("btc,vc->btv", x, self.wte.embedding)

        return logits

    @classmethod
    def from_pretrained(cls, model_type: str):
        assert model_type in ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]

        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]
        config = GPT2Config(**config_args)
        model = cls(config)

        # 2. load pretrained weights from Hugging Face
        from transformers import FlaxAutoModelForCausalLM

        hf_model = FlaxAutoModelForCausalLM.from_pretrained(model_type)

        hf_params = unfreeze(hf_model.params)
        params = {}

        # embeddings + final LN
        params["wte"] = hf_params["transformer"]["wte"]
        params["wpe"] = hf_params["transformer"]["wpe"]
        params["ln_f"] = hf_params["transformer"]["ln_f"]

        for i, hf_blk in hf_params["transformer"]["h"].items():
            hf_blk["attn"]["c_attn"]["kernel"] = hf_blk["attn"]["c_attn"]["kernel"].T
            hf_blk["attn"]["c_proj"]["kernel"] = hf_blk["attn"]["c_proj"]["kernel"].T
            hf_blk["mlp"]["c_fc"]["kernel"] = hf_blk["mlp"]["c_fc"]["kernel"].T
            hf_blk["mlp"]["c_proj"]["kernel"] = hf_blk["mlp"]["c_proj"]["kernel"].T
            params[f"h_{i}"] = hf_blk

        return model, freeze({"params": params})

    @classmethod
    def from_config(
        cls, config: GPT2Config, rng: jrandom.PRNGKey
    ) -> tuple["GPT", FrozenDict]:
        model = cls(config)
        params = model.init(rng, jnp.ones((1, config.block_size), jnp.int32))
        return model, freeze(params)
