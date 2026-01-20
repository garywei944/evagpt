import logging
import math
from typing import Callable

import attrs
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Float, Array, jaxtyped, Int, PRNGKeyArray
from beartype import beartype as typechecker
import optax

logger = logging.getLogger(__name__)


@attrs.define
class GPTConfig:
    block_size: int = 1024
    # GPT-2 vocab size of 50257, rounded up to nearest multiple of 64 for efficiency
    vocab_size: int = 50304
    n_layers: int = 12
    n_heads: int = 12
    n_embd: int = 768
    dropout: float = 0.1
    bias: bool = True


class MLP(eqx.Module):
    config: GPTConfig
    c_fc: eqx.nn.Linear
    c_proj: eqx.nn.Linear
    dropout: eqx.nn.Dropout

    def __init__(self, config: GPTConfig, *, key: PRNGKeyArray):
        super().__init__()

        self.config = config
        key, key_fc, key_proj = jrandom.split(key, 3)
        std_init = jax.nn.initializers.truncated_normal(0.02)
        scaled_init = jax.nn.initializers.truncated_normal(0.02 / math.sqrt(2 * config.n_layers))

        self.c_fc = eqx.nn.Linear(
            config.n_embd, 4 * config.n_embd, use_bias=config.bias, key=key_fc
        )
        # replace weight with standard initialization
        self.c_fc = eqx.tree_at(
            lambda m: m.weight,
            self.c_fc,
            std_init(key_fc, self.c_fc.weight.shape, self.c_fc.weight.dtype),
        )
        self.c_fc = eqx.tree_at(
            lambda m: m.bias,
            self.c_fc,
            jnp.zeros((4 * config.n_embd,), dtype=self.c_fc.weight.dtype),
        )

        self.c_proj = eqx.nn.Linear(
            4 * config.n_embd, config.n_embd, use_bias=config.bias, key=key_proj
        )
        # apply special scaled initialization, as in GPT-2 paper
        self.c_proj = eqx.tree_at(
            lambda m: m.weight,
            self.c_proj,
            scaled_init(key_proj, self.c_proj.weight.shape, self.c_proj.weight.dtype),
        )
        self.c_proj = eqx.tree_at(
            lambda m: m.bias,
            self.c_proj,
            jnp.zeros((config.n_embd,), dtype=self.c_proj.weight.dtype),
        )

        self.dropout = eqx.nn.Dropout(p=config.dropout)

    @jaxtyped(typechecker=typechecker)
    def __call__(self, x: Float[Array, "*B T C"], *, key: PRNGKeyArray) -> Float[Array, "*B T C"]:
        x = self.c_fc(x)
        x = jax.nn.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x, key=key)
        return x


class CausalSelfAttention(eqx.Module):
    config: GPTConfig

    c_attn: eqx.nn.Linear
    c_proj: eqx.nn.Linear
    dropout: eqx.nn.Dropout

    def __init__(self, config: GPTConfig, *, key: PRNGKeyArray):
        super().__init__()

        self.config = config
        key, key_c_attn, key_c_proj = jrandom.split(key, 3)
        std_init = jax.nn.initializers.truncated_normal(0.02)
        scaled_init = jax.nn.initializers.truncated_normal(0.02 / math.sqrt(2 * config.n_layers))

        # QKV projection
        self.c_attn = eqx.nn.Linear(
            config.n_embd, 3 * config.n_embd, use_bias=config.bias, key=key_c_attn
        )
        self.c_attn = eqx.tree_at(
            lambda m: m.weight,
            self.c_attn,
            std_init(key_c_attn, self.c_attn.weight.shape, self.c_attn.weight.dtype),
        )
        self.c_attn = eqx.tree_at(
            lambda m: m.bias,
            self.c_attn,
            jnp.zeros((3 * config.n_embd,), dtype=self.c_attn.weight.dtype),
        )

        self.c_proj = eqx.nn.Linear(
            config.n_embd, config.n_embd, use_bias=config.bias, key=key_c_proj
        )
        self.c_proj = eqx.tree_at(
            lambda m: m.weight,
            self.c_proj,
            scaled_init(key_c_proj, self.c_proj.weight.shape, self.c_proj.weight.dtype),
        )
        self.c_proj = eqx.tree_at(
            lambda m: m.bias,
            self.c_proj,
            jnp.zeros((config.n_embd,), dtype=self.c_proj.weight.dtype),
        )

        self.dropout = eqx.nn.Dropout(p=config.dropout)

    @jaxtyped(typechecker=typechecker)
    def __call__(self, x: Float[Array, "*B T C"], *, key: PRNGKeyArray) -> Float[Array, "*B T C"]:
        B, T, C = x.shape
        nh, hs = self.config.n_heads, C // self.config.n_heads

        q, k, v = jnp.split(self.c_attn(x), 3, axis=-1)
        q = q.reshape(B, T, nh, hs)
        k = k.reshape(B, T, nh, hs)
        v = v.reshape(B, T, nh, hs)

        # out: (B, T, nh, hs)
        out = jax.nn.dot_product_attention(q, k, v, is_causal=True, implementation="cudnn")
        out = out.reshape(B, T, C)
        out = self.dropout(out, key=key)
        return out


class Block(eqx.Module):
    ln1: eqx.nn.LayerNorm
    ln1_batched: Callable
    attn: CausalSelfAttention
    ln2: eqx.nn.LayerNorm
    ln2_batched: Callable
    mlp: MLP

    def __init__(self, config: GPTConfig, *, key: PRNGKeyArray):
        super().__init__()
        key, key_attn, key_mlp = jrandom.split(key, 3)
        self.ln1 = eqx.nn.LayerNorm(shape=config.n_embd, use_bias=config.bias)
        self.ln1_batched = jax.vmap(jax.vmap(self.ln1))
        self.attn = CausalSelfAttention(config=config, key=key_attn)
        self.ln2 = eqx.nn.LayerNorm(shape=config.n_embd, use_bias=config.bias)
        self.ln2_batched = jax.vmap(jax.vmap(self.ln2))
        self.mlp = MLP(config=config, key=key_mlp)

    @jaxtyped(typechecker=typechecker)
    def __call__(self, x: Float[Array, "*B T C"], *, key: PRNGKeyArray) -> Float[Array, "*B T C"]:
        key, key_attn, key_mlp = jrandom.split(key, 3)
        # x = x + self.attn(jax.vmap(jax.vmap(self.ln1))(x), key=key_attn)
        # x = x + self.mlp(jax.vmap(jax.vmap(self.ln2))(x), key=key_mlp)
        x = x + self.attn(self.ln1_batched(x), key=key_attn)
        x = x + self.mlp(self.ln2_batched(x), key=key_mlp)
        return x


class GPT2(eqx.Module):
    config: GPTConfig
    wte: eqx.nn.Embedding
    wpe: eqx.nn.Embedding
    dropout: eqx.nn.Dropout
    h: list[Block]
    ln_f: eqx.nn.LayerNorm
    ln_f_batched: Callable

    def __init__(self, config: GPTConfig, *, key: PRNGKeyArray):
        super().__init__()

        keys = jrandom.split(key, 2 + config.n_layers)
        init = jax.nn.initializers.truncated_normal(0.02)

        self.config = config
        self.wte = eqx.nn.Embedding(weight=init(keys[0], (config.vocab_size, config.n_embd)))
        self.wpe = eqx.nn.Embedding(weight=init(keys[1], (config.block_size, config.n_embd)))
        self.dropout = eqx.nn.Dropout(p=config.dropout)
        self.h = [Block(config=config, key=keys[i + 2]) for i in range(config.n_layers)]
        self.ln_f = eqx.nn.LayerNorm(shape=config.n_embd, use_bias=config.bias)
        self.ln_f_batched = jax.vmap(jax.vmap(self.ln_f))

    @jaxtyped(typechecker=typechecker)
    def __call__(
        self,
        input_ids: Int[Array, "*B T"],
        labels: Int[Array, "*B T"] | None = None,
        *,
        key: PRNGKeyArray,
    ) -> tuple[Float[Array, "*B T C"], Float[Array, "*B T"] | None]:
        B, T = input_ids.shape
        keys = jrandom.split(key, self.config.n_layers + 1)

        tok_emb = jax.vmap(jax.vmap(self.wte))(input_ids)
        pos_emb = jax.vmap(self.wpe)(jnp.arange(T))
        x = self.dropout(tok_emb + pos_emb, key=keys[0])

        for i, block in enumerate(self.h):
            x = block(x, key=keys[i + 1])
        x = self.ln_f_batched(x)

        logits = jnp.einsum("btc,vc->btv", x, self.wte.weight)

        # compute loss
        if labels is None:
            return logits, None

        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        loss = optax.softmax_cross_entropy_with_integer_labels(shift_logits, shift_labels)
        return logits, loss.mean()


if __name__ == "__main__":
    key = jrandom.key(0)
    config = GPTConfig()

    key, key_model, key_data = jrandom.split(key, 3)

    model = GPT2(config=config, key=key_model)
    x = jrandom.randint(key_data, (2, 1024), 0, config.vocab_size)
    y = x

    logits, loss = model(x, labels=y, key=key)
    print("logits shape:", logits.shape)
    print("loss:", loss)

    print(logits[:, :10, :10])
